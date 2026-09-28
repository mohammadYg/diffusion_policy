"""
Back ported methods: call, set_attr from v0.26
Disabled auto-reset after done
Added render method.
"""


import numpy as np
import multiprocessing as mp
import time
import sys
from enum import Enum
from copy import deepcopy

from gym import logger
from gym.vector.vector_env import VectorEnv
from gym.error import (
    AlreadyPendingCallError,
    NoAsyncCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
)
from gym.vector.utils import (
    create_shared_memory,
    create_empty_array,
    write_to_shared_memory,
    read_from_shared_memory,
    concatenate,
    CloudpickleWrapper,
    clear_mpi_env_vars,
)

__all__ = ["AsyncVectorEnv"]


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorEnv(VectorEnv):
    """Vectorized environment that runs multiple environments in parallel. It
    uses `multiprocessing` processes, and pipes for communication.
    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.
    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.
    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.
    shared_memory : bool (default: `True`)
        If `True`, then the observations from the worker processes are
        communicated back through shared variables. This can improve the
        efficiency if the observations are large (e.g. images).
    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    context : str, optional
        Context for multiprocessing. If `None`, then the default context is used.
        Only available in Python 3.
    daemon : bool (default: `True`)
        If `True`, then subprocesses have `daemon` flag turned on; that is, they
        will quit if the head process quits. However, `daemon=True` prevents
        subprocesses to spawn children, so for some environments you may want
        to have it set to `False`
    worker : function, optional
        WARNING - advanced mode option! If set, then use that worker in a subprocess
        instead of a default one. Can be useful to override some inner vector env
        logic, for instance, how resets on done are handled. Provides high
        degree of flexibility and a high chance to shoot yourself in the foot; thus,
        if you are writing your own worker, it is recommended to start from the code
        for `_worker` (or `_worker_shared_memory`) method below, and add changes
    """

    def __init__(
        self,
        env_fns,
        dummy_env_fn=None,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
        tolerate_step_errors=False,
    ):
        """
        tolerate_step_errors : bool (default: `False`)
            If `False` (the default, and the only behavior available before
            this flag existed), a worker raising during `step()` (e.g. a
            MuJoCo NaN/Inf instability) still raises back to the caller and
            discards the whole batch's results for that step, exactly like
            upstream gym.vector - the safe default for any caller that
            doesn't inspect `infos[i]['crashed']`.
            If `True`, `step_wait()` instead isolates the crashed worker
            (closes its pipe, retires it permanently) and returns real
            results for every other worker plus a `done=True,
            infos[i]={'crashed': True, 'error': ...}` placeholder for the
            dead slot - see `step_wait()`'s docstring. Only pass `True` if
            the caller actually checks `infos[i]['crashed']` and excludes
            that slot's in-progress episode itself (e.g.
            `RobomimicLowdimRunner._run_streaming`); otherwise a crash will
            be silently miscounted as a normal episode that scored 0
            instead of being excluded or raised.
        """
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy
        self.tolerate_step_errors = tolerate_step_errors

        # Added dummy_env_fn to fix OpenGL error in Mujoco
        # disable any OpenGL rendering in dummy_env_fn, since it
        # will conflict with OpenGL context in the forked child process
        if dummy_env_fn is None:
            dummy_env_fn = env_fns[0]
        dummy_env = dummy_env_fn()
        self.metadata = dummy_env.metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space
        dummy_env.close()
        del dummy_env
        super(AsyncVectorEnv, self).__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.single_observation_space, n=self.num_envs, ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    _obs_buffer, self.single_observation_space, n=self.num_envs
                )
            except CustomSpaceError:
                raise ValueError(
                    "Using `shared_memory=True` in `AsyncVectorEnv` "
                    "is incompatible with non-standard Gym observation spaces "
                    "(i.e. custom spaces inheriting from `gym.Space`), and is "
                    "only compatible with default Gym spaces (e.g. `Box`, "
                    "`Tuple`, `Dict`) for batching. Set `shared_memory=False` "
                    "if you use custom observation spaces."
                )
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = _worker_shared_memory if self.shared_memory else _worker
        target = worker or target
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name="Worker<{0}>-{1}".format(type(self).__name__, idx),
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_observation_spaces()

        # Indices whose worker has permanently died (e.g. a MuJoCo NaN/Inf
        # crash during step_wait - see step_wait()/step_async() below). Once
        # True, that pipe is closed and set to None in self.parent_pipes;
        # callers must never target a dead index with step/reset/call again.
        self._failed = [False] * self.num_envs
        self._last_error = [None] * self.num_envs

    def seed(self, seeds=None):
        self._assert_is_running()
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `seed` while waiting "
                "for a pending call to `{0}` to complete.".format(self._state.value),
                self._state.value,
            )

        for pipe, seed in zip(self.parent_pipes, seeds):
            pipe.send(("seed", seed))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def reset_async(self):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `reset_async` while waiting "
                "for a pending call to `{0}` to complete".format(self._state.value),
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("reset", None))
        self._state = AsyncState.WAITING_RESET

    def reset_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `reset_wait` times out. If
            `None`, the call to `reset_wait` never times out.
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                "The call to `reset_wait` has timed out after "
                "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            self.observations = concatenate(
                results, self.observations, self.single_observation_space
            )

        return deepcopy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        """
        Parameters
        ----------
        actions : iterable of samples from `action_space`
            List of actions.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `step_async` while waiting "
                "for a pending call to `{0}` to complete.".format(self._state.value),
                self._state.value,
            )

        # Dead pipes (self._failed[i]) never receive a command again - the
        # worker process behind them has already exited (see step_wait()).
        for pipe, action in zip(self.parent_pipes, actions):
            if pipe is not None:
                pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `step_wait` times out. If
            `None`, the call to `step_wait` never times out.
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment. A
            slot whose worker has died (this call or a previous one) holds
            stale data from its last successful step - callers must ignore
            it (see `infos[i]['crashed']`).
        rewards : `np.ndarray` instance (dtype `np.float_`)
            A vector of rewards from the vectorized environment. `0.0` for
            a crashed slot.
        dones : `np.ndarray` instance (dtype `np.bool_`)
            A vector whose entries indicate whether the episode has ended.
            Always `True` for a crashed slot, so callers that harvest on
            `done` still see it - they must check `infos[i]['crashed']`
            to tell a real episode end from a lost one.
        infos : list of dict
            A list of auxiliary diagnostic information. A crashed slot's
            entry is `{'crashed': True, 'error': <str>}` instead of the
            env's own info dict.

        Unlike the upstream gym.vector implementation, a single worker
        crashing (e.g. a MuJoCo NaN/Inf instability) does NOT raise and does
        NOT discard the other workers' already-received results for this
        step: that worker's pipe is closed and permanently retired
        (`self._failed[i] = True`), its slot reports `done=True` with
        `infos[i]['crashed']=True` from here on, and every other slot's
        real result for this step is still returned normally. Callers are
        responsible for excluding crashed slots from whatever they're
        aggregating (see `RobomimicLowdimRunner._run_streaming`) rather
        than treating the whole batch as failed.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                "The call to `step_wait` has timed out after "
                "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
            )

        if not self.tolerate_step_errors:
            # Default, backward-compatible path: identical to upstream
            # gym.vector - any single worker's error raises and discards
            # the whole batch's results for this step. Safe for any caller
            # that doesn't inspect infos[i]['crashed'].
            results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
            self._raise_if_errors(successes)
            self._state = AsyncState.DEFAULT
            observations_list, rewards, dones, infos = zip(*results)

            if not self.shared_memory:
                self.observations = concatenate(
                    observations_list, self.observations, self.single_observation_space
                )

            return (
                deepcopy(self.observations) if self.copy else self.observations,
                np.array(rewards),
                np.array(dones, dtype=np.bool_),
                infos,
            )

        raw = [
            pipe.recv() if pipe is not None else (None, True)
            for pipe in self.parent_pipes
        ]
        results, successes = zip(*raw)
        self._state = AsyncState.DEFAULT

        newly_failed = [
            i for i, (pipe, ok) in enumerate(zip(self.parent_pipes, successes))
            if pipe is not None and not ok
        ]
        # error_queue delivers entries in wall-clock crash order, not in
        # ascending worker-index order, so if two workers crash within the
        # same step_wait() call, a positional `index == i` assumption can
        # pair the wrong worker's index with the wrong exception. Drain all
        # of this step's errors first, keyed by their own index, then look
        # each newly-failed slot up by that key.
        errors_by_index = {}
        for _ in newly_failed:
            index, exctype, value = self.error_queue.get()
            errors_by_index[index] = (exctype, value)
        for i in newly_failed:
            exctype, value = errors_by_index[i]
            logger.warn(
                "Worker-{0} crashed during step ({1}: {2}); excluding its "
                "in-progress episode and retiring that slot for the rest "
                "of this rollout.".format(i, exctype.__name__, value)
            )
            self.parent_pipes[i].close()
            self.parent_pipes[i] = None
            self._failed[i] = True
            self._last_error[i] = "{0}: {1}".format(exctype.__name__, value)

        placeholder_info = lambda i: {"crashed": True, "error": self._last_error[i]}
        observations_list, rewards, dones, infos = [], [], [], []
        for i, (pipe, result) in enumerate(zip(self.parent_pipes, results)):
            if pipe is None:
                # Either just failed above, or was already dead from an
                # earlier step_wait() call - its shared-memory observation
                # slot (if shared_memory=True) is simply stale and unused
                # by callers that check infos[i]['crashed'] first.
                observations_list.append(self.observations[i] if not self.shared_memory else None)
                rewards.append(0.0)
                dones.append(True)
                infos.append(placeholder_info(i))
            else:
                obs_i, reward_i, done_i, info_i = result
                observations_list.append(obs_i)
                rewards.append(reward_i)
                dones.append(done_i)
                infos.append(info_i)

        if not self.shared_memory:
            self.observations = concatenate(
                observations_list, self.observations, self.single_observation_space
            )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def close_extras(self, timeout=None, terminate=False):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `close` times out. If `None`,
            the call to `close` never times out. If the call to `close` times
            out, then all processes are terminated.
        terminate : bool (default: `False`)
            If `True`, then the `close` operation is forced and all processes
            are terminated.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    "Calling `close` while waiting for a pending "
                    "call to `{0}` to complete.".format(self._state.value)
                )
                function = getattr(self, "{0}_wait".format(self._state.value))
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for i, pipe in enumerate(self.parent_pipes):
            if pipe is None:
                # Permanently dead from an earlier crash (self._failed[i]) -
                # nothing to wait on for this slot, not a reason to time out.
                continue
            delta = max(end_time - time.perf_counter(), 0)
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _check_observation_spaces(self):
        self._assert_is_running()
        for pipe in self.parent_pipes:
            pipe.send(("_check_observation_space", self.single_observation_space))
        same_spaces, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        if not all(same_spaces):
            raise RuntimeError(
                "Some environments have an observation space "
                "different from `{0}`. In order to batch observations, the "
                "observation spaces from all environments must be "
                "equal.".format(self.single_observation_space)
            )

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                "Trying to operate on `{0}`, after a "
                "call to `close()`.".format(type(self).__name__)
            )

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for _ in range(num_errors):
            index, exctype, value = self.error_queue.get()
            logger.error(
                "Received the following error from Worker-{0}: "
                "{1}: {2}".format(index, exctype.__name__, value)
            )
            logger.error("Shutting down Worker-{0}.".format(index))
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

        logger.error("Raising the last exception back to the main process.")
        raise exctype(value)

    def _raise_if_errors_at(self, indices, successes):
        # Like _raise_if_errors, but for a call that only targeted `indices`
        # (not all num_envs pipes) - can't reuse _raise_if_errors, since its
        # num_errors = num_envs - sum(successes) assumes a full-vector call.
        if all(successes):
            return
        for i, success in zip(indices, successes):
            if not success:
                index, exctype, value = self.error_queue.get()
                logger.error(
                    "Received the following error from Worker-{0}: "
                    "{1}: {2}".format(index, exctype.__name__, value)
                )
                logger.error("Shutting down Worker-{0}.".format(index))
                self.parent_pipes[index].close()
                self.parent_pipes[index] = None
                raise exctype(value)

    def reset_at(self, indices):
        """Reset only the sub-environments at `indices`, leaving all others
        untouched - unlike reset()/reset_async(), which always reset the
        whole vector. Synchronous; must not be called while a step_async()/
        call_async() is pending. Returns the fresh observations for
        `indices`, in that order.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `reset_at` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )
        for i in indices:
            self.parent_pipes[i].send(("reset", None))
        results, successes = zip(*[self.parent_pipes[i].recv() for i in indices]) \
            if indices else ((), ())
        self._raise_if_errors_at(indices, successes)

        if self.shared_memory:
            # workers already wrote straight into the shared observation
            # buffer at their own index - just read those rows back.
            obs = deepcopy(self.observations) if self.copy else self.observations
            return [obs[i] for i in indices]
        return list(results)

    def call_at(self, indices, name: str, *args, **kwargs):
        """Call a method / get a property from only the sub-environments at
        `indices`, without touching or waiting on any other environment -
        unlike call()/call_each(), which always address the whole vector.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_at` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )
        for i in indices:
            self.parent_pipes[i].send(("_call", (name, args, kwargs)))
        results, successes = zip(*[self.parent_pipes[i].recv() for i in indices]) \
            if indices else ((), ())
        self._raise_if_errors_at(indices, successes)
        return list(results)

    def call_async(self, name: str, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            name: Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout = None) -> list:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to `step_wait` times out.
                If `None` (default), the call to `step_wait` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling `call_wait` without any prior call to `call_async`.
            TimeoutError: The call to `call_wait` has timed out after timeout second(s).
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def call(self, name: str, *args, **kwargs):
        """Call a method, or get a property, from each parallel environment.

        Args:
            name (str): Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Returns:
            List of the results of the individual calls to the method or property for each environment.
        """
        self.call_async(name, *args, **kwargs)
        return self.call_wait()
    

    def call_each(self, name: str, 
            args_list: list=None, 
            kwargs_list: list=None, 
            timeout = None):
        n_envs = len(self.parent_pipes)
        if args_list is None:
            args_list = [[]] * n_envs
        assert len(args_list) == n_envs

        if kwargs_list is None:
            kwargs_list = [dict()] * n_envs
        assert len(kwargs_list) == n_envs

        # send
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for i, pipe in enumerate(self.parent_pipes):
            pipe.send(("_call", (name, args_list[i], kwargs_list[i])))
        self._state = AsyncState.WAITING_CALL

        # receive
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results


    def set_attr(self, name: str, values):
        """Sets an attribute of the sub-environments.

        Args:
            name: Name of the property to be set in each individual environment.
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
            AlreadyPendingCallError: Calling `set_attr` while waiting for a pending call to complete.
        """
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `set_attr` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(("_setattr", (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def render(self, *args, **kwargs):
        return self.call('render', *args, **kwargs)



def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                pipe.send((observation, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                # if done:
                #     observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))

            elif command == "_check_observation_space":
                pipe.send((data == env.observation_space, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send((None, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                # if done:
                #     observation = env.reset()
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send(((None, reward, done, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_observation_space":
                pipe.send((data == observation_space, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()