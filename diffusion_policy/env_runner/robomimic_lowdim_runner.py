import os
import wandb
import gym
from gym import logger
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv, AsyncState
from mujoco_py.builder import MujocoException
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_pac_policy import BaseLowdimPacPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from mimicgen.envs.robosuite import *

def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env

class ObservationNoiseWrapper(gym.Wrapper):
    def __init__(self, env, relative_noise=0.0):
        super().__init__(env)
        self.relative_noise = relative_noise
        self.rng = None

    def configure(self, *, seed: int, enabled: bool):
        self.enabled = enabled
        if enabled:
            self.rng = np.random.RandomState(seed)

    def reset(self, **kwargs):
        obs = self.env.reset()
        return self._add_noise(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._add_noise(obs)
        return obs, reward, done, info

    def _add_noise(self, obs):
        if not self.enabled or self.relative_noise == 0:
            return obs

        obs = obs.copy()

        eps = 1e-6
        scale = self.relative_noise * np.maximum(np.abs(obs), eps)
        obs = obs + scale * self.rng.randn(*obs.shape)

        return obs

class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            noise_enabled=False,
            relative_noise=0.0,
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            env = RobomimicLowdimWrapper(
                    env=robomimic_env,
                    obs_keys=obs_keys,
                    init_state=None,
                    render_hw=render_hw,
                    render_camera_name=render_camera_name
                )
            
            env = ObservationNoiseWrapper(
                env,
                relative_noise=relative_noise
            )
                
            env = VideoRecordingWrapper(
                                env,
                                video_recoder=VideoRecorder.create_h264(
                                    fps=fps,
                                    codec="h264",
                                    input_pix_fmt="rgb24",
                                    crf=crf,
                                    thread_type="FRAME",
                                    thread_count=1
                                ),
                                file_path=None,
                                steps_per_render=steps_per_render
                            )

            return MultiStepWrapper(
                                env,
                                n_obs_steps=env_n_obs_steps,
                                n_action_steps=env_n_action_steps,
                                max_episode_steps=max_steps
                            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render, train_idx=train_idx):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        epoch = getattr(self, "current_epoch", 0)
                        name = self.make_video_filename(epoch=epoch,idx=train_idx)
                        # filename = pathlib.Path(output_dir).joinpath(
                        #     'media', wv.util.generate_id() + ".mp4")
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', name + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env.env, RobomimicLowdimWrapper)
                    env.env.env.env.init_state = init_state

                    # configure noise wrapper
                    noise_wrapper = env.env.env                           # unwrap MultiStep → Video → ObservationNoise
                    assert isinstance(noise_wrapper, ObservationNoiseWrapper)

                    noise_wrapper.configure(
                        seed=train_idx,
                        enabled=noise_enabled
                    )

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    epoch = getattr(self, "current_epoch", 0)
                    name = self.make_video_filename(
                        epoch=epoch,
                        idx=seed
                    )
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', name + ".mp4")
                    # filename = pathlib.Path(output_dir).joinpath(
                    #     'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env.env, RobomimicLowdimWrapper)
                env.env.env.env.init_state = None
                env.seed(seed)
                noise_wrapper = env.env.env
                assert isinstance(noise_wrapper, ObservationNoiseWrapper)

                noise_wrapper.configure(
                    seed=seed,
                    enabled=noise_enabled
                )

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        # tolerate_step_errors=True: _run_streaming() checks
        # infos[i]['crashed'] and excludes that slot's episode itself,
        # instead of the whole batch aborting on one worker's MuJoCo
        # NaN/Inf - see AsyncVectorEnv.__init__'s docstring for that flag.
        # (_run_chunked, used only by stateful policies which this project
        # doesn't evaluate, does NOT check infos[i]['crashed'] - if it ever
        # is used, this flag would need _run_chunked fixed to match first.)
        env = AsyncVectorEnv(env_fns, tolerate_step_errors=True)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.current_epoch = 0
        self.rng = torch.Generator(device="cuda")
        self.rng.manual_seed(100)

    def make_video_filename(self, *, epoch, idx):
            return f"epoch={epoch:04d}_seed={idx:05d}.mp4"
    
    def run(self, policy, stochastic=False):
        """Stateless policies (drift/diffusion/flow, plain or PAC) stream all
        n_inits rollouts through n_envs parallel workers continuously: as
        soon as one slot's episode ends, it's immediately reassigned the
        next pending init instead of waiting for every other slot in a
        fixed-size chunk to also finish - avoids the straggler problem where
        a chunk's wall-time is bounded by its slowest member. Policies that
        carry cross-step state across the whole batch (policy.is_stateful,
        e.g. RobomimicLowdimPolicy's RNN hidden state) can't have individual
        rows reassigned mid-batch without corrupting that state, so they
        keep the original synchronized chunk-by-chunk rollout.
        """
        # A previous run() call tolerated a worker crash and left that slot's
        # pipe permanently None. Every subsequent env.reset()/call_each(...)
        # sends to ALL pipes unconditionally, so without rebuilding here the
        # very next run() call would raise an uncaught AttributeError on the
        # dead pipe instead of a MujocoException - i.e. one tolerated crash
        # would otherwise silently turn into a hard failure of the next
        # rollout/checkpoint eval. `_failed` only gets set by a crash inside
        # step() (see AsyncVectorEnv.step_wait's tolerate branch) - a crash
        # inside reset_at/call_at/call_each (_raise_if_errors_at) instead
        # nulls the pipe without setting `_failed`, and can also leave
        # `_state` stuck away from DEFAULT, so check for those directly too.
        # A worker that dies in a way step_wait can't even parse (e.g. its
        # exception value fails to unpickle, or it's OOM-killed/segfaults
        # while idle) leaves none of the above set - `is_alive()` is the
        # only signal left for that case, so check the actual OS processes
        # too rather than trusting the higher-level bookkeeping alone.
        if (any(self.env._failed)
                or any(p is None for p in self.env.parent_pipes)
                or self.env._state != AsyncState.DEFAULT
                or not all(p.is_alive() for p in self.env.processes)):
            try:
                self.env.close(terminate=True)
            except Exception:
                # Best-effort: close() itself can fail if the env is in a
                # sufficiently broken state (see close_extras' own note on
                # draining a stuck call) - fall back to reaping the OS
                # processes by hand so this doesn't leak them, then rebuild
                # regardless, since env_fns is enough to construct a fresh
                # env independent of the old one's state.
                for process in self.env.processes:
                    if process.is_alive():
                        process.terminate()
                for process in self.env.processes:
                    process.join(timeout=5)
            self.env = AsyncVectorEnv(self.env_fns, tolerate_step_errors=True)
        if getattr(policy, 'is_stateful', False):
            all_rewards = self._run_chunked(policy, stochastic)
        else:
            all_rewards = self._run_streaming(policy, stochastic)
        return self._aggregate_log(policy, stochastic, all_rewards)

    def _predict_and_step(self, policy, obs, past_action, stochastic):
        """One policy-inference + env.step() round for the whole vector."""
        np_obs_dict = {
            # handle n_latency_steps by discarding the last n_latency_steps
            'obs': obs[:,:self.n_obs_steps].astype(np.float32)
        }
        if self.past_action and (past_action is not None):
            # TODO: not tested
            np_obs_dict['past_action'] = past_action[
                :,-(self.n_obs_steps-1):].astype(np.float32)

        obs_dict = dict_apply(np_obs_dict,
            lambda x: torch.from_numpy(x).to(device=policy.device))

        with torch.no_grad():
            if isinstance(policy, BaseLowdimPacPolicy):
                action_dict = policy.predict_action(obs_dict, stochastic=stochastic)
            else:
                action_dict = policy.predict_action(obs_dict)

        np_action_dict = dict_apply(action_dict,
            lambda x: x.detach().to('cpu').numpy())

        # handle latency_steps, we discard the first n_latency_steps actions
        # to simulate latency
        action = np_action_dict['action'][:,self.n_latency_steps:]
        if not np.all(np.isfinite(action)):
            print(action)
            raise RuntimeError("Nan or Inf action")

        env_action = action
        if self.abs_action:
            env_action = self.undo_transform_action(action)

        # info[i]['crashed'] is set by AsyncVectorEnv.step_wait() for any
        # slot whose worker died this step (e.g. MuJoCo NaN/Inf) - see
        # _run_streaming(), which excludes just that slot's in-progress
        # episode instead of aborting the whole rollout.
        obs, reward, done, info = self.env.step(env_action)
        return obs, reward, done, action, info

    def _run_chunked(self, policy, stochastic):
        """Original synchronized rollout: n_inits processed in fixed-size
        chunks of n_envs, each chunk running until every slot in it is done.
        Required for stateful policies - see run()'s docstring.

        NOTE: unlike _run_streaming(), this path does not gracefully exclude
        a single crashed env from the batch - a MuJoCo NaN/Inf here still
        aborts the whole chunk (raising MujocoException explicitly below,
        the same way the non-tolerant default path used to for every
        crash) instead of continuing with that slot excluded. None of this
        project's own report results use a stateful policy (is_stateful is
        set by e.g. RobomimicLowdimPolicy, the upstream BC-RNN baseline -
        see train_robomimic_lowdim_workspace.yaml - which this path IS
        reachable through if that workspace/config is ever used), so
        fixing it the same way as _run_streaming (excluding just the
        crashed slot and continuing) is future work; this at least
        restores the pre-tolerance behavior instead of leaving the crashed
        worker's null pipe to be hit unconditionally by env.render()/
        env.call(...) below, which raised an uncaught AttributeError.
        """
        # This path always either completes with zero crashes (it raises
        # MujocoException above on any crash, so _aggregate_log below never
        # runs otherwise) or doesn't get this far at all - reset explicitly
        # so a crash count from an earlier _run_streaming call on this same
        # runner instance is never stale-reported here.
        self._last_n_crashed = 0
        self._last_n_unstarted = 0

        env = self.env
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                obs, reward, done_arr, action, _info = self._predict_and_step(
                    policy, obs, past_action, stochastic)
                crashed = [i for i, inf in enumerate(_info) if inf.get('crashed')]
                if crashed:
                    # A stateful policy can't have individual slots excluded
                    # mid-batch (see run()'s docstring), and this loop's own
                    # env.render()/env.call(...) below address every slot
                    # unconditionally - so a crashed slot's now-null pipe
                    # would otherwise be hit by those calls, raising an
                    # uncaught AttributeError instead of the MujocoException
                    # callers already know how to handle.
                    raise MujocoException(
                        f"Worker(s) {crashed} crashed during a chunked "
                        "(stateful-policy) rollout; this path can't exclude "
                        "a single crashed slot, so the whole chunk aborts."
                    )
                done = np.all(done_arr)
                past_action = action
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        return all_rewards

    def _run_streaming(self, policy, stochastic):
        """Continuous rollout for stateless policies: every vector slot is
        kept busy on a new init the moment its previous one finishes,
        instead of only refilling at fixed chunk boundaries.

        A slot whose worker crashes mid-episode (info[i]['crashed'], set by
        AsyncVectorEnv.step_wait() on a MuJoCo NaN/Inf or similar) has its
        in-progress episode excluded entirely - not counted as success or
        failure - and is permanently retired for the rest of this call (its
        subprocess is dead; see step_wait()'s docstring). The remaining
        active slots keep streaming through whatever inits haven't been
        assigned yet, so a crash costs one slot's worth of parallelism, not
        the whole rollout. `all_rewards[g]` stays `None` for a lost init's
        global index; `_aggregate_log` skips those, so the reported success
        rate is the mean over the survivors (1000-N successful envs, if N
        crashed), not a whole-checkpoint failure.
        """
        env = self.env
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)

        all_rewards = [None] * n_inits
        n_lost = 0
        # n_lost conflates two different causes (used for the loop's own
        # harvested+n_lost==n_inits bookkeeping, where the distinction
        # doesn't matter) - n_crashed tracks only real worker crashes, so
        # callers reporting "N episodes crashed" aren't also counting
        # inits that were simply never assigned a slot because every
        # worker had already died.
        n_crashed = 0

        # slot_init[i] = global init index currently assigned to vector-slot
        # i, or None once that slot has no more work left to do (either it
        # ran out of pending inits, or its worker crashed).
        n_start = min(n_envs, n_inits)
        slot_init = list(range(n_start)) + [None] * (n_envs - n_start)
        active = [g is not None for g in slot_init]
        next_init_idx = n_start

        # slots beyond n_inits (only when n_inits < n_envs) get a dummy init
        # so every slot has something to run - their results are never
        # harvested (active starts False for them), mirroring the padding
        # the old chunked path used for its last, partial chunk.
        init_fns = [self.env_init_fn_dills[g] if g is not None else self.env_init_fn_dills[0]
                    for g in slot_init]
        env.call_each('run_dill_function', args_list=[(x,) for x in init_fns])
        obs = env.reset()
        past_action = None
        policy.reset()

        env_name = self.env_meta['env_name']
        pbar = tqdm.tqdm(total=n_inits, desc=f"Eval {env_name}Lowdim",
            leave=False, mininterval=self.tqdm_interval_sec)

        harvested = 0
        while harvested + n_lost < n_inits:
            obs, reward, done_arr, action, info = self._predict_and_step(
                policy, obs, past_action, stochastic)
            past_action = action

            reinit_indices = []
            reinit_args = []
            for i in range(n_envs):
                if active[i] and done_arr[i]:
                    g = slot_init[i]

                    if info[i].get('crashed'):
                        # Worker is dead (AsyncVectorEnv already closed its
                        # pipe) - don't call_at/render/get_attr on it, don't
                        # count this init, and never reinit this slot again.
                        logger.warn(
                            f"Eval {env_name}Lowdim: excluding init {g} "
                            f"(vector slot {i}) - worker crashed: "
                            f"{info[i].get('error')}"
                        )
                        n_lost += 1
                        n_crashed += 1
                        pbar.update(1)
                        slot_init[i] = None
                        active[i] = False
                        continue

                    # stops/flushes that slot's video recorder (if enabled)
                    # and reads back its total episode reward.
                    env.call_at([i], 'render')
                    all_rewards[g] = env.call_at([i], 'get_attr', 'reward')[0]
                    harvested += 1
                    pbar.update(1)

                    if next_init_idx < n_inits:
                        slot_init[i] = next_init_idx
                        reinit_indices.append(i)
                        reinit_args.append((self.env_init_fn_dills[next_init_idx],))
                        next_init_idx += 1
                    else:
                        slot_init[i] = None
                        active[i] = False

            if reinit_indices:
                # each slot gets a different init_fn, so these can't share
                # one call_at() (which sends identical args to every index).
                try:
                    for i, (fn,) in zip(reinit_indices, reinit_args):
                        env.call_at([i], 'run_dill_function', fn)
                    fresh_obs = env.reset_at(reinit_indices)
                    for i, o in zip(reinit_indices, fresh_obs):
                        obs[i] = o
                        # avoid leaking the finished rollout's action into
                        # the freshly reset one's near-term obs['past_action'].
                        if past_action is not None:
                            past_action[i] = 0
                except MujocoException as e:
                    # A crash here (e.g. robosuite's reset() calling
                    # sim.forward()) previously aborted the WHOLE run(),
                    # discarding every already-harvested episode's reward
                    # this call - not just the crashed slot's. Retry the
                    # batch one slot at a time (only in this rare fallback
                    # path, so the common no-crash case keeps the cheap
                    # single batched call above) so a crash on one index
                    # doesn't cost the others' already-computed results too.
                    logger.warn(
                        f"Eval {env_name}Lowdim: reinit batch crashed "
                        f"({e}) - retrying its {len(reinit_indices)} "
                        "slot(s) one at a time to isolate the failure."
                    )
                    for i, (fn,) in zip(reinit_indices, reinit_args):
                        if env.parent_pipes[i] is None:
                            # Already dead - either the slot whose crash
                            # triggered this fallback, or another one that
                            # failed earlier in the same batched attempt
                            # above. Nothing left to retry for it.
                            logger.warn(
                                f"Eval {env_name}Lowdim: excluding init "
                                f"{slot_init[i]} (vector slot {i}) - "
                                "worker crashed during reinit."
                            )
                            n_lost += 1
                            n_crashed += 1
                            pbar.update(1)
                            slot_init[i] = None
                            active[i] = False
                            continue
                        try:
                            env.call_at([i], 'run_dill_function', fn)
                            fresh_obs_i = env.reset_at([i])
                        except MujocoException as e2:
                            logger.warn(
                                f"Eval {env_name}Lowdim: excluding init "
                                f"{slot_init[i]} (vector slot {i}) - "
                                f"worker crashed during reinit: {e2}"
                            )
                            n_lost += 1
                            n_crashed += 1
                            pbar.update(1)
                            slot_init[i] = None
                            active[i] = False
                            continue
                        obs[i] = fresh_obs_i[0]
                        if past_action is not None:
                            past_action[i] = 0

            if not any(active) and next_init_idx < n_inits:
                # Every slot is dead or spent, but inits remain that were
                # never even assigned to a slot - with no one left to pick
                # them up, count them as lost too instead of looping forever.
                n_unstarted = n_inits - next_init_idx
                logger.warn(
                    f"Eval {env_name}Lowdim: all vector slots crashed or "
                    f"finished with {n_unstarted} init(s) never started; "
                    "counting them as lost too rather than hanging."
                )
                n_lost += n_unstarted
                pbar.update(n_unstarted)
                next_init_idx = n_inits
        pbar.close()
        if n_lost:
            logger.warn(
                f"Eval {env_name}Lowdim: {n_lost}/{n_inits} episode(s) "
                f"excluded ({n_crashed} from worker crashes, "
                f"{n_lost - n_crashed} never started once every slot had "
                "died); success rate below is averaged over the remaining "
                f"{n_inits - n_lost}."
            )
        self._last_n_crashed = n_crashed
        self._last_n_unstarted = n_lost - n_crashed
        return all_rewards

    def _aggregate_log(self, policy, stochastic, all_rewards):
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        log_data['n_crashed_episodes'] = getattr(self, '_last_n_crashed', 0)
        # Distinct from n_crashed_episodes: inits that were never even
        # assigned a slot because every worker had already died - not
        # themselves a crash, but still missing from the reported score.
        log_data['n_unstarted_episodes'] = getattr(self, '_last_n_unstarted', 0)
        n_inits = len(self.env_init_fn_dills)
        log_data['n_total_episodes'] = n_inits
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            if all_rewards[i] is None:
                # Excluded by _run_streaming due to a worker crash - not a
                # success and not a failure, just not counted at all, so
                # the mean below is over the surviving episodes only.
                continue
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)

        # Every caller's score_key_for() reads 'test/mean_score...' - only
        # that prefix's total wipeout is a caller-facing failure (its key
        # would otherwise be silently missing from log_data below, crashing
        # the caller's runner_log[...] lookup). A 'train/' (or any other
        # non-'test/' prefix) wipeout just omits that prefix's own key,
        # which nothing downstream reads, so it doesn't need to abort an
        # otherwise-valid checkpoint's real test/ score.
        if 'test/' in set(self.env_prefixs) and len(max_rewards.get('test/', [])) == 0:
            raise MujocoException(
                "All 'test/' episodes were lost to worker crashes; no valid "
                "score to report for this checkpoint."
            )

        # log aggregate metrics
        if isinstance(policy, BaseLowdimPacPolicy):
            if stochastic==False:
                for prefix, value in max_rewards.items():
                    name = prefix+'mean_score_deterministic'
                    value = np.mean(value)
                    log_data[name] = value
            else:
                for prefix, value in max_rewards.items():
                    name = prefix+'mean_score_stochastic'
                    value = np.mean(value)
                    log_data[name] = value
        else:
            for prefix, value in max_rewards.items():
                name = prefix+'mean_score'
                value = np.mean(value)
                log_data[name] = value
        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
