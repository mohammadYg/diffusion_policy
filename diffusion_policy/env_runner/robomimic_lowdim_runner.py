import os
import wandb
import gym
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
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
        
        env = AsyncVectorEnv(env_fns)
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

        obs, reward, done, info = self.env.step(env_action)
        return obs, reward, done, action

    def _run_chunked(self, policy, stochastic):
        """Original synchronized rollout: n_inits processed in fixed-size
        chunks of n_envs, each chunk running until every slot in it is done.
        Required for stateful policies - see run()'s docstring.
        """
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
                obs, reward, done_arr, action = self._predict_and_step(
                    policy, obs, past_action, stochastic)
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
        """
        env = self.env
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)

        all_rewards = [None] * n_inits

        # slot_init[i] = global init index currently assigned to vector-slot
        # i, or None once that slot has no more work left to do.
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
        while harvested < n_inits:
            obs, reward, done_arr, action = self._predict_and_step(
                policy, obs, past_action, stochastic)
            past_action = action

            reinit_indices = []
            reinit_args = []
            for i in range(n_envs):
                if active[i] and done_arr[i]:
                    g = slot_init[i]
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
                for i, (fn,) in zip(reinit_indices, reinit_args):
                    env.call_at([i], 'run_dill_function', fn)
                fresh_obs = env.reset_at(reinit_indices)
                for i, o in zip(reinit_indices, fresh_obs):
                    obs[i] = o
                    # avoid leaking the finished rollout's action into the
                    # freshly reset one's near-term obs['past_action'].
                    if past_action is not None:
                        past_action[i] = 0
        pbar.close()
        return all_rewards

    def _aggregate_log(self, policy, stochastic, all_rewards):
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        n_inits = len(self.env_init_fn_dills)
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)

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
