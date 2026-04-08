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

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.policy.base_lowdim_prob_policy import BaseLowdimProbPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


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

class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            render_camera_name='robot0_robotview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):

        super().__init__(output_dir)
        
        self.current_epoch = 0
        
        if n_envs is None:
            n_envs = n_test

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
                    epoch = self.current_epoch
                    name = self.make_video_filename(
                        epoch=epoch,
                        idx=seed
                    )
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', name + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)
                
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
        

    def make_video_filename(self, *, epoch, idx):
            return f"epoch={epoch:04d}_seed={idx:05d}.mp4"
    
    def _get_epoch(self, epoch):
        self.current_epoch = epoch
    
    def action_reconst_loss(self, simulation_dict, dataset_dict, cfg):
        """
        This function returns the action reconstruction L2 loss by comparing the generated action to
        the nearest action in the training dataset.
        It compares the states obtained from simulation with training dataset to find nearest state and its correspoding action in the training dataset.

        simulation_dict: {action:(M, Ta, Da), obs:(M, To, Do)}          M = number of simulations
        dataset_dict: {action:(N, Ta, Da), obs:(N, To, Do)}             N = number training data

        return:
            l2_loss: (M,)
        """
        device = simulation_dict["action"].device
        if dataset_dict["action"].device != device:
            raise RuntimeError(f"training data is not on device:{device} as the simulation data")

        M, To, Do = simulation_dict["obs"].shape
        _, Ta, Da = simulation_dict["action"].shape

        obs_train = dataset_dict['obs'][:, :To, :]  
        obs_simulation = simulation_dict["obs"]

        N = obs_train.shape[0]

        # Flatten for distance computation
        obs_train = obs_train.reshape(N, -1)
        obs_simulation = obs_simulation.reshape(M, -1)

        # Compute pairwise distances
        # (M, N)
        dists = torch.cdist(obs_simulation, obs_train, p=2.0)

        # Get nearest neighbors
        _, idx = torch.topk(dists, k=1, dim=1, largest=False)
        idx = idx.squeeze()

        # Get corresponding action
        nearest_actions = dataset_dict['action'][idx]
        if cfg.pred_action_steps_only:
            start = To
            if cfg.oa_step_convention:
                start = To - 1
            end = start + cfg.n_action_steps
            nearest_actions = dataset_dict['action'][idx, start:end, :]

        # compute loss
        l2_loss = torch.linalg.norm(
                                    simulation_dict["action"] - nearest_actions,
                                    ord=2,
                                    dim=(1, 2))
        return l2_loss

    def compute_manifold_adherence(self, simulation_dict, dataset_dict, k, cfg):
        """
        This funtion computes off-manifold norm. Firts it compares the current state to find k
        nearest states in the dataset and its corresponding actions. Then it projects the generated action
        to the subspace spanned by the k nearest actions to evaluate how far the generated action is from
        the subspace
        """

        device = simulation_dict["action"].device
        if dataset_dict["action"].device != device:
            raise RuntimeError(f"training data is not on device:{device} as the simulation data")

        M, To, Do = simulation_dict["obs"].shape
        _, Ta, Da = simulation_dict["action"].shape

        obs_train = dataset_dict['obs'][:, :To, :]  
        obs_simulation = simulation_dict["obs"]

        N = obs_train.shape[0]

        # Flatten for distance computation
        obs_train = obs_train.reshape(N, -1)
        obs_simulation = obs_simulation.reshape(M, -1)

        # Compute pairwise distances
        # (M, N)
        dists = torch.cdist(obs_simulation, obs_train, p=2.0)

        # Get nearest neighbors
        _, knn_idx = torch.topk(dists, k=k, dim=1, largest=False)
        nearest_actions = list()    
        for i in range(M):
            nearest_actions.append(dataset_dict['action'][knn_idx[i]])  # (k, n_action_step, Da)
        nearest_actions = torch.stack(nearest_actions, dim=0)  # (M, k, n_action_step, Da)  

        if cfg.pred_action_steps_only:
            start = To
            if cfg.oa_step_convention:
                start = To - 1
            end = start + cfg.n_action_steps
            nearest_actions = nearest_actions[..., start:end, :]

        pred = simulation_dict["action"].reshape(M, -1)  # (M, T*Da)
        knn = nearest_actions.reshape(M, k, -1)  # (M, k, T*Da)

        errors = []
        for i in range(M):
            a = pred[i]  # (T*Da,)
            A = knn[i].T  # (T*Da, k)
            # Solve least squares: min_c ||a - A @ c||_2
            sol = torch.linalg.lstsq(A, a)  
            c = sol.solution            # (k, 1)
            proj = A @ c  # (T*Da, 1)
            error = torch.norm(a - proj, p=2)
            errors.append(error)
        errors = torch.stack(errors)
        metric = errors.mean()
        return metric, errors

    def run(self, policy, train_dataset, cfg):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        loss_reconst_all_envs = [[] for _ in range(n_inits)]
        manifold_adh = [[] for _ in range(n_inits)]
        all_pred_actions = []

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
            
            simulation_dict = {"action":None, "obs":None}
            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[:,:self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # Store the observation
                simulation_dict ["obs"]= obs_dict["obs"]

                # run policy
                with torch.no_grad():
                    if isinstance(policy, BaseLowdimProbPolicy):   
                        policy.model.sample_weights()
                        action_dict = policy.predict_action(obs_dict, stochastic=cfg.eval.stochastic)
                        policy.model.clear_sampled_weights()
                    else:                 
                        action_dict = policy.predict_action(obs_dict)

                # store generated actions
                simulation_dict ["action"] = action_dict["action_pred"]
                all_pred_actions.append(simulation_dict ["action"].cpu().numpy())

                # compute action reconstruction loss and manifold adherence
                loss_reconst = self.action_reconst_loss(simulation_dict, train_dataset, cfg)
                _, adherence = self.compute_manifold_adherence(simulation_dict, train_dataset, 50, cfg)

                for i in range (this_n_active_envs):
                    loss_reconst_all_envs[start+i].append(loss_reconst[i].item())
                    manifold_adh[start+i].append(adherence[i].item())
                
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]

                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])

            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path, format="gif")
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        return log_data, np.stack(loss_reconst_all_envs), np.stack (manifold_adh), np.stack(all_pred_actions, axis=1)

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