import os
import gym
import numpy as np
import torch
import h5py
import dill
import math

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_prob_policy import BaseLowdimProbPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import _data_to_obs

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

class ActionPerturbationEvaluator:
    def __init__(self, 
            dataset_path,
            obs_keys,
            n_test=100,
            max_steps=10,
            n_obs_steps=2,
            n_action_steps=8,
            abs_action=False,
            n_envs=None,
            eps=0.05,
        ):

        if n_envs is None:
            n_envs = n_test

        env_n_obs_steps = n_obs_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)

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
                )

            return MultiStepWrapper(
                                env,
                                n_obs_steps=env_n_obs_steps,
                                n_action_steps=env_n_action_steps,
                                max_episode_steps=max_steps
                            )

        env_fns = [env_fn] * n_envs
        env_init_fn_dills = list()
        states = []
        actions = []
        self.nominal_actions = []
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_test):
                rng = np.random.RandomState(i)
                # sample random demo and a random timestep from the demo, and use the corresponding state as initial state
                demo_idx = rng.randint(0, 10)   
                demo = f[f'data/demo_{demo_idx}']
                if demo_idx==0:
                    a,b = 190, 250
                elif demo_idx==1:
                    a,b = 170, 220
                elif demo_idx==2:
                    a,b = 170, 230
                elif demo_idx==3:
                    a,b = 170, 230
                elif demo_idx==4:
                    a,b = 200, 260
                elif demo_idx==5:
                    a,b = 190, 290
                elif demo_idx==6:
                    a,b = 200, 230
                elif demo_idx==7:
                    a,b = 190, 260
                elif demo_idx==8:
                    a,b = 150, 200
                elif demo_idx==9:
                    a,b = 160, 290
                data_idx = rng.randint(a, b-n_action_steps) 
                init_state = demo['states'][data_idx]
                data = _data_to_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer)

            # perturb action sequence with noise    
                action = data["action"][data_idx:data_idx+n_action_steps]
                self.nominal_actions.append(action) 
                
                # create init_fn for each initial state
                def init_fn(env, init_state=init_state):
                    # switch to init_state reset
                    assert isinstance(env.env, RobomimicLowdimWrapper)
                    env.env.init_state = init_state

                env_init_fn_dills.append(dill.dumps(init_fn))
        
        self.nominal_actions = np.array(self.nominal_actions, dtype=np.float32)

        env = AsyncVectorEnv(env_fns)
        # env = SyncVectorEnv(env_fns)

        self.n_envs = n_envs
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_test = n_test
        self.max_steps = max_steps
        self.eps = eps
        self.rng = np.random.RandomState(42)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_init_fn_dills = env_init_fn_dills
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action

    def run(self, policy, cfg):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        all_pred_actions = []
        all_obs = []

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
            policy.reset()

            # perturbs normalized action
            normalized_action = policy.normalizer['action'].normalize(self.nominal_actions[this_global_slice]) 
            perturbed_nomalized_action = normalized_action.detach().to('cpu').numpy() + self.eps * self.rng.randn(*normalized_action.shape)
            
            # unormalized the action
            perturbed_action = policy.normalizer['action'].unnormalize(perturbed_nomalized_action)
            perturbed_action = perturbed_action.detach().to('cpu').numpy()
            
            if this_n_active_envs < n_envs:
                pad = n_envs - this_n_active_envs
                perturbed_action = np.concatenate(
                    [perturbed_action, np.repeat(perturbed_action[:1], pad, axis=0)],
                    axis=0
                )

            env_action = perturbed_action
            if self.abs_action:
                env_action = self.undo_transform_action(perturbed_action)
            obs, _, _, _ = self.env.step(env_action)
            
            # prepare obs
            np_obs_dict = {
                'obs': obs[:, :self.n_obs_steps].astype(np.float32)
            }
            
            all_obs.append(np_obs_dict['obs'][:this_n_active_envs])
            
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=device))

            # run policy
            with torch.no_grad():              
                if isinstance(policy, BaseLowdimProbPolicy):   
                    policy.model.sample_weights()
                    action_dict = policy.predict_action(obs_dict, stochastic=cfg.eval.stochastic)
                    policy.model.clear_sampled_weights()
                else:
                    action_dict = policy.predict_action(obs_dict)

            # save OOD pred actions 
            action_pred = action_dict['action_pred'].detach().cpu().numpy()
            action_pred = action_pred[:this_n_active_envs]

            all_pred_actions.append(action_pred)

        # concatenate all chunks
        data_dict = {'obs': np.concatenate(all_obs, axis=0),
                    'action': np.concatenate(all_pred_actions, axis=0)}

        return data_dict
    
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
