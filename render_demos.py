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
import hydra
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
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            abs_action=False,
            n_envs=None,
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps 
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)

        print (f"env_meta: {env_meta}")
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

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

        self.env_meta = env_meta
        self.env = env
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.current_epoch = 0
        self.dataset_path = dataset_path
    
    def run(self):
        env = self.env
        
        # allocate data
        video = []

        with h5py.File(self.dataset_path, "r") as f:
            demo_idx = 9
            demo = f[f"data/demo_{demo_idx}"]
            
            for i in range (160, 290):
                env.init_state = demo["states"][i]
                obs = env.reset()
                video.append(env.render())

        # visualize
        from IPython.display import Video
        from skvideo.io import vwrite
        vwrite(f'./data/demos_tool_hang/demos{demo_idx}.mp4', video)
        Video(f'./data/demos_tool_hang/demos{demo_idx}.mp4', embed=True, width=256, height=256)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("diffusion_policy/diffusion_policy/config")), 
    config_name=pathlib.Path(__file__).stem)

def main(cfg):
    env_runner: BaseLowdimRunner
    env_runner = RobomimicLowdimRunner(
        output_dir='./render_demos_output',
        dataset_path=cfg.task.dataset_path,
        obs_keys=cfg.task.obs_keys,
        max_steps=cfg.task.env_runner.max_steps,
        n_obs_steps=cfg.task.env_runner.n_obs_steps,
        n_action_steps=cfg.task.env_runner.n_action_steps,
        render_hw=cfg.task.env_runner.render_hw,
        render_camera_name='agentview',
        fps=cfg.task.env_runner.fps,
        crf=cfg.task.env_runner.crf,
        abs_action=cfg.task.env_runner.abs_action,
        n_envs=1
    )

    env_runner.run()
    
if __name__ == "__main__":
    main()
