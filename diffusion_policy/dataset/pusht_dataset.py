from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class PushTLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            train_episodes_for_PAC_Bayes= 0,
            offset = 0.0
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        
        train_mask_init = ~val_mask
        train_mask_final = np.zeros(self.replay_buffer.n_episodes, dtype=bool)
        n_samples=0
        while n_samples<int(max_train_episodes):
            train_mask = downsample_mask(
                        mask=train_mask_init, 
                        max_n=10, 
                        seed=seed)
            train_idx = np.where(train_mask)[0]
            train_mask_final [train_idx] = True
            train_mask_init [train_idx] = False 
            n_samples+=10 
        
        # val_mask = ~train_mask
        pac_mask = np.zeros(self.replay_buffer.n_episodes, dtype=bool)

        if train_episodes_for_PAC_Bayes>0:    
            # Get the indices of validation episodes
            train_indices = np.where(train_mask_final)[0]
            # Reproducible random generator
            rng = np.random.default_rng(seed)
            # Randomly choose demos from train dataset
            n_train_pac = min(train_episodes_for_PAC_Bayes, len(train_indices))
            train_pac_indices = rng.choice(train_indices, size=n_train_pac, replace=False)
            pac_mask [train_pac_indices] = True
            train_mask_final [train_pac_indices] = False
        
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask_final
            )
        
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.train_mask = train_mask_final
        self.val_mask = val_mask
        self.pac_mask = pac_mask
        self.offset = offset

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.train_mask = self.val_mask
        return val_set

    def get_pac_bayes_dataset(self):
        pac_set = copy.copy(self)
        pac_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.pac_mask
            )
        pac_set.train_mask = self.pac_mask
        return pac_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)

        data = {
            "obs": obs + self.offset, # T, D_o
            "action": sample[self.action_key] + self.offset,  # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
