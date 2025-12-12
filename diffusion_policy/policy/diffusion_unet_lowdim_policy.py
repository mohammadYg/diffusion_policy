from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.SDE import VPSDE
from diffusion_policy.common.likelihood import get_likelihood_fn

from omegaconf import OmegaConf
import collections
import numpy as np
import math 
import tqdm

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred_normalized = naction_pred
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
            'action_pred_normalized': action_pred_normalized
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, train = True):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        if train:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=trajectory.device
            ).long()
        else:
            timesteps = torch.ones(bsz, device=trajectory.device).long() * 10
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    # ========= Compute Reconstruction Loss of PAC-Bayes Bounds for case where we do inference from last step ============
    def compute_action_reconst_loss(self, init_noise, batch, loss_type, normalized):
        
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        naction = nbatch['action']

        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = naction

        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_mask = torch.zeros(size=shape, device=device, dtype=torch.bool)
            init_noise = init_noise[...,:Da]
        
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            init_noise = init_noise[...,:Da]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = naction[:,start:end]
                init_noise = init_noise[:, start:end]
            cond_mask = torch.zeros(size=shape, device=device, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_mask = torch.zeros(size=shape, device=device, dtype=torch.bool)
            cond_mask[:,:To,Da:] = True
            trajectory = torch.cat([naction, nobs], dim=-1)
        
        # set timestep to last noising step for each sanmple
        timestep = self.noise_scheduler.timesteps[0]

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, init_noise, timestep)
        
        # set step values
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            
            # 1. apply conditioning
            noisy_trajectory[cond_mask] = trajectory[cond_mask]
    
            # 2. predict model output
            model_output = self.model(noisy_trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            noisy_trajectory = self.noise_scheduler.step(
                model_output, t, noisy_trajectory,
                generator = None,
                **self.kwargs
                ).prev_sample
               
        # finally make sure conditioning is enforced
        noisy_trajectory[cond_mask] = trajectory[cond_mask]
        
        # extract only predicted actions
        if not normalized:
            action_pred = self.normalizer['action'].unnormalize(noisy_trajectory[...,:Da])
            action_ref  = self.normalizer['action'].unnormalize(trajectory[...,:Da])
        else:
            action_pred = noisy_trajectory[...,:Da]
            action_ref = trajectory[...,:Da]

        # compute loss
        if loss_type == "MSE":
            loss = F.mse_loss(action_pred, action_ref)

        elif loss_type == "RMSE":
            loss = torch.sqrt(F.mse_loss(action_pred, action_ref))
        else: 
            loss = torch.linalg.norm(action_pred - action_ref, ord=2, dim=(1, 2)).mean()
            
        return loss
  
    ## ========= Memorization Evaluation ============
    def eval_memorization(self, dataloader_eval, dataloader_train, normalized, device, threshold=0.3):
        # -----------------------------------------------------------
        # 1. Load all training actions once
        # -----------------------------------------------------------
        train_actions_list = []
        with torch.inference_mode():
            with tqdm.tqdm(dataloader_train, desc=f"Memorization Eval: Load all training actions once", 
                                leave=False, mininterval=1.0) as tepoch:
                for batch in tepoch:
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if normalized:
                        batch = self.normalizer.normalize(batch)
                    train_actions_list.append(batch['action'])
            train_actions = torch.cat(train_actions_list, dim=0)   # (N_train, H, A)

        # -----------------------------------------------------------
        # 2. Compute nearest-neighbor ratios for all eval samples
        # -----------------------------------------------------------
        all_ratios = []
        with torch.inference_mode():
            with tqdm.tqdm(dataloader_train, desc=f"Memorization Eval: Compute nearest-neighbor ratios for all eval samples", 
                                leave=False, mininterval=1.0) as tepoch:
                for batch in tepoch:
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    # Predict actions
                    action = self.predict_action({'obs': batch['obs']})
                    gen_actions = action["action_pred_normalized"] if normalized else action["action_pred"]
                    gen_actions = gen_actions.to(device)

                    # Compute pairwise distances: (B_eval, N_train)
                    diff = gen_actions[:, None, :, :] - train_actions[None, :, :, :]
                    dist = torch.sqrt(torch.sum(diff ** 2, dim=(2, 3)))   # (B_eval, N_train)

                    # Extract 2 nearest distances
                    smallest_vals, _ = torch.topk(dist, k=2, largest=False)
                    # smallest_vals: (B_eval, 2)

                    # Compute ratio d1 / d2
                    ratios = smallest_vals[:, 0] / smallest_vals[:, 1]
                    all_ratios.append(ratios.cpu())

        # Concatenate all batches
        all_ratios = torch.cat(all_ratios, dim=0)   # (N_eval,)

        # -----------------------------------------------------------
        # 3. Compute required metrics
        # -----------------------------------------------------------
        mean_ratio = all_ratios.mean().item()

        # Memorized samples
        memorized_mask = all_ratios < threshold
        memorized_count = memorized_mask.sum().item()

        # Memorization fraction
        total_generated = all_ratios.numel()
        memorization_fraction = 100.0*memorized_count / total_generated

        return mean_ratio, memorized_count, memorization_fraction


    # ========= Lipschitz Constant of the Denoising ============
    def lip_const(self, obs):
        nobs = self.normalizer['obs'].normalize(obs)
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        model = self.model
        scheduler = self.noise_scheduler

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        trajectory = torch.randn(
            size=cond_data.shape, 
            dtype=cond_data.dtype,
            device=device,
            generator = None)
    
        # set step values
        scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        # set a variable to save denoising results
        denoise_step = collections.deque(maxlen=self.noise_scheduler.config.num_train_timesteps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[cond_mask] = cond_data[cond_mask]

            # 1.1 save the output of each denoising step
            denoise_step.append(trajectory)
            
            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=None,
                **self.kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[cond_mask] = cond_data[cond_mask]      

        # save last step of denoising 
        denoise_step.append(trajectory)

        denoise_step = torch.stack(list(denoise_step))

        delta_g_theta = denoise_step[1:,0] - denoise_step[1:,1]
        delta_XY = denoise_step[:-1,0] - denoise_step[:-1,1]
        k_theta = torch.linalg.norm(delta_g_theta, ord=2, dim=(1, 2))/ torch.linalg.norm(delta_XY, ord=2, dim=(1, 2))
        k_theta_prod = torch.prod(k_theta)
        
        return torch.flip(k_theta, dims=[0]), k_theta_prod
    
    # ========= Compute Reconstruction Loss of PAC-Bayes Bounds for case where we do inference from random step ============
    def compute_reconst_loss_Tt(self, batch, timestep, loss_type):
        
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        naction = nbatch['action']

        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = naction

        if self.obs_as_local_cond:
            # condition through local feature. all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_mask = torch.zeros(size=shape, device=device, dtype=torch.bool)
        
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = naction[:,start:end]
            cond_mask = torch.zeros(size=shape, device=device, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_mask = torch.zeros(size=shape, device=device, dtype=torch.bool)
            cond_mask[:,:To,Da:] = True
            trajectory = torch.cat([naction, nobs], dim=-1)

        # Sample noise that we'll add to the input
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # Add noise to clean data to construct reference noisy data at chosen t
        noisy_trajectory_ref = self.noise_scheduler.add_noise(trajectory, noise, timestep)
        noisy_trajectory_ref[cond_mask] = trajectory[cond_mask]

        # set step values
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)

        # Reconstruction loss from T to t
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, self.noise_scheduler.timesteps[0])
        
        i=0
        loss_T = 0

        while not self.noise_scheduler.timesteps[i]==timestep:
            # 1. apply conditioning
            noisy_trajectory[cond_mask] = trajectory[cond_mask]
        
            t = self.noise_scheduler.timesteps[i]

            model_output = self.model(noisy_trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            noisy_trajectory = self.noise_scheduler.step(
                model_output, t, noisy_trajectory,
                generator = None,
                **self.kwargs
                ).prev_sample
            
            i+=1
        
        # finally make sure conditioning is enforced
        noisy_trajectory[cond_mask] = trajectory[cond_mask]
        naction_pred = noisy_trajectory[...,:Da]

        # compute loss
        if loss_type == "MSE":
            loss_T = F.mse_loss(naction_pred, noisy_trajectory_ref[...,:Da])
        else:
            loss_T = torch.linalg.norm(naction_pred - noisy_trajectory_ref[...,:Da], ord=2, dim=(1, 2)).mean()

        # Reconstruction loss from t to 0
        timesteps = torch.arange(timestep.item(), -1, -1, device=timestep.device)
        noisy_trajectory = noisy_trajectory_ref.clone()
        
        loss_t=0
        for t in timesteps:

            # 1. apply conditioning
            noisy_trajectory[cond_mask] = trajectory[cond_mask]
        
            # 2. predict model output
            model_output = self.model(noisy_trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            noisy_trajectory = self.noise_scheduler.step(
                model_output, t, noisy_trajectory,
                generator = None,
                **self.kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        noisy_trajectory[cond_mask] = trajectory[cond_mask]
        
        naction_pred = noisy_trajectory[...,:Da]

        # compute loss
        if loss_type == "MSE":
            loss_t = F.mse_loss(naction_pred, trajectory[...,:Da])
        else:
            loss_t = torch.linalg.norm(naction_pred - trajectory[...,:Da], ord=2, dim=(1, 2)).mean()

        return loss_t + loss_T
    
    # ========= Compute Reconstruction Loss of PAC-Bayes Bounds for case where we do inference from random step ============
    def compute_reconst_loss_t(self, batch, timestep, loss_type):
        
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        naction = nbatch['action']

        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = naction

        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_mask = torch.zeros(size=shape, device=device, dtype=torch.bool)
        
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = naction[:,start:end]
            cond_mask = torch.zeros(size=shape, device=device, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_mask = torch.zeros(size=shape, device=device, dtype=torch.bool)
            cond_mask[:,:To,Da:] = True
            trajectory = torch.cat([naction, nobs], dim=-1)

        # Sample noise that we'll add to the input
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timestep)
        
        # set step values
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        timesteps = torch.arange(timestep.item(), -1, -1, device=timestep.device)
        
        for t in timesteps:
            # 1. apply conditioning
            noisy_trajectory[cond_mask] = trajectory[cond_mask]

            # 2. Compute the GroundTruth denoising
            GT_denoised = self.noise_scheduler.groundT_denoise(
                noisy_trajectory, t, trajectory)
        
            # 2. predict model output
            model_output = self.model(noisy_trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            noisy_trajectory = self.noise_scheduler.step(
                model_output, t, noisy_trajectory,
                generator = None,
                **self.kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        noisy_trajectory[cond_mask] = trajectory[cond_mask]
        
        # prediction
        #! what should I do for the case in which the actions and conditions are
        #! concatenated to eachother, should I compare the whole trajectory or 
        #! the actions
        #naction_pred = noisy_trajectory[...,:Da]
        
        # compute loss mask
        loss_mask = ~cond_mask

        # compute loss
        if loss_type == "MSE":
            loss = F.mse_loss(noisy_trajectory, trajectory, reduction='none')
            loss = loss * loss_mask.type(loss.dtype)
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
        else:
            loss = (noisy_trajectory - trajectory) ** 2   # squared error
            loss = loss * loss_mask.type(loss.dtype)
            loss = torch.sqrt(loss.sum(dim=(1, 2)))   # L2 norm over dims (1,2)
            loss = loss.mean()  # average over batch

        return loss
    
    # ========= PAC-Bayes Bounds ============
    # def PAC_Bayes_Bounds(self, data_loader, cfg: OmegaConf):
        
    #     def reconstruction_loss(data_loader):
    #         emp_risk = 0
    #         alpha_bar_T = self.noise_scheduler.alpha_bar[-1]
            
    #         for batch_idx, batch in enumerate(data_loader):               
    #             # device transfer
    #             device = cfg.device
    #             batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    #             obs_dict = {"obs": batch["obs"]}
    #             gt_action = batch["action"]
                
    #             loss = 0
    #             for _ in range (cfg.training.num_expect):
    #                 result = self.predict_action(obs_dict)
    #                 if cfg.pred_action_steps_only:
    #                     pred_action = result["action"]
    #                     start = cfg.n_obs_steps - 1
    #                     end = start + cfg.n_action_steps
    #                     gt_action = gt_action[:, start:end]
    #                 else:
    #                     pred_action = result["action_pred"]
                    
    #                 # compute loss
    #                 if cfg.training.loss_type == "MSE":
    #                     loss += torch.nn.functional.mse_loss(pred_action, gt_action) * gt_action.shape[0]
    #                 else:
    #                     loss += torch.linalg.norm(pred_action - gt_action, ord=2, dim=(1, 2))
                    
    #                 emp_risk +=  torch.sum(loss)

    #         return emp_risk.item() / (num_expect * (batch_idx+1))

    def noisy_channel(self, x, logsnr):
        """
        Forward diffusion process: match each logSNR to the nearest DDIM alpha_cumprod
        and generate noisy samples + return timestep and the noise.
        """
        # Compute alpha from logSNR and find nearest scheduler timestep
        alpha = torch.sigmoid(logsnr)  # (N,)
        scheduler_alpha = self.noise_scheduler.alphas_cumprod.to(self.device)  # (T,)

        # Compute |alpha - alpha_t|
        diff = torch.abs(alpha.unsqueeze(1) - scheduler_alpha.unsqueeze(0))
        timesteps = diff.argmin(dim=1)  # (N,)

        # Sample noise
        eps = torch.randn((len(logsnr),) + self.shape, dtype=x.dtype, device=x.device)

        # DDPM forward step
        noisy_x = self.noise_scheduler.add_noise(x, eps, timesteps)
        return noisy_x, timesteps, eps

    def mse(self, batch, logsnr, mse_type='epsilon', xinterval=None):
        """
        Compute per-sample MSE between predicted noise eps_hat and true eps.
        If mse_type='epsilon', returns ||eps - eps_hat||^2.
        """
        x = batch[0].to(self.device)
        z, timesteps, eps = self.noisy_channel(x, logsnr)
        
        # Unpack conditioning
        local_cond, global_cond, cond_data, cond_mask = batch[1:]

        # Apply imputation mask
        z[cond_mask] = cond_data[cond_mask]

        # ---------------------------
        # 2. Predict noise
        # ---------------------------
        eps_hat = self.model(
            z, timesteps,
            local_cond=local_cond,
            global_cond=global_cond
        )

        # ---------------------------
        # 3. Compute MSE
        # ---------------------------
        # compute loss mask
        loss_mask = (~cond_mask).float()        # same shape as eps

        sse = ((eps - eps_hat)**2 * loss_mask)  # masked squared error
        mse_eps = sse.flatten(start_dim=1).sum(dim=1)

        return mse_eps

    def nll(self, batch, logsnr_samples_per_x=1, xinterval=None):
        """Monte-Carlo estimate of -log p(x)."""
        total = 0
        B = len(batch[0])

        for _ in range(logsnr_samples_per_x):
            logsnr, w = self.logistic_integrate(
                B, *self.loc_scale, device=self.device
            )
            mses = self.mse(batch, logsnr, mse_type='epsilon', xinterval=xinterval)
            total += self.loss(mses, logsnr, w)

        return total / logsnr_samples_per_x

    def loss(self, mses, logsnr, w):
        """
        MSE → NLL conversion using analytic Gaussian MMSE gap.
        """
        mmse_gap = mses - self.mmse_g(logsnr)
        return self.h_g + 0.5 * (w * mmse_gap).mean()

    @torch.no_grad()
    def test_nll(self, dataloader, epoch, npoints=100, xinterval=None):
        """Calculate expected NLL on data at test time.  Main difference is that we can clamp the integration
        range, because negative values of MMSE gap we can switch to Gaussian decoder to get zero.
        npoints - number of points to use in integration
        delta - if the data is discrete, delta is the gap between discrete values.
        E.g. delta = 1/127.5 for CIFAR data (0, 255) scaled to -1, 1 range
        xinterval - a tuple of the range of the discrete values, e.g. (-1, 1) for CIFAR10 normalized
        soft -  using soft discretization if True.
        """
        if self.model.training:
            print("Warning - estimating test NLL but model is in train mode")
        results = {}  # Return multiple forms of results in a dictionary
        clip = 4
        loc, scale = self.loc_scale
        logsnr, w = self.logistic_integrate(npoints, loc=loc, scale=scale, clip=clip, device=self.device, deterministic=True)

        # sort logsnrs along with weights
        logsnr, idx = logsnr.sort()
        w = w[idx].to('cpu')

        results['logsnr'] = logsnr.to('cpu')
        results['w'] = w
        mses = []  # Store all MSEs, per sample, logsnr, in an array
        total_samples = 0
        val_loss = 0
        with tqdm.tqdm(dataloader, desc=f"NLL computation at epoch = {epoch}", 
                        leave=False) as tepoch:
            for batch in tepoch:
                nbatch = self.normalizer.normalize(batch)
                nobs = nbatch['obs']
                naction = nbatch['action']

                B, _, Do = nobs.shape
                To = self.n_obs_steps
                assert Do == self.obs_dim
                T = self.horizon
                Da = self.action_dim

                # build input
                device = self.device
                dtype = self.dtype

                # handle different ways of passing observation
                local_cond = None
                global_cond = None
                trajectory = naction

                if self.obs_as_local_cond:
                    # condition through local feature
                    # all zero except first To timesteps
                    local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
                    local_cond[:,:To] = nobs[:,:To]
                    shape = (B, T, Da)
                    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

                elif self.obs_as_global_cond:
                    # condition throught global feature
                    global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
                    shape = (B, T, Da)
                    if self.pred_action_steps_only:
                        shape = (B, self.n_action_steps, Da)
                        start = To
                        if self.oa_step_convention:
                            start = To - 1
                        end = start + self.n_action_steps
                        trajectory = naction[:,start:end]
                    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                else:
                    # condition through impainting
                    shape = (B, T, Da+Do)
                    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                    cond_data[:,:To,Da:] = nobs[:,:To]
                    cond_mask[:,:To,Da:] = True
                    trajectory = torch.cat([naction, nobs], dim=-1)

                # construct a batch of data in the form that diffusion model of IT expects
                batch = [trajectory, local_cond, global_cond, cond_data, cond_mask]
                data = batch[0].to(self.device)  # assume iterator gives other things besides x in list
                n_samples = len(data)
                total_samples += n_samples

                val_loss += self.nll([data, ] + batch[1:], xinterval=xinterval).cpu() * n_samples

                mses.append(torch.zeros(n_samples, len(logsnr)))
                for j, this_logsnr in enumerate(logsnr):
                    this_logsnr_broadcast = this_logsnr * torch.ones(len(data), device=self.device)

                    # Regular MSE, clamps predictions, but does not discretize
                    this_mse = self.mse([data, ] + batch[1:], this_logsnr_broadcast, mse_type='epsilon',
                                        xinterval=xinterval).cpu()
                    mses[-1][:, j] = this_mse

        val_loss /= total_samples

        mses = torch.cat(mses, dim=0)  # Concatenate the batches together across axis 0
        results['mses-all'] = mses  # Store array of mses for each sample, logsnr
        mses = mses.mean(dim=0)  # Average across samples, giving MMSE(logsnr)

        results['mses'] = mses
        results['mmse_g'] = self.mmse_g(logsnr.to(self.device)).to('cpu')

        # here the results of the integral is clamped to 0 if the MMSE is negative
        results['nll (nats)'] = torch.mean(self.h_g - 0.5 * w * torch.clamp(results['mmse_g'] - mses, 0.))
        results['nll (bpd)'] = results['nll (nats)'] / math.log(2) / self.d
 
        ## Variance (of the mean) calculation - via CLT, it's the variance of the samples (over epsilon, x, logsnr) / n samples.
        ## n_samples is number of x samples * number of logsnr samples per x
        inds = (results['mmse_g'] - results[
            'mses']) > 0  # we only give nonzero estimates in this region (for continuous estimators)
        n_samples = results['mses-all'].numel()
        wp = w[inds]
        results['nll (nats) - var'] = torch.var(
            0.5 * wp * (results['mmse_g'][inds] - results['mses-all'][:, inds])) / n_samples
        
        results['nll (bpd) - std'] = torch.sqrt(results['nll (nats) - var']) / math.log(2) / self.d
        
        return results['nll (bpd)']

    @property
    def loc_scale(self):
        """Return the parameters defining a normal distribution over logsnr, inferred from data statistics."""
        return self.loc_logsnr, self.scale_logsnr

    def dataset_info(self, dataloader, covariance_spectrum=None, diagonal=False):
        """covariance_spectrum can provide precomputed spectrum to speed up frequent experiments.
           diagonal: {False, True}  approximates covariance as diagonal, useful for very high-d data.
        """
        # logger.log("Getting dataset statistics, including eigenvalues,")
        # data = nbatch['action'].to("cpu")  # assume iterator gives other things besides x in list
        # logger.log('using # samples given:', len(data))
        
        for batch in dataloader:
            break

        # normalize the data
        nbatch = self.normalizer.normalize(batch)
        data = nbatch["action"].to(self.device)
        
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            cond = nbatch["obs"].to(self.device)
            data = torch.cat([data, cond], dim=-1)

        self.d = len(data[0].flatten())
        if not diagonal:
            assert len(data) > self.d, f"Use a batch with more samples {len(data[0])} than dimensions {self.d}"
        self.shape = data[0].shape
        self.left = (-1,) + (1,) * (len(self.shape))  # View for left multiplying a batch of samples
        x = data.flatten(start_dim=1)
        if covariance_spectrum:  # May save in cache to avoid processing for every experiment
            self.mu, self.U, self.log_eigs = covariance_spectrum
        else:
            var, self.mu = torch.var_mean(x, 0)
            x = x - self.mu
            if diagonal:
                self.log_eigs = torch.log(var)
                self.U = None  # "U" in this diagonal approximation should be identity, but would need to be handled specially to avoid storing large matrix.
            else:
                _, eigs, self.U = torch.linalg.svd(x, full_matrices=False)  # U.T diag(eigs^2/(n-1)) U = covariance
                self.log_eigs = 2 * torch.log(eigs) - math.log(len(x) - 1)  # Eigs of covariance are eigs**2/(n-1)  of SVD
            # t.save((self.mu, self.U, self.log_eigs), './covariance/cifar_covariance.pt')  # Save like this

        self.log_eigs = self.log_eigs.to(self.device)
        self.mu = self.mu.to(self.device)
        if self.U is not None:
            self.U = self.U.to(self.device)

        # Used to estimate good range for integration
        self.loc_logsnr = -self.log_eigs.mean().item()
        if diagonal:
            # A heuristic, since we won't get good variance estimate from diagonal - use loc/scale from CIFAR.
            self.loc_logsnr, self.scale_logsnr = 6.261363983154297, 3.0976245403289795
        else:
            self.scale_logsnr = torch.sqrt(1 + 3. / math.pi * self.log_eigs.var()).item()

    @property
    def h_g(self):
        """Differential entropy for a N(mu, Sigma), where Sigma matches data, with same dimension as data."""
        return 0.5 * self.d * math.log(2 * math.pi * math.e) + 0.5 * self.log_eigs.sum().item()

    def mmse_g(self, logsnr):
        """The analytic MMSE for a Gaussian with the same eigenvalues as the data in a Gaussian noise channel."""
        return torch.sigmoid(logsnr + self.log_eigs.view((-1, 1))).sum(axis=0)  # *logsnr integration, see note
    
    def logistic_integrate(self, npoints, loc, scale, clip=4., device='cpu', deterministic=False):
        """Return sample point and weights for integration, using
        a truncated logistic distribution as the base, and importance weights.
        These logsnr values are the integration points.
        """
        loc, scale, clip = torch.tensor(loc, device=device), torch.tensor(scale, device=device), torch.tensor(clip, device=device)

        # IID samples from uniform, use inverse CDF to transform to target distribution
        # Create a dedicated random generator
        if deterministic:
            generator = torch.Generator(device=device)
            generator.manual_seed(500)
            ps = torch.rand(npoints, dtype=loc.dtype, device=device, generator=generator)
        else:
            ps = torch.rand(npoints, dtype=loc.dtype, device=device)
        ps = torch.sigmoid(-clip) + (torch.sigmoid(clip) - torch.sigmoid(-clip)) * ps  # Scale quantiles to clip
        logsnr = loc + scale * torch.logit(ps)  # Using quantile function for logistic distribution

        # importance weights
        weights = scale * torch.tanh(clip / 2) / (torch.sigmoid((logsnr - loc)/scale) * torch.sigmoid(-(logsnr - loc)/scale))
        return logsnr, weights

    def nll_sde(self, dataloader):
        SDE = VPSDE(self.noise_scheduler, beta_min=0.1, beta_max=20.0, N=100)
        logp_fn = get_likelihood_fn(SDE, continuous = False, exact = False)

        NLL = 0
        n_samples = 0
        with tqdm.tqdm(dataloader, desc=f"NLL computation_score based = ", 
                        leave=False) as tepoch:
            for batch in tepoch:
                n_samples += len(batch["action"])

                nbatch = self.normalizer.normalize(batch)
                nobs = nbatch['obs']
                naction = nbatch['action']

                B, _, Do = nobs.shape
                To = self.n_obs_steps
                assert Do == self.obs_dim
                T = self.horizon
                Da = self.action_dim

                # build input
                device = self.device
                dtype = self.dtype

                # handle different ways of passing observation
                local_cond = None
                global_cond = None
                trajectory = naction

                if self.obs_as_local_cond:
                    # condition through local feature
                    # all zero except first To timesteps
                    local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
                    local_cond[:,:To] = nobs[:,:To]
                    shape = (B, T, Da)
                    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

                elif self.obs_as_global_cond:
                    # condition throught global feature
                    global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
                    shape = (B, T, Da)
                    if self.pred_action_steps_only:
                        shape = (B, self.n_action_steps, Da)
                        start = To
                        if self.oa_step_convention:
                            start = To - 1
                        end = start + self.n_action_steps
                        trajectory = naction[:,start:end]
                    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                else:
                    # condition through impainting
                    shape = (B, T, Da+Do)
                    cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
                    cond_data[:,:To,Da:] = nobs[:,:To]
                    cond_mask[:,:To,Da:] = True
                    trajectory = torch.cat([naction, nobs], dim=-1)

                logp, z, nfe = logp_fn (self.model, trajectory, local_cond, global_cond)
                NLL+=logp.sum()

        return NLL/n_samples
    
    


    # @torch.no_grad()
    # def test_nll(self, dataloader, epoch, npoints=100, xinterval=None):
    #     """Calculate expected NLL on data at test time.  Main difference is that we can clamp the integration
    #     range, because negative values of MMSE gap we can switch to Gaussian decoder to get zero.
    #     npoints - number of points to use in integration
    #     delta - if the data is discrete, delta is the gap between discrete values.
    #     E.g. delta = 1/127.5 for CIFAR data (0, 255) scaled to -1, 1 range
    #     xinterval - a tuple of the range of the discrete values, e.g. (-1, 1) for CIFAR10 normalized
    #     soft -  using soft discretization if True.
    #     """
        
    #     IT_model.set_noise_scheduler(self.noise_scheduler)
    #     if IT_model.model.training:
    #         print("Warning - estimating test NLL but model is in train mode")
    #     results = {}  # Return multiple forms of results in a dictionary
    #     clip = IT_model.clip
    #     loc, scale = IT_model.loc_scale
    #     logsnr, w = IT_model.logistic_integrate(npoints, loc=loc, scale=scale, clip=clip, device=self.device, deterministic=True)

    #     # sort logsnrs along with weights
    #     logsnr, idx = logsnr.sort()
    #     w = w[idx].to('cpu')

    #     results['logsnr'] = logsnr.to('cpu')
    #     results['w'] = w
    #     mses = []  # Store all MSEs, per sample, logsnr, in an array
    #     total_samples = 0
    #     val_loss = 0
    #     with tqdm.tqdm(dataloader, desc=f"NLL computation {epoch}", 
    #                     leave=False) as tepoch:
    #         for batch in tepoch:
    #             batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
    #             nbatch = self.normalizer.normalize(batch)
    #             nobs = nbatch['obs']
    #             naction = nbatch['action']

    #             B, _, Do = nobs.shape
    #             To = self.n_obs_steps
    #             assert Do == self.obs_dim
    #             T = self.horizon
    #             Da = self.action_dim

    #             # build input
    #             device = self.device
    #             dtype = self.dtype

    #             # handle different ways of passing observation
    #             local_cond = None
    #             global_cond = None
    #             trajectory = naction

    #             if self.obs_as_local_cond:
    #                 # condition through local feature
    #                 # all zero except first To timesteps
    #                 local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
    #                 local_cond[:,:To] = nobs[:,:To]
    #                 shape = (B, T, Da)
    #                 cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
    #                 cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

    #             elif self.obs_as_global_cond:
    #                 # condition throught global feature
    #                 global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
    #                 shape = (B, T, Da)
    #                 if self.pred_action_steps_only:
    #                     shape = (B, self.n_action_steps, Da)
    #                     start = To
    #                     if self.oa_step_convention:
    #                         start = To - 1
    #                     end = start + self.n_action_steps
    #                     trajectory = naction[:,start:end]
    #                 cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
    #                 cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    #             else:
    #                 # condition through impainting
    #                 shape = (B, T, Da+Do)
    #                 cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
    #                 cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    #                 cond_data[:,:To,Da:] = nobs[:,:To]
    #                 cond_mask[:,:To,Da:] = True
    #                 trajectory = torch.cat([naction, nobs], dim=-1)

    #             # construct a batch of data in the form that diffusion model of IT expects
    #             batch = [trajectory, local_cond, global_cond, cond_data, cond_mask]
    #             data = batch[0].to(batch[0].device)
    #             n_samples = len(data)
    #             total_samples += n_samples

    #             val_loss += IT_model.nll([data, ] + batch[1:], xinterval=xinterval) * n_samples

    #             mses.append(torch.zeros(n_samples, len(logsnr)))
    #             for j, this_logsnr in enumerate(logsnr):
    #                 this_logsnr_broadcast = this_logsnr * torch.ones(len(data), device=device)

    #                 # Regular MSE, clamps predictions, but does not discretize
    #                 this_mse = IT_model.mse([data, ] + batch[1:], this_logsnr_broadcast, mse_type='epsilon',
    #                                     xinterval=xinterval).cpu()
    #                 mses[-1][:, j] = this_mse

    #     val_loss /= total_samples

    #     mses = torch.cat(mses, dim=0)  # Concatenate the batches together across axis 0
    #     results['mses-all'] = mses  # Store array of mses for each sample, logsnr
    #     mses = mses.mean(dim=0)  # Average across samples, giving MMSE(logsnr)

    #     results['mses'] = mses
    #     results['mmse_g'] = IT_model.mmse_g(logsnr.to(device)).to('cpu')

    #     # here the results of the integral is clamped to 0 if the MMSE is negative
    #     results['nll (nats)'] = torch.mean(IT_model.h_g - 0.5 * w * torch.clamp(results['mmse_g'] - mses, 0.))
    #     results['nll (bpd)'] = results['nll (nats)'] / math.log(2) / IT_model.d
        
    #     ## Variance (of the mean) calculation - via CLT, it's the variance of the samples (over epsilon, x, logsnr) / n samples.
    #     ## n_samples is number of x samples * number of logsnr samples per x
    #     inds = (results['mmse_g'] - results[
    #         'mses']) > 0  # we only give nonzero estimates in this region (for continuous estimators)
    #     n_samples = results['mses-all'].numel()
    #     wp = w[inds]
    #     results['nll (nats) - var'] = torch.var(
    #         0.5 * wp * (results['mmse_g'][inds] - results['mses-all'][:, inds])) / n_samples
        
    #     results['nll (bpd) - std'] = torch.sqrt(results['nll (nats) - var']) / math.log(2) / IT_model.d
        
    #     return results['nll (bpd)']
    

    