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
            'action_pred': action_pred
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

    def compute_loss(self, batch):
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
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
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
    def compute_reconst_loss_T(self, batch, noise, loss_type):
        
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

        # # Sample noise that we'll add to the input
        # noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # # set timestep to last noising step for each image
        # timestep = self.noise_scheduler.timesteps[0]

        # # Add noise to the clean images according to the noise magnitude at each timestep
        # # (this is the forward diffusion process)
        # noisy_trajectory = self.noise_scheduler.add_noise(
        #     trajectory, noise, timestep)

        noisy_trajectory = noise.clone()
        
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

        # # finally make sure conditioning is enforced
        #noisy_trajectory[cond_mask] = trajectory[cond_mask]
        
        # prediction
        #! what should I do for the case in which the actions and conditions are
        #! concatenated to eachother, should I compare the whole trajectory or 
        #! the actions. if I only consider the actions, then I can simply consider 
        #! the predicted actions and then compute the loss, there is no need for 
        #! loss_mask
        naction_pred = noisy_trajectory[...,:Da]
        
        # get action
        # if self.pred_action_steps_only:
        #     action = naction_pred
        # else:
        #     start = To
        #     if self.oa_step_convention:
        #         start = To - 1
        #     end = start + self.n_action_steps
        #     action = naction_pred[:,start:end]
        
        # # compute loss mask
        #loss_mask = ~cond_mask

        # ## Loss of Lipshicts
        # rand_sample1 = np.random.randint(0, len(batch))
        # #rand_sample2 = np.random.randint(0, len(batch))
        # obs1 = batch["obs"][rand_sample1]
        # obs2 = batch['obs'][rand_sample1]
        # obs = torch.stack((obs1, obs2))
        # train_lips_const, train_lips_const_prod = self.lip_const(obs)

        # # compute loss
        if loss_type == "MSE":
            loss = F.mse_loss(naction_pred, trajectory[...,:Da])
            # loss_lips = F.mse_loss(train_lips_const, self.noise_scheduler.k_lip_t[1:].to(device))
            #  loss = F.mse_loss(noisy_trajectory, trajectory, reduction='none')
            #  loss = loss * loss_mask.type(loss.dtype)
            #  loss = reduce(loss, 'b ... -> b (...)', 'mean')
            #  loss = loss.mean()
        elif loss_type == "RMSE":
            loss = torch.sqrt(F.mse_loss(naction_pred, trajectory[...,:Da]))
            #loss = torch.linalg.norm(naction_pred - trajectory[...,:Da], ord=2, dim=(1, 2)).mean()
            # loss_lips = torch.linalg.norm(train_lips_const - self.noise_scheduler.k_lip_t[1:].to(device))

            # loss = (naction_pred - naction) ** 2   # squared error
            # #loss = loss * loss_mask.type(loss.dtype)
            # loss = torch.sqrt(loss.sum(dim=(1, 2)))   # L2 norm over dims (1,2)
            # loss = loss.mean()  # average over batch

        else: 
            loss = torch.linalg.norm(naction_pred - trajectory[...,:Da], ord=2, dim=(1, 2)).mean()
        # loss_sum = 0
        # for t in self.noise_scheduler.timesteps:
            
        #     # 1. Compute the GroundTruth denoising
        #     GT_denoised = self.noise_scheduler.groundT_denoise(
        #         noisy_trajectory, t, trajectory)
            
        #     # 2.2 apply conditioning
        #     GT_denoised[cond_mask] = trajectory[cond_mask]
        
        #     # 3. predict model output
        #     model_output = self.model(noisy_trajectory, t, 
        #         local_cond=local_cond, global_cond=global_cond)

        #     # 4. compute previous image: x_t -> x_t-1
        #     noisy_trajectory = self.noise_scheduler.step(
        #         model_output, t, noisy_trajectory,
        #         generator = None,
        #         **self.kwargs
        #         ).prev_sample

        #     # 5. apply conditioning
        #     noisy_trajectory[cond_mask] = trajectory[cond_mask]

        #     # 6.compute loss mask
        #     loss_mask = ~cond_mask

        #     # 7.compute loss
        #     if loss_type == "MSE":
        #         loss = F.mse_loss(noisy_trajectory, GT_denoised, reduction='none')
        #         loss = loss * loss_mask.type(loss.dtype)
        #         loss = reduce(loss, 'b ... -> b (...)', 'mean')
        #         loss = loss.mean()
        #     else:
        #         loss = (noisy_trajectory - GT_denoised) ** 2   # squared error
        #         loss = loss * loss_mask.type(loss.dtype)
        #         loss = torch.sqrt(loss.sum(dim=(1, 2)))   # L2 norm over dims (1,2)
        #         loss = loss.mean()  # average over batch

        #     loss_sum += loss
        
        return loss 
    
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
        scheduler.set_timesteps(self.num_inference_steps)
        
        # set a variable to save denoising results
        denoise_step = collections.deque(maxlen=self.num_inference_steps)

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
    

    # ========= PAC-Bayes Bounds ============
    def test_nll(self, IT_model, dataloader, epoch, npoints=100, xinterval=None):
        """Calculate expected NLL on data at test time.  Main difference is that we can clamp the integration
        range, because negative values of MMSE gap we can switch to Gaussian decoder to get zero.
        npoints - number of points to use in integration
        delta - if the data is discrete, delta is the gap between discrete values.
        E.g. delta = 1/127.5 for CIFAR data (0, 255) scaled to -1, 1 range
        xinterval - a tuple of the range of the discrete values, e.g. (-1, 1) for CIFAR10 normalized
        soft -  using soft discretization if True.
        """
        IT_model.set_noise_scheduler(self.noise_scheduler)
        if IT_model.model.training:
            print("Warning - estimating test NLL but model is in train mode")
        results = {}  # Return multiple forms of results in a dictionary
        clip = IT_model.clip
        loc, scale = IT_model.loc_scale
        logsnr, w = IT_model.logistic_integrate(npoints, loc=loc, scale=scale, clip=clip, device=self.device, deterministic=True)

        # sort logsnrs along with weights
        logsnr, idx = logsnr.sort()
        w = w[idx].to('cpu')

        results['logsnr'] = logsnr.to('cpu')
        results['w'] = w
        mses = []  # Store all MSEs, per sample, logsnr, in an array
        total_samples = 0
        val_loss = 0
        with tqdm.tqdm(dataloader, desc=f"NLL computation {epoch}", 
                        leave=False) as tepoch:
            for batch in tepoch:
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
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
                data = batch[0].to(batch[0].device)
                n_samples = len(data)
                total_samples += n_samples

                val_loss += IT_model.nll([data, ] + batch[1:], xinterval=xinterval) * n_samples

                mses.append(torch.zeros(n_samples, len(logsnr)))
                for j, this_logsnr in enumerate(logsnr):
                    this_logsnr_broadcast = this_logsnr * torch.ones(len(data), device=device)

                    # Regular MSE, clamps predictions, but does not discretize
                    this_mse = IT_model.mse([data, ] + batch[1:], this_logsnr_broadcast, mse_type='epsilon',
                                        xinterval=xinterval).cpu()
                    mses[-1][:, j] = this_mse

        val_loss /= total_samples

        mses = torch.cat(mses, dim=0)  # Concatenate the batches together across axis 0
        results['mses-all'] = mses  # Store array of mses for each sample, logsnr
        mses = mses.mean(dim=0)  # Average across samples, giving MMSE(logsnr)

        results['mses'] = mses
        results['mmse_g'] = IT_model.mmse_g(logsnr.to(device)).to('cpu')

        # here the results of the integral is clamped to 0 if the MMSE is negative
        results['nll (nats)'] = torch.mean(IT_model.h_g - 0.5 * w * torch.clamp(results['mmse_g'] - mses, 0.))
        results['nll (bpd)'] = results['nll (nats)'] / math.log(2) / IT_model.d
        
        ## Variance (of the mean) calculation - via CLT, it's the variance of the samples (over epsilon, x, logsnr) / n samples.
        ## n_samples is number of x samples * number of logsnr samples per x
        inds = (results['mmse_g'] - results[
            'mses']) > 0  # we only give nonzero estimates in this region (for continuous estimators)
        n_samples = results['mses-all'].numel()
        wp = w[inds]
        results['nll (nats) - var'] = torch.var(
            0.5 * wp * (results['mmse_g'][inds] - results['mses-all'][:, inds])) / n_samples
        
        results['nll (bpd) - std'] = torch.sqrt(results['nll (nats) - var']) / math.log(2) / IT_model.d
        
        return results['nll (bpd)']
    
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
