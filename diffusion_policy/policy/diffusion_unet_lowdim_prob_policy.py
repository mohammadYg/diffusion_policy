from typing import Dict
from matplotlib.pyplot import cla
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_prob_policy import BaseLowdimProbPolicy
from diffusion_policy.model.diffusion.conditional_prob_unet1d import BayesianConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.SDE import VPSDE
from diffusion_policy.common.likelihood import get_likelihood_fn

import collections
import math 
import tqdm
import numpy as np

class DiffusionUnetLowdimProbPolicy(BaseLowdimProbPolicy):
    def __init__(self, 
            model: BayesianConditionalUnet1D,
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
            stochastic = False, clamping = True,
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
                local_cond=local_cond, global_cond=global_cond, 
                stochastic = stochastic, clamping = clamping)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], stochastic=False, clamping=True) -> Dict[str, torch.Tensor]:
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
            stochastic=stochastic,
            clamping = clamping,
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

    def compute_bound(self, batch, n_bound, delta, kl_weight, mc_sampling=1000, stochastic = True, clamping = False, bounded = False):
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
        # if not self.model.training:
        #     error_mc = 0.0
        #     for _ in range (mc_sampling):
        #         pred = self.model(noisy_trajectory, timesteps, 
        #             local_cond=local_cond, global_cond=global_cond,
        #             stochastic = stochastic, clamping = clamping)

        #         pred_type = self.noise_scheduler.config.prediction_type 
        #         if pred_type == 'epsilon':
        #             target = noise
        #         elif pred_type == 'sample':
        #             target = trajectory
        #         else:
        #             raise ValueError(f"Unsupported prediction type {pred_type}")

        #         loss_dm = F.mse_loss(pred, target, reduction='none')
        #         loss_dm = loss_dm * loss_mask.type(loss_dm.dtype)
        #         loss_dm = reduce(loss_dm, 'b ... -> b (...)', 'mean')
        #         loss_dm = loss_dm.mean()
        #         error_mc += loss_dm
        #     loss_emp = error_mc/mc_sampling
        # else:
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond,
            stochastic = stochastic, clamping = clamping)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss_dm = F.mse_loss(pred, target, reduction='none')
        loss_dm = loss_dm * loss_mask.type(loss_dm.dtype)
        loss_dm = reduce(loss_dm, 'b ... -> b (...)', 'mean')
        loss_dm = loss_dm.mean()
        loss_emp = loss_dm
        
        kl = self.model.compute_kl()
        
        loss_kl = torch.div(
                2*(kl_weight*kl + np.log((2*np.sqrt(n_bound))/delta)), n_bound)

        # scale the empirical risk to be inside [0,1]
        scale = 2.0
        if bounded:
            loss_sum = loss_emp/scale + loss_kl + torch.sqrt((loss_emp/scale)*loss_kl)
        else:
            loss_sum = loss_emp + loss_kl + torch.sqrt(loss_emp*loss_kl)

        # loss_kl = torch.div(
        #         kl_weight*kl + np.log((2*np.sqrt(n_bound))/delta), 2*n_bound)
        # # scale the empirical risk to be inside [0,1]
        # scale = 2.0
        # if bounded:
        #     loss_sum = loss_emp/scale + torch.sqrt(loss_kl)
        # else:
        #     loss_sum = loss_emp + torch.sqrt(loss_kl)
        
        return loss_sum, loss_emp, kl

    # ========= Compute Reconstruction Loss of PAC-Bayes Bounds for case where we do inference from last step ============
    def compute_reconst_loss_T(self, batch, loss_type, stochastic=False, clamping=False):
        
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
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        
        # # set timestep to last noising step for each image
        timestep = self.noise_scheduler.timesteps[0]

        # # Add noise to the clean images according to the noise magnitude at each timestep
        # # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timestep)
        
        # set step values
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
    
        for t in self.noise_scheduler.timesteps:
            
            # 1. apply conditioning
            noisy_trajectory[cond_mask] = trajectory[cond_mask]
    
            # 2. predict model output
            model_output = self.model(noisy_trajectory, t, 
                local_cond=local_cond, global_cond=global_cond,
                stochastic = stochastic, clamping = clamping)
            
            # 3. compute previous image: x_t -> x_t-1
            noisy_trajectory = self.noise_scheduler.step(
                model_output, t, noisy_trajectory,
                generator = None,
                **self.kwargs
                ).prev_sample
               
        #finally make sure conditioning is enforced
        noisy_trajectory[cond_mask] = trajectory[cond_mask]
        
        # prediction
        naction_pred = noisy_trajectory[...,:Da]

        # # compute loss
        if loss_type == "MSE":
            loss = F.mse_loss(naction_pred, trajectory[...,:Da])
        elif loss_type == "RMSE":
            loss = torch.sqrt(F.mse_loss(naction_pred, trajectory[...,:Da]))
        else: 
            loss = torch.linalg.norm(naction_pred - trajectory[...,:Da], ord=2, dim=(1, 2)).mean()
        
        return loss 
    
    # ========================== Lipschitz Constant of the Denoising ==========================
    def lip_const(self, obs, stochastic=False, clamping=False):
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
                local_cond=local_cond, global_cond=global_cond,
                stochastic= stochastic, clamping=clamping)

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
    
# ========================== Compute upper bound on nll ==========================
    def noisy_channel(self, x, logsnr):
        '''
        The noise channel is working as the forward process of the DDPM
        here, we need to computet the corresponding diffusion timestep to the logsnr
        '''
        # compare each sigmoid(logsnr) (or sigmoid(alpha)) with alpha_cumprod of DDIM to find the closest value of alpha_cumprod to each sigmoid(logsnr)
        # Then find the correspoing t of logsnr
        diff = torch.abs(torch.sigmoid(logsnr).unsqueeze(1) - self.noise_scheduler.alphas_cumprod.unsqueeze(0).to(self.device))
        timestep = torch.argmin(diff, dim=1)   # shape (N,)

        eps = torch.randn((len(logsnr),) + self.shape, dtype=x.dtype, device=x.device)
        noisy_data = self.noise_scheduler.add_noise(x, eps, timestep)
        return noisy_data, timestep, eps

    def mse(self, batch, logsnr, mse_type='epsilon', xinterval=None, stochastic=False, clamping = False):
        """Return MSE curve either conditioned or not on y.
        x_hat = z/sqrt(snr) - eps_hat(z, snr)/sqrt(snr),
        so x_hat - x = (eps - eps_hat(z, snr))/sqrt(snr).
        And we actually reparametrize eps_hat to depend on eps(z/sqrt(1+snr), snr)
        Options are:
        mse_type: {"epsilon" or "x"), for error in predicting noise, or predicting data.
        xinterval: Whether predictions should be clamped to some interval.
        delta: If provided, hard round to the nearest discrete value
        soft: If provided, soft round to the nearest discrete value
        """
        x = batch[0].to(self.device)  # assume iterator gives other things besides x in list (e.g. y)
        z, timesteps, eps = self.noisy_channel(x, logsnr)


        local_cond = batch[1]
        global_cond = batch[2]
        cond_data = batch[3]
        cond_mask = batch[4]

        z[cond_mask] = cond_data[cond_mask]

        eps_hat = self.model(z, timesteps, local_cond=local_cond, global_cond=global_cond,
                             stochastic = stochastic, clamping= clamping)
        
        err_eps = (eps - eps_hat).flatten(start_dim=1) 
        mse_eps = torch.einsum('ij,ij->i', err_eps, err_eps)  # MSE for epsilon
        # if mse_type == 'epsilon':
        #     return mse_x * torch.exp(logsnr)  # Special form of x_hat, eps_hat leads to this relation
        # elif mse_type == 'x':
        #     return mse_x
        
        # x_hat = torch.sqrt(1 + torch.exp(-logsnr.view(self.left))) * z - eps_hat * torch.exp(-logsnr.view(self.left) / 2)
        # if xinterval:
        #     x_hat = torch.clamp(x_hat, xinterval[0], xinterval[1])  # clamp predictions to not fall outside of range
        # err = (x - x_hat).flatten(start_dim=1)  # Flatten for, e.g., image data
        # mse_x = torch.einsum('ij,ij->i', err, err)  # MSE for epsilon
        # if mse_type == 'epsilon':
        #     return mse_x * torch.exp(logsnr)  # Special form of x_hat, eps_hat leads to this relation
        # elif mse_type == 'x':
        #     return mse_x
        return mse_eps

    def nll(self, batch, logsnr_samples_per_x=1, xinterval=None, stochastic = False, clamping = False):
        """-log p(x) (or -log p(x|y) if y is provided) estimated for a batch."""
        nll = 0
        for _ in range(logsnr_samples_per_x):
            logsnr, w = self.logistic_integrate(len(batch[0]), *self.loc_scale, device=self.device)
            mses = self.mse(batch, logsnr, mse_type='epsilon', xinterval=xinterval, stochastic=stochastic, clamping=clamping)
            nll += self.loss(mses, logsnr, w) / logsnr_samples_per_x
        return nll

    def loss(self, mses, logsnr, w):
        """
        Returns the (per-sample) losses from MSEs, for convenience adding constants to match
        with NLL expression.
        :param mses:  Mean square error for *epsilon*
        :param logsnr:  Log signal to noise ratio
        :param w:  Integration weights
        :return: loss, -log p(x) estimate
        """
        mmse_gap = mses - self.mmse_g(
            logsnr)  # The "scale" does not change the integral, but may be more numerically stable.
        loss = self.h_g + 0.5 * (w * mmse_gap).mean()
        return loss  # *logsnr integration, see paper

    @torch.no_grad()
    def test_nll(self, dataloader, epoch, npoints=100, xinterval=None, stochastic = False, clamping=False):
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

                val_loss += self.nll([data, ] + batch[1:], xinterval=xinterval, stochastic=stochastic, clamping=clamping).cpu() * n_samples

                mses.append(torch.zeros(n_samples, len(logsnr)))
                for j, this_logsnr in enumerate(logsnr):
                    this_logsnr_broadcast = this_logsnr * torch.ones(len(data), device=self.device)

                    # Regular MSE, clamps predictions, but does not discretize
                    this_mse = self.mse([data, ] + batch[1:], this_logsnr_broadcast, mse_type='epsilon',
                                        xinterval=xinterval, stochastic=stochastic, clamping=clamping).cpu()
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

    def nll_sde(self, dataloader, stochastic=False, clamping=False):
        SDE = VPSDE(self.noise_scheduler, beta_min=0.1, beta_max=20.0, N=1000)
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

                logp, z, nfe = logp_fn (self.model, trajectory, local_cond, global_cond, stochastic=stochastic, clamping=clamping)
                NLL+=logp.sum()

        return NLL/n_samples
    # # ========= PAC-Bayes Bounds ============
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
    

    