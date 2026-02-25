from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_prob_policy import BaseLowdimProbPolicy
from diffusion_policy.model.diffusion.conditional_prob1_unet1d import BayesianConditionalUnet1D
from diffusion_policy.model.diffusion.conditional_prob2_unet1d import BayesianConditionalUnet1D
from diffusion_policy.model.diffusion.conditional_prob3_unet1d import BayesianConditionalUnet1D
from diffusion_policy.model.diffusion.conditional_prob4_unet1d import BayesianConditionalUnet1D
from diffusion_policy.model.diffusion.conditional_prob5_unet1d import BayesianConditionalUnet1D

from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel


from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.SDE import VPSDE
#from diffusion_policy.common.likelihood import get_likelihood_fn
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to

import math 
import tqdm
import numpy as np
import hydra

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
            stochastic = False,
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
                local_cond=local_cond, global_cond=global_cond,stochastic = stochastic)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], stochastic=False) -> Dict[str, torch.Tensor]:
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

    def compute_loss(self, batch, stochastic=False, train=True):
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
        
        if train:    
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=trajectory.device).long()
        else:
            generator = torch.Generator(device=trajectory.device)
            generator.manual_seed(42)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (bsz,), device=trajectory.device, generator=generator).long()
        
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
            local_cond=local_cond, global_cond=global_cond,
            stochastic = stochastic)

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
    
    def compute_bound(self, batch, n_bound, objective = "fquad", delta = 0.025, kl_penalty = 0.005, mc_sampling=1000, stochastic = True, bounded = False, train=True):
        
        # DM emprical risk
        loss_emp = self.compute_loss(batch, stochastic=stochastic, train=train)
        scale = 2.0
        if bounded:
            loss_emp_scaled = loss_emp/scale
        else:
            loss_emp_scaled = loss_emp
        
        if objective == "fquad":
            # compute kl divergence of the network
            kl = self.model.compute_kl()
            # compute the PAC-Bayes bound
            kl_ratio = torch.div((kl*kl_penalty + np.log((2*np.sqrt(n_bound))/delta)), 2*n_bound)
            # scale the empirical risk to be inside [0,1]
            first_term = torch.sqrt(loss_emp_scaled + kl_ratio)
            second_term = torch.sqrt(kl_ratio)
            loss_sum = torch.pow(first_term + second_term, 2)
        
        elif objective == "classic":
            # compute kl divergence of the network
            kl = self.model.compute_kl()
            # compute the PAC-Bayes bound
            kl_ratio = torch.div((kl*kl_penalty + np.log((2 * np.sqrt(n_bound)) / delta)), 2*n_bound)
            loss_sum = loss_emp_scaled + torch.sqrt(kl_ratio)
        
        elif objective == "friendly":
            # ipdb.set_trace()
            kl = self.model.compute_kl()
            # compute the PAC-Bayes bound
            kl_ratio = torch.div((kl*kl_penalty + np.log((2 * np.sqrt(n_bound)) / delta)), n_bound)
            first_term = torch.sqrt(2*loss_emp_scaled * kl_ratio)
            second_term = 2*kl_ratio
            loss_sum = loss_emp_scaled + first_term + second_term

        elif objective == "bbb":
            # ipdb.set_trace()
            kl = self.model.compute_kl()
            loss_sum = loss_emp_scaled + kl_penalty * (kl / n_bound)
        else:
            raise RuntimeError(f"Wrong objective {self.objective}")

        return loss_sum, loss_emp, kl

# ========================== Compute upper bound on nll ==========================
    def noisy_channel(self, x, logsnr):
        """
        Vectorized DDPM forward process.
        x: (B, K, ...)
        logsnr: (B, K)
        """
        alpha = torch.sigmoid(logsnr) # (B, K)

        # Map alpha → nearest scheduler timestep
        scheduler_alpha = self.noise_scheduler.alphas_cumprod.to(self.device) # (T,)
        diff = torch.abs(alpha[..., None] - scheduler_alpha)
        timesteps = diff.argmin(dim=-1)  # (B, K)

        eps = torch.randn_like(x)

        noisy_x = self.noise_scheduler.add_noise(
            x.flatten(0, 1),
            eps.flatten(0, 1),
            timesteps.flatten(),
        ).view_as(x)

        return noisy_x, timesteps, eps

    def mse(self, batch, logsnr, stochastic=False):
        """
        Compute per-sample MSE between predicted noise eps_hat and true eps.
        """
        x = batch[0].to(self.device)
        # Unpack conditioning
        local_cond, global_cond, cond_data, cond_mask = batch[1:]

        B = x.shape[0]
        K = logsnr.shape[0]

        # Expand x and logsnr
        x = x[:, None].expand(B, K, *x.shape[1:])
        logsnr = logsnr[None].expand(B, K)

        # Forward diffusion
        z, timesteps, eps = self.noisy_channel(x, logsnr)

        # Apply imputation
        local_cond = local_cond[:, None, ...].expand(B,K,*local_cond.shape[1:]) if local_cond is not None else None  # (B, K, T, D)
        global_cond = global_cond[:, None, ...].expand(B,K,*global_cond.shape[1:]) if global_cond is not None else None # (B, K, T, D)
        cond_mask = cond_mask[:, None, ...].expand(B, K, *cond_mask.shape[1:])
        cond_data = cond_data[:, None, ...].expand(B, K, *cond_data.shape[1:])
        
        z[cond_mask] = cond_data[cond_mask]

        # Noise Prediction
        output = self.model(
            z.flatten(0, 1),
            timesteps.flatten(),
            local_cond=local_cond.flatten(0, 1) if local_cond is not None else None ,
            global_cond=global_cond.flatten(0, 1) if global_cond is not None else None,
            stochastic=stochastic,
        ).view_as(eps)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = eps
        elif pred_type == 'sample':
            target = x
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # MSE loss
        loss_mask = (~cond_mask).float()
        sse = ((target - output) ** 2) * loss_mask

        return sse.flatten(start_dim=2).sum(dim=2)  # (B, K)

    @torch.no_grad()
    def nll_bound(self, dataloader, epoch, npoints=100, stochastic = False):
        """Calculate expected NLL on data at test time.  Main difference is that we can clamp the integration
        range, because negative values of MMSE gap we can switch to Gaussian decoder to get zero.
        npoints -> number of points to use in integration
        """
        if self.model.training:
            print("Warning - estimating test NLL but model is in train mode")
        
        results = {}  # Return multiple forms of results in a dictionary
        clip = 4
        loc, scale = self.loc_scale
        logsnr, w = self.logistic_integrate(npoints, loc=loc, scale=scale, clip=clip, device=self.device, deterministic=True) # logsnr:(K,), w:(K,)

        # sort logsnrs along with weights
        logsnr, idx = logsnr.sort()
        w = w[idx]

        mses = []  # Store all MSEs, per sample, logsnr, in an array
        with tqdm.tqdm(dataloader, desc=f"NLL computation at epoch = {epoch}", 
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
                this_mse = self.mse(batch, logsnr, stochastic=stochastic)
                mses.append(this_mse)
            
        mses = torch.cat(mses, dim=0)       # Concatenate the batches together across axis 0
        results['mses-all'] = mses          # Store array of mses for each sample, logsnr
        mses = mses.mean(dim=0)             # Average across samples, giving MMSE(logsnr)

        results['mses'] = mses
        results['mmse_g'] = self.mmse_g(logsnr)

        # here the results of the integral is clamped to 0 if the MMSE is negative
        results['nll (nats)'] = self.h_g - torch.mean(0.5 * w * torch.clamp(results['mmse_g'] - mses, 0.))
        results['nll (bpd)'] = results['nll (nats)'] / (math.log(2) * self.d)

        # results['logsnr'] = logsnr.to('cpu')
        # results['w'] = w.to('cpu')
        
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
        mmse = torch.sigmoid(logsnr[None, :] + self.log_eigs[:, None]).sum(dim=0)   # (K,)
        return mmse
    
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

    def rho_stats(self):
        stats = {}
        for name, p in self.model.named_parameters():
            if name.endswith(".weight.rho") or name.endswith(".bias.rho"):
                sigma = torch.nn.functional.softplus(p).detach()
                stats[name] = {
                    "mean": sigma.mean().item(),
                    "median": sigma.median().item(),
                    "max": sigma.max().item()
                }
        return stats
        
    def train_prior(self, prior_dataset, cfg):

        global_step = 0
        epoch = 0

        model: DiffusionUnetLowdimPolicy
        model = hydra.utils.instantiate(cfg.prior_policy)
        # ema_model: DiffusionUnetLowdimPolicy = None
        # if cfg.prior_training.use_ema:
        #     ema_model = copy.deepcopy(model)

        # configure training state
        optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=model.parameters())
        
        # configure dataset and normalizer
        train_dataloader = DataLoader(prior_dataset, **cfg.prior_dataloader)
        normalizer = prior_dataset.get_normalizer()
        model.set_normalizer(normalizer)
        # if cfg.prior_training.use_ema:
        #     ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.prior_training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.prior_training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.prior_training.num_epochs) \
                    // cfg.prior_training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=global_step-1
        )

        # # configure ema
        # ema: EMAModel = None
        # if cfg.prior_training.use_ema:
        #     ema = hydra.utils.instantiate(
        #         cfg.ema,
        #         model=ema_model)
        
        # device transfer
        device = torch.device(cfg.prior_training.device)
        model.to(device)
        # if ema_model is not None:
        #     ema_model.to(device)
        optimizer_to(optimizer, device)

        for local_epoch_idx in range(cfg.prior_training.num_epochs):
            with tqdm.tqdm(train_dataloader, desc=f"Prior Training epoch {local_epoch_idx}",
                leave=False, mininterval=cfg.prior_training.tqdm_interval_sec) as tepoch:
                    
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        raw_loss = model.compute_loss(batch, train=True)
                        loss = raw_loss / cfg.prior_training.gradient_accumulate_every
                        loss.backward()
                        raw_loss_cpu = raw_loss.item()

                        # step optimizer
                        if global_step % cfg.prior_training.gradient_accumulate_every == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # # update ema
                        # if cfg.prior_training.use_ema:
                        #     ema.step(model)

                        # logging
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        global_step += 1

        return model.model.eval()
    
    def prior_post_initialize(
    self,
    prior_model,
    rho_post,
    rho_prior=-5.0,
    initialize_from_prior=True
    ):
        with torch.no_grad():
            for name, param in self.model.state_dict().items():
                if name.endswith(".weight_prior.mu"):
                    w0_name = name.replace(".weight_prior.mu", ".weight")
                    param.copy_(prior_model.state_dict()[w0_name])

                elif name.endswith(".bias_prior.mu"):
                    b0_name = name.replace(".bias_prior.mu", ".bias")
                    param.copy_(prior_model.state_dict()[b0_name])

                elif name.endswith(".weight.mu"):
                    if initialize_from_prior:
                        w0_name = name.replace(".weight.mu", ".weight")
                        param.copy_(prior_model.state_dict()[w0_name])
                
                elif name.endswith(".bias.mu"):
                    if initialize_from_prior:
                        b0_name = name.replace(".bias.mu", ".bias")
                        param.copy_(prior_model.state_dict()[b0_name])
                
                elif name.endswith(".bias_prior.rho") or name.endswith(".weight_prior.rho"):
                    param.fill_(rho_prior)

                elif name.endswith(".bias.rho") or name.endswith(".weight.rho"):
                    param.fill_(rho_post)

                else:
                    if name in prior_model.state_dict():
                        param.copy_(prior_model.state_dict()[name])
                    else:
                        # Allow unmatched params (e.g. new Bayesian-only params)
                        pass

    # =============== Compute Reconstruction Loss of PAC-Bayes Bounds of Mbacke =======================
    # see https://arxiv.org/pdf/2312.05989
    
    @torch.no_grad()
    def compute_action_reconst_loss(self, dataloader, cfg):
        loss_rec = 0
        n_total_samples = 0
        
        with tqdm.tqdm(dataloader, desc=f"Reconstruction Loss", 
                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:

                for batch in tepoch:
                    batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                    
                    # extract observation
                    obs_dict = {'obs': batch['obs']}
                    ref_action = batch["action"]

                    B = ref_action.shape[0]
                    n_total_samples += B

                    if cfg.pred_action_steps_only:
                        start = cfg.n_obs_steps - 1
                        end = start + cfg.n_action_steps
                        ref_action = ref_action[:, start:end]

                    for _ in range (cfg.num_mc_samples):
                        result = self.predict_action(obs_dict, cfg.eval.stochastic)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                        else:
                            pred_action = result['action_pred']

                        batch_loss = torch.linalg.norm(
                                                pred_action - ref_action,
                                                ord=2,
                                                dim=(1, 2)
                                            )  # (B,)
                        # compute reconstruction loss
                        loss_rec += batch_loss.sum()
        
        return loss_rec/(cfg.num_mc_samples*n_total_samples)

    
    # def nll_sde(self, dataloader, stochastic=False):
    #     SDE = VPSDE(self.noise_scheduler, beta_min=0.1, beta_max=20.0, N=1000)
    #     logp_fn = get_likelihood_fn(SDE, continuous = False, exact = False)

    #     NLL = 0
    #     n_samples = 0
    #     with tqdm.tqdm(dataloader, desc=f"NLL computation_score based = ", 
    #                     leave=False) as tepoch:
    #         for batch in tepoch:
    #             n_samples += len(batch["action"])

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

    #             logp, z, nfe = logp_fn (self.model, trajectory, local_cond, global_cond, stochastic=stochastic)
    #             NLL+=logp.sum()

    #     return NLL/n_samples
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
    

    