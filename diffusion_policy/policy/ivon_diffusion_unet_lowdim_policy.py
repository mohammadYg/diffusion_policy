from typing import Dict
import math
import torch
import torch.nn.functional as F
from einops import reduce
import tqdm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_pac_policy import BaseLowdimPacPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply


class IvonDiffusionUnetLowdimPolicy(BaseLowdimPacPolicy):
    """
    IVON counterpart of DiffusionUnetLowdimPolicy: same model
    (ConditionalUnet1D), same noise scheduler, same compute_loss/
    predict_action/nll_bound/compute_action_reconst_loss logic - the only
    real difference is that this network is trained by the IVON optimizer
    instead of AdamW, which turns it into a Bayesian (mean-field Gaussian
    posterior) network at the optimizer level, with no architectural
    changes at all: no ProbConv1d/ProbLinear/Gaussian layers, no learnable
    `rho`, no explicit weight_prior/bias_prior buffers anywhere.

    The mean-field Gaussian posterior q(theta) = N(mu, diag(sigma^2)) is
    owned entirely by the IVON optimizer that trains this policy's model:
        - mu      = this model's own current parameters (nn.Parameter values)
        - sigma^2 = 1 / (ess * (hess + weight_decay))
    where `hess` is IVON's internally-tracked Hessian estimate (see
    ivon._ivon.IVON._new_hess / ._sample_params). ess/weight_decay are set
    by the workspace as ess=len(dataset), weight_decay=1/(ess*prior_std^2)
    for an implicit N(0, prior_std^2*I) prior (default prior_std=0.3,
    matching the IVON paper's own CIFAR-10/100/TinyImageNet setting - see
    train_ivon_diffusion_unet_lowdim_workspace.py's docstring for why
    prior_std=1 was tried first and found to let `hess` collapse over long
    runs, and for the paper's actual per-experiment prior_std values).
    The prior's mean really is exactly 0: IVON's mean update
    (ivon._ivon.IVON._new_param_averages) is
    `mu <- mu - lr*(momentum/debias + weight_decay*mu)/(hess+weight_decay)`
    - the only prior-related term is `weight_decay*mu`, proportional to mu
    itself with no additive offset, so with zero gradient signal it decays
    any mu toward exactly 0 (verified empirically, not just by inspection).

    Usage (see train_ivon_diffusion_unet_lowdim_workspace.py):
        policy = IvonDiffusionUnetLowdimPolicy(...)
        # Scoped to policy.model.parameters(), NOT policy.parameters(): the
        # policy also owns non-network parameters (e.g. the normalizer's)
        # that never appear in the loss graph, and IVON (unlike AdamW)
        # raises if any tracked parameter never receives a gradient.
        optimizer = ivon.IVON(policy.model.parameters(), lr=..., ess=ess,
                               weight_decay=1.0 / (ess * prior_std ** 2))
        policy.set_ivon_optimizer(optimizer)   # needed for stochastic sampling

        for _ in range(num_mc_samples):
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                policy.compute_loss(batch).backward()
        optimizer.step()
    """

    def __init__(self,
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            pred_action_steps_only=False,
            oa_step_convention=False,
            **kwargs):
        super().__init__()
        assert obs_as_global_cond, (
            "IvonDiffusionUnetLowdimPolicy only supports obs_as_global_cond=True: "
            "the plain ConditionalUnet1D it wraps has no local_cond pathway "
            "(unlike the Bayesian BayesianConditionalUnet1D variants)."
        )
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
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
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # Set via set_ivon_optimizer() once the workspace has built the
        # IVON optimizer (which needs self.parameters(), so it can only be
        # constructed after this policy exists). Needed for the stochastic
        # sampling in predict_action()/predict_action_bma().
        self.ivon_optimizer = None

    def set_ivon_optimizer(self, optimizer):
        self.ivon_optimizer = optimizer

    # ========= inference  ============
    def conditional_sample(self, condition_data, condition_mask,
            global_cond=None, generator=None, **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, global_cond=global_cond)
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator, **kwargs).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], stochastic=False) -> Dict[str, torch.Tensor]:
        """
        stochastic=True draws ONE posterior weight sample from IVON (via
        `ivon_optimizer.sampled_params`) and holds it fixed for the entire
        denoising loop (all `num_inference_steps` calls to self.model) -
        matching the "one coherent network sample per generated
        trajectory" convention used by the Bayes-by-Backprop policies
        elsewhere in this repo. A fresh sample is drawn every time
        predict_action() is called - it is NOT held across separate calls
        (e.g. across successive env-rollout steps).

        stochastic=False (the default, and what every caller in this file
        uses - matching DiffusionUnetLowdimPolicy.predict_action, which has
        no such option at all) uses the posterior mean, i.e. this model's
        current parameters exactly as they sit - the plain, deterministic
        behaviour. If stochastic=True but no IVON optimizer has been
        attached (e.g. this policy is being used standalone, outside
        training), this silently falls back to the same mean behaviour.
        """
        assert 'obs' in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        device = self.device
        dtype = self.dtype

        global_cond = nobs[:, :To].reshape(B, -1)
        shape = (B, T, Da)
        if self.pred_action_steps_only:
            shape = (B, self.n_action_steps, Da)
        cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if stochastic and self.ivon_optimizer is not None:
            with self.ivon_optimizer.sampled_params(train=False):
                nsample = self.conditional_sample(
                    cond_data, cond_mask, global_cond=global_cond, **self.kwargs)
        else:
            nsample = self.conditional_sample(
                cond_data, cond_mask, global_cond=global_cond, **self.kwargs)

        naction_pred = nsample[..., :Da]
        action_pred_normalized = naction_pred
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1 if self.oa_step_convention else To
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred,
            'action_pred_normalized': action_pred_normalized,
        }

    @torch.no_grad()
    def predict_action_bma(self, obs_dict: Dict[str, torch.Tensor], num_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Bayesian model averaging: draws `num_samples` independent posterior
        weight samples from IVON, runs a full (independent) denoising loop
        per sample, and returns the per-sample predictions plus their mean
        and std. Not used by the training workspace (which mirrors
        DiffusionUnetLowdimPolicy's fully deterministic evaluation) - kept
        as an available capability for anyone who wants to exploit the
        posterior IVON gives you.
        """
        assert self.ivon_optimizer is not None, (
            "predict_action_bma() needs an IVON optimizer - call "
            "set_ivon_optimizer() first."
        )
        all_actions = [
            self.predict_action(obs_dict, stochastic=True)['action_pred']
            for _ in range(num_samples)
        ]
        stacked = torch.stack(all_actions, dim=0)  # (S, B, T, Da)
        return {
            'action_pred_samples': stacked,
            'action_pred_mean': stacked.mean(dim=0),
            'action_pred_std': stacked.std(dim=0),
        }

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, train=True):
        """
        Plain diffusion noise-prediction MSE loss - identical to
        DiffusionUnetLowdimPolicy.compute_loss. This is the only thing that
        should be passed to `.backward()` under IVON, inside a
        `sampled_params(train=True)` block (see the training loop in
        train_ivon_diffusion_unet_lowdim_workspace.py) - whether that ends
        up evaluated at a stochastic posterior sample or at the posterior
        mean is decided entirely by whether the caller is currently inside
        that block when this is invoked, not by anything in this method.
        """
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        global_cond = obs[:, :self.n_obs_steps, :].reshape(obs.shape[0], -1)
        trajectory = action
        if self.pred_action_steps_only:
            To = self.n_obs_steps
            start = To - 1 if self.oa_step_convention else To
            end = start + self.n_action_steps
            trajectory = action[:, start:end]

        condition_mask = (
            torch.zeros_like(trajectory, dtype=torch.bool)
            if self.pred_action_steps_only
            else self.mask_generator(trajectory.shape)
        )

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        if train:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=trajectory.device).long()
        else:
            generator = torch.Generator(device=trajectory.device)
            generator.manual_seed(42)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=trajectory.device, generator=generator).long()

        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

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
        return loss.mean()

    # ========================== Compute upper bound on nll ==========================
    # Ported verbatim from DiffusionUnetLowdimPolicy - orthogonal to which
    # optimizer trains the network, so unchanged under IVON.
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

    def mse(self, batch, logsnr):
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
            global_cond=global_cond.flatten(0, 1) if global_cond is not None else None
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
    def nll_bound(self, dataloader, epoch, npoints=100):
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

                device = self.device
                dtype = self.dtype

                global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
                shape = (B, T, Da)
                if self.pred_action_steps_only:
                    shape = (B, self.n_action_steps, Da)
                    start = To - 1 if self.oa_step_convention else To
                    end = start + self.n_action_steps
                    trajectory = naction[:, start:end]
                else:
                    trajectory = naction
                cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
                cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

                # construct a batch of data in the form the NLL estimator expects
                batch = [trajectory, None, global_cond, cond_data, cond_mask]
                this_mse = self.mse(batch, logsnr)
                mses.append(this_mse)

        mses = torch.cat(mses, dim=0)       # Concatenate the batches together across axis 0
        results['mses-all'] = mses          # Store array of mses for each sample, logsnr
        mses = mses.mean(dim=0)             # Average across samples, giving MMSE(logsnr)

        results['mses'] = mses
        results['mmse_g'] = self.mmse_g(logsnr)

        # here the results of the integral is clamped to 0 if the MMSE is negative
        results['nll (nats)'] = self.h_g - torch.mean(0.5 * w * torch.clamp(results['mmse_g'] - mses, 0.))
        results['nll (bpd)'] = results['nll (nats)'] / (math.log(2) * self.d)

        return results['nll (bpd)']

    @property
    def loc_scale(self):
        """Return the parameters defining a normal distribution over logsnr, inferred from data statistics."""
        return self.loc_logsnr, self.scale_logsnr

    def dataset_info(self, dataloader, covariance_spectrum=None, diagonal=False):
        """covariance_spectrum can provide precomputed spectrum to speed up frequent experiments.
           diagonal: {False, True}  approximates covariance as diagonal, useful for very high-d data.
        """
        for batch in dataloader:
            break

        # normalize the data
        nbatch = self.normalizer.normalize(batch)
        data = nbatch["action"].to(self.device)

        if not self.obs_as_global_cond:
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
                self.U = None
            else:
                _, eigs, self.U = torch.linalg.svd(x, full_matrices=False)  # U.T diag(eigs^2/(n-1)) U = covariance
                self.log_eigs = 2 * torch.log(eigs) - math.log(len(x) - 1)  # Eigs of covariance are eigs**2/(n-1) of SVD

        self.log_eigs = self.log_eigs.to(self.device)
        self.mu = self.mu.to(self.device)
        if self.U is not None:
            self.U = self.U.to(self.device)

        # Used to estimate good range for integration
        self.loc_logsnr = -self.log_eigs.mean().item()
        if diagonal:
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

    # =============== Compute Reconstruction Loss of PAC-Bayes Bounds of Mbacke =======================
    # see https://arxiv.org/pdf/2312.05989 - identical to
    # DiffusionUnetLowdimPolicy.compute_action_reconst_loss (same
    # signature, same cfg.num_mc_samples loop, fully deterministic - not
    # something IVON changes).
    @torch.no_grad()
    def compute_action_reconst_loss(self, dataloader, cfg):
        total_loss_rec = 0
        for _ in range(cfg.num_mc_samples):
            loss_rec = 0
            with tqdm.tqdm(dataloader, desc=f"Reconstruction Loss",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch in tepoch:
                    batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

                    obs_dict = {'obs': batch['obs']}
                    ref_action = batch["action"]

                    result = self.predict_action(obs_dict)

                    if self.pred_action_steps_only:
                        pred_action = result['action']
                        To = self.n_obs_steps
                        start = To - 1 if self.oa_step_convention else To
                        end = start + self.n_action_steps
                        ref_action = ref_action[:, start:end]
                    else:
                        pred_action = result['action_pred']

                    batch_loss = torch.linalg.norm(
                                            pred_action - ref_action,
                                            ord=2,
                                            dim=(1, 2)
                                        )  # (B,)
                    loss_rec += batch_loss.sum()

            total_loss_rec += loss_rec

        return total_loss_rec / (cfg.num_mc_samples * len(dataloader.dataset))
