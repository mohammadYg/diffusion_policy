from typing import Dict
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from functools import partial


from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

from diffusion_policy.CFM.ode_solver import ODESolver
from diffusion_policy.CFM.path.affine import normal_log_prob
from diffusion_policy.CFM.path.affine import CondOTProbPath
from diffusion_policy.CFM.cfm_model import CFMVectorField
from diffusion_policy.CFM.utils import skewed_timestep_sample


class FlowUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            FM: CondOTProbPath,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            prior_std = 1.0,
            **kwargs
            ):
        
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        
        self.model = model
        
        # wrapper around the model
        self.vector_field = CFMVectorField(self.model)
        
        # solver is constructed ONCE
        self.solver = ODESolver(self.vector_field)

        self.FM = FM
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
        self.prior_std = prior_std
        self.kwargs = kwargs

    def sample_x1_vf_batch(self, dataset, batch_size: int, device)-> torch.Tensor:
        total_samples = len(dataset)
        if total_samples == 0:
            return torch.empty(0, device=device)


        random_indices = torch.randint(0, total_samples, (batch_size,))
        batch_data = torch.stack(
            [dataset[idx.item()]['action'] for idx in random_indices]
        )
        return batch_data.to(device)
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            global_cond=None,
            generator=None,
            # keyword arguments to integrator
            **kwargs
            ):
        
        x_0 = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)* self.prior_std

        time_grid=torch.tensor(
                [0.0, 1.0],
                device=x_0.device,
            )
        # integrate the ODE backwards in time
        x_1 = self.solver.sample(
            x_init=x_0,
            time_grid=time_grid,
            **kwargs,
            global_cond=global_cond
        )
        return x_1


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
        global_cond = None
    
        if self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
        else:
            raise ValueError("Impainting is not supported for lowdim policy yet. Please use obs_as_global_cond=True.")
            # # condition through impainting
            # shape = (B, T, Da+Do)
            # cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            # cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            # cond_data[:,:To,Da:] = nobs[:,:To]
            # cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
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

    def compute_loss(self, batch, x1_vf_batch=None, skewed_timesteps=False, 
                     debug=False):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        if x1_vf_batch is not None:
            # normalize x1_vf_batch
            x1_vf_batch = self.normalizer['action'].normalize(x1_vf_batch)

        # handle different ways of passing observation
        global_cond = None
        x_1 = action
        if self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                x_1 = action[:,start:end]
        else:
            x_1 = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(x_1, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(x_1.shape)

        # Sample noise that we'll add to the images
        x_0 = torch.randn(x_1.shape, device=x_1.device)*self.prior_std
        if skewed_timesteps:
            t = skewed_timestep_sample(x_1.shape[0], device=x_1.device)
        else:
            t = torch.rand(x_1.shape[0], device=x_1.device)

        if debug:
            out, first_element_prob, norm_score = self.FM.sample(x_0, x_1, t, x1_vf_batch, prior_std = self.prior_std, debug=debug) 
        else:
            out = self.FM.sample(x_0, x_1, t, x1_vf_batch, prior_std = self.prior_std, debug=debug) 
        x_t = out.x_t
        x_1 = out.x_1
        u_t = out.dx_t

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        x_t[condition_mask] = x_1[condition_mask]
        
        # Predict the noise residual
        vt_pred = self.model(x_t, t, global_cond=global_cond)
       
        loss = F.mse_loss(vt_pred, u_t, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def compute_nll(self, batch, exact_divergence = True):
        # normalize input
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        global_cond = None
        x1 = action
        if self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                x1 = action[:,start:end]
        else:
            #! inpainting is not supported for lowdim policy yet. Please use obs_as_global_cond=True.
            x1 = torch.cat([action, obs], dim=-1)

        _, logp = self.solver.compute_likelihood(
            x_1=x1,
            log_p0=partial(normal_log_prob, std=self.prior_std),
            global_cond=global_cond,
            exact_divergence = exact_divergence,
            **self.kwargs,
        )

        # action_normalizer = self.normalizer['action']
        # scale = action_normalizer.params_dict['scale'].to(
        #     device=x1.device,
        #     dtype=x1.dtype,
        # )
        # # The action normalizer applies an affine transform z = s * x + b.
        # # For a change of variables, the density in the original space gets an
        # # extra log-absolute-Jacobian term: sum(log |s|) per action coordinate.
        # # Since the transform is applied independently at each action step, the
        # # total Jacobian for a trajectory with T action steps is T * sum(log |s|).
        # log_abs_det = x1.shape[1] * torch.log(scale.abs()).sum()

        # logp = logp + log_abs_det

        # Normalize by the number of action dimensions so the reported NLL is
        # comparable across different action shapes and horizons.
        num_dims = x1[0].numel()
        nll_ = -logp.mean() / num_dims
        return nll_

