from sched import scheduler
from typing import Dict
import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torchdyn.core import NeuralODE

from diffusion_policy.flow_matcher.conditional_flow_matching import ConditionalFlowMatcher
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.flow_matcher.cnf_wrapper import CFMVectorField

import numpy as np
import math 
import tqdm

class FlowUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            FM: ConditionalFlowMatcher,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            integrate_steps = 100,
            # parameters passed to integrator
            **kwargs):
        
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
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
        self.integrate_steps = integrate_steps
        self.kwargs = kwargs

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            global_cond=None,
            generator=None,
            # keyword arguments to integrator
            **kwargs
            ):
        
        model = self.model

        X0 = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # define ODE function for flow matching
        odefunc = CFMVectorField(model, global_cond=global_cond)
        node = NeuralODE(
                odefunc,
                **kwargs
            )

        # integrate the ODE backwards in time
        t_span = torch.linspace(0, 1, self.integrate_steps, device=X0.device)
        traj = node(X0, t_span=t_span)
        X1 = traj[-1]
        return X1


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

    def compute_loss(self, batch, train=True):
        # normalize input
        assert 'valid_mask' not in batch
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
            x1 = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(x1, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(x1.shape)

        # Sample noise that we'll add to the images
        x0 = torch.randn(x1.shape, device=x1.device)
        t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, x1)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        xt[condition_mask] = x1[condition_mask]
        
        # Predict the noise residual
        vt_pred = self.model(xt, t, global_cond=global_cond)
       
        loss = F.mse_loss(vt_pred, ut, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    # @torch.no_grad()
    # def compute_action_reconst_loss(self, dataloader, cfg):
    #     total_loss_rec = 0
    #     for _ in range (cfg.num_mc_samples): 
    #         loss_rec=0
    #         with tqdm.tqdm(dataloader, desc=f"Reconstruction Loss", 
    #                     leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
    #             for batch in tepoch:
    #                 batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                    
    #                 # extract observation
    #                 obs_dict = {'obs': batch['obs']}
    #                 ref_action = batch["action"]
                    
    #                 # reconstruct action 
    #                 result = self.predict_action(obs_dict)

    #                 if self.pred_action_steps_only:
    #                     pred_action = result['action']
    #                     start = To
    #                     if self.oa_step_convention:
    #                         start = To - 1
    #                     end = start + self.n_action_steps
    #                     ref_action = ref_action[:, start:end]
    #                 else:
    #                     pred_action = result['action_pred']

    #                 batch_loss = torch.linalg.norm(
    #                                         pred_action - ref_action,
    #                                         ord=2,
    #                                         dim=(1, 2)
    #                                     )  # (B,)
    #                 # compute reconstruction loss
    #                 loss_rec += batch_loss.sum()
                
    #         total_loss_rec += loss_rec

    #     return total_loss_rec/(cfg.num_mc_samples*len(dataloader.dataset))