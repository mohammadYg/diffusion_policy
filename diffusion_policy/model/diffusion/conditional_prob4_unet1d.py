import torch
import torch.nn as nn
import numpy as np
import math

import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from typing import Union

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

import logging
logger = logging.getLogger(__name__)


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :N(mu, std^2)
    with values outside :[a, b] redrawn until they are within
    the bounds. The method used works best if 'mu' is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
   
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        # Here the CDF is computed based on the relationship between error function (math.erf) and normal CDF
        # This CDF is for standard normal distribution: N(0,1)
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Get upper and lower cdf values
        # Here, we first standardize the values a and b that are from 
        # a normal distribution with mean and std
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1. - eps), max=(1. - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor    

class Gaussian(nn.Module):
    """Implementation of a Gaussian random variable, using softplus for
    the standard deviation and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Gaussian.

    rho : Tensor of floats
        Scale parameter of the Gaussian (to be transformed to std
        via the softplus function)

    fixed : bool
        Boolean indicating whether the Gaussian is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))
        
    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size(), device=self.mu.device)
        #epsilon = torch.randn(self.sigma.size())
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(
            torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div


class Laplace(nn.Module):
    """Implementation of a Laplace random variable, using softplus for
    the scale parameter and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Laplace distr.

    rho : Tensor of floats
        Scale parameter for the distribution (to be transformed
        via the softplus function)

    fixed : bool
        Boolean indicating whether the distribution is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)

    @property
    def scale(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Laplace distribution
        # we do scaling due to numerical issues
        epsilon = (0.999*torch.rand(self.scale.size(), device=self.mu.device)-0.49999)
        result = self.mu - torch.mul(torch.mul(self.scale, torch.sign(epsilon)),
                                     torch.log(1-2*torch.abs(epsilon)))
        return result

    def compute_kl(self, other):
        # Compute KL divergence between two Laplaces distr. (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = self.scale
        b0 = other.scale
        term1 = torch.log(torch.div(b0, b1))
        aux = torch.abs(self.mu - other.mu)
        term2 = torch.div(aux, b0)
        term3 = torch.div(b1, b0) * torch.exp(torch.div(-aux, b1))

        kl_div = (term1 + term2 + term3 - 1).sum()
        return kl_div

class ProbLinear(nn.Module):
    """Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_init : float
        scale hyperparmeter (to initialise the scale of
        the posterior)

    rho_prior : float
        scale hyperparmeter (to set the scale of
        the prior)
    
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    """

    def __init__(self, in_features, out_features, rho_init = -3.0, rho_prior=-3.0, prior_dist='gaussian'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        # Initialise posterior distribution means using truncated normal
        weights_mu_init = trunc_normal_(torch.Tensor(
            out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
        bias_mu_init = torch.zeros(out_features)
        weights_rho_init = torch.ones(out_features, in_features) * rho_init
        bias_rho_init = torch.ones(out_features) * rho_init
        
        # Prior Initialization
        weights_mu_prior = torch.zeros(out_features, in_features)
        bias_mu_prior = torch.zeros(out_features) 
        weights_rho_prior = torch.ones(out_features, in_features) * rho_prior
        bias_rho_prior = torch.ones(out_features) * rho_prior

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), fixed=False)
        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_prior.clone(), fixed=True)
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_prior.clone(), fixed=True)

        self.kl_div = 0

    def forward(self, input, stochastic=False):
        #if self.training or stochastic:
        if stochastic:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                self.bias.compute_kl(self.bias_prior)

        return F.linear(input, weight, bias)

class ProbConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        rho_init=-3.0,
        rho_prior=-3.0,
        prior_dist='gaussian'
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        
        # Probabilistic conditioning pathway
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            ProbLinear(
                cond_dim, cond_channels, 
                rho_init=rho_init,
                rho_prior=rho_prior, 
                prior_dist=prior_dist
            ),
            Rearrange("batch t -> batch t 1"),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond, stochastic=False):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)

        # FiLM conditioning with probabilistic linear layer
        embed = self.cond_encoder[0](cond)  # Mish activation
        embed = self.cond_encoder[1](embed, stochastic=stochastic)  # ProbLinear
        embed = self.cond_encoder[2](embed)  # Rearrange

        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    
    def compute_kl(self):  # Renamed for consistency
        """Get the KL divergence for the conditioning linear layer""" 
        return self.cond_encoder[1].kl_div
    
class BayesianConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=None,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        use_dropout=False,
        # Bayesian parameters
        rho_init=-3.0,
        rho_prior=-3.0,
        prior_dist='gaussian'
    ):
        super().__init__()
        

        if output_dim is None:
            output_dim = input_dim
        #print("output_dim", output_dim)

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
 
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
  
        # Probabilistic local conditioning
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            
            local_cond_encoder = nn.ModuleList(
                [
                    # down encoder
                    ProbConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                        rho_init=rho_init,
                        rho_prior=rho_prior,
                        prior_dist=prior_dist,
                    ),
                    # up encoder
                    ProbConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                        rho_init=rho_init,
                        rho_prior=rho_prior,
                        prior_dist=prior_dist
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ProbConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    rho_init=rho_init,
                    rho_prior=rho_prior,
                    prior_dist=prior_dist
                ),
                ProbConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    rho_init=rho_init,
                    rho_prior=rho_prior,
                    prior_dist=prior_dist
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            
            down_modules.append(
                nn.ModuleList(
                    [
                        ProbConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            rho_init=rho_init,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist
                        ),
                        ProbConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            rho_init=rho_init,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity()
                    ]
                )
            )
        
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ProbConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            rho_init=rho_init,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist
                        ),
                        ProbConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            rho_init=rho_init,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity()
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        # Store Bayesian parameters for reference
        self.rho_prior = rho_prior
        self.rho_init = rho_init
        self.prior_dist = prior_dist

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int] = None,
        local_cond=None,
        global_cond=None,
        stochastic=False,  # Added stochastic parameter
        **kwargs,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,output_dim)
        """
        sample = einops.rearrange(sample, "b h t -> b t h")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        # use deterministic encoding for timesteps
        global_feature = self.diffusion_step_encoder(timesteps)
        
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature, stochastic=stochastic)
            h_local.append(x)
            x = resnet2(local_cond, global_feature, stochastic=stochastic)
            h_local.append(x)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature, stochastic=stochastic)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature, stochastic=stochastic)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature, stochastic=stochastic)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature, stochastic=stochastic)
            if idx == (len(self.up_modules) - 1) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature, stochastic=stochastic)
            x = upsample(x)

        # This works if self.dropout flag is True
        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x
    
    def compute_kl(self):
        """Compute total KL divergence from all probabilistic components"""
        kl_div = 0
           
        # KL from local condition encoder
        if self.local_cond_encoder is not None:
            for layer in self.local_cond_encoder:
                kl_div += layer.compute_kl()
        
        # KL from mid modules
        for layer in self.mid_modules:
            kl_div += layer.compute_kl()
        
        # KL from down modules
        for module_list in self.down_modules:
            for layer in module_list:
                if hasattr(layer, 'compute_kl'):
                    kl_div += layer.compute_kl()
                elif hasattr(layer, 'kl_div'):
                    kl_div += layer.kl_div
        
        # KL from up modules
        for module_list in self.up_modules:
            for layer in module_list:
                if hasattr(layer, 'compute_kl'):
                    kl_div += layer.compute_kl()
                elif hasattr(layer, 'kl_div'):
                    kl_div += layer.kl_div
        return kl_div