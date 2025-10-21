import torch
import torch.nn as nn
import numpy as np
import math

import torch.nn.functional as F
from einops.layers.torch import Rearrange
from typing import Union
import os

import einops
import dill
import copy
import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
def regression_output_transform(x, clamping=True, min_val=-1.0, max_val=1.0):
    """Transform output for regression with clamping between min_val and max_val"""
    if clamping:
        x = torch.clamp(x, min_val, max_val)
    return x

# Or use a smooth clamping function like tanh
def tanh_output_transform(x, clamping=True):
    """Use tanh to smoothly constrain outputs between -1 and 1"""
    if clamping:
        x = torch.tanh(x)
    return x

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

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the Gaussian is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))
        

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(self.device)
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

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the distribution is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def scale(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        m = nn.Softplus()
        return m(self.rho)

    def sample(self):
        # Return a sample from the Laplace distribution
        # we do scaling due to numerical issues
        epsilon = (0.999*torch.rand(self.scale.size())-0.49999).to(self.device)
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


class Lambda_var(nn.Module):
    """Class for the lambda variable included in the objective
    flambda

    Parameters
    ----------
    lamb : float
        initial value

    n : int
        Scaling parameter (lamb_scaled is between 1/sqrt(n) and 1)

    """

    def __init__(self, lamb, n):
        super().__init__()
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=True)
        self.min = 1/np.sqrt(n)

    @property
    def lamb_scaled(self):
        # We restrict lamb_scaled to be between 1/sqrt(n) and 1.
        m = nn.Sigmoid()
        return (m(self.lamb) * (1-self.min) + self.min)
    
class ProbLinear(nn.Module):
    """Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_layer : Linear object
        Linear layer object used to initialise the prior

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(self, in_features, out_features, rho_prior=-3.0, prior_dist='gaussian', 
                 device='cuda', init_prior='weights', init_layer=None, 
                 init_layer_prior=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            # Initialise distribution means using truncated normal
            weights_mu_init = trunc_normal_(torch.Tensor(
                out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_features)

        weights_rho_init = torch.ones(out_features, in_features) * rho_prior
        bias_rho_init = torch.ones(out_features) * rho_prior

        if init_prior == 'zeros':
            bias_mu_prior = torch.zeros(out_features) 
            weights_mu_prior = torch.zeros(out_features, in_features)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(
                out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_prior = torch.zeros(out_features) 
        elif init_prior == 'weights': 
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                # otherwise initialise to posterior weights
                weights_mu_prior = weights_mu_init.clone()
                bias_mu_prior = bias_mu_init.clone()
        else: 
            raise RuntimeError(f'Wrong type of prior initialisation!')

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), device=device, fixed=False)
        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0

    def forward(self, input, stochastic=False):
        if self.training or stochastic:
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

class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)
    
class ProbConv1d(nn.Module):
    """Probabilistic 1D Convolutional Layer.
    Each weight and bias is modeled as a Gaussian random variable.
    during the training the kl divergence is updated every time the layer is called

    Parameters
    ----------
    in_channels : int
        Number of input channels for the layer

    out_channels : int
        Number of output channels for the layer

    kernel_size : int
        size of the convolutional kernel

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    stride : int
        Stride of the convolution

    padding: int
        Zero-padding added to both sides of the input

    dilation: int
        Spacing between kernel elements

    init_layer : Linear object
        Linear layer object used to initialise the prior

    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_prior=-3.0,
                 prior_dist='gaussian', device='cuda', stride=1, padding=0, dilation=1,
                groups = 1, init_prior='weights', init_layer=None, init_layer_prior=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # He-style sigma for initialization
        in_features = self.in_channels
        out_features = self.out_channels
        sigma_weights = 1. / np.sqrt(in_channels * kernel_size)

        # posterior init
        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            weights_mu_init = trunc_normal_(torch.Tensor(out_channels, in_channels, kernel_size),
                                        0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_channels)

        weights_rho_init = torch.ones(out_channels, in_channels, kernel_size) * rho_prior
        bias_rho_init = torch.ones(out_channels) * rho_prior

        if init_prior == 'zeros':
            bias_mu_prior = torch.zeros(out_features) 
            weights_mu_prior = torch.zeros(out_features, in_features, kernel_size)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(
                out_channels, in_channels, kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_prior = torch.zeros(out_features) 
        elif init_prior == 'weights': 
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                # otherwise initialise to posterior weights
                weights_mu_prior = weights_mu_init.clone()
                bias_mu_prior = bias_mu_init.clone()
        else: 
            raise RuntimeError(f'Wrong type of prior initialisation!')
        
        # priors = fixed, posteriors = learnable
        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Unknown prior_dist {prior_dist}')

        self.weight = dist(weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=False)
        self.bias = dist(bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0

    def forward(self, x, stochastic=False):
        if self.training or stochastic:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training:
            self.kl_div = self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(self.bias_prior)

        return F.conv1d(x, weight, bias, stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups)
    
class ProbConvTranspose1d(nn.Module):
    """Probabilistic 1D Transposed Convolutional Layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the layer

    out_channels : int
        Number of output channels for the layer

    kernel_size : int
        size of the convolutional kernel

    rho_prior : float
        prior scale hyperparameter (to initialise the scale of the posterior)

    prior_dist : string
        string that indicates the type of distribution for the prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    stride : int
        Stride of the convolution

    padding: int
        Zero-padding added to both sides of the input

    output_padding: int
        Additional padding added to the output

    dilation: int
        Spacing between kernel elements

    init_prior : string
        How to initialize the prior ('zeros', 'random', 'weights')

    init_layer : Linear object
        Linear layer object used to initialise the posterior

    init_layer_prior : Linear object  
        Linear layer object used to initialise the prior
    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_prior=-3.0,
                 prior_dist='gaussian', device='cuda', stride=1, padding=0, 
                 output_padding=0, dilation=1, groups=1,
                 init_prior='weights', init_layer=None, init_layer_prior=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        # He-style sigma for initialization
        sigma_weights = 1. / np.sqrt(in_channels * kernel_size)

        # posterior init
        if init_layer:
            weights_mu_init = init_layer.weight
            bias_mu_init = init_layer.bias
        else:
            weights_mu_init = trunc_normal_(torch.Tensor(
                in_channels, out_channels, kernel_size),  # Note: transposed conv has different weight arrangement
                0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_channels)

        weights_rho_init = torch.ones(in_channels, out_channels, kernel_size) * rho_prior
        bias_rho_init = torch.ones(out_channels) * rho_prior

        # prior init
        if init_prior == 'zeros':
            bias_mu_prior = torch.zeros(out_channels)  # Fixed: out_channels
            weights_mu_prior = torch.zeros(in_channels, out_channels, kernel_size)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(
                in_channels, out_channels, kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_prior = torch.zeros(out_channels)
        elif init_prior == 'weights': 
            if init_layer_prior:
                weights_mu_prior = init_layer_prior.weight
                bias_mu_prior = init_layer_prior.bias
            else:
                # otherwise initialise to posterior weights
                weights_mu_prior = weights_mu_init
                bias_mu_prior = bias_mu_init
        else: 
            raise RuntimeError(f'Wrong type of prior initialisation!')
        
        # priors = fixed, posteriors = learnable
        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Unknown prior_dist {prior_dist}')

        self.weight = dist(weights_mu_init.clone(), weights_rho_init.clone(), device=device, fixed=False)
        self.bias = dist(bias_mu_init.clone(), bias_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(weights_mu_prior.clone(), weights_rho_init.clone(), device=device, fixed=True)
        self.bias_prior = dist(bias_mu_prior.clone(), bias_rho_init.clone(), device=device, fixed=True)

        self.kl_div = 0

    def forward(self, x, stochastic=False):
        if self.training or stochastic:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training:
            self.kl_div = self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(self.bias_prior)

        return F.conv_transpose1d(x, weight, bias, stride=self.stride, padding=self.padding,
                                 output_padding=self.output_padding, dilation=self.dilation, 
                                 groups=self.groups)

class ProbDownsample1d(nn.Module):
    ''' This class is initialized with nn.conv1D layer from a deterministic network
    the init_layer and init_layer_prior must be 'Downsample1d' layer from deterministic network
    '''
    def __init__(self, dim, rho_prior=-3.0, prior_dist='gaussian', device='cuda',
                 init_prior='weights', init_downsample1d=None, init_downsample1d_prior=None):
        super().__init__()
        
        # Extract the conv layer from init_upsample1d if provided
        init_conv = init_downsample1d.conv if init_downsample1d else None
        init_conv_prior = init_downsample1d_prior.conv if init_downsample1d_prior else None
        
        self.conv = ProbConv1d(
            dim, dim, kernel_size=3, stride=2, padding=1,
            rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            init_prior=init_prior, init_layer=init_conv, init_layer_prior=init_conv_prior
        )

    def forward(self, x, stochastic=False):
        return self.conv(x, stochastic=stochastic)

    def compute_kl(self):
        #! make sure this kl divergence is used only during training 
        return self.conv.kl_div

    
class ProbUpsample1d(nn.Module):
    def __init__(self, dim, rho_prior=-3.0, prior_dist='gaussian', device='cuda',
                 init_prior='weights', init_upsample1d=None, init_upsample1d_prior=None):
        super().__init__()
        
        # Extract the conv layer from init_upsample1d if provided
        init_conv = init_upsample1d.conv if init_upsample1d else None
        init_conv_prior = init_upsample1d_prior.conv if init_upsample1d_prior else None
        
        self.conv = ProbConvTranspose1d(
            dim, dim, kernel_size=4, stride=2, padding=1,
            rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            init_prior=init_prior, init_layer=init_conv, init_layer_prior=init_conv_prior
        )

    def forward(self, x, stochastic=False):
        return self.conv(x, stochastic=stochastic)
    
    def compute_kl(self):
        return self.conv.kl_div

class ProbConv1dBlock(nn.Module):
    """
    Probabilistic Conv1d --> GroupNorm --> Mish
    The calss takes an initial deterministic conv1dblock.
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8, 
                 rho_prior=-3.0, prior_dist='gaussian', device='cuda',
                 init_prior='weights', init_conv1dblock=None, init_conv1dblock_prior=None):
        super().__init__()

        # Extract the conv layers from init_conv1dblovk if provided
        init_conv = init_conv1dblock.block[0] if init_conv1dblock else None
        init_conv_prior = init_conv1dblock_prior.block[0] if init_conv1dblock_prior else None

        self.conv = ProbConv1d(
            inp_channels, out_channels, kernel_size, 
            rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            padding=kernel_size // 2,  # Maintain same padding
            init_prior=init_prior, init_layer=init_conv, init_layer_prior=init_conv_prior
        )
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.activation = nn.Mish()

    def forward(self, x, stochastic=False):
        # Apply probabilistic convolution
        x = self.conv(x, stochastic=stochastic)
        
        # Apply group norm and activation
        x = self.norm(x)
        x = self.activation(x)     
        return x
    
    def compute_kl(self):
        return self.conv.kl_div
    

class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
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
    
class ProbConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        rho_prior=-3.0,
        prior_dist='gaussian',
        device='cuda',
        init_prior='weights', 
        init_condresblock=None,
        init_condresblock_prior=None
    ):
        super().__init__()

        # Handle init_layer for the convolutional blocks
        init_conv1dblock1 = init_condresblock.blocks[0] if init_condresblock else None
        init_conv1dblock_prior1 = init_condresblock_prior.blocks[0] if init_condresblock_prior else None
        init_conv1dblock2 = init_condresblock.blocks[1] if init_condresblock else None
        init_conv1dblock_prior2 = init_condresblock_prior.blocks[1] if init_condresblock_prior else None
        
        self.blocks = nn.ModuleList(
            [
                ProbConv1dBlock(
                    in_channels, out_channels, kernel_size, 
                    n_groups=n_groups, rho_prior=rho_prior,
                    prior_dist=prior_dist, device=device,
                    init_prior=init_prior, 
                    init_conv1dblock=init_conv1dblock1,
                    init_conv1dblock_prior=init_conv1dblock_prior1
                ),
                ProbConv1dBlock(
                    out_channels, out_channels, kernel_size,
                    n_groups=n_groups, rho_prior=rho_prior,
                    prior_dist=prior_dist, device=device,
                    init_prior=init_prior,
                    init_conv1dblock=init_conv1dblock2,
                    init_conv1dblock_prior=init_conv1dblock_prior2
                ),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        
        # Handle init_layer for conditioning linear
        init_cond_encoder_linear_layer = init_condresblock.cond_encoder[1] if init_condresblock.cond_encoder else None
        init_cond_encoder_prior_linear_layer = init_condresblock_prior.cond_encoder[1] if init_condresblock_prior.cond_encoder else None

        # Probabilistic conditioning pathway
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            ProbLinear(
                cond_dim, cond_channels, 
                rho_prior=rho_prior, 
                prior_dist=prior_dist, 
                device=device,
                init_prior=init_prior,
                init_layer=init_cond_encoder_linear_layer,
                init_layer_prior=init_cond_encoder_prior_linear_layer
            ),
            Rearrange("batch t -> batch t 1"),
        )

        # Residual connection with initialization options
        # Handle init_layer for residual convolution
        init_residual = init_condresblock.residual_conv if (init_condresblock.residual_conv and not isinstance(init_condresblock.residual_conv, nn.Identity)) else None
        init_residual_prior = init_cond_encoder_prior.residual_conv if (init_cond_encoder_prior.residual_conv and not isinstance(init_cond_encoder_prior.residual_conv, nn.Identity)) else None
        
        if in_channels != out_channels:
            self.residual_conv = ProbConv1d(
                in_channels, out_channels, kernel_size=1,
                rho_prior=rho_prior, prior_dist=prior_dist,
                device=device, padding=0,
                init_prior=init_prior,
                init_layer=init_residual,
                init_layer_prior=init_residual_prior
            )
        else:
            self.residual_conv = nn.Identity()


    def forward(self, x, cond, stochastic=False):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        # First convolutional block
        out = self.blocks[0](x, stochastic=stochastic)
        
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
        
        # Second convolutional block
        out = self.blocks[1](out, stochastic=stochastic)
        
        # Residual connection
        if isinstance(self.residual_conv, nn.Identity):
            residual = self.residual_conv(x)
        else:
            residual = self.residual_conv(x, stochastic=stochastic)
        
        out = out + residual
        return out
    
    def compute_kl(self):  # Renamed for consistency
        """Get total KL divergence from all probabilistic components"""
        kl_div = 0
        
        # KL from convolutional blocks
        kl_div += self.blocks[0].conv.kl_div
        kl_div += self.blocks[1].conv.kl_div
        
        # KL from conditioning linear layer
        kl_div += self.cond_encoder[1].kl_div
        
        # KL from residual convolution (if present)
        if not isinstance(self.residual_conv, nn.Identity):
            kl_div += self.residual_conv.kl_div
            
        return kl_div
    

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
        rho_prior=-3.0,
        prior_dist='gaussian',
        device='cuda',
        init_prior='weights',
        init_net=None,
        init_net_prior=None,  # Separate prior initialization
        # Output transformation parameters
        output_transform_type='tanh',  # 'clamp', 'tanh', or 'none'
        output_clamp_min=-1.0,
        output_clamp_max=1.0,
    ):
        super().__init__()
        
        # configure initial model
        if init_net is not None:
            if not os.path.isfile(init_net):
                raise ValueError(f"No such file: {init_net}")
            print("Initial model from checkpoint:", init_net)
            
            payload = torch.load(open(init_net, 'rb'), pickle_module=dill)
            cfg_init_net = payload['cfg']
            cls = hydra.utils.get_class(cfg_init_net._target_)
            workspace = cls(cfg_init_net)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            
            # get policy from workspace
            init_net = copy.deepcopy(workspace.model.model)
            if cfg_init_net.training.use_ema:
                init_net = copy.deepcopy(workspace.ema_model.model)
            
            init_net.to(device)
            init_net.eval()

            # configure the init_net in the probabilstic model
            del workspace
        
        if init_net_prior:
            if not os.path.isfile(init_net_prior):
                raise ValueError(f"No such file: {init_net_prior}")
            print("Initial model prior from checkpoint:", init_net_prior)

            payload = torch.load(open(init_net_prior, 'rb'), pickle_module=dill)
            cfg_init_net_prior = payload['cfg']
            cls = hydra.utils.get_class(cfg_init_net_prior._target_)
            workspace = cls(cfg_init_net_prior)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            
            # get policy from workspace
            init_net_prior = copy.deepcopy(workspace.model.model)
            if cfg_init_net_prior.training.use_ema:
                init_net_prior = copy.deepcopy(workspace.ema_model.model)
            
            init_net_prior.to(device)
            init_net_prior.eval()
            del workspace

        # Store output transformation parameters
        self.output_transform_type = output_transform_type
        self.output_clamp_min = output_clamp_min
        self.output_clamp_max = output_clamp_max

        if output_dim is None:
            output_dim = input_dim
        print("output_dim", output_dim)

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        
        # Deterministic encoding of the timestep
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # # Probabilistic timestep encoding
        # init_step_encoder = init_net.diffusion_step_encoder if init_net.diffusion_step_encoder else None
        # init_step_encoder_prior = init_net_prior.diffusion_step_encoder if init_net_prior.diffusion_step_encoder else None
        
        # init_step_encoder_linearlayer1 = init_step_encoder[1] if init_step_encoder else None
        # init_step_encoder_linearlayer2 = init_step_encoder[3] if init_step_encoder else None
        # init_step_encoder_linearlayer1_prior = init_step_encoder_prior[1] if init_step_encoder_prior else None
        # init_step_encoder_linearlayer2_prior = init_step_encoder_prior[3] if init_step_encoder_prior else None

        # diffusion_step_encoder = nn.Sequential(
        #     SinusoidalPosEmb(dsed),
        #     ProbLinear(dsed, dsed * 4, rho_prior=rho_prior, prior_dist=prior_dist, 
        #               device=device, init_prior=init_prior, 
        #               init_layer=init_step_encoder_linearlayer1, 
        #               init_layer_prior=init_step_encoder_linearlayer1_prior),
        #     nn.Mish(),
        #     ProbLinear(dsed * 4, dsed, rho_prior=rho_prior, prior_dist=prior_dist,
        #               device=device, init_prior=init_prior,
        #               init_layer=init_step_encoder_linearlayer2, 
        #               init_layer_prior=init_step_encoder_linearlayer2_prior),
        # )
        
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # Deterministic local conditioning
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList(
                [
                    # down encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                    # up encoder
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        cond_predict_scale=cond_predict_scale,
                    ),
                ]
            )
        
        # # Probabilistic local conditioning
        # local_cond_encoder = None
        # if local_cond_dim is not None:
        #     _, dim_out = in_out[0]
        #     dim_in = local_cond_dim
            
        #     # Handle initialization from pretrained nets
        #     init_local_cond = init_net.local_cond_encoder if init_net.local_cond_encoder else None
        #     init_local_cond_prior = init_net_prior.local_cond_encoder if init_net_prior.local_cond_encoder else None
            
        #     init_local_condresblock_down = init_local_cond[0] if init_local_cond else None
        #     init_local_condresblock_down_prior = init_local_cond_prior[0] if init_local_cond_prior else None
        #     init_local_condresblock_up = init_local_cond[1] if init_local_cond else None
        #     init_local_condresblock_up_prior = init_local_cond_prior[1] if init_local_cond_prior else None
            
        #     local_cond_encoder = nn.ModuleList(
        #         [
        #             # down encoder
        #             ProbConditionalResidualBlock1D(
        #                 dim_in,
        #                 dim_out,
        #                 cond_dim=cond_dim,
        #                 kernel_size=kernel_size,
        #                 n_groups=n_groups,
        #                 cond_predict_scale=cond_predict_scale,
        #                 rho_prior=rho_prior,
        #                 prior_dist=prior_dist,
        #                 device=device,
        #                 init_prior=init_prior,
        #                 init_condresblock=init_local_condresblock_down,
        #                 init_condresblock_prior=init_local_condresblock_down_prior,
        #             ),
        #             # up encoder
        #             ProbConditionalResidualBlock1D(
        #                 dim_in,
        #                 dim_out,
        #                 cond_dim=cond_dim,
        #                 kernel_size=kernel_size,
        #                 n_groups=n_groups,
        #                 cond_predict_scale=cond_predict_scale,
        #                 rho_prior=rho_prior,
        #                 prior_dist=prior_dist,
        #                 device=device,
        #                 init_prior=init_prior,
        #                 init_condresblock=init_local_condresblock_up,
        #                 init_condresblock_prior=init_local_condresblock_up_prior,
        #             ),
        #         ]
        #     )

        mid_dim = all_dims[-1]
        # Handle initialization for mid modules
        init_mid_modules = init_net.mid_modules if init_net else None
        init_mid_modules_prior = init_net_prior.mid_modules if init_net_prior else None
        
        init_mid1 = init_mid_modules[0] if init_mid_modules else None
        init_mid1_prior = init_mid_modules_prior[0] if init_mid_modules_prior else None
        init_mid2 = init_mid_modules[1] if init_mid_modules else None
        init_mid2_prior = init_mid_modules_prior[1] if init_mid_modules_prior else None
        print ("ali")
        self.mid_modules = nn.ModuleList(
            [
                ProbConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    rho_prior=rho_prior,
                    prior_dist=prior_dist,
                    device=device,
                    init_prior=init_prior,
                    init_condresblock=init_mid1,
                    init_condresblock_prior=init_mid1_prior,
                ),
                ProbConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                    rho_prior=rho_prior,
                    prior_dist=prior_dist,
                    device=device,
                    init_prior=init_prior,
                    init_condresblock=init_mid2,
                    init_condresblock_prior=init_mid2_prior,
                ),
            ]
        )
        print ("gholi")
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            
            # Handle initialization for each down module
            init_down = init_net.down_modules[ind] if init_net else None
            init_down_prior = init_net_prior.down_modules[ind] if init_net_prior else None
            
            init_condresblock_1 = init_down[0] if init_down else None
            init_condresblock_1_prior = init_down_prior[0] if init_down_prior else None
            init_condresblock_2 = init_down[1] if init_down else None
            init_condresblock_2_prior = init_down_prior[1] if init_down_prior else None
            init_downsample = init_down[2] if init_down else None
            init_downsample_prior = init_down_prior[2] if init_down_prior else None
            
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
                            rho_prior=rho_prior,
                            prior_dist=prior_dist,
                            device=device,
                            init_prior=init_prior,
                            init_condresblock=init_condresblock_1,
                            init_condresblock_prior=init_condresblock_1_prior,
                        ),
                        ProbConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist,
                            device=device,
                            init_prior=init_prior,
                            init_condresblock=init_condresblock_2,
                            init_condresblock_prior=init_condresblock_2_prior,
                        ),
                        ProbDownsample1d(
                            dim_out, 
                            rho_prior=rho_prior,
                            prior_dist=prior_dist,
                            device=device,
                            init_prior=init_prior,
                            init_downsample1d=init_downsample,
                            init_downsample1d_prior=init_downsample_prior
                        ) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            
            # Handle initialization for each up module
            init_up = init_net.up_modules[ind] if init_net else None
            init_up_prior = init_net_prior.up_modules[ind] if init_net_prior else None
            
            init_condresblock_1 = init_up[0] if init_up else None
            init_condresblock_1_prior = init_up_prior[0] if init_up_prior else None
            init_condresblock_2 = init_up[1] if init_up else None
            init_condresblock_2_prior = init_up_prior[1] if init_up_prior else None
            init_upsample = init_up[2] if init_up else None
            init_upsample_prior = init_up_prior[2] if init_up_prior else None
            
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
                            rho_prior=rho_prior,
                            prior_dist=prior_dist,
                            device=device,
                            init_prior=init_prior,
                            init_condresblock=init_condresblock_1,
                            init_condresblock_prior=init_condresblock_1_prior,
                        ),
                        ProbConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist,
                            device=device,
                            init_prior=init_prior,
                            init_condresblock=init_condresblock_2,
                            init_condresblock_prior=init_condresblock_2_prior,
                        ),
                        ProbUpsample1d(
                            dim_in,
                            rho_prior=rho_prior,
                            prior_dist=prior_dist,
                            device=device,
                            init_prior=init_prior,
                            init_upsample1d=init_upsample,
                            init_upsample1d_prior=init_upsample_prior
                        ) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.dropout = nn.Dropout(0.25) if use_dropout else nn.Identity()

        # Replace final conv with probabilistic version
        init_final_conv = init_net.final_conv if init_net else None
        init_final_conv_prior = init_net_prior.final_conv if init_net_prior else None
        
        init_conv_block = init_final_conv[0].block[0] if init_final_conv else None
        init_conv_block_prior = init_final_conv_prior[0].block[0] if init_final_conv_prior else None
        init_final_layer = init_final_conv[1] if init_final_conv else None
        init_final_layer_prior = init_final_conv_prior[1] if init_final_conv_prior else None
        
        final_conv = nn.Sequential(
            ProbConv1dBlock(
                start_dim, start_dim, kernel_size=kernel_size,
                n_groups=n_groups, rho_prior=rho_prior,
                prior_dist=prior_dist, device=device,
                init_prior=init_prior, 
                init_conv1dblock=init_conv_block,
                init_conv1dblock_prior=init_conv_block_prior
            ),
            ProbConv1d(
                start_dim, output_dim, kernel_size=1,
                rho_prior=rho_prior, prior_dist=prior_dist,
                device=device, init_prior=init_prior,
                init_layer=init_final_layer,
                init_layer_prior=init_final_layer_prior
            ),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        # Store Bayesian parameters for reference
        self.rho_prior = rho_prior
        self.prior_dist = prior_dist
        self.device = device

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int] = None,
        local_cond=None,
        global_cond=None,
        stochastic=False,  # Added stochastic parameter
        clamping = True,
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
        if timestep is not None:
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
            
            # # Use stochastic sampling for diffusion step encoder
            # global_feature = self.diffusion_step_encoder[0](timesteps)  # SinusoidalPosEmb
            # global_feature = self.diffusion_step_encoder[1](global_feature, stochastic=stochastic)
            # global_feature = self.diffusion_step_encoder[2](global_feature)  # Mish
            # global_feature = self.diffusion_step_encoder[3](global_feature, stochastic=stochastic)

            if global_cond is not None:
                global_feature = torch.cat([global_feature, global_cond], axis=-1)
        elif global_cond is not None:
            global_feature = global_cond
        else:
            raise ValueError("Either timestep or global_cond must be provided")

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
            if not isinstance(downsample, nn.Identity):
                x = downsample(x, stochastic=stochastic)
            else:
                x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature, stochastic=stochastic)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature, stochastic=stochastic)
            if idx == (len(self.up_modules) - 1) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature, stochastic=stochastic)
            if not isinstance(upsample, nn.Identity):
                x = upsample(x, stochastic=stochastic)
            else:
                x = upsample(x)

        x = self.dropout(x)
        
        # Apply final convolution with stochastic sampling
        x = self.final_conv[0](x, stochastic=stochastic)
        x = self.final_conv[1](x, stochastic=stochastic)

        # Apply output transformation
        if clamping:
            x = self._apply_output_transform(x)

        x = einops.rearrange(x, "b t h -> b h t")
        return x
    
    def _apply_output_transform(self, x):
        """Apply the selected output transformation"""
        if self.output_transform_type == 'clamp':
            return regression_output_transform(
                x, 
                clamping=True, 
                min_val=self.output_clamp_min, 
                max_val=self.output_clamp_max
            )
        elif self.output_transform_type == 'tanh':
            return tanh_output_transform(x, clamping=True)
        elif self.output_transform_type == 'none':
            return x
        else:
            raise ValueError(f"Unknown output_transform_type: {self.output_transform_type}")

    def compute_kl(self):
        """Compute total KL divergence from all probabilistic components"""
        kl_div = 0
        
        # KL from diffusion step encoder
        for layer in self.diffusion_step_encoder:
            if hasattr(layer, 'kl_div'):
                kl_div += layer.kl_div
        
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
        
        # KL from final convolution
        kl_div += self.final_conv[0].compute_kl()
        kl_div += self.final_conv[1].kl_div
        
        return kl_div

    def set_stochastic(self, stochastic=True):
        """Convenience method to set stochastic sampling mode for all layers"""
        # This method can be used to easily toggle between sampling modes
        self.stochastic_mode = stochastic