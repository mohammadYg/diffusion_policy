import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def sigma_to_rho(sigma):
    """Inverse of softplus: returns rho such that softplus(rho) == sigma.

    Used to initialize a layer's posterior/prior rho from a target std
    (sigma) instead of an absolute rho value - see `post_sigma_scale` /
    `prior_sigma_scale` on the Prob* layers below, which set the initial
    sampling-noise std to a fraction of the layer's own (fan-in-scaled)
    deterministic weight-init std, rather than one constant shared by every
    layer regardless of width.
    """
    sigma_t = torch.as_tensor(sigma, dtype=torch.float32)
    return torch.log(torch.expm1(sigma_t))

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
        return F.softplus(self.rho)

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size(), device=self.mu.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(torch.pow(self.mu - other.mu, 2), b0)
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

    rho_post : float
        scale hyperparmeter (to initialise the scale of
        the posterior)

    rho_prior : float
        scale hyperparmeter (to set the scale of
        the prior)
    
    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    
    init_post : string
        string that indicates the way to initialise the posterior:
    
    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    post_sigma_scale : float, optional
        When set, overrides `rho_post` for the (always learnable) posterior
        rho: its initial sampling-noise std is set
        to `post_sigma_scale * sigma_weights` (fan-in-scaled, like the mu
        init) instead of the fixed absolute `softplus(rho_post)` shared by
        every layer regardless of width. Fixes wide/deep layers ending up
        proportionally far noisier than narrow ones under a single global
        rho_post. None (default) preserves the original absolute-rho_post
        behaviour.

    prior_sigma_scale : float, optional
        Same idea as `post_sigma_scale`, but for the (fixed) prior's rho_prior.

    local_reparam : bool, optional
        Default used by `forward()` when its own `local_reparam` argument is
        not explicitly overridden per-call - see `forward`'s docstring.
        Config-driven (threaded down from the U-Net's constructor), so a
        whole model can be switched between local and global (Blundell-
        style) reparameterization via a single yaml/CLI flag. Default True.

    """

    def __init__(self, in_features, out_features, rho_post = -3.0, rho_prior=-3.0,
                 prior_dist='gaussian', init_post = 'random', init_prior = 'zeros',
                 post_sigma_scale=None, prior_sigma_scale=None, local_reparam=True):

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sampled_weight = None
        self.sampled_bias = None
        self.local_reparam = local_reparam


        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        # prior initialization
        if init_prior == 'zeros':
            weights_mu_prior = torch.zeros(out_features, in_features)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
        else:
            raise RuntimeError(f'Wrong prior initialization. It should be either "zeros" or "random", but got {init_prior}')

        bias_mu_prior = torch.zeros(out_features)
        if prior_sigma_scale is not None:
            rho_prior_value = sigma_to_rho(prior_sigma_scale * sigma_weights).item()
        else:
            rho_prior_value = rho_prior
        weights_rho_prior = torch.ones(out_features, in_features) * rho_prior_value
        bias_rho_prior = torch.ones(out_features) * rho_prior_value

        # Posterior initialization
        if init_post == 'zeros':
            weights_mu_init = torch.zeros(out_features, in_features)
        elif init_post == 'random':
            if init_prior == 'random':
                weights_mu_init = weights_mu_prior.clone()
            else:
                weights_mu_init = trunc_normal_(torch.Tensor(out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
        else:
            raise RuntimeError(f'Wrong posterior initialization. It should be either "zeros" or "random", but got {init_post}')

        bias_mu_init = torch.zeros(out_features)
        if post_sigma_scale is not None:
            rho_post_value = sigma_to_rho(post_sigma_scale * sigma_weights).item()
        else:
            rho_post_value = rho_post
        weights_rho_post = torch.ones(out_features, in_features) * rho_post_value
        bias_rho_post = torch.ones(out_features) * rho_post_value

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_post.clone(), fixed=False)
        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_post.clone(), fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_prior.clone(), fixed=True)
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_prior.clone(), fixed=True)

        self.kl_div = 0

    def sample_weights(self):
        self.sampled_weight = self.weight.sample()
        self.sampled_bias = self.bias.sample()

    def clear_sample(self):
        self.sampled_weight = None
        self.sampled_bias = None

    def _forward_local_reparam(self, input):
        """Local reparameterization trick (Kingma, Salimans & Welling, 2015).

        Instead of sampling a full weight tensor W ~ q(W) and computing
        y = x @ W + b, sample the pre-activation y directly from its
        induced Gaussian. Since the weight entries are independent
        (mean-field posterior), y_j = sum_i x_i * w_ij + b_j is itself
        Gaussian with:
            mean(y) = x @ mu_W + mu_b
            var(y)  = x^2 @ sigma_W^2 + sigma_b^2
        (variance of a sum of independent terms is the sum of variances;
        squaring x and convolving/matmul-ing with the weight *variance*
        gives exactly that sum). This is mathematically exact for a linear
        layer (no approximation) and gives lower-variance gradient
        estimates than sampling W directly, since the noise is now
        per-example instead of shared across the whole batch.
        """
        mean = F.linear(input, self.weight.mu, self.bias.mu)
        var = F.linear(input ** 2, self.weight.sigma ** 2, self.bias.sigma ** 2)
        # Analytically var >= 0 always (sum of nonnegative terms), but clamp
        # defensively against fp round-off before the sqrt.
        var = var.clamp(min=1e-8)
        eps = torch.randn_like(mean)
        return mean + torch.sqrt(var) * eps

    def forward(self, input, stochastic=False, local_reparam: bool = None):
        if local_reparam is None:
            local_reparam = self.local_reparam
        if stochastic:
            if self.sampled_weight is not None:
                # A specific weight sample was cached via sample_weights()
                # (e.g. to hold one coherent network sample fixed across
                # several forward calls). Local reparam draws fresh
                # per-call activation noise instead of reusing that exact
                # sample, so it would silently break that guarantee -
                # fall back to using the cached sample directly.
                result = F.linear(input, self.sampled_weight, self.sampled_bias)
            elif local_reparam:
                result = self._forward_local_reparam(input)
            else:
                weight = self.weight.sample()
                bias = self.bias.sample()
                result = F.linear(input, weight, bias)
        else:
            result = F.linear(input, self.weight.mu, self.bias.mu)

        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                self.bias.compute_kl(self.bias_prior)

        return result

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

    rho_post : float
        scale hyperparmeter (to initialise the scale of
        the posterior)

    rho_prior : float
        scale hyperparmeter (to set the scale of
        the prior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior
    
    init_post : string
        string that indicates the way to initialise the posterior:
    
    stride : int
        Stride of the convolution

    padding: int
        Zero-padding added to both sides of the input

    dilation: int
        Spacing between kernel elements

    post_sigma_scale, prior_sigma_scale : float, optional
        See ProbLinear - fan-in-scaled alternative to an absolute rho_post/
        rho_prior. None (default) preserves the original behaviour.

    local_reparam : bool, optional
        See ProbLinear - default used by `forward()` when its own
        `local_reparam` argument is not explicitly overridden per-call.
        Default True.

    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_post = -3.0, rho_prior=-3.0,
                 prior_dist='gaussian', init_post = 'random', init_prior = 'zeros', stride=1,
                 padding=0, dilation=1, groups = 1,
                 post_sigma_scale=None, prior_sigma_scale=None, local_reparam=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.sampled_weight = None
        self.sampled_bias = None
        self.local_reparam = local_reparam

        # He-style sigma for initialization
        in_features = self.in_channels
        out_features = self.out_channels
        sigma_weights = 1. / np.sqrt(in_channels * kernel_size)

        # Prior Initialization
        if init_prior == 'zeros':
            weights_mu_prior = torch.zeros(out_features, in_features, kernel_size)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(out_features, in_features, kernel_size),
                                    0, sigma_weights, -2*sigma_weights, 2*sigma_weights)   
        else:
            raise RuntimeError(f'Wrong prior initialization. It should be either "zeros" or "random", but got {init_prior}')

        bias_mu_prior = torch.zeros(out_features)
        if prior_sigma_scale is not None:
            rho_prior_value = sigma_to_rho(prior_sigma_scale * sigma_weights).item()
        else:
            rho_prior_value = rho_prior
        weights_rho_prior = torch.ones(out_channels, in_channels, kernel_size) * rho_prior_value
        bias_rho_prior = torch.ones(out_channels) * rho_prior_value

        # Posterior Initialization
        if init_post == 'zeros':
            weights_mu_init = torch.zeros(out_channels, in_channels, kernel_size)
        elif init_post == 'random':
            if init_prior == 'random':
                weights_mu_init = weights_mu_prior.clone()
            else:
                weights_mu_init = trunc_normal_(torch.Tensor(out_features, in_features, kernel_size),
                                        0, sigma_weights, -2*sigma_weights, 2*sigma_weights)   
        else:
            raise RuntimeError(f'Wrong posterior initialization. It should be either "zeros" or "random", but got {init_post}')
        
        bias_mu_init = torch.zeros(out_channels)
        if post_sigma_scale is not None:
            rho_post_value = sigma_to_rho(post_sigma_scale * sigma_weights).item()
        else:
            rho_post_value = rho_post
        weights_rho_post = torch.ones(out_channels, in_channels, kernel_size) * rho_post_value
        bias_rho_post = torch.ones(out_channels) * rho_post_value

        # priors = fixed, posteriors = learnable
        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Unknown prior_dist {prior_dist}')

        self.weight = dist(weights_mu_init.clone(), weights_rho_post.clone(), fixed=False)
        self.bias = dist(bias_mu_init.clone(), bias_rho_post.clone(), fixed=False)
        self.weight_prior = dist(weights_mu_prior.clone(), weights_rho_prior.clone(), fixed=True)
        self.bias_prior = dist(bias_mu_prior.clone(), bias_rho_prior.clone(), fixed=True)

        self.kl_div = 0

    def sample_weights(self):
        self.sampled_weight = self.weight.sample()
        self.sampled_bias = self.bias.sample()

    def clear_sample(self):
        self.sampled_weight = None
        self.sampled_bias = None

    def _forward_local_reparam(self, x):
        """See ProbLinear._forward_local_reparam for the identity being used
        here. Applied via conv1d instead of a matmul: each output position
        is still a sum of independent, weight-scaled input terms, so
        convolving x^2 with the *weight variance* "kernel" gives exactly
        the pre-activation variance at every position.
        """
        mean = F.conv1d(x, self.weight.mu, self.bias.mu, stride=self.stride, padding=self.padding,
                         dilation=self.dilation, groups=self.groups)
        var = F.conv1d(x ** 2, self.weight.sigma ** 2, self.bias.sigma ** 2, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)
        var = var.clamp(min=1e-8)
        eps = torch.randn_like(mean)
        return mean + torch.sqrt(var) * eps

    def forward(self, x, stochastic=False, local_reparam: bool = None):
        if local_reparam is None:
            local_reparam = self.local_reparam
        if stochastic:
            if self.sampled_weight is not None:
                # See ProbLinear.forward - reuse the cached sample as-is
                # rather than drawing fresh per-call activation noise.
                result = F.conv1d(x, self.sampled_weight, self.sampled_bias, stride=self.stride,
                                   padding=self.padding, dilation=self.dilation, groups=self.groups)
            elif local_reparam:
                result = self._forward_local_reparam(x)
            else:
                weight = self.weight.sample()
                bias = self.bias.sample()
                result = F.conv1d(x, weight, bias, stride=self.stride, padding=self.padding,
                                    dilation=self.dilation, groups=self.groups)
        else:
            result = F.conv1d(x, self.weight.mu, self.bias.mu, stride=self.stride, padding=self.padding,
                                dilation=self.dilation, groups=self.groups)

        if self.training:
            self.kl_div = self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(self.bias_prior)

        return result

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

    rho_post : float
        scale hyperparameter (to initialise the scale of the posterior)

    rho_prior : float
        scale hyperparameter (to set the scale of the prior)

    prior_dist : string
        string that indicates the type of distribution for the prior and posterior

    init_post : string
        string that indicates the way to initialise the posterior:

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

    post_sigma_scale, prior_sigma_scale : float, optional
        See ProbLinear - fan-in-scaled alternative to an absolute rho_post/
        rho_prior. None (default) preserves the original behaviour.

    local_reparam : bool, optional
        See ProbLinear - default used by `forward()` when its own
        `local_reparam` argument is not explicitly overridden per-call.
        Default True.

    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_post = -3.0, rho_prior=-3.0,
                 prior_dist='gaussian', init_post = 'random', init_prior = 'zeros', stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1,
                 post_sigma_scale=None, prior_sigma_scale=None, local_reparam=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.sampled_weight = None
        self.sampled_bias = None
        self.local_reparam = local_reparam

        # He-style sigma for initialization
        sigma_weights = 1. / np.sqrt(in_channels * kernel_size)
        
        # prior init
        if init_prior == 'zeros':
            weights_mu_prior = torch.zeros(in_channels, out_channels, kernel_size)
        elif init_prior == 'random':
            weights_mu_prior = trunc_normal_(torch.Tensor(
                in_channels, out_channels, kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
        else:
            raise RuntimeError(f'Wrong prior initialization. It should be either "zeros" or "random", but got {init_prior}')

        bias_mu_prior = torch.zeros(out_channels)  # Fixed: out_channels
        if prior_sigma_scale is not None:
            rho_prior_value = sigma_to_rho(prior_sigma_scale * sigma_weights).item()
        else:
            rho_prior_value = rho_prior
        weights_rho_prior = torch.ones(in_channels, out_channels, kernel_size) * rho_prior_value
        bias_rho_prior = torch.ones(out_channels) * rho_prior_value

        # posterior init
        if init_post == 'zeros':
            weights_mu_init = torch.zeros(in_channels, out_channels, kernel_size)
        elif init_post == 'random':
            if init_prior == 'random':
                weights_mu_init = weights_mu_prior.clone()
            else:
                weights_mu_init = trunc_normal_(torch.Tensor(
                    in_channels, out_channels, kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
        else:
            raise RuntimeError(f'Wrong posterior initialization. It should be either "zeros" or "random", but got {init_post}')

        bias_mu_init = torch.zeros(out_channels)

        if post_sigma_scale is not None:
            rho_post_value = sigma_to_rho(post_sigma_scale * sigma_weights).item()
        else:
            rho_post_value = rho_post
        weights_rho_post = torch.ones(in_channels, out_channels, kernel_size) * rho_post_value
        bias_rho_post = torch.ones(out_channels) * rho_post_value

        # priors = fixed, posteriors = learnable
        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Unknown prior_dist {prior_dist}')

        self.weight = dist(weights_mu_init.clone(), weights_rho_post.clone(), fixed=False)
        self.bias = dist(bias_mu_init.clone(), bias_rho_post.clone(), fixed=False)
        self.weight_prior = dist(weights_mu_prior.clone(), weights_rho_prior.clone(), fixed=True)
        self.bias_prior = dist(bias_mu_prior.clone(), bias_rho_prior.clone(), fixed=True)

        self.kl_div = 0

    def sample_weights(self):
        self.sampled_weight = self.weight.sample()
        self.sampled_bias = self.bias.sample()

    def clear_sample(self):
        self.sampled_weight = None
        self.sampled_bias = None

    def _forward_local_reparam(self, x):
        """See ProbLinear._forward_local_reparam for the identity being used
        here, applied via conv_transpose1d instead of conv1d/matmul. Each
        output position is still a sum of independent, weight-scaled input
        terms (transposed convolution is a linear map like any other), so
        the same "convolve the squared input with the weight variance"
        trick gives the exact pre-activation variance.
        """
        mean = F.conv_transpose1d(x, self.weight.mu, self.bias.mu, stride=self.stride, padding=self.padding,
                                   output_padding=self.output_padding, dilation=self.dilation, groups=self.groups)
        var = F.conv_transpose1d(x ** 2, self.weight.sigma ** 2, self.bias.sigma ** 2, stride=self.stride,
                                  padding=self.padding, output_padding=self.output_padding,
                                  dilation=self.dilation, groups=self.groups)
        var = var.clamp(min=1e-8)
        eps = torch.randn_like(mean)
        return mean + torch.sqrt(var) * eps

    def forward(self, x, stochastic=False, local_reparam: bool = None):
        if local_reparam is None:
            local_reparam = self.local_reparam
        if stochastic:
            if self.sampled_weight is not None:
                # See ProbLinear.forward - reuse the cached sample as-is
                # rather than drawing fresh per-call activation noise.
                result = F.conv_transpose1d(x, self.sampled_weight, self.sampled_bias, stride=self.stride,
                                             padding=self.padding, output_padding=self.output_padding,
                                             dilation=self.dilation, groups=self.groups)
            elif local_reparam:
                result = self._forward_local_reparam(x)
            else:
                weight = self.weight.sample()
                bias = self.bias.sample()
                result = F.conv_transpose1d(x, weight, bias, stride=self.stride, padding=self.padding,
                                             output_padding=self.output_padding, dilation=self.dilation,
                                             groups=self.groups)
        else:
            result = F.conv_transpose1d(x, self.weight.mu, self.bias.mu, stride=self.stride, padding=self.padding,
                                         output_padding=self.output_padding, dilation=self.dilation,
                                         groups=self.groups)

        if self.training:
            self.kl_div = self.weight.compute_kl(self.weight_prior) + self.bias.compute_kl(self.bias_prior)

        return result

class ProbDownsample1d(nn.Module):
    ''' This class is initialized with nn.conv1D layer from a deterministic network
    the init_layer and init_layer_prior must be 'Downsample1d' layer from deterministic network
    '''
    def __init__(self, dim, rho_post=-3.0, rho_prior=-3.0, prior_dist='gaussian',
                 init_post='random', init_prior='zeros',
                 post_sigma_scale=None, prior_sigma_scale=None, local_reparam=True):
        super().__init__()

        self.conv = ProbConv1d(
            dim, dim, kernel_size=3, stride=2, padding=1, rho_post=rho_post,
            rho_prior=rho_prior, prior_dist=prior_dist, init_post=init_post,
            init_prior=init_prior,
            post_sigma_scale=post_sigma_scale, prior_sigma_scale=prior_sigma_scale,
            local_reparam=local_reparam
        )
    def sample_weights(self):
        self.conv.sample_weights()

    def clear_sample(self):
        self.conv.clear_sample()

    def forward(self, x, stochastic=False):
        return self.conv(x, stochastic=stochastic)

    def compute_kl(self):
        #! make sure this kl divergence is used only during training
        return self.conv.kl_div


class ProbUpsample1d(nn.Module):
    def __init__(self, dim, rho_post=-3.0, rho_prior=-3.0, prior_dist='gaussian',
                  init_post='random', init_prior='zeros',
                  post_sigma_scale=None, prior_sigma_scale=None, local_reparam=True):
        super().__init__()

        self.conv = ProbConvTranspose1d(
            dim, dim, kernel_size=4, stride=2, padding=1, rho_post=rho_post,
            rho_prior=rho_prior, prior_dist=prior_dist, init_post=init_post, init_prior=init_prior,
            post_sigma_scale=post_sigma_scale, prior_sigma_scale=prior_sigma_scale,
            local_reparam=local_reparam
        )

    def sample_weights(self):
        self.conv.sample_weights()

    def clear_sample(self):
        self.conv.clear_sample()

    def forward(self, x, stochastic=False):
        return self.conv(x, stochastic=stochastic)

    def compute_kl(self):
        return self.conv.kl_div

class ProbConv1dBlock(nn.Module):
    """
    Probabilistic Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8,
                 rho_post=-3.0, rho_prior=-3.0, prior_dist='gaussian',
                 init_post='random', init_prior='zeros',
                 post_sigma_scale=None, prior_sigma_scale=None, local_reparam=True):
        super().__init__()

        self.block = nn.Sequential(
            ProbConv1d(
            inp_channels, out_channels, kernel_size, rho_post = rho_post,
            rho_prior=rho_prior, prior_dist=prior_dist, init_post=init_post, init_prior=init_prior,
            padding=kernel_size // 2,                        # Maintain same padding,
            post_sigma_scale=post_sigma_scale, prior_sigma_scale=prior_sigma_scale,
            local_reparam=local_reparam
        ),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )
    
    def sample_weights(self):
        self.block[0].sample_weights()
    def clear_sample(self):
        self.block[0].clear_sample()

    def forward(self, x, stochastic=False):
        # Apply probabilistic convolution
        x = self.block[0](x, stochastic=stochastic)

        # Apply group norm and activation
        x = self.block[1](x)
        x = self.block[2](x)
        return x
    
    def compute_kl(self):
        return self.block[0].kl_div