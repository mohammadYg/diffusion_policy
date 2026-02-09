import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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
        return torch.nn.functional.softplus(self.rho)

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
        epsilon = (0.999*torch.rand(self.scale.size())-0.49999)
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

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(self, in_features, out_features, rho_post = -3.0, rho_prior=-3.0, prior_dist='gaussian'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        # prior initialization
        weights_mu_prior = torch.zeros(out_features, in_features)
        bias_mu_prior = torch.zeros(out_features) 
        weights_rho_prior = torch.ones(out_features, in_features) * rho_prior
        bias_rho_prior = torch.ones(out_features) * rho_prior

        # Posterior initialization 
        weights_mu_init = trunc_normal_(torch.Tensor(
                out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
        bias_mu_init = torch.zeros(out_features)
        # weights_mu_init = weights_mu_prior.clone()
        # bias_mu_init = bias_mu_prior.clone()    
        weights_rho_post = torch.ones(out_features, in_features) * rho_post
        bias_rho_post = torch.ones(out_features) * rho_post
       
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

    stride : int
        Stride of the convolution

    padding: int
        Zero-padding added to both sides of the input

    dilation: int
        Spacing between kernel elements

    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_post = -3.0, rho_prior=-3.0,
                 prior_dist='gaussian', stride=1, padding=0, dilation=1, groups = 1):
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

        # Prior Initialization
        weights_mu_prior = torch.zeros(out_features, in_features, kernel_size)
        bias_mu_prior = torch.zeros(out_features) 
        weights_rho_prior = torch.ones(out_channels, in_channels, kernel_size) * rho_prior
        bias_rho_prior = torch.ones(out_channels) * rho_prior

        # Posterior Initialization
        weights_mu_init = trunc_normal_(torch.Tensor(out_channels, in_channels, kernel_size),
                                    0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
        bias_mu_init = torch.zeros(out_channels)
        # weights_mu_init = weights_mu_prior.clone()
        # bias_mu_init = bias_mu_prior.clone()    
        weights_rho_post = torch.ones(out_channels, in_channels, kernel_size) * rho_post
        bias_rho_post = torch.ones(out_channels) * rho_post

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

    def forward(self, x, stochastic=False):
        #if self.training or stochastic:
        if stochastic:
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

    rho_post : float
        scale hyperparameter (to initialise the scale of the posterior)

    rho_prior : float
        scale hyperparameter (to set the scale of the prior)

    prior_dist : string
        string that indicates the type of distribution for the prior and posterior

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

    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_post = -3.0, rho_prior=-3.0,
                 prior_dist='gaussian', stride=1, padding=0, 
                 output_padding=0, dilation=1, groups=1):
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
        
        # prior init
        weights_mu_prior = torch.zeros(in_channels, out_channels, kernel_size)
        bias_mu_prior = torch.zeros(out_channels)  # Fixed: out_channels
        weights_rho_prior = torch.ones(in_channels, out_channels, kernel_size) * rho_prior
        bias_rho_prior = torch.ones(out_channels) * rho_prior
        
        # posterior init
        weights_mu_init = trunc_normal_(torch.Tensor(
            in_channels, out_channels, kernel_size),  # Note: transposed conv has different weight arrangement
            0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
        bias_mu_init = torch.zeros(out_channels)
        # weights_mu_init = weights_mu_prior.clone()
        # bias_mu_init = bias_mu_prior.clone()    
        weights_rho_post = torch.ones(in_channels, out_channels, kernel_size) * rho_post
        bias_rho_post = torch.ones(out_channels) * rho_post

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

    def forward(self, x, stochastic=False):
        #if self.training or stochastic:
        if stochastic:
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
    def __init__(self, dim, rho_post=-3.0, rho_prior=-3.0, prior_dist='gaussian'):
        super().__init__()

        self.conv = ProbConv1d(
            dim, dim, kernel_size=3, stride=2, padding=1, rho_post=rho_post,
            rho_prior=rho_prior, prior_dist=prior_dist
        )

    def forward(self, x, stochastic=False):
        return self.conv(x, stochastic=stochastic)

    def compute_kl(self):
        #! make sure this kl divergence is used only during training 
        return self.conv.kl_div

    
class ProbUpsample1d(nn.Module):
    def __init__(self, dim, rho_post=-3.0, rho_prior=-3.0, prior_dist='gaussian'):
        super().__init__()
        
        self.conv = ProbConvTranspose1d(
            dim, dim, kernel_size=4, stride=2, padding=1, rho_post = rho_post,
            rho_prior=rho_prior, prior_dist=prior_dist)

    def forward(self, x, stochastic=False):
        return self.conv(x, stochastic=stochastic)
    
    def compute_kl(self):
        return self.conv.kl_div

class ProbConv1dBlock(nn.Module):
    """
    Probabilistic Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8, 
                 rho_post=-3.0, rho_prior=-3.0, prior_dist='gaussian'):
        super().__init__()

        self.block = nn.Sequential(
            ProbConv1d(
            inp_channels, out_channels, kernel_size, rho_post = rho_post,
            rho_prior=rho_prior, prior_dist=prior_dist,
            padding=kernel_size // 2                        # Maintain same padding
        ),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x, stochastic=False):
        # Apply probabilistic convolution
        x = self.block[0](x, stochastic=stochastic)

        # Apply group norm and activation
        x = self.block[1](x)
        x = self.block[2](x)
        return x
    
    def compute_kl(self):
        return self.block[0].kl_div