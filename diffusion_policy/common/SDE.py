import torch
import numpy as np
import abc
import math

################################################# SDE #################################################
class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t, local_cond, global_cond, stochastic = False, clamping = False):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        # print ("drift : ", drift)
        # print ("diffusion : ", diffusion)
        score = score_fn(x, t, local_cond, global_cond, stochastic=stochastic, clamping=clamping)
        drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t, local_cond, global_cond, stochastic = False, clamping = False):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None] ** 2 * score_fn(x, t, local_cond, global_cond, stochastic=stochastic, clamping=clamping) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()

class VPSDE(SDE):
  def __init__(self, noise_scheduler, beta_min=0.1, beta_max= 20.0, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.scheduler = noise_scheduler
    self.N = self.scheduler.config.num_train_timesteps
    self.discrete_betas = self.scheduler.betas
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
    self.beta_0 = beta_min
    self.beta_1 = beta_max

  @property
  def T(self):
    return 1
  
  # This interpolate beta among all descrite betas
  # def get_beta_fn(self, betas):
  #     def beta_t(t):
  #         t = t.clamp(0, 1)
  #         scaled_t = t * (len(betas) - 1)
  #         low = torch.floor(scaled_t).long()
  #         high = torch.clamp(low + 1, max=len(betas) - 1)
  #         w = scaled_t - low.float()
  #         return (1 - w) * betas[low] + w * betas[high]
  #     return beta_t
  
  def beta_for_alpha_bar(self, tt, min_beta = 0.1, max_beta=200.0):
    """
    Enhanced continuous beta(t) with safety checks.
    
    Args:
        t: Tensor in [0,1]
        eps: Prevents singularity at t=1 (1e-5 matches paper)
        max_beta: Clips beta for extreme cases (optional)
    """
    tt = torch.clamp(tt, max=1 - 1e-6)

    theta = (tt * math.pi / 2)
    theta_prime = math.pi
    beta = theta_prime * torch.tan(theta)
    
    # # Optional: Clip for extreme numerical cases
    if max_beta is not None:
      beta = torch.clamp(beta, max=max_beta)
    if min_beta is not None:
         beta = torch.clamp(beta, min=min_beta)
    return beta
  
  def sde(self, x, t):
    if not isinstance(t, (float, torch.Tensor)):
        raise TypeError(f"Expected t to be float or tensor, got {type(t)}")
    
    # t is already between [eps, 1-eps)
    if self.scheduler.config.beta_schedule=="linear":
      beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    elif self.scheduler.config.beta_schedule=="squaredcos_cap_v2":
      beta_t = self.beta_for_alpha_bar(t)
    else:
      raise ValueError("Beta scheduler is not squaredcos_cap_v2 or linear")

    drift = -0.5 * beta_t[:, None, None] * x
    # print ("beta:", beta_t)
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    if self.scheduler.config.beta_schedule=="linear":
      log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
      mean = torch.exp(log_mean_coeff[:, None, None]) * x
      std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

    elif self.scheduler.config.beta_schedule=="squaredcos_cap_v2":
      theta = (t * math.pi / 2)
      torch.log(torch.cos(theta))
      log_mean_coeff = torch.log(torch.cos(theta))
      mean = torch.exp(log_mean_coeff[:, None, None]) * x
      std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    else:
      raise ValueError("Beta scheduler is not squaredcos_cap_v2 or linear")
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1,2)) / 2.
    return logps
  
  #! I think this is for sampling and do not need that 
  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None] * x - x
    G = sqrt_beta
    return f, G
