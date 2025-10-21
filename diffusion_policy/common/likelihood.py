import numpy as np 
import torch
from scipy import integrate
from diffusion_policy.common.SDE import VPSDE

def get_model_fn(model, train=False):
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels, local_cond, global_cond):
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(sample = x, timestep = labels, local_cond=local_cond, global_cond=global_cond)
    else:
      model.train()
      return model(sample = x, timestep = labels, local_cond=local_cond, global_cond=global_cond)

  return model_fn

def get_score_fn(sde, model, train=False, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  if train:
    model_fn = model.train()
  else:
    model_fn = model.eval()

  if isinstance(sde, VPSDE):
    def score_fn(x, t, local_cond, global_cond):
      # Scale neural network output by standard deviation and flip sign
      if continuous:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        noise = model_fn(sample = x, timestep = labels, local_cond=local_cond, global_cond=global_cond)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        noise = model_fn(sample = x, timestep = labels, local_cond=local_cond, global_cond=global_cond)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -noise / std[:, None, None]
      return score
    
  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


######################################### Function to compute the offset for log likelihood #############################
def get_data_inverse_scaler(centered = True):
  """Inverse data normalizer."""
  if centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

########################################### Compute the estimate trace of the jacobian matrix #################
def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, local_cond, global_cond, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t, local_cond, global_cond) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn

########################################### Compute the exact trace of the jacobian matrix #################
def get_exact_div_fn(fn):
    """
    Computes the exact trace of the Jacobian of fn(x, t) w.r.t. x, per batch item.
    """
    def div_fn(x, t, local_cond, global_cond):
        B, T, F = x.shape
        with torch.enable_grad():
            x.requires_grad_(True)
            output = fn(x, t, local_cond, global_cond)  # Shape: (B, T, F)
            trace = torch.zeros(B, device=x.device)

            flat_size = T * F

            for i in range(flat_size):
                grad_outputs = torch.zeros_like(output)
                grad_outputs.reshape(B, -1)[:, i] = 1.0
                grad = torch.autograd.grad(
                    outputs=output,
                    inputs=x,
                    grad_outputs=grad_outputs,
                    create_graph=False,
                    retain_graph=True
                )[0]  # (B, T, F)
                trace += grad.reshape(B, -1)[:, i]
        x.requires_grad_(False)
        return trace  # (B,)
    return div_fn

############################################################ likeliood ################################################
def get_likelihood_fn(sde, continuous=False, exact = True, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """
    inverse_scaler = get_data_inverse_scaler()
    
    def drift_fn(model, x, t, local_cond, global_cond):
        """The drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=continuous)
        # Probability flow ODE is a special case of Reverse SDE
        rsde = sde.reverse(score_fn, probability_flow=True)

        # Here you just need the drift not the diffusion coefficient
        return rsde.sde(x, t, local_cond, global_cond)[0]

    
    def div_fn(model, x, t, local_cond, global_cond, noise):
        """
        This function returns the Hutchinson-based divergence estimate of the drift function
        drift_fn(model, xx, tt) returns the vector field whose divergence we want to compute.

        get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))
        → returns a function: div_fn(x, t, noise)

        We immediately call that returned function with (x, t, noise)
        """
        if exact:
           return get_exact_div_fn(lambda xx, tt, local_condd, global_condd: drift_fn(model, xx, tt, local_condd, global_condd))(x, t, local_cond, global_cond)
        else:
            return get_div_fn(lambda xx, tt, local_condd, global_condd: drift_fn(model, xx, tt, local_condd, global_condd))(x, t, local_cond, global_cond, noise)
    

    def likelihood_fn(model, action, local_cond, global_cond):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
            model: A score model.
            data: A PyTorch tensor.

        Returns:
            bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
            z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
            nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """

        with torch.no_grad():
            shape = action.shape
            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(action)
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(action, low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                """ It defines how the state evolves over time under the probability flow ODE.
                
                """
                sample = from_flattened_numpy(x[:-shape[0]], shape).to(action.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t

                drift = to_flattened_numpy(drift_fn(model, sample, vec_t, local_cond, global_cond))
                logp_grad = to_flattened_numpy(div_fn(model, sample, vec_t, local_cond, global_cond, epsilon))

                return np.concatenate([drift, logp_grad], axis=0)

            init = np.concatenate([to_flattened_numpy(action), np.zeros((shape[0],))], axis=0)
            # Times at which to store the computed solution
            #t_eval = np.arange(0.0,1.01,0.01)
            #t_eval[0] = eps
            #solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, t_eval = t_eval, rtol=rtol, atol=atol, method=method)
            solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            zp = solution.y[:, -1]
            z = from_flattened_numpy(zp[:-shape[0]], shape).to(action.device).type(torch.float32)
            delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(action.device).type(torch.float32)
            prior_logp = sde.prior_logp(z)
            bpd = -(prior_logp + delta_logp) / np.log(2)
            N = np.prod(shape[1:])
            bpd = bpd / N
            # get the loglikelihood in dataspace 
            # here we assume the data has been normalized between -1 and 1
            # offset = np.log(2/(stats["max"]-stats['min']))
            # bpd = bpd.cpu() + offset
            return bpd, z, nfe

    return likelihood_fn