import math
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional, Tuple, Union

from diffusion_policy.CFM.path.scheduler import CondOTScheduler, Scheduler
from diffusion_policy.CFM.path.path import ProbPath
from diffusion_policy.CFM.path.path_sample import PathSample
from diffusion_policy.CFM.utils import expand_tensor_like

def normal_log_prob(value, std):
    var = std**2
    log_scale = math.log(std) if isinstance(std, float) else std.log()

    logp = -((value) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))

    return logp.sum(dim=-1)

def vectorized_computation(xt, t, x1, x1_vf_batch, prior_std, sigma=0.0):
    """
    Vectorized implementation to replace the for-loop.

    Args:
        xt (torch.Tensor): Batch of noisy data points. Shape: [B, D]
        t (torch.Tensor): Batch of time steps. Shape: [B]
        x1 (torch.Tensor): Batch of original data points. Shape: [B, D]
        x1_vf_batch (torch.Tensor): A fixed set of candidate points. Shape: [N, D]
        p0: A probability distribution object with a log_prob method.
        sigma (float): A scalar hyperparameter.

    Returns:
        ut (torch.Tensor): The estimated vector field. Shape: [B, D]
    """
    b, h, d = xt.shape              # This is for state-based control
    
    bvf, *_ = x1_vf_batch.shape

    xt = xt.view(b, -1)
    x1 = x1.view(b, -1)
    x1_vf_batch = x1_vf_batch.view(bvf, -1)

    # Get batch size (B), number of candidates (N), and feature dimension (D)
    B, D = xt.shape
    N = x1_vf_batch.shape[0]

    # --- 1. Pre-calculate batch-wise scalars ---
    deno = 1 - (1 - sigma) * t  # Shape: [B]

    # --- 2. Construct the expanded candidate tensor for x1 ---
    # This step replaces the loop's logic of `x1_vf_batch[0] = x1[i]`.
    # We create a tensor of shape [B, N, D] where for each item in the batch,
    # the first candidate is its corresponding x1 value.

    # [B, D] -> [B, 1, D]
    x1_batch_specific = x1.unsqueeze(1)
    # [N-1, D] -> [1, N-1, D] -> [B, N-1, D]
    other_candidates = x1_vf_batch[1:].unsqueeze(0).expand(B, -1, -1)
    # Concatenate to form the full candidate set for the batch.
    # Shape: [B, N, D]
    x1_vf_batch_expanded = torch.cat([x1_batch_specific, other_candidates], dim=1)

    # --- 3. Prepare tensors for broadcasting ---
    # Reshape tensors to enable element-wise operations with the [B, N, D] tensor.
    xt_exp = xt.unsqueeze(1)  # Shape: [B, 1, D]
    t_exp = t.view(B, 1, 1)  # Shape: [B, 1, 1]
    deno_exp = deno.view(B, 1, 1)  # Shape: [B, 1, 1]

    # --- 4. Batch computation of x0 ---
    # This single line computes x0_xbar for all batch items and all candidates.
    # Resulting shape: [B, N, D]
    x0_xbar_batch = (xt_exp - t_exp * x1_vf_batch_expanded) / deno_exp

    # --- 5. Batch computation of log probabilities ---
    # p0.log_prob expects input of shape [B*N, D]
    # logprob_batch = p0.log_prob(x0_xbar_batch.view(B * N, D)).view(
    #     B, N
    # )  # Shape: [B, N]
    logprob_batch = normal_log_prob(x0_xbar_batch.view(B * N, D), prior_std).view(B, N)

    # --- 6. Batch stable softmax ---
    max_logprob_batch, _ = torch.max(logprob_batch, dim=1, keepdim=True)
    logprob_batch = logprob_batch - max_logprob_batch

    prob_batch = torch.exp(logprob_batch)
    # Normalize probabilities for each batch item across all its candidates
    norm_prob_batch = prob_batch / torch.sum(prob_batch, dim=1, keepdim=True)  # Shape: [B, N]

    # --- 7. Batch computation of the vector field estimate `ut` ---
    # First, calculate the vector field `v` for all candidates
    # Shape: [B, N, D]
    #! make  sure the vector field is computed correctly for all candidates
    v_batch_expanded = (x1_vf_batch_expanded - (1 - sigma) * xt_exp) / deno_exp
    #v_batch_expanded = x1_vf_batch_expanded - x0_xbar_batch
    
    # Finally, compute the weighted average of `v` using the normalized probabilities.
    # This is an element-wise multiplication followed by a sum over the candidate dimension.
    # ( [B, N, 1] * [B, N, D] ) -> sum(dim=1) -> [B, D]
    ut = torch.sum(norm_prob_batch.unsqueeze(2) * v_batch_expanded, dim=1)

    v_i = v_batch_expanded[:, 0, :]

    deviation = v_i - ut

    # restore below if you want the normalized norm
    # norm_score = deviation.norm(dim=-1) / v_i.norm(dim=-1)
    # compute the cosine similarity of v_i and deviation
    cosine_similarity = torch.nn.functional.cosine_similarity(v_i, ut, dim=-1)
    norm_score = cosine_similarity

    ut = ut.view(b, h, d)
    # print (norm_prob_batch.shape)
    # print ( norm_prob_batch[:, 1])
    # print (norm_score)

    return ut, norm_prob_batch[:, 0], norm_score


class AffineProbPath(ProbPath):
    r"""The ``AffineProbPath`` class represents a specific type of probability path where the transformation between distributions is affine.
    An affine transformation can be represented as:

    .. math::

        X_t = \alpha_t X_1 + \sigma_t X_0,

    where :math:`X_t` is the transformed data point at time `t`. :math:`X_0` and :math:`X_1` are the source and target data points, respectively. :math:`\alpha_t` and :math:`\sigma_t` are the parameters of the affine transformation at time `t`.

    The scheduler is responsible for providing the time-dependent parameters :math:`\alpha_t` and :math:`\sigma_t`, as well as their derivatives, which define the affine transformation at any given time `t`.

    Using ``AffineProbPath`` in the flow matching framework:

    .. code-block:: python

        # Instantiates a probability path
        my_path = AffineProbPath(...)
        mse_loss = torch.nn.MSELoss()

        for x_1 in dataset:
            # Sets x_0 to random noise
            x_0 = torch.randn()

            # Sets t to a random value in [0,1]
            t = torch.rand()

            # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
            path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)

            # Computes the MSE loss w.r.t. the velocity
            loss = mse_loss(path_sample.dx_t, my_model(x_t, t))
            loss.backward()

    Args:
        scheduler (Scheduler): An instance of a scheduler that provides the parameters :math:`\alpha_t`, :math:`\sigma_t`, and their derivatives over time.

    """

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler
        
    def sample(
        self,
        x_0: Tensor,
        x_1: Tensor,
        t: Tensor,
        x1_vf_batch: Optional[Dataset],
        prior_std: float,
        debug=False,
    ) -> Union[PathSample, Tuple[PathSample, torch.Tensor]]:
        r"""Sample from the affine probability path:

        | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
        | return :math:`X_0, X_1, X_t = \alpha_t X_1 + \sigma_t X_0`, and the conditional velocity at :math:`X_t, \dot{X}_t = \dot{\alpha}_t X_1 + \dot{\sigma}_t X_0`.

        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            PathSample: a conditional sample at :math:`X_t \sim p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        scheduler_output = self.scheduler(t)

        alpha_t = expand_tensor_like(input_tensor=scheduler_output.alpha_t, expand_to=x_1)
        sigma_t = expand_tensor_like(input_tensor=scheduler_output.sigma_t, expand_to=x_1)
        d_alpha_t = expand_tensor_like(input_tensor=scheduler_output.d_alpha_t, expand_to=x_1)
        d_sigma_t = expand_tensor_like(input_tensor=scheduler_output.d_sigma_t, expand_to=x_1)

        # # construct xt ~ p_t(x|x1).
        x_t = sigma_t * x_0 + alpha_t * x_1

        if x1_vf_batch is not None:
            dx_t = torch.zeros_like(x_t)
            dx_t, first_element_prob, norm_score = vectorized_computation(x_t, t, x_1, x1_vf_batch, prior_std, sigma=self.scheduler.sigma)
        else:
            dx_t = d_sigma_t * x_0 + d_alpha_t * x_1
            first_element_prob = None
            norm_score = None

        if debug:
            return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t), first_element_prob, norm_score
        else:
            return PathSample(x_t=x_t, dx_t=dx_t, x_1=x_1, x_0=x_0, t=t)

    def target_to_velocity(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from x_1 representation to velocity.

        | given :math:`X_1`.
        | return :math:`\dot{X}_t`.

        Args:
            x_1 (Tensor): target data point.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: velocity.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = d_sigma_t / sigma_t
        b_t = (d_alpha_t * sigma_t - d_sigma_t * alpha_t) / sigma_t

        return a_t * x_t + b_t * x_1

    def epsilon_to_velocity(self, epsilon: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from epsilon representation to velocity.

        | given :math:`\epsilon`.
        | return :math:`\dot{X}_t`.

        Args:
            epsilon (Tensor): noise in the path sample.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: velocity.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = d_alpha_t / alpha_t
        b_t = (d_sigma_t * alpha_t - d_alpha_t * sigma_t) / alpha_t

        return a_t * x_t + b_t * epsilon

    def velocity_to_target(self, velocity: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from velocity to x_1 representation.

        | given :math:`\dot{X}_t`.
        | return :math:`X_1`.

        Args:
            velocity (Tensor): velocity at the path sample.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: target data point.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = -d_sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)
        b_t = sigma_t / (d_alpha_t * sigma_t - d_sigma_t * alpha_t)

        return a_t * x_t + b_t * velocity

    def epsilon_to_target(self, epsilon: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from epsilon representation to x_1 representation.

        | given :math:`\epsilon`.
        | return :math:`X_1`.

        Args:
            epsilon (Tensor): noise in the path sample.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: target data point.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t

        a_t = 1 / alpha_t
        b_t = -sigma_t / alpha_t

        return a_t * x_t + b_t * epsilon

    def velocity_to_epsilon(self, velocity: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from velocity to noise representation.

        | given :math:`\dot{X}_t`.
        | return :math:`\epsilon`.

        Args:
            velocity (Tensor): velocity at the path sample.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: noise in the path sample.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        d_alpha_t = scheduler_output.d_alpha_t
        sigma_t = scheduler_output.sigma_t
        d_sigma_t = scheduler_output.d_sigma_t

        a_t = -d_alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)
        b_t = alpha_t / (d_sigma_t * alpha_t - d_alpha_t * sigma_t)

        return a_t * x_t + b_t * velocity

    def target_to_epsilon(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert from x_1 representation to velocity.

        | given :math:`X_1`.
        | return :math:`\epsilon`.

        Args:
            x_1 (Tensor): target data point.
            x_t (Tensor): path sample at time t.
            t (Tensor): time in [0,1].

        Returns:
            Tensor: noise in the path sample.
        """
        scheduler_output = self.scheduler(t)

        alpha_t = scheduler_output.alpha_t
        sigma_t = scheduler_output.sigma_t

        a_t = 1 / sigma_t
        b_t = -alpha_t / sigma_t

        return a_t * x_t + b_t * x_1


class CondOTProbPath(AffineProbPath):
    r"""The ``CondOTProbPath`` class represents a conditional optimal transport probability path.

    This class is a specialized version of the ``AffineProbPath`` that uses a conditional optimal transport scheduler to determine the parameters of the affine transformation.

    The parameters :math:`\alpha_t` and :math:`\sigma_t` for the conditional optimal transport path are defined as:

    .. math::

        \alpha_t = t \quad \text{and} \quad \sigma_t = 1 - t.
    """

    def __init__(self, sigma: float = 0.0):
        self.scheduler = CondOTScheduler(sigma=sigma)