import math
import torch

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
    # v_batch_expanded = (x1_vf_batch_expanded - (1 - sigma) * xt_exp) / deno_exp
    v_batch_expanded = x1_vf_batch_expanded - x0_xbar_batch
    
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

    return ut, norm_prob_batch[:, 0], norm_score