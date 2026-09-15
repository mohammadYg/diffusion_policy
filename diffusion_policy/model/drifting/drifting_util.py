import math
import torch
import torch.nn.functional as F


def _cdist(x, y, eps=1e-8):
    """Pairwise L2 distance: [B, N, D] x [B, M, D] -> [B, N, M].

    Exact port of official JAX cdist (dot-product formula + eps clamp).
    """
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return torch.sqrt(torch.clamp(sq_dist, min=eps))


def drift_loss(gen, fixed_pos, fixed_neg=None, weight_gen=None, weight_pos=None,
               weight_neg=None, R_list=(0.02, 0.05, 0.2)):
    """Faithful port of official drifting loss (drifting/drift_loss.py, JAX -> PyTorch).

    Args:
        gen: [B, C_g, S] generated samples
        fixed_pos: [B, C_p, S] positive (real) samples
        fixed_neg: [B, C_n, S] negative samples (optional, None = no explicit negatives)
        weight_gen: [B, C_g] (optional, default 1)
        weight_pos: [B, C_p] (optional, default 1)
        weight_neg: [B, C_n] (optional, default 1)
        R_list: tuple of temperature values
    Returns:
        loss: [B]
        info: dict with 'scale' and 'loss_{R}' entries
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1) # [B, C_g+C_n+C_p]

    # Goal computation (no gradients)
    with torch.no_grad():
        info = {}
        # Pairwise distances from each generated point to every target [B, C_g, C_g+C_n+C_p]
        # This includes distances from gen points to each other (self-connections)
        dist = _cdist(old_gen, targets)
        # it weights the distances by the target weights
        # So weighted_dist[b, i, j] = distance from generated point i to target j, 
        # scaled by target j's importance weight
        weighted_dist = dist * targets_w[:, None, :] # [B, 1, C_g+C_n+C_p]
        # This is a weighted average distance, normalized by the average weight
        # without this normalization, scale would grow just because weights are large, not because distances are large.
        scale = weighted_dist.mean() / targets_w.mean()
        info["scale"] = scale
        info["scale_clamped"] = torch.clamp(scale, min=1e-3)
        scale_inputs = scale / (S ** 0.5)
        info["scale_inputs"] = scale_inputs
        scale_inputs = torch.clamp(scale_inputs, min=1e-3)
        info["scale_inputs_clamped"] = scale_inputs

        # Raw (unrenormalized) accuracy/diversity diagnostics, free from `dist`
        # (computed before any /scale division, so - unlike `scale`/`loss_{R}`/
        # `force_scale_{R}` below - these stay meaningful even when the batch's
        # own scale collapses or the force-renormalization saturates, e.g. at
        # small gen_per_label; see the gen_per_label=1 case where train_loss
        # was pinned at a constant while these would have kept moving).
        gen_pos_dist = dist[:, :, -C_p:]  # [B, C_g, C_p]: generated -> true-action distance
        info["mean_dist_to_pos"] = gen_pos_dist.mean()
        info["best_dist_to_pos"] = gen_pos_dist.min(dim=1).values.mean()   # best-of-G ("min-of-N") accuracy
        info["worse_dist_to_pos"] = gen_pos_dist.max(dim=1).values.mean() # "max-of-G" / worse-hypothesis accuracy
        if C_g > 1:
            gen_gen_dist = dist[:, :, :C_g]  # [B, C_g, C_g] (diagonal is exactly 0 - self-distance)
            off_diag = 1.0 - torch.eye(C_g, device=gen.device, dtype=gen.dtype)
            info["diversity"] = (gen_gen_dist * off_diag).sum(dim=(-1, -2)) / off_diag.sum()

        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs

        dist_normed = dist / torch.clamp(scale, min=1e-3)

        # Mask self-connections for gen block
        mask_val = 100.0
        diag_mask = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block_mask = F.pad(diag_mask, (0, C_n + C_p))
        block_mask = block_mask.unsqueeze(0)
        # Adds 100 to the diagonal entries of the gen-to-gen block
        # Since these feed into softmax(-dist/R), 
        # adding 100 to the distance makes exp(-100/R) ≈ 0 — 
        # effectively zeroing out self-attraction.
        dist_normed = dist_normed + block_mask * mask_val

        # Force loop over temperatures
        force_across_R = torch.zeros_like(old_gen_scaled) # [B, C_g, S]

        # Small R = sharp/local forces, large R = soft/global forces.
        for R in R_list:
            logits = -dist_normed / R   #[B, C_g, C_g+C_n+C_p]

            # Row-wise softmax: for each generated point, 
            # a probability distribution over all targets.
            affinity = torch.softmax(logits, dim=-1)    # [B, C_g, C_g+C_n+C_p]
            # Column-wise softmax: for each target, a distribution over generated points
            aff_transpose = torch.softmax(logits, dim=-2)    # [B, C_g, C_g+C_n+C_p]
            # Takes the geometric mean of the two softmaxes
            # a high value only when the generated point "wants" the target 
            # AND the target "wants" the generated point
            affinity = torch.sqrt(torch.clamp(affinity * aff_transpose, min=1e-6)) # [B, C_g, C_g+C_n+C_p]

            # Per-temperature kernel-sharpness diagnostic: renormalize `affinity`
            # (the geometric-mean affinity, before the targets_w reweighting below)
            # into a proper per-generated-point distribution over targets, then
            # measure its (normalized) entropy and top-1 mass. entropy_R -> 1.0
            # means this R behaves near-uniform/soft over targets; -> 0.0 means
            # it's collapsed onto a single nearest target (hard nearest-neighbor).
            # See the diagnose_R.py-style offline sweep this mirrors, but computed
            # live on the network's actual current outputs during real training.
            row = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            n_targets = C_g + C_n + C_p
            info[f"entropy_{R}"] = -(row * row.clamp_min(1e-12).log()).sum(dim=-1) / math.log(n_targets)
            info[f"top1_{R}"] = row.max(dim=-1).values

            affinity = affinity * targets_w[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]    # [B, C_g, C_g+C_n]
            aff_pos = affinity[:, :, split_idx:]    # [B, C_g, C_p]

            sum_pos = aff_pos.sum(dim=-1, keepdim=True) # [B, C_g, 1]
            r_coeff_neg = -aff_neg * sum_pos            # [B, C_g, C_g+C_n]
            sum_neg = aff_neg.sum(dim=-1, keepdim=True) # [B, C_g, 1]
            r_coeff_pos = aff_pos * sum_neg             # [B, C_g, C_p]
            # sum_neg is exactly the "negative/self mass" that drives the
            # attraction-toward-positives coefficient (r_coeff_pos = aff_pos *
            # sum_neg) - this is the quantity that collapses to the 1e-6 clamp
            # floor (and kills the attraction force) when gen_per_label=1 and
            # there are no explicit negatives. Logging it directly exposes that
            # mechanism live instead of needing to re-derive it from symptoms.
            info[f"sum_neg_{R}"] = sum_neg
            info[f"sum_pos_{R}"] = sum_pos

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)  #  [B, C_g, C_g+C_n+C_p]

            total_force_R = torch.einsum("biy,byx->bix", R_coeff, targets_scaled) # [B, C_g, S]

            total_coeffs = R_coeff.sum(dim=-1)  # [B, C_g]
            # Numerical sanity check only: total_coeffs is algebraically guaranteed
            # to be exactly 0 (it's -sum_neg*sum_pos + sum_pos*sum_neg). Logging its
            # magnitude is a free correctness/stability tripwire - if it ever drifts
            # meaningfully off 0, something upstream (e.g. mixed precision) is wrong.
            info[f"total_coeffs_abs_{R}"] = total_coeffs.abs()
            total_force_R = total_force_R - total_coeffs.unsqueeze(-1) * old_gen_scaled
            f_norm_val = (total_force_R ** 2).mean()
            info[f"loss_{R}"] = f_norm_val

            force_scale = torch.sqrt(torch.clamp(f_norm_val, min=1e-8))
            info[f"force_scale_{R}"] = force_scale
            force_across_R = force_across_R + total_force_R / force_scale   # [B, C_g, S]

        # Raw (signed) drift term, summed across all temperatures - this is
        # exactly the vector added to old_gen_scaled to form the goal below.
        info["force_across_R_L2_norm"] = (force_across_R ** 2).mean(dim=(-1, -2))
        goal_scaled = old_gen_scaled + force_across_R

    # Loss with gradients through gen
    gen_scaled = gen / scale_inputs.detach()
    # gen_scaled is built from `gen` (requires grad) and old_gen_scaled from
    # `old_gen = gen.detach()` - same underlying values, so this is ~0 at the
    # moment drift_loss is called, by construction, regardless of training
    # progress: it isn't "the network's output changing", it's just gen vs.
    # its own detached copy before backward()/optimizer.step() touch anything.
    # Logged (detached, so it doesn't retain an extra autograd subgraph) to
    # make that explicit rather than inferred: diff = gen_scaled - goal_scaled
    # = (this, ~0) - force_across_R, i.e. loss ≈ mean(force_across_R**2) - the
    # step's`` loss value is essentialg the magnitude of this stely entirely the drift term itself.
    #info["gen_movement"] = ((gen_scaled.detach() - old_gen_scaled) ** 2).mean(dim=(-1, -2))
    
    # goal_scaled is a moving, self-referential target — 
    # recomputed fresh from gen.detach() every single call
    # the model is trained to chase a target derived from its own current state, 
    # and each individual regression step can succeed perfectly while the reported 
    # per-step loss stays roughly constant, because the target keeps regenerating itself 
    # from wherever the model currently is.


    diff = gen_scaled - goal_scaled.detach()
    loss = (diff ** 2).mean(dim=(-1, -2))

    info = {k: v.mean() for k, v in info.items()}

    return loss, info