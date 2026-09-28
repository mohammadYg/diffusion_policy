from typing import Dict, Callable, List
import collections
import torch
import torch.nn as nn

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def pad_remaining_dims(x, target):
    assert x.shape == target.shape[:len(x.shape)]
    return x.reshape(x.shape + (1,)*(len(target.shape) - len(x.shape)))

def dict_apply_split(
        x: Dict[str, torch.Tensor], 
        split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results

def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


def action_sample_diversity(samples: torch.Tensor) -> float:
    """Mean pairwise distance among K repeated action samples per
    observation - the deployed-time single-sample stochasticity a real
    rollout would actually see (one predict_action() call per control
    step), as opposed to a generative policy's own training-time
    candidate diversity (e.g. Drift Policy's `diversity` diagnostic,
    measured among its G training-time candidates, not at deployment).

    samples: [B, K, Ta, Da] - K independent samples (e.g. from K repeated
        predict_action() calls with fresh noise on the same observation)
        for each of B observations.
    Returns a single scalar, averaged over B and over all K*(K-1)
    off-diagonal pairs.
    """
    K = samples.shape[1]
    assert K >= 2, f"action_sample_diversity needs at least 2 samples to form a pair, got K={K}"
    diffs = samples.unsqueeze(2) - samples.unsqueeze(1)    # [B, K, K, Ta, Da]
    dists = diffs.pow(2).sum(dim=(-1, -2)).sqrt()          # [B, K, K]
    off_diag = ~torch.eye(K, dtype=torch.bool, device=dists.device)
    return dists[:, off_diag].mean().item()


def action_reconstruction_loss(samples: torch.Tensor, reference: torch.Tensor) -> float:
    """Mean squared reconstruction error between K independent action
    samples and the single reference (ground-truth demonstrated) action,
    computed per-sample via vmap (matching drift_unet_lowdim_policy.py's
    own vmap convention for its per-timestep loss) and averaged over the
    K samples. The deployed-time counterpart to drift_loss's own
    `mean_dist_to_pos` training diagnostic, which measures this among the
    G training-time candidates instead of K deployed predict_action()
    samples - a complement to action_sample_diversity() above: this says
    how close the deployed samples are to the real thing, diversity says
    how spread out they are from each other.

    samples: [B, K, Ta, Da] - K independent action samples per observation.
    reference: [B, Ta, Da] - the one demonstrated ("reference") action per
        observation, in the same (unnormalized) units as `samples`.
    Returns a single scalar, averaged over the K samples and the batch.
    """
    assert samples.shape[0] == reference.shape[0] and samples.shape[2:] == reference.shape[1:], (
        f"action_reconstruction_loss: samples {tuple(samples.shape)} and "
        f"reference {tuple(reference.shape)} must agree on batch size (dim 0) "
        "and Ta/Da (samples' dims 2: vs. reference's dims 1:)"
    )
    def _mse(sample, ref):
        return (sample - ref).pow(2).mean()
    per_sample = torch.vmap(_mse, in_dims=(1, None))(samples, reference)  # [K]
    return per_sample.mean().item()
