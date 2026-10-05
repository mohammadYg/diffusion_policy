"""Flatness of the denoising loss for standard DP and PAC-DP, measured the same way for both.

Call ``flatness_report`` every 1e4 steps in BOTH training runs, on the same fixed train
batches and the same fixed test batches (``make_fixed_batches``). For PAC-DP also call
``noise_contribution_report`` to see how the learned sigma_i interact with the curvature.

What is measured is always the DEPLOYED network: DP's weights theta, and PAC-DP's posterior
means mu (plus its deterministic conditioning layers). Rho and the prior are never perturbed
in ``flatness_report``. The loss is the raw noise-prediction MSE (not the PAC bound, not the
bounded data term), with the diffusion noise, timesteps and probe vectors pinned by seeds, so
the two runs and successive steps see exactly the same function.

Scale-invariant ("relative") metrics are the ones to compare between DP and PAC-DP. The U-Net's
conv layers are followed by GroupNorm, so scaling their weights by c leaves the network unchanged
but scales their raw Hessian by 1/c^2; PAC-DP's KL keeps weights near the prior while DP's can
grow, so raw curvature mixes flatness with weight size. Relative metrics perturb each weight in
proportion to its own size, which removes that effect. Raw metrics are reported for reference.

Metrics (prefix e.g. ``flat/train/`` or ``flat/test/``):
  loss                          L(theta)
  sharp_rel_{a}                 E_xi[L(theta + a*|theta|*xi)] - L(theta), xi ~ N(0, I), antithetic
                                (expected relative sharpness, forward passes only; ~ a^2/2 * trace_rel)
  trace_rel                     sum_i theta_i^2 H_ii           (Hutchinson)
  top_eig_rel                   largest eigenvalue of D H D, D = diag|theta| (power iteration)
  trace_raw, top_eig_raw        the same without the |theta| scaling (reference only)
  block/<name>/trace_rel        trace_rel restricted to one block of the U-Net
PAC-DP only (``noise_contribution_report``, prefix ``noise/``):
  gap_learned                   E_xi[L(mu + sigma*xi)] - L(mu): what the learned noise costs
  gap_initial                   the same with the noise sigma had at the start of training
  sigma_curv                    sum_i sigma_i^2 H_ii (second-order prediction of 2 * gap_learned)
  awareness                     (sum sigma^2 H / sum sigma^2) / (sum H / n): < 1 means the noise
                                sits in flatter-than-average directions (curvature-aware)
  block/<name>/gap_learned      gap when only that block's weights are noisy
"""
import contextlib
import inspect
from typing import Dict, List

import torch

_SHARP_RADII = (5e-3, 1e-2, 2e-2)


# ----------------------------------------------------------------------------- helpers
def deployed_params(policy):
    """(name, tensor) of the network as deployed: DP's weights; PAC-DP's posterior means and
    deterministic layers (rho and prior parameters excluded). Selected by name, not by
    requires_grad, because the EMA copy has requires_grad switched off everywhere."""
    return [(n, p) for n, p in policy.model.named_parameters()
            if not n.endswith(".rho") and "_prior." not in n]


@contextlib.contextmanager
def _grads_on(params):
    """Temporarily enable requires_grad (needed for Hessian-vector products on the EMA copy)."""
    old = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)
    try:
        yield
    finally:
        for p, r in zip(params, old):
            p.requires_grad_(r)


def _block(name: str) -> str:
    parts = name.replace(".weight.mu", ".weight").replace(".bias.mu", ".bias").split(".")
    return ".".join(parts[:2]) if parts[0] in ("down_modules", "up_modules", "mid_modules") else parts[0]


def make_fixed_batches(dataset, n_batches: int = 2, batch_size: int = 64, seed: int = 0, device="cpu") -> List[dict]:
    """Fixed batches drawn once by index from a dataset. Use the SAME dataset object (the full
    training dataset, and the validation dataset) and seed in the DP and the PAC-DP runs."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[: n_batches * batch_size].tolist()
    batches = []
    for b in range(n_batches):
        items = [dataset[i] for i in idx[b * batch_size:(b + 1) * batch_size]]
        batches.append({k: torch.stack([torch.as_tensor(it[k]) for it in items]).to(device) for k in items[0]})
    return batches


def _loss(policy, batch, seed: int):
    """Raw denoising loss of the deployed (mean) network with pinned noise and timesteps."""
    dev = next(policy.parameters()).device
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(seed)
        if "stochastic" in inspect.signature(policy.compute_loss).parameters:
            return policy.compute_loss(batch, stochastic=False, train=False)
        return policy.compute_loss(batch, train=False)


@torch.no_grad()
def _perturbed_loss(policy, batches, params, scales, seed: int, draw: int) -> float:
    """Mean loss over batches after adding scale * xi (xi ~ N(0,I)) to params, then restoring.
    The loss uses the same pinned noise/timesteps (``seed``) as the unperturbed loss, so the
    difference measures the weight perturbation only; ``draw`` selects the xi sample."""
    gen = torch.Generator(device=params[0].device).manual_seed(10_000 + draw)
    deltas = [s * torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) for p, s in zip(params, scales)]
    # antithetic pair (+xi, -xi): the first-order term g.delta cancels exactly, so what is left
    # is the curvature part, 0.5 * delta^T H delta + O(delta^4), even with few draws
    vals = []
    for sign in (1.0, -1.0):
        for p, d in zip(params, deltas):
            p.add_(sign * d)
        try:
            vals.append(sum(_loss(policy, b, seed + i).item() for i, b in enumerate(batches)) / len(batches))
        finally:
            for p, d in zip(params, deltas):
                p.sub_(sign * d)
    return 0.5 * (vals[0] + vals[1])


def _hvp(policy, batches, params, vecs, seed: int):
    """Hessian-vector product of the mean loss over batches."""
    total = [torch.zeros_like(p) for p in params]
    for i, b in enumerate(batches):
        loss = _loss(policy, b, seed + i)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g * v).sum() for g, v in zip(grads, vecs))
        hv = torch.autograd.grad(dot, params)
        for t, h in zip(total, hv):
            t += h.detach() / len(batches)
    return total


def _hutchinson(policy, batches, params, scales, n_probes, seed, gen):
    """Per-parameter estimates of scale_i^2 * H_ii (Rademacher probes u = scale * v)."""
    est = [torch.zeros_like(p) for p in params]
    for _ in range(n_probes):
        vs = [torch.randint(0, 2, p.shape, generator=gen, device=p.device, dtype=p.dtype) * 2 - 1 for p in params]
        us = [s * v for s, v in zip(scales, vs)]
        hu = _hvp(policy, batches, params, us, seed)
        for e, u, h in zip(est, us, hu):
            e += (u * h).detach() / n_probes
    return est


def _top_eig(policy, batches, params, scales, iters, seed, gen) -> float:
    """Largest-magnitude eigenvalue of D H D with D = diag(scales), by power iteration."""
    v = [torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) for p in params]
    lam = 0.0
    for _ in range(iters):
        nrm = torch.sqrt(sum((x ** 2).sum() for x in v))
        v = [x / nrm for x in v]
        hv = _hvp(policy, batches, params, [s * x for s, x in zip(scales, v)], seed)
        dhdv = [s * h for s, h in zip(scales, hv)]
        lam = sum((a * b).sum() for a, b in zip(v, dhdv)).item()
        v = dhdv
    return lam


# ----------------------------------------------------------------------------- reports
def flatness_report(policy, batches, prefix: str = "flat/", seed: int = 0, n_probes: int = 32,
                    power_iters: int = 10, n_sharp: int = 8, radii=_SHARP_RADII, raw: bool = True) -> Dict[str, float]:
    """Identical flatness metrics for DP and PAC-DP (see module docstring)."""
    was_training = policy.training
    policy.eval()
    named = deployed_params(policy)
    names, params = [n for n, _ in named], [p for _, p in named]
    rel = [p.detach().abs() for p in params]
    one = [torch.ones_like(p) for p in params]
    gen = torch.Generator(device=params[0].device).manual_seed(seed)
    out = {}
    with torch.no_grad():
        out[f"{prefix}loss"] = sum(_loss(policy, b, seed + i).item() for i, b in enumerate(batches)) / len(batches)
    for a in radii:
        vals = [_perturbed_loss(policy, batches, params, [a * r for r in rel], seed, draw=k) for k in range(n_sharp)]
        out[f"{prefix}sharp_rel_{a:g}"] = sum(vals) / n_sharp - out[f"{prefix}loss"]
    grad_ctx = _grads_on(params)
    grad_ctx.__enter__()
    diag_rel = _hutchinson(policy, batches, params, rel, n_probes, seed, gen)
    out[f"{prefix}trace_rel"] = sum(d.sum().item() for d in diag_rel)
    blocks: Dict[str, float] = {}
    for n, d in zip(names, diag_rel):
        blocks[_block(n)] = blocks.get(_block(n), 0.0) + d.sum().item()
    for b, v in blocks.items():
        out[f"{prefix}block/{b}/trace_rel"] = v
    if power_iters:
        out[f"{prefix}top_eig_rel"] = _top_eig(policy, batches, params, rel, power_iters, seed, gen)
    if raw:
        out[f"{prefix}trace_raw"] = sum(d.sum().item() for d in _hutchinson(policy, batches, params, one, n_probes, seed, gen))
        if power_iters:
            out[f"{prefix}top_eig_raw"] = _top_eig(policy, batches, params, one, power_iters, seed, gen)
    grad_ctx.__exit__(None, None, None)
    policy.train(was_training)
    return out


class NoiseContribution:
    """PAC-DP only: how the learned posterior noise sigma_i interacts with the curvature.
    Create it once from the model at the start of training (or from a freshly built, untrained
    model with the same config): it stores that model's sigma on the CPU, by layer name, as the
    "initial" noise. ``report`` can then be called on any model with the same architecture."""

    def __init__(self, policy, use_prior: bool = False, sigma_fn=None):
        """use_prior=True: take the "initial" noise from the PRIOR Gaussians of ``policy`` - right
        for a data-dependent prior, where posterior training starts at the prior (Q = P).
        sigma_fn(gaussian) -> std: how a Gaussian's std follows from rho (default: its own
        .sigma); pass e.g. ``lambda g: torch.exp(g.rho)`` for checkpoints trained when the code
        used sigma = exp(rho). It is used for both the initial and the learned noise."""
        self.sigma_fn = sigma_fn or (lambda g: g.sigma)
        self.sigma0 = {}
        for n, m in self._pairs(policy):
            for kind in ("weight", "bias"):
                g = getattr(m, kind + "_prior" if use_prior else kind, None)
                if g is not None and hasattr(g, "sigma"):
                    self.sigma0[f"{n}.{kind}"] = self.sigma_fn(g).detach().cpu()

    @staticmethod
    def _pairs(policy):
        return [(n, m) for n, m in policy.model.named_modules()
                if hasattr(m, "weight") and hasattr(m, "weight_prior") and hasattr(m.weight, "sigma")]

    def available(self, policy) -> bool:
        return len(self._pairs(policy)) > 0

    def _tensors(self, policy, use_initial: bool):
        params, scales, names = [], [], []
        for n, m in self._pairs(policy):
            for kind in ("weight", "bias"):
                g = getattr(m, kind, None)
                if g is None or not hasattr(g, "sigma"):
                    continue
                name = f"{n}.{kind}"
                s = self.sigma0[name].to(g.mu.device) if use_initial else self.sigma_fn(g).detach()
                params.append(g.mu)
                scales.append(s)
                names.append(name)
        return names, params, scales

    def report(self, policy, batches, prefix: str = "noise/", seed: int = 0, n_draws: int = 8,
               n_probes: int = 32, per_block: bool = True) -> Dict[str, float]:
        was_training = policy.training
        policy.eval()
        out = {}
        with torch.no_grad():
            base = sum(_loss(policy, b, seed + i).item() for i, b in enumerate(batches)) / len(batches)
        names, params, sig = self._tensors(policy, use_initial=False)
        _, _, sig0 = self._tensors(policy, use_initial=True)
        gap = lambda scales: sum(_perturbed_loss(policy, batches, params, scales, seed, draw=k)
                                 for k in range(n_draws)) / n_draws - base
        out[f"{prefix}gap_learned"] = gap(sig)
        out[f"{prefix}gap_initial"] = gap(sig0)
        gen = torch.Generator(device=params[0].device).manual_seed(seed)
        with _grads_on(params):
            sh = _hutchinson(policy, batches, params, sig, n_probes, seed, gen)      # sigma_i^2 H_ii
            h = _hutchinson(policy, batches, params, [torch.ones_like(s) for s in sig], n_probes, seed, gen)  # H_ii
        sum_sh = sum(x.sum().item() for x in sh)
        sum_s2 = sum((s ** 2).sum().item() for s in sig)
        mean_h = sum(x.sum().item() for x in h) / sum(x.numel() for x in h)
        out[f"{prefix}sigma_curv"] = sum_sh
        out[f"{prefix}awareness"] = (sum_sh / sum_s2) / mean_h if mean_h != 0 else float("nan")
        if per_block:
            groups: Dict[str, List[int]] = {}
            for i, n in enumerate(names):
                groups.setdefault(_block(n), []).append(i)
            for b, ids in groups.items():
                only = [s if i in ids else torch.zeros_like(s) for i, s in enumerate(sig)]
                out[f"{prefix}block/{b}/gap_learned"] = gap(only)
        policy.train(was_training)
        return out
