"""Monitor the learned posterior noise of PAC-DP (Bayesian) layers relative to the prior.

Works with any model whose probabilistic layers hold posterior/prior Gaussians named
``weight``/``weight_prior`` and ``bias``/``bias_prior``, each with ``.mu``, ``.rho`` and a
``.sigma`` property (see ``model/diffusion/bayes_conv1d_components.py``). Pass the trained
model (``self.model`` in the workspace), not the EMA copy: the optimiser updates rho there.

What is reported (keys are wandb-friendly):
  sigma/...       ratio r = sigma / sigma_prior per weight: quantiles, fraction shrunk (r < 0.5),
                  unchanged (|r - 1| < 1%) and above the prior (r > 1.01), per-layer medians,
                  and the split of KL(Q||P) into its sigma part and its mean part.
  sigma_pull/...  the two forces on rho in the last step: the data-term pull and the KL pull.
                  A positive rho-gradient lowers sigma under gradient descent. Also the
                  effective KL weight lambda_eff = (dT/dKL) / (dT/dR) of the PAC-Bayes objective.

Online use (in the training loop, only every few hundred / thousand steps):

    probe = SigmaPullProbe(self.model)
    ...
    raw_loss, emp_risk_train, kl_train = self.model.compute_bound(...)
    log_sigma = (self.global_step % sigma_every) == 0
    if log_sigma:
        probe.before_backward(raw_loss, loss_emp_bounded, kl_train)  # cheap: no network backward
    raw_loss.backward()
    if log_sigma:
        sigma_log = probe.after_backward()                          # reads the .grad of rho
        sigma_log.update(sigma_summary(self.model))
    ...                                                             # optimizer.step() etc.
    if log_sigma:
        step_log.update(sigma_log)

Offline use (from a saved checkpoint):

    python -m diffusion_policy.common.sigma_monitor path/to/ckpt [--sigma softplus|exp]

Checkpoints written before commit 71a1470 (2026-08-07) use sigma = softplus(rho); later ones
use sigma = exp(rho). The offline report must be told which one applies (default: softplus).
"""
import argparse
import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

_SHRUNK = 0.5          # r below this: noise shrunk to less than half the prior scale
_TOL = 0.01            # |r - 1| below this: noise unchanged (within optimiser noise)
_MAX_QUANTILE = 2_000_000


def _is_gaussian(x) -> bool:
    return all(hasattr(x, a) for a in ("mu", "rho", "sigma"))


def gaussian_pairs(model: torch.nn.Module) -> List[Tuple[str, object, object]]:
    """(name, posterior, prior) for every probabilistic weight and bias in the model."""
    pairs = []
    for name, module in model.named_modules():
        for kind in ("weight", "bias"):
            post = getattr(module, kind, None)
            prior = getattr(module, kind + "_prior", None)
            if post is not None and prior is not None and _is_gaussian(post) and _is_gaussian(prior):
                pairs.append((f"{name}.{kind}" if name else kind, post, prior))
    return pairs


def _summarise(items: Iterable[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
               prefix: str, per_layer: bool) -> Dict[str, float]:
    """items: (name, sigma, sigma_prior, mu, mu_prior) tensors, all on the same device."""
    out: Dict[str, float] = {}
    ratios, kl_sigma, kl_mean, n = [], 0.0, 0.0, 0
    for name, s, s_p, mu, mu_p in items:
        s, s_p, mu, mu_p = s.double(), s_p.double(), mu.double(), mu_p.double()
        r = (s / s_p).flatten()
        ratios.append(r)
        n += r.numel()
        # KL(Q||P) of a diagonal Gaussian = sigma part + mean part (eq. 5 of the review)
        kl_sigma += (torch.log(s_p / s) + s ** 2 / (2 * s_p ** 2) - 0.5).sum().item()
        kl_mean += ((mu - mu_p) ** 2 / (2 * s_p ** 2)).sum().item()
        if per_layer and name.endswith("weight"):
            out[f"{prefix}layer/{name}/median_ratio"] = r.median().item()
            out[f"{prefix}layer/{name}/median_sigma"] = s.median().item()
    if n == 0:
        return out
    r = torch.cat(ratios)
    sample = r if r.numel() <= _MAX_QUANTILE else r[torch.randint(r.numel(), (_MAX_QUANTILE,), device=r.device)]
    qs = torch.quantile(sample, torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], dtype=sample.dtype, device=sample.device))
    for q, v in zip(("q01", "q05", "median", "q95", "q99"), qs.tolist()):
        out[f"{prefix}ratio_{q}"] = v
    out[f"{prefix}frac_shrunk"] = (r < _SHRUNK).double().mean().item()
    out[f"{prefix}frac_unchanged"] = ((r - 1).abs() < _TOL).double().mean().item()
    out[f"{prefix}frac_above_prior"] = (r > 1 + _TOL).double().mean().item()
    out[f"{prefix}mean_log_ratio"] = torch.log(r).mean().item()
    out[f"{prefix}kl_sigma_part"] = kl_sigma
    out[f"{prefix}kl_mean_part"] = kl_mean
    out[f"{prefix}n_params"] = float(n)
    return out


@torch.no_grad()
def sigma_summary(model: torch.nn.Module, prefix: str = "sigma/", per_layer: bool = True) -> Dict[str, float]:
    """Cheap statistics of sigma / sigma_prior over all probabilistic parameters."""
    items = ((name, post.sigma, prior.sigma, post.mu, prior.mu) for name, post, prior in gaussian_pairs(model))
    return _summarise(items, prefix, per_layer)


@torch.no_grad()
def sigma_histograms(model: torch.nn.Module, prefix: str = "sigma_hist/", max_points: int = 100_000):
    """wandb histograms of log10(sigma / sigma_prior), one per weight tensor (call rarely)."""
    import wandb
    out, everything = {}, []
    for name, post, prior in gaussian_pairs(model):
        if not name.endswith("weight"):
            continue
        r = torch.log10(post.sigma / prior.sigma).flatten().float()
        if r.numel() > max_points:
            r = r[torch.randint(r.numel(), (max_points,), device=r.device)]
        everything.append(r)
        out[f"{prefix}{name}"] = wandb.Histogram(r.cpu().numpy())
    if everything:
        out[f"{prefix}all"] = wandb.Histogram(torch.cat(everything).cpu().numpy())
    return out


class SigmaPullProbe:
    """Splits the rho-gradient into the data-term pull and the KL pull without an extra
    backward pass through the network, and turns the data pull into a free estimate of the
    curvature averaged over the noisy weights.

    For the PAC-Bayes objective T(R, KL), with R the data term that actually enters the bound
    (``loss_emp_bounded`` from ``compute_bound``):
        dT/drho = (dT/dR) * dR/drho + (dT/dKL) * dKL/drho.
    The KL part only involves (mu, rho), so the data part is total gradient minus KL part.

    Curvature (review eq. 13, Price's theorem): E_Q[dR/dsigma_i] = sigma_i * Hbar_ii with
    Hbar_ii = E_Q[d2R/dw_i2], and dsigma/drho = sigmoid(rho) for sigma = softplus(rho). Hence
        Hbar_ii ~ (data pull on rho_i) / (dT/dR * sigma_i * sigmoid(rho_i)),
    unbiased but noisy for one step; it is averaged over each layer and over calls (EMA).
    At equilibrium (eq. 13) it equals the curvature implied by the learned noise,
        H_implied_ii = beta * (1/sigma_i^2 - 1/sigma_P,i^2),  beta = lambda_eff.
    measured/implied > 1: the data term still pushes sigma down; < 1: sigma is still growing.
    Needs stochastic=True training (with sigma fixed at mu the data term has no rho-gradient).
    """

    def __init__(self, model: torch.nn.Module, prefix: str = "sigma_pull/", ema: float = 0.9):
        self.model = model
        self.prefix = prefix
        self.ema = ema
        self._ctx = None
        self._hbar_ema: Dict[str, float] = {}

    def before_backward(self, loss: torch.Tensor, data_term: torch.Tensor, kl: torch.Tensor) -> None:
        pairs = gaussian_pairs(self.model)
        rhos = [post.rho for _, post, _ in pairs]
        if not kl.requires_grad or not data_term.requires_grad or not rhos:
            self._ctx = None
            return
        dT_dR, dT_dKL = torch.autograd.grad(loss, [data_term, kl], retain_graph=True)
        g_kl = torch.autograd.grad(kl, rhos, retain_graph=True, allow_unused=True)
        self._ctx = (pairs, dT_dR.detach(), dT_dKL.detach(), [None if g is None else g.detach() for g in g_kl])

    @torch.no_grad()
    def after_backward(self, per_layer: bool = True) -> Dict[str, float]:
        if self._ctx is None:
            return {}
        pairs, dT_dR, dT_dKL, g_kl = self._ctx
        self._ctx = None
        p = self.prefix
        beta = (dT_dKL / dT_dR).item()
        out = {f"{p}dT_dR": dT_dR.item(), f"{p}dT_dKL": dT_dKL.item(), f"{p}lambda_eff": beta}
        n = data_down = kl_up = 0
        sq_data = sq_kl = 0.0
        for (name, post, prior), gk in zip(pairs, g_kl):
            if post.rho.grad is None or gk is None:
                continue
            kl_pull = dT_dKL * gk                       # KL contribution to dT/drho
            data_pull = post.rho.grad - kl_pull         # data-term contribution
            n += data_pull.numel()
            data_down += (data_pull > 0).sum().item()   # data term lowers sigma here
            kl_up += (kl_pull < 0).sum().item()         # KL raises sigma here
            sq_data += data_pull.double().pow(2).sum().item()
            sq_kl += kl_pull.double().pow(2).sum().item()
            if per_layer and name.endswith("weight"):
                s, s_p = post.sigma.double(), prior.sigma.double()
                hbar = (data_pull.double() / (dT_dR * s * torch.sigmoid(post.rho.double()))).mean().item()
                prev = self._hbar_ema.get(name)
                self._hbar_ema[name] = hbar if prev is None else self.ema * prev + (1 - self.ema) * hbar
                implied = (beta * (1 / s ** 2 - 1 / s_p ** 2)).mean().item()
                out[f"{p}layer/{name}/hbar_step"] = hbar
                out[f"{p}layer/{name}/hbar_ema"] = self._hbar_ema[name]
                out[f"{p}layer/{name}/h_implied"] = implied
        if n:
            out.update({f"{p}frac_data_lowers_sigma": data_down / n,
                        f"{p}frac_kl_raises_sigma": kl_up / n,
                        f"{p}norm_data_pull": math.sqrt(sq_data),
                        f"{p}norm_kl_pull": math.sqrt(sq_kl),
                        f"{p}kl_to_data_pull_ratio": math.sqrt(sq_kl) / max(math.sqrt(sq_data), 1e-30)})
        return out


def _bayes_means(model):
    pairs = [(n, post) for n, post, _ in gaussian_pairs(model) if n.endswith("weight")]
    return [n for n, _ in pairs], [post.mu for _, post in pairs]


def hessian_probe(policy, batch, n_probes: int = 8, stochastic: bool = False, n_weight_samples: int = 1,
                  seed: int = 0, power_iters: int = 0, prefix: str = "hessian/") -> Dict[str, float]:
    """Direct curvature of the denoising loss w.r.t. the posterior means of the Bayesian
    weights, via Hessian-vector products (double backward). Call every ~10k steps on a FIXED
    held-out batch: the seed pins the diffusion noise, timesteps and weight samples, so
    successive measurements are comparable.

    stochastic=False: Hessian at the mean, H(mu) (what eq. 11 uses).
    stochastic=True : average over n_weight_samples posterior draws, Hbar = E_Q[H(w)] (eq. 13).
    Reports, per weight tensor, the mean Hessian diagonal (= trace / n, Hutchinson with
    Rademacher probes) and the global trace; optionally the top eigenvalue (power iteration).
    Note: this is the curvature of the raw loss_emp; the probe above measures the bounded
    data term (divide by loss_scale to compare when the clamp is not active).
    """
    names, mus = _bayes_means(policy)
    was_training = policy.training
    policy.eval()
    diag_sum = [torch.zeros_like(m) for m in mus]

    def hvp(vs):
        with torch.random.fork_rng(devices=[mus[0].device] if mus[0].is_cuda else []):
            torch.manual_seed(seed + hvp.k)
            loss = policy.compute_loss(batch, stochastic=stochastic, train=False)
        grads = torch.autograd.grad(loss, mus, create_graph=True)
        dot = sum((g * v).sum() for g, v in zip(grads, vs))
        return torch.autograd.grad(dot, mus)

    # probe vectors from their own seeded generator: repeated calls at the same weights give
    # identical numbers, so changes over training reflect the model, not the estimator.
    gen = torch.Generator(device=mus[0].device)
    gen.manual_seed(seed)
    total = n_probes * (n_weight_samples if stochastic else 1)
    for s_i in range(n_weight_samples if stochastic else 1):
        hvp.k = s_i
        for _ in range(n_probes):
            vs = [torch.randint(0, 2, m.shape, generator=gen, device=m.device, dtype=m.dtype) * 2 - 1
                  for m in mus]
            hv = hvp(vs)
            for d, v, h in zip(diag_sum, vs, hv):
                d += (v * h).detach()
    out, trace = {}, 0.0
    for name, d in zip(names, diag_sum):
        d = d / total
        trace += d.sum().item()
        out[f"{prefix}layer/{name}/mean_diag"] = d.mean().item()
    out[f"{prefix}trace"] = trace
    if power_iters:
        hvp.k = 0
        v = [torch.randn(m.shape, generator=gen, device=m.device, dtype=m.dtype) for m in mus]
        lam = 0.0
        for _ in range(power_iters):
            nrm = torch.sqrt(sum((x ** 2).sum() for x in v))
            v = [x / nrm for x in v]
            hv = hvp(v)
            lam = sum((a * b).sum() for a, b in zip(v, hv)).item()
            v = [h.detach() for h in hv]
        out[f"{prefix}top_eigenvalue"] = lam
    policy.train(was_training)
    return out


def checkpoint_summary(path: str, sigma: str = "softplus", which: str = "model") -> Dict[str, float]:
    """Same statistics as sigma_summary, read directly from a saved workspace checkpoint."""
    import dill
    payload = torch.load(open(path, "rb"), pickle_module=dill, map_location="cpu", weights_only=False)
    sd = payload["state_dicts"][which]
    to_sigma = F.softplus if sigma == "softplus" else torch.exp
    items = []
    for key in sd:
        if key.endswith(".rho") and "_prior." not in key:
            base = key[: -len(".rho")]
            prior = base + "_prior"
            if prior + ".rho" in sd:
                items.append((base, to_sigma(sd[base + ".rho"]), to_sigma(sd[prior + ".rho"]),
                              sd[base + ".mu"], sd[prior + ".mu"]))
    return _summarise(items, prefix="sigma/", per_layer=True)


def main():
    parser = argparse.ArgumentParser(description="Report sigma/sigma_prior of a PAC-DP checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--sigma", choices=["softplus", "exp"], default="softplus",
                        help="parameterisation used when the checkpoint was trained")
    parser.add_argument("--which", default="model", help="state dict to read: model or ema_model")
    parser.add_argument("--layers", type=int, default=6, help="number of most-shrunk layers to list")
    args = parser.parse_args()
    stats = checkpoint_summary(args.checkpoint, sigma=args.sigma, which=args.which)
    layers = sorted(((k, v) for k, v in stats.items() if "/layer/" in k), key=lambda kv: kv[1])
    for k, v in stats.items():
        if "/layer/" not in k:
            print(f"{k:28s} {v:.6g}")
    print(f"\nmost-shrunk layers (median sigma/sigma_prior):")
    for k, v in layers[: args.layers]:
        print(f"  {k.split('/layer/')[1]:60s} {v:.3f}")


if __name__ == "__main__":
    main()
