"""Per-layer view of trained DP / PAC-DP runs: which Bayesian layers use their noise.

    python analyze_layers.py RUN_LIST [OUT_ROOT] [OUT_JSON] [--sigma softplus|exp]

For every run in RUN_LIST (one name per line, relative to OUT_ROOT):
  * from its last checkpoint (state dict "model", where rho is trained), per Bayesian tensor:
    number of weights, quantiles of r = sigma / sigma_prior, fraction shrunk (r < 0.5),
    fraction unchanged (|r - 1| < 1%), median |mu - mu_prior| / sigma_prior, and the KL split
    KL = sum[ log(sP/s) + s^2/(2 sP^2) - 1/2 ]  (sigma part)  +  sum[ (mu - muP)^2 / (2 sP^2) ]  (mean part);
  * from its newest flat_eval/flatness_log_*.json, the per-block metrics (block/*/trace_rel and
    noise/block/*/gap_learned), averaged over the evaluated checkpoints.
Writes one JSON with everything; summarise it with any notebook or the review's tables.
"""
import glob
import json
import re
import sys

import dill
import torch
import torch.nn.functional as F

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SIGMA = "exp" if "--sigma" in sys.argv and sys.argv[sys.argv.index("--sigma") + 1] == "exp" else "softplus"
RUN_LIST = args[0]
OUT = args[1] if len(args) > 1 else "/mnt/proj1/eu-26-52/data/outputs"
OUT_JSON = args[2] if len(args) > 2 else "layer_analysis.json"
to_sigma = F.softplus if SIGMA == "softplus" else torch.exp


def ckpt_index(path):
    m = re.search(r"(epoch|step)=(\d+)", path)
    return int(m.group(2)) if m else -1


@torch.no_grad()
def tensor_stats(mu, rho, mu_p, rho_p):
    s, sp = to_sigma(rho.double()), to_sigma(rho_p.double())
    r = (s / sp).flatten()
    q = torch.quantile(r[torch.randint(r.numel(), (min(r.numel(), 2_000_000),))] if r.numel() > 2_000_000 else r,
                       torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=r.dtype))
    shift = ((mu.double() - mu_p.double()).abs() / sp).flatten()
    kl_sigma = (torch.log(sp / s) + s ** 2 / (2 * sp ** 2) - 0.5).sum().item()
    kl_mean = ((mu.double() - mu_p.double()) ** 2 / (2 * sp ** 2)).sum().item()
    return {"n": r.numel(), "r_q05": q[0].item(), "r_q25": q[1].item(), "r_median": q[2].item(),
            "r_q75": q[3].item(), "r_q95": q[4].item(), "frac_shrunk": (r < 0.5).double().mean().item(),
            "frac_unchanged": ((r - 1).abs() < 0.01).double().mean().item(),
            "frac_above": (r > 1.01).double().mean().item(),
            "shift_median": shift.median().item(), "kl_sigma": kl_sigma, "kl_mean": kl_mean,
            "sigma_prior_median": sp.flatten().median().item()}


def flat_blocks(run_dir):
    files = sorted(glob.glob(f"{run_dir}/flat_eval/flatness_log_*.json"))
    if not files:
        return {}
    d = json.load(open(files[-1]))
    keys = [k for k in d if k.startswith("model_at_")]
    acc = {}
    for k in keys:
        for name, v in d[k].get("train", {}).items():
            if "block/" in name:
                acc.setdefault(name, []).append(v)
    return {name: sum(v) / len(v) for name, v in acc.items()}


result = {}
for run in [l.strip() for l in open(RUN_LIST) if l.strip()]:
    run_dir = f"{OUT}/{run}"
    ckpts = sorted((p for p in glob.glob(f"{run_dir}/checkpoints/*.ckpt") if "latest" not in p), key=ckpt_index)
    entry = {"blocks": flat_blocks(run_dir), "layers": {}}
    if ckpts:
        entry["checkpoint"] = ckpts[-1].split("/")[-1]
        payload = torch.load(open(ckpts[-1], "rb"), pickle_module=dill, map_location="cpu", weights_only=False)
        sd = payload["state_dicts"]["model"]
        for key in sd:
            if key.endswith(".rho") and "_prior." not in key:
                base = key[: -len(".rho")]
                if base + "_prior.rho" in sd:
                    entry["layers"][base] = tensor_stats(sd[base + ".mu"], sd[key], sd[base + "_prior.mu"],
                                                         sd[base + "_prior.rho"])
        del payload, sd
    result[run] = entry
    print(f"{run}: {len(entry['layers'])} Bayesian tensors, {len(entry['blocks'])} block metrics", flush=True)

json.dump({"sigma_param": SIGMA, "runs": result}, open(OUT_JSON, "w"))
print("written", OUT_JSON)
