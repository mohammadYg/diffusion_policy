"""Aggregate flatness (run-flat-eval.sh) and success rate of a list of runs.

    python aggregate_flatness.py RUN_LIST [OUT_ROOT] [TAG] [PREFIX]

Per run: flatness from <run>/<TAG>/flatness_log_*.json (newest file). Success rate, at the SAME
checkpoints as the flatness, from every <run>/eval{,_output}/eval_log_*.json (one column per log, labelled by
its eval_config, or by its timestamp when the log has none), or else from test/mean_score in the
training log (logs.json.txt), matched by epoch or global step. Every value is given as
    lastN = mean over the evaluated checkpoints,   final = last checkpoint.

Writes PREFIX.md (group table and per-run table), PREFIX.csv and PREFIX.json. A group is the run
name with its seed replaced by '*'; group rows are mean +- std over seeds of the per-run lastN.
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

RUN_LIST = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/proj1/eu-26-52/data/outputs"
TAG = sys.argv[3] if len(sys.argv) > 3 else "flat_eval"
PREFIX = sys.argv[4] if len(sys.argv) > 4 else "flatness_summary"

FLAT = ["loss", "grad_norm_rel", "sharp_rel_0.005", "sharp_rel_0.01", "sharp_rel_0.02", "trace_rel",
        "top_eig_rel", "sam_rel_0.05", "sam_rel_0.1", "sam_rel_0.2", "sam_curv_rel_0.05",
        "sam_curv_rel_0.1", "sam_curv_rel_0.2", "trace_raw", "top_eig_raw", "sam_raw_0.05",
        "noise/gap_learned", "noise/gap_initial", "noise/sigma_curv", "noise/awareness"]
SHOW = ["train/loss", "train/sharp_rel_0.01", "test/sharp_rel_0.01", "train/trace_rel",
        "train/top_eig_rel", "test/top_eig_rel", "train/sam_curv_rel_0.1", "train/sam_rel_0.1",
        "train/grad_norm_rel", "train/noise/awareness", "train/noise/gap_learned",
        "train/noise/gap_initial"]


def parse(run):
    model = "PAC-DP" if "pac-dp" in run else "DP"
    rho = re.search(r"_rho_(-?[\d.]+)_", run)
    kl = re.search(r"_kl_([\d.]+e[+-]?\d+|[\d.]+)_", run)
    return model, (float(rho.group(1)) if rho else None), (float(kl.group(1)) if kl else None)


def key_index(key):
    """('epoch', 1107) from model_at_epoch_1107, ('step', 10000) from model_at_step_010000."""
    m = re.match(r"model_at_(epoch|step)_(\d+)$", key)
    return (m.group(1), int(m.group(2))) if m else None


def success_sources(run_dir):
    """{label: {(kind, n): success_rate}}."""
    out = {}
    for f in sorted(glob.glob(f"{run_dir}/eval/eval_log_*.json") + glob.glob(f"{run_dir}/eval_output/eval_log_*.json"),
                    key=os.path.basename):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        cfg = d.get("eval_config")
        if cfg:
            label = ("stoch" if cfg.get("eval_stochastic") else "det") + (
                "[" + ",".join(o for o in cfg.get("overrides", []) if "n_envs" not in o and "n_test" not in o) + "]")
        else:
            label = "eval@" + os.path.basename(f)[len("eval_log_"):-len(".json")]
        sr = {}
        for k, v in d.items():
            idx = key_index(k)
            if idx and isinstance(v, dict) and "success_rate" in v:
                sr[idx] = v["success_rate"]
        if sr:
            out[label] = sr
    if not out and os.path.exists(f"{run_dir}/logs.json.txt"):
        sr = {}
        for line in open(f"{run_dir}/logs.json.txt"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            score = next((r[k] for k in ("test/mean_score", "test/mean_score_deterministic") if k in r), None)
            if score is not None and score == score:
                sr[("epoch", int(r.get("epoch", -1)))] = score
                sr[("step", int(r.get("global_step", -1)))] = score
        if sr:
            out["train_log"] = sr
    return out


def lookup(sr, idx):
    kind, n = idx
    for m in ((n,) if kind == "epoch" else (n, n - 1, n + 1)):
        if (kind, m) in sr:
            return sr[(kind, m)]
    return None


runs = [l.strip() for l in open(RUN_LIST) if l.strip()]
per_run = []
for run in runs:
    files = sorted(glob.glob(f"{OUT}/{run}/{TAG}/flatness_log_*.json"))
    if not files:
        print("no flatness log:", run, file=sys.stderr)
        continue
    d = json.load(open(files[-1]))
    keys = sorted((k for k in d if key_index(k)), key=key_index)
    if not keys:
        print("empty flatness log:", run, file=sys.stderr)
        continue
    model, rho, kl = parse(run)
    seed = re.search(r"seed_(\d+)", run)
    row = {"run": run, "group": re.sub(r"_seed_\d+", "_seed_*", run), "model": model, "rho": rho, "kl": kl,
           "seed": int(seed.group(1)) if seed else None, "n_ckpt": len(keys),
           "ckpts": f"{key_index(keys[0])[1]}-{key_index(keys[-1])[1]} ({key_index(keys[0])[0]})"}
    for split in ("train", "test"):
        for m in FLAT:
            vals = [d[k].get(split, {}).get(m) for k in keys]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"{split}/{m}"] = (float(np.mean(vals)), vals[-1])
    for label, sr in success_sources(f"{OUT}/{run}").items():
        vals = [lookup(sr, key_index(k)) for k in keys]
        got = [v for v in vals if v is not None]
        if got:
            row[f"success:{label}"] = (float(np.mean(got)), vals[-1] if vals[-1] is not None else float("nan"))
            row[f"success:{label}:n"] = len(got)
    per_run.append(row)

succ = sorted({k for r in per_run for k in r if k.startswith("success:") and not k.endswith(":n")})
# old eval logs are labelled by timestamp, which differs per run: show them as eval#1, eval#2 (by time)
for r in per_run:
    stamps = sorted(k for k in r if k.startswith("success:eval@") and not k.endswith(":n"))
    for i, k in enumerate(stamps):
        r[f"success:eval#{i + 1}"] = r[k]
        r[f"success:eval#{i + 1}:src"] = k.split("@", 1)[1]
succ = sorted({k for r in per_run for k in r if k.startswith("success:") and "@" not in k
               and not k.endswith(":n") and not k.endswith(":src")})
cols = succ + SHOW
order = lambda r: (r["model"] != "DP", r["rho"] or 0, r["kl"] or 0, r["group"], r["seed"] or 0)
per_run.sort(key=order)
groups = {}
for r in per_run:
    groups.setdefault(r["group"], []).append(r)


def ms(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "-"
    sd = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
    return f"{np.mean(vals):.3g} ± {sd:.2g}"


f3 = lambda x: "nan" if x != x else f"{x:.3g}"
md = ["# Flatness and success rate", "",
      "lastN = mean over the evaluated checkpoints of a run; final = its last checkpoint. "
      "A group is the run name with the seed replaced by `*`; group rows are mean ± std over seeds "
      "of the per-run lastN values. Success columns: `eval#1`, `eval#2` = the run's eval logs in "
      "time order (no eval_config recorded), `train_log` = test/mean_score of the training log.", "",
      "## Per group", "",
      "| group | model | rho | kl | seeds | " + " | ".join(c.replace("success:", "success ") for c in cols) + " |",
      "|" + "---|" * (len(cols) + 5)]
gjson = []
for g, rs in sorted(groups.items(), key=lambda kv: order(kv[1][0])):
    cells = [ms([r[c][0] if c in r else None for r in rs]) for c in cols]
    r0 = rs[0]
    seeds = ",".join(str(r["seed"]) for r in rs)
    md.append(f"| {g} | {r0['model']} | {r0['rho']} | {r0['kl']} | {len(rs)} ({seeds}) | " + " | ".join(cells) + " |")
    gjson.append({"group": g, "model": r0["model"], "rho": r0["rho"], "kl": r0["kl"],
                  "runs": [r["run"] for r in rs], **dict(zip(cols, cells))})
md += ["", "## Per run (lastN / final)", "",
       "| run | seed | ckpts | " + " | ".join(c.replace("success:", "success ") for c in cols) + " |",
       "|" + "---|" * (len(cols) + 3)]
for r in per_run:
    cells = [f"{f3(r[c][0])} / {f3(r[c][1])}" if c in r else "-" for c in cols]
    md.append(f"| {r['run']} | {r['seed']} | {r['n_ckpt']}: {r['ckpts']} | " + " | ".join(cells) + " |")
open(f"{PREFIX}.md", "w").write("\n".join(md) + "\n")

metric_keys = sorted({k for r in per_run for k in r if isinstance(r[k], tuple)})
base = ["run", "group", "model", "rho", "kl", "seed", "n_ckpt", "ckpts"]
with open(f"{PREFIX}.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(base + [f"{k}|lastN" for k in metric_keys] + [f"{k}|final" for k in metric_keys])
    for r in per_run:
        w.writerow([r[k] for k in base] + [r[k][0] if k in r else "" for k in metric_keys]
                   + [r[k][1] if k in r else "" for k in metric_keys])
json.dump({"groups": gjson, "runs": per_run}, open(f"{PREFIX}.json", "w"), indent=1)
print(f"{len(per_run)} runs, {len(groups)} groups -> {PREFIX}.md/.csv/.json")
