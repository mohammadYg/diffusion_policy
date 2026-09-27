"""
Generates all figures for report.tex from the dp/flow/drift comparison results
collected in this project's experiment log. Run once from the `report/` dir:
    python make_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- validated categorical palette (dataviz skill, first 3 slots, light mode) ----
COLOR = {"dp": "#2a78d6", "flow": "#eb6834", "drift": "#1baf7a"}
LABEL = {"dp": "Diffusion Policy", "flow": "Flow Policy", "drift": "Drift Policy"}
METHODS = ["dp", "flow", "drift"]

TASKS = ["pusht_lowdim", "lift_lowdim_abs", "can_lowdim_abs",
         "square_lowdim_abs", "tool_hang_lowdim_abs", "transport_lowdim_abs"]
TASK_LABEL = {
    "pusht_lowdim": "PushT",
    "lift_lowdim_abs": "Lift",
    "can_lowdim_abs": "Can",
    "square_lowdim_abs": "Square",
    "tool_hang_lowdim_abs": "Tool Hang",
    "transport_lowdim_abs": "Transport",
}

# mean, std triples, from the 3-seed aggregation in this experiment's logs
SUCCESS = {
    "pusht_lowdim":         {"dp": (0.896, 0.007), "flow": (0.922, 0.010), "drift": (0.918, 0.003)},
    "lift_lowdim_abs":      {"dp": (0.999, 0.001), "flow": (1.000, 0.000), "drift": (1.000, 0.000)},
    "can_lowdim_abs":       {"dp": (0.990, 0.002), "flow": (0.990, 0.002), "drift": (0.992, 0.001)},
    "square_lowdim_abs":    {"dp": (0.935, 0.004), "flow": (0.930, 0.008), "drift": (0.941, 0.004)},
    "tool_hang_lowdim_abs": {"dp": (0.628, 0.004), "flow": (0.540, 0.031), "drift": (0.693, 0.017)},
    "transport_lowdim_abs": {"dp": (0.788, 0.012), "flow": (0.789, 0.020), "drift": (0.771, 0.016)},
}
TRAIN_H = {
    "pusht_lowdim":         {"dp": (2.43, 0.03), "flow": (2.56, 0.02), "drift": (7.31, 0.02)},
    "lift_lowdim_abs":      {"dp": (2.54, 0.03), "flow": (2.80, 0.05), "drift": (7.43, 0.02)},
    "can_lowdim_abs":       {"dp": (2.33, 0.01), "flow": (2.58, 0.05), "drift": (7.30, 0.03)},
    "square_lowdim_abs":    {"dp": (2.32, 0.01), "flow": (2.55, 0.03), "drift": (7.30, 0.01)},
    "tool_hang_lowdim_abs": {"dp": (2.28, 0.01), "flow": (2.51, 0.01), "drift": (7.24, 0.03)},
    "transport_lowdim_abs": {"dp": (2.29, 0.01), "flow": (2.47, 0.02), "drift": (7.25, 0.10)},
}
EVAL_MIN = {
    "pusht_lowdim":         {"dp": (7.73, 0.26),   "flow": (8.82, 0.12),   "drift": (4.73, 0.11)},
    "lift_lowdim_abs":      {"dp": (35.25, 2.03),  "flow": (31.01, 2.24),  "drift": (27.45, 0.65)},
    "can_lowdim_abs":       {"dp": (41.21, 0.70),  "flow": (35.75, 2.59),  "drift": (33.85, 1.51)},
    "square_lowdim_abs":    {"dp": (42.04, 0.87),  "flow": (37.28, 1.84),  "drift": (39.97, 3.34)},
    "tool_hang_lowdim_abs": {"dp": (104.68, 1.30), "flow": (92.31, 2.64),  "drift": (89.53, 6.88)},
    "transport_lowdim_abs": {"dp": (136.17, 8.91), "flow": (125.46, 2.22), "drift": (139.48, 10.66)},
}
# isolated predict_action() latency, ms, batch=96 (GPU), from the microbenchmark
LATENCY_MS = {
    "pusht_lowdim":         {"dp": (64.91, 0.09), "flow": (62.33, 0.08), "drift": (6.31, 0.01)},
    "lift_lowdim_abs":      {"dp": (66.31, 0.41), "flow": (62.44, 0.08), "drift": (6.30, 0.01)},
    "can_lowdim_abs":       {"dp": (65.46, 0.08), "flow": (62.41, 0.07), "drift": (6.33, 0.01)},
    "square_lowdim_abs":    {"dp": (65.45, 0.07), "flow": (62.60, 0.07), "drift": (6.32, 0.01)},
    "tool_hang_lowdim_abs": {"dp": (65.53, 0.07), "flow": (62.50, 0.07), "drift": (6.32, 0.01)},
    "transport_lowdim_abs": {"dp": (65.90, 0.08), "flow": (62.82, 0.06), "drift": (6.34, 0.01)},
}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.7,
    "axes.edgecolor": "#8a8a86",
})


def grouped_bar(data, ylabel, title, out_path, pct=False, logy=False):
    x = np.arange(len(TASKS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, m in enumerate(METHODS):
        means = [data[t][m][0] for t in TASKS]
        stds = [data[t][m][1] for t in TASKS]
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2.5,
               color=COLOR[m], label=LABEL[m], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if pct:
        ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def latency_bar(out_path):
    means = [np.mean([LATENCY_MS[t][m][0] for t in TASKS]) for m in METHODS]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    bars = ax.bar(METHODS, means, color=[COLOR[m] for m in METHODS],
                   edgecolor="white", linewidth=0.6, width=0.55)
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=10, ha="right")
    ax.set_ylabel("predict_action() latency (ms)")
    ax.set_title("Isolated inference latency\n(batch = 96, averaged over 6 tasks)")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f} ms",
                ha="center", va="bottom", fontsize=9, color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def tradeoff_scatter(out_path):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    markers = {"dp": "o", "flow": "s", "drift": "^"}
    for m in METHODS:
        xs = [EVAL_MIN[t][m][0] for t in TASKS]
        ys = [SUCCESS[t][m][0] for t in TASKS]
        ax.scatter(xs, ys, s=60, color=COLOR[m], label=LABEL[m], marker=markers[m],
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Full-rollout evaluation time per run (min, log scale)")
    ax.set_ylabel("Mean success rate")
    ax.set_title("Success rate vs. evaluation wall-clock cost")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



# --- Appendix C: full 3-seed confirmatory sweep (Tool Hang, updates=2.5e5,
# no training rollout, matching the main comparison's protocol exactly) and
# the drift_loss diagnostic metrics explaining its non-monotonic shape.
# G=5/False, G=6/True, G=7/True are corrected: their raw eval logs default
# to a literal 0.0 for a checkpoint where a MuJoCo NaN/Inf event hit before
# any checkpoint in that run had yet succeeded (nothing to carry forward
# from), so we back-fill from that seed's next successful checkpoint
# instead of keeping the misleading 0.0 in the per-seed mean. ---
DRIFT_3SEED_SR = {
    2:  {"True": (0.5817, 0.0165), "False": (0.3876, 0.0301)},
    3:  {"True": (0.6934, 0.0174), "False": (0.5507, 0.0187)},
    4:  {"True": (0.6936, 0.0235), "False": (0.6046, 0.0313)},
    5:  {"True": (0.6554, 0.0071), "False": (0.6217, 0.0375)},
    6:  {"True": (0.5969, 0.0239), "False": (0.6114, 0.0303)},
    7:  {"True": (0.6007, 0.0073), "False": (0.6089, 0.0218)},
    8:  {"True": (0.5349, 0.0163), "False": (0.5960, 0.0096)},
    12: {"True": (0.4885, 0.0245), "False": (0.3298, 0.0344)},
    16: {"True": (0.4817, 0.0237), "False": (0.3403, 0.0276)},
}
# tail-averaged (last 10,000 logged steps), 3-seed mean, entropy at each R
DRIFT_ENTROPY_002 = {
    2: {"True": 0.140, "False": 0.159}, 3: {"True": 0.247, "False": 0.283},
    4: {"True": 0.264, "False": 0.319}, 5: {"True": 0.270, "False": 0.340},
    6: {"True": 0.274, "False": 0.359}, 7: {"True": 0.277, "False": 0.360},
    8: {"True": 0.279, "False": 0.345}, 12: {"True": 0.290, "False": 0.384},
    16: {"True": 0.297, "False": 0.459},
}
DRIFT_ENTROPY_005 = {
    2: {"True": 0.169, "False": 0.274}, 3: {"True": 0.263, "False": 0.441},
    4: {"True": 0.280, "False": 0.511}, 5: {"True": 0.296, "False": 0.552},
    6: {"True": 0.311, "False": 0.583}, 7: {"True": 0.323, "False": 0.588},
    8: {"True": 0.336, "False": 0.566}, 12: {"True": 0.381, "False": 0.496},
    16: {"True": 0.401, "False": 0.481},
}
DRIFT_ENTROPY_02 = {
    2: {"True": 0.4009, "False": 0.4917}, 3: {"True": 0.5215, "False": 0.6746},
    4: {"True": 0.5722, "False": 0.7494}, 5: {"True": 0.6099, "False": 0.7939},
    6: {"True": 0.6397, "False": 0.8162}, 7: {"True": 0.6617, "False": 0.8390},
    8: {"True": 0.6830, "False": 0.8568}, 12: {"True": 0.7378, "False": 0.8745},
    16: {"True": 0.7651, "False": 0.8575},
}
# tail-averaged (last 10,000 logged steps), 3-seed mean, top1 (peak softmax
# probability mass) at each R
DRIFT_TOP1_002 = {
    2: {"True": 0.930, "False": 0.927}, 3: {"True": 0.846, "False": 0.836},
    4: {"True": 0.823, "False": 0.794}, 5: {"True": 0.812, "False": 0.764},
    6: {"True": 0.803, "False": 0.736}, 7: {"True": 0.794, "False": 0.725},
    8: {"True": 0.788, "False": 0.732}, 12: {"True": 0.761, "False": 0.611},
    16: {"True": 0.743, "False": 0.486},
}
DRIFT_TOP1_005 = {
    2: {"True": 0.924, "False": 0.866}, 3: {"True": 0.849, "False": 0.731},
    4: {"True": 0.822, "False": 0.654}, 5: {"True": 0.797, "False": 0.600},
    6: {"True": 0.773, "False": 0.555}, 7: {"True": 0.752, "False": 0.536},
    8: {"True": 0.731, "False": 0.541}, 12: {"True": 0.656, "False": 0.520},
    16: {"True": 0.612, "False": 0.468},
}
DRIFT_TOP1_02 = {
    2: {"True": 0.790, "False": 0.711}, 3: {"True": 0.676, "False": 0.526},
    4: {"True": 0.607, "False": 0.429}, 5: {"True": 0.549, "False": 0.364},
    6: {"True": 0.500, "False": 0.323}, 7: {"True": 0.460, "False": 0.289},
    8: {"True": 0.423, "False": 0.264}, 12: {"True": 0.319, "False": 0.211},
    16: {"True": 0.263, "False": 0.201},
}
DRIFT_DIVERSITY = {
    2: {"True": 0.04509, "False": 0.46223}, 3: {"True": 0.03027, "False": 0.35952},
    4: {"True": 0.02691, "False": 0.33603}, 5: {"True": 0.02409, "False": 0.30682},
    6: {"True": 0.02271, "False": 0.31711}, 7: {"True": 0.02211, "False": 0.27070},
    8: {"True": 0.02117, "False": 0.21331}, 12: {"True": 0.01988, "False": 0.00419},
    16: {"True": 0.01971, "False": 0.00356},
}
DRIFT_GRADNORM = {
    2: {"True": 82.89, "False": 119.75}, 3: {"True": 55.99, "False": 125.89},
    4: {"True": 45.40, "False": 106.35}, 5: {"True": 46.94, "False": 96.32},
    6: {"True": 48.91, "False": 90.67}, 7: {"True": 45.06, "False": 96.73},
    8: {"True": 45.90, "False": 84.20}, 12: {"True": 48.93, "False": 1768.42},
    16: {"True": 55.42, "False": 1746.98},
}


def drift_3seed_success_sweep(out_path):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, label, marker in [("False", "per\\_timestep\\_loss = False", "o"),
                                ("True", "per\\_timestep\\_loss = True", "^")]:
        xs = sorted(DRIFT_3SEED_SR)
        means = [DRIFT_3SEED_SR[x][key][0] for x in xs]
        stds = [DRIFT_3SEED_SR[x][key][1] for x in xs]
        color = COLOR["dp"] if key == "False" else COLOR["drift"]
        ax.errorbar(xs, means, yerr=stds, marker=marker, color=color,
                    label=label.replace("\\_", "_"), linewidth=1.6, markersize=6,
                    capsize=3)
    ax.set_xlabel("gen_per_label (G)")
    ax.set_ylabel("Success rate (mean $\\pm$ std, 3 seeds)")
    ax.set_title("Drift Policy: success rate vs. $G$, 3-seed confirmatory sweep\n(Tool Hang, $2.5\\times10^5$ updates, no training rollout)")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_kernel_metric_lines(ax, data):
    """Shared True/False line-plot body used by both entropy and top1 panels."""
    for key, label, marker in [("False", "per\\_timestep\\_loss = False", "o"),
                                ("True", "per\\_timestep\\_loss = True", "^")]:
        xs = sorted(data)
        ys = [data[x][key] for x in xs]
        color = COLOR["dp"] if key == "False" else COLOR["drift"]
        ax.plot(xs, ys, marker=marker, color=color, label=label.replace("\\_", "_"),
                linewidth=1.6, markersize=6)
    ax.set_xlabel("gen_per_label (G)")
    ax.set_ylim(0, 1.0)


def drift_entropy_sweep(out_path, data, R_label):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _plot_kernel_metric_lines(ax, data)
    ax.set_ylabel(f"entropy$_{{R={R_label}}}$ (0 = sharp kernel, 1 = uniform)")
    ax.set_title(f"Attraction/repulsion kernel sharpness vs. $G$ ($R={R_label}$)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def drift_top1_sweep(out_path, data, R_label):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    _plot_kernel_metric_lines(ax, data)
    ax.set_ylabel(f"top1$_{{R={R_label}}}$ (peak softmax probability mass)")
    ax.set_title(f"Attraction/repulsion kernel top-1 mass vs. $G$ ($R={R_label}$)")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def drift_collapse_sweep(out_path_diversity, out_path_gradnorm):
    for data, ylabel, title, out_path in [
        (DRIFT_DIVERSITY, "diversity (mean pairwise candidate distance, log scale)",
         "Candidate diversity vs. $G$", out_path_diversity),
        (DRIFT_GRADNORM, "grad_norm (log scale)", "Gradient norm vs. $G$", out_path_gradnorm),
    ]:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for key, label, marker in [("False", "per\\_timestep\\_loss = False", "o"),
                                    ("True", "per\\_timestep\\_loss = True", "^")]:
            xs = sorted(data)
            ys = [data[x][key] for x in xs]
            color = COLOR["dp"] if key == "False" else COLOR["drift"]
            ax.plot(xs, ys, marker=marker, color=color, label=label.replace("\\_", "_"),
                    linewidth=1.6, markersize=6)
        ax.set_xlabel("gen_per_label (G)")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(frameon=False, loc="center left")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    grouped_bar(SUCCESS, "Success rate", "Success rate by task and method",
                "figures/success_rate.pdf", pct=True)
    grouped_bar(TRAIN_H, "Training time (h)", "Training wall-clock time by task and method",
                "figures/train_time.pdf")
    grouped_bar(EVAL_MIN, "Evaluation time (min, log scale)",
                "Full-rollout evaluation wall-clock time by task and method",
                "figures/eval_time.pdf", logy=True)
    latency_bar("figures/inference_latency.pdf")
    tradeoff_scatter("figures/tradeoff.pdf")
    drift_3seed_success_sweep("figures/drift_3seed_success.pdf")
    drift_entropy_sweep("figures/drift_entropy_r002.pdf", DRIFT_ENTROPY_002, "0.02")
    drift_entropy_sweep("figures/drift_entropy_r005.pdf", DRIFT_ENTROPY_005, "0.05")
    drift_entropy_sweep("figures/drift_entropy_r02.pdf", DRIFT_ENTROPY_02, "0.2")
    drift_top1_sweep("figures/drift_top1_r002.pdf", DRIFT_TOP1_002, "0.02")
    drift_top1_sweep("figures/drift_top1_r005.pdf", DRIFT_TOP1_005, "0.05")
    drift_top1_sweep("figures/drift_top1_r02.pdf", DRIFT_TOP1_02, "0.2")
    drift_collapse_sweep("figures/drift_diversity_sweep.pdf", "figures/drift_gradnorm_sweep.pdf")
    print("Wrote figures to figures/")
