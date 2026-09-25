"""
Evaluate Drifting-model checkpoints (plain or PAC-Bayes variant).

Works with checkpoints produced by either:
  - train_drift_unet_lowdim_workspace.py      (DriftUnetLowdimPolicy)
  - train_pac_drift_unet_lowdim_workspace.py  (PacDriftUnetLowdimPolicy)
which of the two a checkpoint came from is auto-detected per-checkpoint from
the loaded policy's type (isinstance(..., BaseLowdimPacPolicy)), so a single
--ckpts_dir of either kind works with this one script.

Note on NLL: unlike the diffusion/flow-matching eval scripts (eval_ckpts_dp.py,
eval_ckpts_flow.py), this script does NOT compute a negative log-likelihood.
The drifting generative process is a single deterministic forward pass
(noise -> action via one U-Net call at timesteps=0, see
drift_unet_lowdim_policy.py/pac_drift_unet_lowdim_policy.py) trained
with drift_loss's particle-attraction/repulsion objective - there is no
log-det-Jacobian, no invertibility, and no noise-level-indexed denoising
channel to integrate over, so neither the flow-matching ODE likelihood nor
the diffusion I-MMSE nll_bound machinery applies here. Instead, this script
reports drift_loss's own diagnostic metrics (scale, loss_{R} per configured
temperature) averaged over the validation set, which is the closest
drift-specific signal actually available.

Usage:
    python eval_ckpts_drift.py --ckpts_dir data/outputs/.../checkpoints -o data/outputs/.../eval
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import dill
import hydra
import numpy as np
import torch
from mujoco_py.builder import MujocoException
from omegaconf import OmegaConf, DictConfig
# Must be registered before any cfg saved by a training workspace (all of which
# use "${eval: ...}" interpolations, e.g. dataset pad_before/pad_after) is
# resolved - hydra.utils.instantiate()/OmegaConf.resolve() need this custom
# resolver, and nothing else in this script's import chain registers it.
OmegaConf.register_new_resolver("eval", eval, replace=True)
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.policy.base_lowdim_pac_policy import BaseLowdimPacPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

logger = logging.getLogger("eval_ckpts_drift")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def list_ckpt_files(ckpts_dir: Path) -> List[Path]:
    """Return sorted list of .ckpt files in ckpts_dir."""
    return sorted(p for p in ckpts_dir.iterdir() if p.suffix == ".ckpt")


def parse_step_from_filename(filename: str) -> Optional[int]:
    """Parse step number from checkpoint filename like 'step=0010-...ckpt'.
    Returns None if no step pattern is found or the file is 'latest.ckpt'.
    """
    if filename == "latest.ckpt":
        return None
    try:
        parts = filename.split("step=")
        if len(parts) < 2:
            return None
        after = parts[1]
        digits = after.split("-")[0].split(".")[0]
        return int(digits)
    except Exception:
        return None


def load_checkpoint_payload(ckpt_path: Path) -> Dict:
    """Load checkpoint payload using dill as the pickle module."""
    with ckpt_path.open("rb") as f:
        return torch.load(f, pickle_module=dill)


def instantiate_workspace(cfg: DictConfig, output_dir: Path) -> BaseWorkspace:
    """Create a workspace from Hydra config."""
    cls = hydra.utils.get_class(cfg._target_)
    return cls(cfg, output_dir=str(output_dir))


def _is_stochastic_policy(policy) -> bool:
    """True for policy families that take a `stochastic=` kwarg on
    compute_loss/predict_action: BaseLowdimPacPolicy subclasses (here,
    PacDriftUnetLowdimPolicy) - there's no separate "prob" policy base
    class in this codebase.
    """
    return isinstance(policy, BaseLowdimPacPolicy)


def compute_policy_loss(policy, batch, is_pac: bool, cfg) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Call compute_loss with the right signature for plain vs. PAC drifting
    policies. Both return (loss, metrics) where metrics is drift_loss's own
    info dict (scale, loss_{R} for each configured temperature - see
    drift_util.py's drift_loss()).
    """
    if is_pac:
        stochastic = bool(OmegaConf.select(cfg, "eval.stochastic", default=False))
        return policy.compute_loss(batch, stochastic=stochastic)
    return policy.compute_loss(batch)


def evaluate_drift_loss(policy, dataloader: DataLoader, cfg, device: torch.device, is_pac: bool) -> Tuple[float, Dict[str, float]]:
    """Evaluate average drift_loss (+ its diagnostic metrics: scale, loss_{R})
    over a dataset, sample-weighted.
    """
    policy.eval()
    total_loss = 0.0
    total_samples = 0
    metric_sums: Dict[str, float] = {}

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Validation drift_loss", leave=False,
                    mininterval=cfg.training.tqdm_interval_sec)
        for batch in pbar:
            n = len(batch["obs"])
            total_samples += n
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            loss, metrics = compute_policy_loss(policy, batch, is_pac, cfg)
            total_loss += loss.item() * n
            for k, v in metrics.items():
                v = v.item() if torch.is_tensor(v) else v
                metric_sums[k] = metric_sums.get(k, 0.0) + v * n

    if total_samples == 0:
        return 0.0, {}
    avg_loss = total_loss / total_samples
    avg_metrics = {k: v / total_samples for k, v in metric_sums.items()}
    return avg_loss, avg_metrics


def score_key_for(policy, stochastic: bool) -> str:
    """Env runners log 'test/mean_score' for plain policies, or
    'test/mean_score_deterministic' / 'test/mean_score_stochastic' for
    BaseLowdimPacPolicy subclasses - see e.g. pusht_keypoints_runner.py /
    robomimic_lowdim_runner.py's run().
    """
    if _is_stochastic_policy(policy):
        return "test/mean_score_stochastic" if stochastic else "test/mean_score_deterministic"
    return "test/mean_score"


def run_env_runner(env_runner, policy, stochastic: bool) -> Tuple[dict, float]:
    """Run the environment runner and return the log dict and mean score.
    NOTE: env_runner.run()'s actual signature is run(self, policy,
    stochastic=False) (see pusht_keypoints_runner.py / robomimic_lowdim_runner.py) -
    NOT run(policy, cfg). Passing cfg positionally there (as the two sibling
    eval scripts eval_ckpts_dp.py/eval_ckpts_flow.py do) would silently bind
    to the `stochastic` parameter instead.
    """
    runner_log = env_runner.run(policy, stochastic=stochastic)
    key = score_key_for(policy, stochastic)
    return runner_log, runner_log[key].item() if torch.is_tensor(runner_log[key]) else runner_log[key]


def save_json_log(out_path: Path, data: Dict) -> None:
    """Write JSON data to file."""
    with out_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def build_env_runner(cfg, output_dir: Path):
    """(Re)instantiate the task's env_runner - needed after a MujocoException,
    since AsyncVectorEnv._raise_if_errors permanently kills the crashed worker's pipe."""
    return hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))


def free_cuda_memory():
    """Clear CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def delete_checkpoint(ckpt_path: Path) -> None:
    """Delete a checkpoint file safely."""
    try:
        ckpt_path.unlink(missing_ok=True)
        logger.info("Deleted checkpoint: %s", ckpt_path.name)
    except Exception as e:
        logger.warning("Failed to delete checkpoint %s: %s", ckpt_path.name, e)


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

@click.command()
@click.option("-c", "--ckpts_dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output_dir", required=False, default=None, type=click.Path(path_type=Path),
              help="Where to write evaluation outputs")
@click.option("-d", "--device", default="cuda:0", help="Torch device string")
@click.option("--stochastic", is_flag=True, default=False,
              help="For PAC checkpoints: sample Bayesian weights during rollout instead of using the posterior mean. Ignored for plain drifting checkpoints.")
@click.option("--override", multiple=True, help="Hydra-style overrides e.g. task.env_runner.n_test=300")
@click.option("--delete_ckpts", is_flag=True, help="Whether to delete checkpoints after evaluation")
def main(ckpts_dir: Path, output_dir: Optional[Path], device: str, stochastic: bool,
         override: Tuple[str, ...], delete_ckpts: bool = False):
    """Evaluate all checkpoints in ckpts_dir and log results."""
    parent_dir = ckpts_dir.parent
    if output_dir is None:
        output_dir = parent_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    out_path = output_dir / f"eval_log_{timestamp}.json"
    device_obj = torch.device(device)

    all_ckpt_files = list_ckpt_files(ckpts_dir)
    if not all_ckpt_files:
        logger.warning("No .ckpt files found in %s", ckpts_dir)
        return

    # Load config from the LAST checkpoint (assumes all have same config)
    cfg = load_checkpoint_payload(all_ckpt_files[-1])["cfg"]
    if override:
        override_cfg = OmegaConf.from_dotlist(list(override))
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Prepare validation dataset and dataloader
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseLowdimDataset)
    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    # Environment runner
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))

    json_log = {}
    step_results = {"steps": [], "success_rates": []}
    sum_success_rates = 0.0
    num_evaluated = 0
    eval_times: List[float] = []
    loss_val = []
    metric_val: Dict[str, List[float]] = {}
    failed_checkpoints: List[Dict] = []
    last_success_rate: Optional[float] = None
    last_success_step: Optional[int] = None

    for ckpt_path in all_ckpt_files:
        step = parse_step_from_filename(ckpt_path.name)
        if step is None:
            continue

        logger.info("Evaluating checkpoint %s (step %d)", ckpt_path.name, step)

        payload = load_checkpoint_payload(ckpt_path)
        if "cfg" not in payload:
            logger.warning("No 'cfg' in payload for %s, skipping", ckpt_path.name)
            continue

        workspace = instantiate_workspace(cfg, output_dir)
        workspace.load_payload(payload, exclude_keys=["optimizer", "model"], include_keys=None)
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.to(device_obj)
        policy.eval()

        is_pac = _is_stochastic_policy(policy)
        use_stochastic = stochastic and is_pac

        try:
            # Run environment evaluation
            eval_start = time.perf_counter()
            _, success_rate = run_env_runner(env_runner, policy, use_stochastic)
            eval_time = time.perf_counter() - eval_start
        except MujocoException as e:
            logger.warning(
                "MuJoCo instability (NaN/Inf) evaluating checkpoint %s (step %d): %s. "
                "Reporting the previous checkpoint's success_rate instead.",
                ckpt_path.name, step, e,
            )
            failed_checkpoints.append({"step": step, "ckpt": ckpt_path.name, "error": str(e)})

            # Carry the previous checkpoint's success_rate forward (0.0 if there is none yet) so mean_scores has no gap/NaN.
            reported_success_rate = last_success_rate if last_success_rate is not None else 0.0
            entry = {
                "error": str(e),
                "is_pac": is_pac,
                "success_rate": reported_success_rate,
                "success_rate_carried_over_from_step": last_success_step,
            }
            json_log[f"model_at_step_{step:06d}"] = entry

            if last_success_rate is None:
                logger.warning(
                    "No previous successful checkpoint to carry a success_rate over from "
                    "for %s (step %d); reporting 0.0.", ckpt_path.name, step,
                )

            step_results["steps"].append(step)
            step_results["success_rates"].append(reported_success_rate)
            sum_success_rates += reported_success_rate
            num_evaluated += 1

            save_json_log(out_path, json_log)

            del policy, workspace, payload
            free_cuda_memory()

            # env_runner's crashed worker pipe is permanently dead - rebuild it.
            try:
                if hasattr(env_runner, "env"):
                    env_runner.env.close(terminate=True)
            except Exception:
                logger.warning("Failed to cleanly close env_runner after MuJoCo error", exc_info=True)
            env_runner = build_env_runner(cfg, output_dir)
            continue

        # Compute validation drift_loss + its diagnostic metrics (scale, loss_{R})
        if len(val_dataloader) == 0:
            loss_val.append(0.0)
            logger.warning("Validation dataloader is empty, skipping loss evaluation.")
        else:
            loss, metrics = evaluate_drift_loss(policy, val_dataloader, cfg, device_obj, is_pac)
            loss_val.append(loss)
            for k, v in metrics.items():
                metric_val.setdefault(k, []).append(v)

        key = f"model_at_step_{step:06d}"
        json_log[key] = {
            "success_rate": success_rate,
            "is_pac": is_pac,
            "eval_time_sec": eval_time,
        }
        step_results["steps"].append(step)
        step_results["success_rates"].append(success_rate)

        sum_success_rates += success_rate
        num_evaluated += 1
        eval_times.append(eval_time)
        last_success_rate = success_rate
        last_success_step = step

        save_json_log(out_path, json_log)

        del policy, workspace, payload
        free_cuda_memory()

    if num_evaluated > 0:
        json_log["loss_val"] = np.mean(loss_val)
        for k, vals in metric_val.items():
            json_log[f"drift_metric_{k}"] = np.mean(vals)
        json_log["mean_scores"] = step_results["success_rates"]
        json_log["num_steps"] = step_results["steps"]
        json_log[f"mean_success_rate_last_{num_evaluated}_checkpoints"] = sum_success_rates / num_evaluated
        if eval_times:
            json_log["eval_times_sec"] = eval_times
            json_log[f"mean_eval_time_sec_last_{len(eval_times)}_checkpoints"] = float(np.mean(eval_times))
    else:
        logger.warning("No valid checkpoints found.")

    if failed_checkpoints:
        json_log["failed_checkpoints"] = failed_checkpoints
        logger.warning(
            "%d checkpoint(s) skipped due to MuJoCo instability: %s",
            len(failed_checkpoints), [f["ckpt"] for f in failed_checkpoints],
        )

    save_json_log(out_path, json_log)
    logger.info("Evaluation complete. Log written to %s", out_path)

    if delete_ckpts:
        logger.info("All evaluations completed. Deleting checkpoints...")
        for ckpt_path in all_ckpt_files:
            delete_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()
