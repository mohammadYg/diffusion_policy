"""
Evaluate Diffusion Policy checkpoints.

Usage:
    python eval_ckpts_dp.py --ckpts_dir data/outputs/.../checkpoints -o data/outputs/.../eval
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
from omegaconf import OmegaConf
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

logger = logging.getLogger("eval_ckpts_dp")
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
    # pattern: 'step=1234'
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


def instantiate_workspace(cfg: OmegaConf, output_dir: Path) -> BaseWorkspace:
    """Create a workspace from Hydra config."""
    cls = hydra.utils.get_class(cfg._target_)
    return cls(cfg, output_dir=str(output_dir))


def _is_stochastic_policy(policy) -> bool:
    """True for policy families that take a `stochastic=` kwarg on
    compute_loss/nll_bound/predict_action: BaseLowdimPacPolicy subclasses -
    both the Bayes-by-backprop PAC-Bayes policies and IvonDiffusionUnetLowdimPolicy
    (the IVON-optimizer-based Bayesian policy also subclasses BaseLowdimPacPolicy
    directly - there's no separate "prob" policy base class in this codebase).
    """
    return isinstance(policy, BaseLowdimPacPolicy)


def evaluate_DM_loss(policy, dataloader: DataLoader, cfg, device: torch.device) -> float:
    """Evaluate average Diffusion Model loss over a dataset."""
    policy.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Validation loss", leave=False, mininterval=cfg.training.tqdm_interval_sec)
        for batch in pbar:
            n = len(batch["obs"])
            total_samples += n
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            if _is_stochastic_policy(policy):
                loss = policy.compute_loss(batch, stochastic=cfg.eval.stochastic, train=False)
            else:
                loss = policy.compute_loss(batch, train=False)
            total_loss += loss.item() * n

    return total_loss / total_samples if total_samples > 0 else 0.0


def evaluate_nll(policy, dataloader: DataLoader, step: int, cfg, device: torch.device) -> float:
    """Compute the negative log‑likelihood lower bound if the policy supports it."""
    if not hasattr(policy, "nll_bound"):
        return 0.0
    policy.eval()
    npoints = 100
    with torch.inference_mode():
        if _is_stochastic_policy(policy):
            nll = policy.nll_bound(dataloader, step, npoints=npoints, stochastic=cfg.eval.stochastic)
        else:
            nll = policy.nll_bound(dataloader, step, npoints=npoints)
    return nll.item()


def score_key_for(policy, stochastic: bool) -> str:
    """Env runners log 'test/mean_score' for plain policies, or
    'test/mean_score_deterministic' / 'test/mean_score_stochastic' for
    BaseLowdimPacPolicy subclasses - see e.g. pusht_keypoints_runner.py /
    robomimic_lowdim_runner.py's run().
    """
    if _is_stochastic_policy(policy):
        return "test/mean_score_stochastic" if stochastic else "test/mean_score_deterministic"
    return "test/mean_score"


def run_env_runner(env_runner, policy, cfg) -> Tuple[dict, float]:
    """Run the environment runner and return the log dict and mean score.

    env_runner.run()'s actual signature is run(self, policy, stochastic=False)
    (see pusht_keypoints_runner.py / robomimic_lowdim_runner.py) - it does NOT
    take cfg positionally. Passing cfg there (as this function used to)
    silently mis-binds it to `stochastic`, which is harmless for plain
    policies (env runners ignore `stochastic` for them) but breaks PAC
    policies (BaseLowdimPacPolicy): the env runner logs
    'test/mean_score_deterministic'/'_stochastic' for those instead of
    'test/mean_score', so runner_log["test/mean_score"] raised KeyError.
    """
    is_pac = _is_stochastic_policy(policy)
    stochastic = bool(getattr(cfg.eval, "stochastic", False)) if is_pac else False
    runner_log = env_runner.run(policy, stochastic=stochastic)
    key = score_key_for(policy, stochastic)
    score = runner_log[key]
    return runner_log, score.item() if torch.is_tensor(score) else score


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
@click.option("--override", multiple=True, help="Hydra-style overrides e.g. task.env_runner.n_test=300")
@click.option("--delete_ckpts", is_flag=True, help="Whether to delete checkpoints after evaluation")
def main(ckpts_dir: Path, output_dir: Optional[Path], device: str, override: Tuple[str, ...], delete_ckpts: bool = False):
    """Evaluate all checkpoints in ckpts_dir (step >= 50) and log results."""
    # Setup paths
    parent_dir = ckpts_dir.parent
    if output_dir is None:
        output_dir = parent_dir / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    out_path = output_dir / f"eval_log_{timestamp}.json"
    device_obj = torch.device(device)

    # List and filter checkpoints
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

    # Dataloader for full dataset (used for covariance spectrum)
    full_dataloader = DataLoader(
        dataset,
        batch_size=len(dataset),
        num_workers=1,
        pin_memory=True,
        persistent_workers=False,
    )

    # Environment runner
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))

    # Prepare containers for results
    json_log = {}
    step_results = {
        "steps": [],
        "success_rates": [],
        #"validation_losses": [],
        #"nll_values": [],
    }
    sum_success_rates = 0.0
    num_evaluated = 0
    eval_times: List[float] = []

    loss_val = []
    nll_val = []
    failed_checkpoints: List[Dict] = []
    last_success_rate: Optional[float] = None
    last_success_step: Optional[int] = None

    # Iterate over checkpoints
    for ckpt_path in all_ckpt_files:
        step = parse_step_from_filename(ckpt_path.name)
        if step is None:
            continue

        logger.info("Evaluating checkpoint %s (step %d)", ckpt_path.name, step)

        # Load checkpoint payload
        payload = load_checkpoint_payload(ckpt_path)
        if "cfg" not in payload:
            logger.warning("No 'cfg' in payload for %s, skipping", ckpt_path.name)
            continue

        # Load workspace and model
        workspace = instantiate_workspace(cfg, output_dir)
        workspace.load_payload(payload, exclude_keys=["optimizer", "model"], include_keys=None)
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.to(device_obj)
        policy.eval()

        try:
            # Run environment evaluation
            eval_start = time.perf_counter()
            _, success_rate = run_env_runner(env_runner, policy, cfg)
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
            json_log[f"model_at_step_{step:06d}"] = {
                "error": str(e),
                "success_rate": reported_success_rate,
                "success_rate_carried_over_from_step": last_success_step,
            }

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

        # Compute Validation metrics (if dataloader is not empty)
        if len(val_dataloader) == 0:
            loss_val.append(0.0)
            nll_val.append(0.0)
            logger.warning("Validation dataloader is empty, skipping loss and nll evaluation.")
        else:
            loss = evaluate_DM_loss(policy, val_dataloader, cfg, device_obj)
            loss_val.append(loss)

            policy.dataset_info(full_dataloader, covariance_spectrum=None, diagonal=False)
            nll = evaluate_nll(policy, val_dataloader, step, cfg, device_obj)
            nll_val.append(nll)

        # Store results
        key = f"model_at_step_{step:06d}"
        json_log[key] = {
            "success_rate": success_rate,
            "eval_time_sec": eval_time,
            #"test": {"loss_val": loss_val, "nll": nll_val},
        }
        step_results["steps"].append(step)
        step_results["success_rates"].append(success_rate)
        #step_results["validation_losses"].append(noise_loss)
        #step_results["nll_values"].append(nll_val)

        sum_success_rates += success_rate
        num_evaluated += 1
        eval_times.append(eval_time)
        last_success_rate = success_rate
        last_success_step = step

        # Save partial log
        save_json_log(out_path, json_log)

        # Cleanup
        del policy, workspace, payload
        free_cuda_memory()

    # Final summary
    if num_evaluated > 0:
        json_log["loss_val"] = np.mean(loss_val)
        json_log["nll_val"] = np.mean(nll_val)
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

    # Delete checkpoints after ALL evaluations are complete
    if delete_ckpts:
        logger.info("All evaluations completed. Deleting checkpoints...")

        for ckpt_path in all_ckpt_files:
            delete_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()
