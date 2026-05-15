"""
Evaluate Diffusion Policy checkpoints.

Usage:
    python eval_ckpts.py --ckpts_dir data/outputs/.../checkpoints -o data/outputs/.../eval
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager  # noqa: F401 (kept for reference)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.policy.base_lowdim_pac_policy import BaseLowdimPacPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Attempt to import probabilistic policy if available (optional feature)
try:
    from diffusion_policy.policy.base_lowdim_prob_policy import BaseLowdimProbPolicy
except ImportError:
    BaseLowdimProbPolicy = None

logger = logging.getLogger("eval_refactor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def list_ckpt_files(ckpts_dir: Path) -> List[Path]:
    """Return sorted list of .ckpt files in ckpts_dir."""
    return sorted(p for p in ckpts_dir.iterdir() if p.suffix == ".ckpt")


def parse_epoch_from_filename(filename: str) -> Optional[int]:
    """Parse epoch number from checkpoint filename like 'epoch=0010-...ckpt'.
    Returns None if no epoch pattern is found or the file is 'latest.ckpt'.
    """
    if filename == "latest.ckpt":
        return None
    # pattern: 'epoch=1234'
    try:
        parts = filename.split("epoch=")
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


def compute_loss(policy, batch: Dict, device: torch.device, cfg) -> float:
    """Compute the Diffusion Model loss for a batch."""
    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    if BaseLowdimProbPolicy is not None and isinstance(policy, BaseLowdimProbPolicy):
        loss = policy.compute_loss(batch, stochastic=cfg.eval.stochastic, train=False)
    else:
        loss = policy.compute_loss(batch, train=False)
    return loss.item()


def evaluate_DM_loss(policy, dataloader: DataLoader, cfg, device: torch.device) -> float:
    """Evaluate average Diffusion Model loss over a dataset."""
    policy.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Evaluating loss", leave=False, mininterval=cfg.training.tqdm_interval_sec)
        for batch in pbar:
            n = len(batch["obs"])
            total_samples += n
            loss = compute_loss(policy, batch, device, cfg)
            total_loss += loss * n

    return total_loss / total_samples if total_samples > 0 else 0.0


def evaluate_nll(policy, dataloader: DataLoader, epoch: int, cfg, device: torch.device) -> float:
    """Compute the negative log‑likelihood lower bound if the policy supports it."""
    if not hasattr(policy, "nll_bound"):
        return 0.0
    policy.eval()
    npoints = getattr(cfg.eval, "npoints", 100)
    with torch.inference_mode():
        if BaseLowdimProbPolicy is not None and isinstance(policy, BaseLowdimProbPolicy):
            stochastic = getattr(cfg.eval, "stochastic", False)
            nll = policy.nll_bound(dataloader, epoch, npoints=npoints, stochastic=stochastic)
        else:
            nll = policy.nll_bound(dataloader, epoch, npoints=npoints)
    return nll.item()


def run_env_runner(env_runner, policy, cfg) -> Tuple[dict, float]:
    """Run the environment runner and return the log dict and mean score."""
    runner_log = env_runner.run(policy, cfg)
    score = runner_log["test/mean_score"]
    return runner_log, score.item()

def save_json_log(out_path: Path, data: Dict) -> None:
    """Write JSON data to file."""
    with out_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def free_cuda_memory():
    """Clear CUDA cache if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

@click.command()
@click.option("-c", "--ckpts_dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output_dir", required=False, default=None, type=click.Path(path_type=Path),
              help="Where to write evaluation outputs")
@click.option("-d", "--device", default="cuda:0", help="Torch device string")
@click.option("--override", multiple=True, help="Hydra-style overrides e.g. task.env_runner.n_test=300")
def main(ckpts_dir: Path, output_dir: Optional[Path], device: str, override: Tuple[str, ...]):
    """Evaluate all checkpoints in ckpts_dir (epoch >= 50) and log results."""
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
    epoch_results = {
        "epochs": [],
        "success_rates": [],
        "validation_losses": [],
        "nll_values": [],
    }
    sum_success_rates = 0.0
    num_evaluated = 0

    # Iterate over checkpoints
    for ckpt_path in all_ckpt_files:
        epoch = parse_epoch_from_filename(ckpt_path.name)
        if epoch is None:
            continue

        logger.info("Evaluating checkpoint %s (epoch %d)", ckpt_path.name, epoch)

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

        # Run environment evaluation
        _, success_rate = run_env_runner(env_runner, policy, cfg)

        # Compute Validation metrics (if dataloader is not empty)
        if len(val_dataloader) == 0:
            noise_loss=0.0
            nll_val=0.0
            logger.warning("Validation dataloader is empty, skipping loss and nll evaluation.")
        else:
            noise_loss = evaluate_DM_loss(policy, val_dataloader, cfg, device_obj)
            policy.dataset_info(full_dataloader, covariance_spectrum=None, diagonal=False)
            nll_val = evaluate_nll(policy, val_dataloader, epoch, cfg, device_obj)

        # Store results
        key = f"model_at_epoch_{epoch:04d}"
        json_log[key] = {
            "success_rate": success_rate,
            "test": {"validation_loss": noise_loss, "nll": nll_val},
        }
        epoch_results["epochs"].append(epoch)
        epoch_results["success_rates"].append(success_rate)
        epoch_results["validation_losses"].append(noise_loss)
        epoch_results["nll_values"].append(nll_val)

        sum_success_rates += success_rate
        num_evaluated += 1

        # Save partial log
        save_json_log(out_path, json_log)

        # Cleanup
        del policy, workspace, payload
        free_cuda_memory()

    # Final summary
    if num_evaluated > 0:
        json_log["validation_losses"] = epoch_results["validation_losses"]
        json_log["mean_scores"] = epoch_results["success_rates"]
        json_log["num_epochs"] = epoch_results["epochs"]
        json_log["nlls"] = epoch_results["nll_values"]
        json_log[f"mean_success_rate_last_{num_evaluated}_checkpoints"] = sum_success_rates / num_evaluated
    else:
        logger.warning("No valid checkpoints found.")

    save_json_log(out_path, json_log)
    logger.info("Evaluation complete. Log written to %s", out_path)


if __name__ == "__main__":
    main()