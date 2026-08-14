"""
Evaluate Flow Matching checkpoints.

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
 
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
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

    

def evaluate_loss(policy, dataloader: DataLoader, cfg, device: torch.device) -> float:
    """Evaluate average Flow matching loss over a dataset."""
    policy.eval()
    total_loss = 0.0
    total_samples = 0
    x1_vf_batch = None

    with torch.inference_mode():
        pbar = tqdm(dataloader, desc="Validation loss", leave=False, mininterval=cfg.training.tqdm_interval_sec)
        for batch in pbar:
            n = len(batch["obs"])
            total_samples += n
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            # Sample x1_vf_batch if x1_vf_bs > 0
            if cfg.training.x1_vf_bs > 0:
                x1_vf_batch = policy.sample_x1_vf_batch(dataloader.dataset, cfg.training.x1_vf_bs, device=device)
                                       
            if BaseLowdimProbPolicy is not None and isinstance(policy, BaseLowdimProbPolicy):
                loss = policy.compute_loss(batch, stochastic=cfg.eval.stochastic, 
                                            x1_vf_batch=x1_vf_batch, 
                                            skewed_timesteps=cfg.training.skewed_timesteps,
                                            debug=False)
            else:
                loss = policy.compute_loss(batch,
                                            x1_vf_batch=x1_vf_batch, 
                                            skewed_timesteps=cfg.training.skewed_timesteps,
                                            debug=False)
            total_loss += loss.item() * n

    return total_loss / total_samples if total_samples > 0 else 99999

def evaluate_nll(policy, dataloader: DataLoader, cfg, device: torch.device) -> float:
    """Compute the negative log‑likelihood if the policy supports it."""
    if not hasattr(policy, "compute_nll"):
        return 0.0
    
    policy.eval()
    total_nll  = 0.0
    total_samples = 0
    pbar = tqdm(dataloader, desc="NLL Computation", leave=False, mininterval=cfg.training.tqdm_interval_sec)
    for batch in pbar:
        n = len(batch["obs"])
        total_samples += n
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        if BaseLowdimProbPolicy is not None and isinstance(policy, BaseLowdimProbPolicy):
            nll = policy.compute_nll(batch, stochastic=cfg.eval.stochastic,
                                    exact_divergence=cfg.eval.exact_divergence,
                                                            )
        else:
             nll = policy.compute_nll(batch, 
                                    exact_divergence=cfg.eval.exact_divergence,
                                                                        )
        total_nll  += nll.item() * n

    return total_nll / total_samples if total_samples > 0 else 99999


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

def delete_checkpoint(ckpt_path: Path) -> None:
    """Delete a checkpoint file safely, except latest.ckpt."""
    
    # if ckpt_path.name == "latest.ckpt":
    #     logger.info("Skipping deletion of %s", ckpt_path.name)
    #     return

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

    loss_val=[]
    nll_val=[]

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

        # Run environment evaluation
        _, success_rate = run_env_runner(env_runner, policy, cfg)

        # Compute Validation metrics (if dataloader is not empty)
        if len(val_dataloader) == 0:
            loss_val.append(0.0)
            nll_val.append(0.0)
            logger.warning("Validation dataloader is empty, skipping loss and nll evaluation.")
        else:
            loss = evaluate_loss(policy, val_dataloader, cfg, device_obj)
            loss_val.append(loss)
            
            nll = evaluate_nll(policy, val_dataloader, cfg, device_obj)
            nll_val.append(nll)

        # Store results
        key = f"model_at_step_{step:06d}"
        json_log[key] = {
            "success_rate": success_rate,
            #"test": {"loss_val": loss_val, "nll": nll_val},
        }
        step_results["steps"].append(step)
        step_results["success_rates"].append(success_rate)
        #step_results["validation_losses"].append(noise_loss)
        #step_results["nll_values"].append(nll_val)

        sum_success_rates += success_rate
        num_evaluated += 1

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
    else:
        logger.warning("No valid checkpoints found.")

    save_json_log(out_path, json_log)
    logger.info("Evaluation complete. Log written to %s", out_path)

    # Delete checkpoints after ALL evaluations are complete
    if delete_ckpts:
        logger.info("All evaluations completed. Deleting checkpoints...")

        for ckpt_path in all_ckpt_files:
            delete_checkpoint(ckpt_path)

if __name__ == "__main__":
    main()