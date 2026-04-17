"""
Usage example:
python eval_refactor.py --ckpts_dir data/image/pusht/diffusion_policy_cnn/train_0/checkpoints -o data/pusht_eval_output
"""

from pathlib import Path
from datetime import datetime
import json
import logging
import math
import sys
from typing import Dict, Tuple, Optional, List

import click
import dill
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import tqdm
import time

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.policy.base_lowdim_prob_policy import BaseLowdimProbPolicy
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.pytorch_util import dict_apply


logger = logging.getLogger("eval_refactor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# ----------------------------- Helpers -----------------------------
def list_ckpt_files(ckpts_dir: Path) -> List[Path]:
    """Return sorted list of .ckpt files in ckpts_dir."""
    return sorted([p for p in ckpts_dir.iterdir() if p.suffix == ".ckpt"])

def parse_epoch_from_ckpt_name(ckpt_name: str) -> Optional[int]:
    """Try to parse an epoch number from filenames like 'epoch=0010-...ckpt'.
    Return None if parsing fails or if it's 'latest.ckpt'.
    """
    if ckpt_name == "latest.ckpt":
        return None
    # look for pattern 'epoch=XXXX' somewhere in name
    try:
        parts = ckpt_name.split("epoch=")
        if len(parts) < 2:
            return None
        after = parts[1]
        digits = after.split("-")[0].split(".")[0]
        return int(digits)
    except Exception:
        return None

def load_payload(ckpt_path: Path) -> Dict:
    """Load checkpoint payload using dill as the pickle module."""
    with ckpt_path.open("rb") as f:
        return torch.load(f, pickle_module=dill)

def instantiate_workspace_from_cfg(cfg: OmegaConf, output_dir: Path) -> BaseWorkspace:
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(output_dir))
    return workspace

def run_env_runner(env_runner, policy, cfg) -> Tuple[dict, float]:
    """Run the env_runner n_repeat times and return mean and variance of "test/mean_score"."""
    runner_log = env_runner.run(policy, cfg)
    score = runner_log["test/mean_score"]
    return runner_log, score.item()


def eval_network(policy, dataloader: DataLoader, cfg, device: torch.device) -> float:
    """Evaluate noise prediction loss and action reconstruction loss across a dataloader.

    Returns: (avg_noise_pred_loss, avg_action_rec_loss)
    Both are averaged per-sample and averaged over n_MC where relevant.
    """
    with torch.inference_mode():
        total_samples = 0
        total_noise_loss = 0.0
        with tqdm.tqdm(dataloader, desc="Validation: noise prediction and action reconstruction error", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch in tepoch:
                n_samples = len(batch["obs"])
                total_samples += n_samples

                # send to device
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                # noise prediction loss
                if isinstance(policy, BaseLowdimProbPolicy):
                    noise_pred_loss = policy.compute_loss(batch, stochastic = cfg.eval.stochastic, train = False)
                else:
                    noise_pred_loss = policy.compute_loss(batch, train = False)
                total_noise_loss += float(noise_pred_loss.item() * n_samples)
        
        noise_loss = total_noise_loss / total_samples

    return noise_loss


def save_json_log(out_path: Path, json_log: Dict):
    with out_path.open("w") as f:
        json.dump(json_log, f, indent=2, sort_keys=True)

# ----------------------------- Main -----------------------------
@click.command()
@click.option("-c", "--ckpts_dir", required=True, type=click.Path(exists=True))
@click.option("-o", "--output", 'output_dir', required=False, default=None, help="Where to write eval outputs")
@click.option("-d", "--device", default="cuda:0", help="Torch device string")
@click.option("--override", multiple=True, help="Hydra-style overrides e.g. task.env_runner.n_test=300")
def main(ckpts_dir, output_dir, device, override):
    ckpts_dir = Path(ckpts_dir)
    parent_dir = ckpts_dir.parent

    if output_dir is None:
        output_dir = parent_dir / "eval_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # configure TopK manager for saving best checkpoint
    topk_manager = TopKCheckpointManager(
        save_dir=str(output_dir),
        monitor_key="test_mean_score",
        mode="max",
        k=1,
        format_str='epoch={epoch:04d}-test_mean_score={test_mean_score:.3f}.ckpt'
    )

    now = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    out_path = output_dir / f"eval_log_{now}.json"
    json_log = {}

    ckpt_files = list_ckpt_files(ckpts_dir)
    device = torch.device(device)

    # Trackers
    results_for_all_epochs = {
                              "test_noise_pred_loss": [],
                              "test_mean_score": [], "num_epochs": [],
                              "nll_test": []}
    sum_success_rate_last_10_epochs = 0.0

    # instantiate env_runner
    cfg = load_payload(ckpt_files[-1])['cfg']
    # apply overrides
    if override:
        override_cfg = OmegaConf.from_dotlist(list(override))
        cfg = OmegaConf.merge(cfg, override_cfg)
        
    # prepare datasets (instantiate per-checkpoint in case cfg changed)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseLowdimDataset)
    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False, pin_memory=False, persistent_workers=False)
    ## configure dataset for covariance_spectrum
    cov_dataloader = DataLoader(dataset, batch_size=len(dataset), 
                                num_workers=1,   pin_memory = True, 
                                persistent_workers = False)
    
    ## extract demos that are not used in training 
    test_indices = np.where(~dataset.train_mask)[0]
    
    # initialize envs
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir), test_mask = test_indices)


    test_pred_noise_loss = 0.0
    NLL_test = 0.0
    for ckpt_path in ckpt_files:
        ckpt_name = ckpt_path.name
        epoch = parse_epoch_from_ckpt_name(ckpt_name)

        # skip latest and epochs we don't want to eval
        if epoch is None:
            logger.info("Skipping %s (no epoch parsed)", ckpt_name)
            continue
        if epoch < 50:
            logger.info("Skipping %s (epoch %d < 50)", ckpt_name, epoch)
            continue

        logger.info("Evaluating checkpoint %s (epoch %d)", ckpt_name, epoch)

        payload = load_payload(ckpt_path)
        if cfg is None:
            logger.warning("No cfg in payload %s, skipping", ckpt_name)
            continue

        # instantiate workspace and load payload
        workspace = instantiate_workspace_from_cfg(cfg, output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=['optimizer','model'], include_keys=None)

        # choose policy (ema if configured)
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        policy.to(device)
        policy.eval()

        # run env_runner repeated evaluations
        runner_log, success_rate = run_env_runner(env_runner, policy, cfg)

        # save best model via topk manager
        success_info = {"test_mean_score": float(success_rate), "epoch": int(epoch)}
        best_path = topk_manager.get_ckpt_path(success_info)
        if best_path is not None:
            workspace.save_checkpoint(path=best_path)

        # update running sum for last 10 epochs (preserve original logic)
        if epoch > cfg.training.num_epochs - 550:
            sum_success_rate_last_10_epochs += success_rate

        # compute covariance_spectrum of the whole training data
        policy.dataset_info(cov_dataloader, covariance_spectrum=None, diagonal=False)
       
        # # test eval
        # test_pred_noise_loss = eval_network(policy, val_dataloader, cfg, device)
        # if isinstance(policy, BaseLowdimProbPolicy):
        #     NLL_test = policy.nll_bound(val_dataloader, epoch, npoints=100, stochastic=cfg.eval.stochastic).item()
        # else:
        #     NLL_test = policy.nll_bound(val_dataloader, epoch, npoints=100).item()
        
        ## NLL evaluation
        epoch_key = f"model_at_epoch_{int(epoch):04d}"
        json_log[epoch_key] = {
            "success_rate": success_rate,
            "test": {
                "loss_noise_pred": test_pred_noise_loss,
                "nll_test": NLL_test
            }
        }

        # keep history for plotting / aggregation
        results_for_all_epochs["test_noise_pred_loss"].append(test_pred_noise_loss)
        results_for_all_epochs["test_mean_score"].append(success_rate)
        results_for_all_epochs["num_epochs"].append(epoch)
        results_for_all_epochs["nll_test"].append(NLL_test)

        # write partial log after each epoch to be robust to crashes
        save_json_log(out_path, json_log)

        # cleanup workspace and payload references
        try:
            del policy
            del workspace
            del payload
        except Exception:
            pass

        # free CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # final metadata
    json_log["test_noise_pred_loss_epochs"] = results_for_all_epochs["test_noise_pred_loss"]
    json_log["test_mean_score_over_epochs"] = results_for_all_epochs["test_mean_score"]
    json_log["num_epochs"] = results_for_all_epochs["num_epochs"]
    json_log["nll_test_over_epochs"] = results_for_all_epochs["nll_test"]
    json_log["mean_success_rate_last_10_checkpoints"] = sum_success_rate_last_10_epochs/10

    save_json_log(out_path, json_log)
    logger.info("Evaluation complete. Log written to %s", str(out_path))


if __name__ == '__main__':
    main()
