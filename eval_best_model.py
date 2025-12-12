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

def run_env_runner_repeat(policy, cfg, output_dir: Path, n_repeat: int) -> Tuple[dict, float, float]:
    """Run the env_runner n_repeat times and return mean and variance of "test/mean_score"."""
    scores = []
    # instantiate runner once per evaluation (cheaper)
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))
    for _ in range(n_repeat):
        if isinstance(policy, BaseLowdimProbPolicy):
            runner_log = env_runner.run_prob(policy, cfg.eval.stochastic)
        else:
            runner_log = env_runner.run(policy)
        scores.append(runner_log["test/mean_score"])

    ddof = 1 if n_repeat > 1 else 0
    return runner_log, float(np.mean(scores)), float(np.var(scores, ddof=ddof))


def eval_network(policy, dataloader: DataLoader, cfg, device: torch.device, n_repeat: int, n_MC: int, normalized = False) -> Tuple[float, float, float, float, float, float]:
    """Evaluate noise prediction loss and action reconstruction loss across a dataloader.

    Returns: (avg_noise_pred_loss, avg_action_rec_loss)
    Both are averaged per-sample and averaged over n_MC where relevant.
    """
    noise_loss_single_step = []
    noise_loss_all_steps = []
    recontruction_loss = []

    with torch.inference_mode():
        for _ in range (n_repeat):
            total_samples = 0
            total_noise_loss_single_step = 0.0
            total_noise_loss_all_steps = 0.0
            total_rec_loss = 0.0
            with tqdm.tqdm(dataloader, desc="Validation: noise prediction and action reconstruction error", leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch in tepoch:
                    n_samples = len(batch["obs"])
                    total_samples += n_samples

                    # send to device
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    # noise prediction loss
                    if isinstance(policy, BaseLowdimProbPolicy):
                        noise_pred_loss_single_step = policy.compute_loss(batch, stochastic = cfg.eval.stochastic, train = False)
                        noise_pred_loss_all_steps = policy.compute_loss(batch, stochastic = cfg.eval.stochastic, train = True)
                    else:
                        noise_pred_loss_single_step = policy.compute_loss(batch, train = False)
                        noise_pred_loss_all_steps = policy.compute_loss(batch, train = True)
                    total_noise_loss_single_step += float(noise_pred_loss_single_step.item() * n_samples)
                    total_noise_loss_all_steps += float(noise_pred_loss_all_steps.item() * n_samples)

                    # action reconstruction loss (possibly probabilistic policy)
                    shape = (n_samples, cfg.horizon, cfg.action_dim + cfg.obs_dim)
                    # produce n_MC noises and average
                    rec_loss_sum = 0.0
                    for _ in range(n_MC):
                        noise = torch.randn(shape, device=batch["action"].device)
                        if isinstance(policy, BaseLowdimProbPolicy):
                            rec_loss = policy.compute_action_reconst_loss(noise, batch, cfg.eval.stochastic, loss_type="MSE", normalized = normalized)
                        else:
                            rec_loss = policy.compute_action_reconst_loss(noise, batch, loss_type="MSE", normalized = normalized)
                        rec_loss_sum += float(rec_loss.item() * n_samples)
                    total_rec_loss += rec_loss_sum / max(1, n_MC)
            
            noise_loss_single_step.append(total_noise_loss_single_step / total_samples)
            noise_loss_all_steps.append(total_noise_loss_all_steps / total_samples)
            recontruction_loss.append(total_rec_loss / total_samples)

    avg_noise_single_step = np.mean(noise_loss_single_step)
    avg_noise_all_steps = np.mean(noise_loss_all_steps)
    avg_rec = np.mean(recontruction_loss)
    var_noise_single_step = np.var(noise_loss_single_step, ddof=1 if n_repeat > 1 else 0)
    var_noise_all_steps = np.var(noise_loss_all_steps, ddof=1 if n_repeat > 1 else 0)
    var_rec = np.var(recontruction_loss, ddof=1 if n_repeat > 1 else 0)        
    return float(avg_noise_single_step), float(var_noise_single_step), float(avg_noise_all_steps), float(var_noise_all_steps), float(avg_rec), float(var_rec)


def save_json_log(out_path: Path, json_log: Dict):
    with out_path.open("w") as f:
        json.dump(json_log, f, indent=2, sort_keys=True)

def eval_memorization_runner(policy, dataloader, gen_act, gen_act_norm, normalized, device, threshold=0.5) -> Tuple[float, float]:
    """
    Evaluate if the model has memorized the training dataset.
    
    Args:
        dataloader: provides reference dataset, each batch with data['action'] of shape (batch, horizon, action_dim)
        generated_actions: tensor of shape (m, horizon, action_dim)
        cfg: config with .horizon, .action_dim, .obs_dim
        threshold: fraction threshold to consider a sample memorized

    Returns:
        mean fraction over all generated samples
    """
    all_distances = []  # store distances to all dataset batches
    generated_actions = gen_act
    if normalized:
        generated_actions = gen_act_norm
    generated_actions = generated_actions.to(device)
    
    for batch in dataloader:
        # send to device
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        # normalize batch
        if normalized:
            batch = policy.normalizer.normalize(batch)
        # batch['action']: shape (l, horizon, action_dim)
        ref_actions = batch['action']  # (l, horizon, cfg.action_dim)
        diff = generated_actions[:, None, :, :] - ref_actions[None, :, :, :]  # (m, l, horizon, action_dim)
        dist_squared = torch.sum(diff ** 2, dim=(2, 3))  # (m, l)
        dist = torch.sqrt(dist_squared)                  # (m, l)
        all_distances.append(dist)

    # Concatenate distances along dataset dimension
    all_distances = torch.cat(all_distances, dim=1)  # (m, total_dataset_size)

    # Find smallest and second smallest distance for each generated sample
    smallest_vals, _ = torch.topk(all_distances, k=2, largest=False)  # (m, 2)

    # Compute fraction
    fraction = smallest_vals[:, 0] / smallest_vals[:, 1]  # (m,)

    # Optionally count memorized samples
    memorized_count = (fraction < threshold).sum().item()

    mean_fraction = fraction.mean().item()
    return mean_fraction, memorized_count

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
        monitor_key="test_mean_score_avg",
        mode="max",
        k=1,
        format_str='epoch={epoch:04d}-test_mean_score_avg={test_mean_score_avg:.3f}.ckpt'
    )

    now = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    out_path = output_dir / f"eval_log_{now}.json"
    json_log = {}

    ckpt_files = list_ckpt_files(ckpts_dir)
    device = torch.device(device)

    # Trackers
    results_for_all_epochs = {"train_act_reconst_loss": [], "train_noise_pred_loss_SS": [],
                              "test_act_reconst_loss": [], "test_noise_pred_loss_SS": [],
                              "train_noise_pred_loss_AS": [], "test_noise_pred_loss_AS": [],
                              "mean_memorization": [], "n_memorized": [],
                              "mean_score_avg": [], "num_epochs": [],
                              "memorization_fraction":[],
                              "nll_test": [], "nll_train": []}
    sum_success_rate_last_10_epochs = 0.0

    for ckpt_path in ckpt_files:
        ckpt_name = ckpt_path.name
        epoch = parse_epoch_from_ckpt_name(ckpt_name)

        # skip latest and epochs we don't want to eval
        if epoch is None:
            logger.info("Skipping %s (no epoch parsed)", ckpt_name)
            continue
        if epoch < -1:
            logger.info("Skipping %s (epoch %d < 50)", ckpt_name, epoch)
            continue

        logger.info("Evaluating checkpoint %s (epoch %d)", ckpt_name, epoch)

        payload = load_payload(ckpt_path)
        cfg = payload.get("cfg")
        if cfg is None:
            logger.warning("No cfg in payload %s, skipping", ckpt_name)
            continue

        # apply overrides
        if override:
            override_cfg = OmegaConf.from_dotlist(list(override))
            cfg = OmegaConf.merge(cfg, override_cfg)

        # instantiate workspace and load payload
        workspace = instantiate_workspace_from_cfg(cfg, output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # choose policy (ema if configured)
        policy = workspace.model
        if getattr(cfg.training, "use_ema", False):
            policy = workspace.ema_model

        policy.to(device)
        policy.eval()

        # run env_runner repeated evaluations
        runner_log, avg_success_rate, var_success_rate = run_env_runner_repeat(policy, cfg, output_dir, cfg.task.n_repeat_runner)

        # save best model via topk manager
        success_info = {"test_mean_score_avg": float(avg_success_rate), "epoch": int(epoch)}
        best_path = topk_manager.get_ckpt_path(success_info)
        if best_path is not None:
            workspace.save_checkpoint(path=best_path)

        # update running sum for last 10 epochs (preserve original logic)
        if epoch > cfg.training.num_epochs - 550:
            sum_success_rate_last_10_epochs += avg_success_rate

        # start_time = time.perf_counter()
        # prepare datasets (instantiate per-checkpoint in case cfg changed)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, pin_memory=False, persistent_workers=False)

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False, pin_memory=False, persistent_workers=False)

        # evaluation of train and test splits
        n_MC = 1

        # train eval
        train_noise_avg_SS, train_noise_var_SS, train_noise_avg_AS, train_noise_var_AS, train_rec_avg, train_rec_var = eval_network(policy, train_dataloader, cfg, device, cfg.task.n_repeat_runner, n_MC, normalized = False)
        # test eval
        test_noise_avg_SS, test_noise_var_SS, test_noise_avg_AS, test_noise_var_AS, test_rec_avg, test_rec_var = eval_network(policy, val_dataloader, cfg, device, cfg.task.n_repeat_runner, n_MC, normalized = False)
        # memorization eval for generated actions in env_runner 
        #avg_memoraized, n_memorized = eval_memorization(policy, train_dataloader, runner_log['generated_actions'], runner_log['generated_actions_normalized'], True, device, threshold=0.5)
        # memorization eval for generated actions given observations from test set
        if isinstance(policy, BaseLowdimProbPolicy):
            avg_memoraized, n_memorized, memorization_frac= policy.eval_memorization(val_dataloader, train_dataloader, stochastic.cfg.eval.stochastic, normalized = False, device, threshold=0.5)
            NLL_test = policy.test_nll(val_dataloader, epoch, npoints=100, xinterval=None stochastic=cfg.eval.stochastic)
            NLL_train = policy.test_nll(train_dataloader, epoch, npoints=100, xinterval=None stochastic=cfg.eval.stochastic)
        else:
            avg_memoraized, n_memorized, memorization_frac= policy.eval_memorization(val_dataloader, train_dataloader, normalized = False, device, threshold=0.5)
            NLL_test = policy.test_nll(val_dataloader, epoch, npoints=100, xinterval=None)
            NLL_train = policy.test_nll(train_dataloader, epoch, npoints=100, xinterval=None)
        # store per-epoch results
        
        # end_time = time.perf_counter()
        # elapsed_time = end_time - start_time
        # print (f"Elapsed time:{elapsed_time} seconds")
        
        ## NLL evaluation
        epoch_key = f"model_at_epoch_{int(epoch):04d}"
        json_log[epoch_key] = {
            "success_rate": {
                "mean_score_avg": avg_success_rate,
                "mean_score_var": var_success_rate,
            },
            "train": {
                "loss_noise_pred_avg_SS": train_noise_avg_SS,
                "loss_noise_pred_var_SS": train_noise_var_SS,
                "loss_noise_pred_sd_SS": np.sqrt(train_noise_var_SS),
                "loss_noise_pred_avg_AS": train_noise_avg_AS,
                "loss_noise_pred_var_AS": train_noise_var_AS,
                "loss_noise_pred_sd_AS": np.sqrt(train_noise_var_AS),
                "action_reconst_loss_avg": train_rec_avg,
                "action_reconst_loss_var": train_rec_var,
                "action_reconst_loss_sd": np.sqrt(train_rec_var),
                "nll_train": NLL_train
            },
            "test": {
                "loss_noise_pred_avg_SS": test_noise_avg_SS,
                "loss_noise_pred_var_SS": test_noise_var_SS,
                "loss_noise_pred_sd_SS": np.sqrt(test_noise_var_SS),
                "loss_noise_pred_avg_AS": test_noise_avg_AS,
                "loss_noise_pred_var_AS": test_noise_var_AS,
                "loss_noise_pred_sd_AS": np.sqrt(test_noise_var_AS),
                "action_reconst_loss_avg": test_rec_avg,
                "action_reconst_loss_var": test_rec_var,
                "action_reconst_loss_sd": np.sqrt(test_rec_var),
                "nll_test": NLL_test
            },
            "memorization": {
                "mean_fraction": avg_memoraized,
                "n_memorized": n_memorized,
                "memorization_fraction":memorization_frac
            }
        }

        # keep history for plotting / aggregation
        results_for_all_epochs["train_act_reconst_loss"].append(train_rec_avg)
        results_for_all_epochs["train_noise_pred_loss_SS"].append(train_noise_avg_SS)
        results_for_all_epochs["train_noise_pred_loss_AS"].append(train_noise_avg_AS)
        results_for_all_epochs["test_act_reconst_loss"].append(test_rec_avg)
        results_for_all_epochs["test_noise_pred_loss_SS"].append(test_noise_avg_SS)
        results_for_all_epochs["test_noise_pred_loss_AS"].append(test_noise_avg_AS)
        results_for_all_epochs["mean_memorization"].append(avg_memoraized)
        results_for_all_epochs["n_memorized"].append(n_memorized)
        results_for_all_epochs["memorization_fraction"].append(memorization_frac)

        results_for_all_epochs["mean_score_avg"].append(avg_success_rate)
        results_for_all_epochs["num_epochs"].append(epoch)

        results_for_all_epochs["nll_test"].append(NLL_test)
        results_for_all_epochs["nll_train"].append(NLL_train)

        # write partial log after each epoch to be robust to crashes
        save_json_log(out_path, json_log)

        # cleanup workspace and payload references
        try:
            del policy
            del workspace
            del dataset
            del val_dataset
            del payload
            del cfg
        except Exception:
            pass

        # free CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # final metadata
    json_log["train_act_reconst_loss_over_epochs"] = results_for_all_epochs["train_act_reconst_loss"]
    json_log["train_noise_pred_loss_SS_over_epochs"] = results_for_all_epochs["train_noise_pred_loss_SS"]
    json_log["train_noise_pred_loss_AS_over_epochs"] = results_for_all_epochs["train_noise_pred_loss_AS"]
    json_log["test_act_reconst_loss_over_epochs"] = results_for_all_epochs["test_act_reconst_loss"]
    json_log["test_noise_pred_loss_SS_over_epochs"] = results_for_all_epochs["test_noise_pred_loss_SS"]
    json_log["test_noise_pred_loss_AS_over_epochs"] = results_for_all_epochs["test_noise_pred_loss_AS"]
    json_log["mean_memorization_over_epochs"] = results_for_all_epochs["mean_memorization"] 
    json_log["n_memorized_over_epochs"] = results_for_all_epochs["n_memorized"]
    json_log["memorization_fraction_over_epochs"] = results_for_all_epochs["memorization_fraction"]
    json_log["mean_score_avg_over_epochs"] = results_for_all_epochs["mean_score_avg"]
    json_log["num_epochs"] = results_for_all_epochs["num_epochs"]
    json_log["nll_test_over_epochs"] = results_for_all_epochs["nll_test"]
    json_log["nll_train_over_epochs"] = results_for_all_epochs["nll_train"]

    save_json_log(out_path, json_log)
    logger.info("Evaluation complete. Log written to %s", str(out_path))


if __name__ == '__main__':
    main()
