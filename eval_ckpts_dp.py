"""
Evaluate Diffusion Policy checkpoints.

Usage:
    python eval_ckpts_dp.py --ckpts_dir data/outputs/.../checkpoints -o data/outputs/.../eval
    # add flatness metrics (written to flatness_log_<timestamp>.json next to eval_log_*.json):
    python eval_ckpts_dp.py -c .../checkpoints --flatness
    # flatness only (no rollouts, no validation loss / NLL):
    python eval_ckpts_dp.py -c .../checkpoints --flatness_only

Flatness metrics (diffusion_policy/common/flatness_monitor.py) are computed on the same
evaluated network (EMA if use_ema), on fixed batches drawn exactly as during training: same
seed, the training split (`dataset`) as "train" and the validation split as "test" (if any).
For PAC-DP checkpoints the noise metrics are added. Their "initial" sigma is the noise at the
start of posterior training:
  - data-dependent prior (training.data_dependent_prior=True): posterior training starts at the
    trained prior (Q = P), so the initial sigma is read exactly from the checkpoint's prior;
  - otherwise: it follows from the config, policy.model.rho_post or, if set,
    policy.model.post_sigma_scale times each layer's fan-in std; it is obtained by building an
    untrained model from that config (the same initialisation code as in training).
--sigma_param exp is needed for checkpoints trained between 2026-08-07 (71a1470) and
2026-09-15 (959fc5a), when the code used sigma = exp(rho) instead of softplus(rho).
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
from diffusion_policy.common.flatness_monitor import flatness_report, make_fixed_batches, NoiseContribution
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


def ckpt_key(filename: str, n: int) -> str:
    """Log key of a checkpoint: model_at_step_001234 for 'step=...' files, and
    model_at_epoch_1234 (as in older eval logs) for 'epoch=...' files."""
    return f"model_at_epoch_{n}" if filename.startswith("epoch=") else f"model_at_step_{n:06d}"


def parse_step_from_filename(filename: str) -> Optional[int]:
    """Parse step number from checkpoint filename like 'step=0010-...ckpt' (or the epoch number
    from 'epoch=0010.ckpt', used by older runs). Returns None if no pattern is found or the file
    is 'latest.ckpt'.
    """
    if filename == "latest.ckpt":
        return None
    # pattern: 'step=1234' or 'epoch=1234'
    try:
        parts = filename.split("step=" if "step=" in filename else "epoch=")
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


def drop_stale_null_model_kwargs(cfg: DictConfig) -> List[str]:
    """Backward compatibility for checkpoints saved before an argument was removed from the
    network class (e.g. ConditionalUnet1D's local_cond_dim, removed in f899039): drop
    policy.model arguments that are null in the saved cfg AND no longer accepted by the class.
    A null argument means the feature was unused, so no weights depend on it."""
    import inspect
    from omegaconf import open_dict
    model_cfg = OmegaConf.select(cfg, "policy.model")
    if model_cfg is None or "_target_" not in model_cfg:
        return []
    params = inspect.signature(hydra.utils.get_class(model_cfg._target_).__init__).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return []
    stale = [k for k in model_cfg if k != "_target_" and k not in params and model_cfg[k] is None]
    with open_dict(cfg):
        for k in stale:
            del cfg.policy.model[k]
    return stale


def instantiate_workspace(cfg: DictConfig, output_dir: Path) -> BaseWorkspace:
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


def compute_flatness(policy, flat_batches: Dict[str, list], noise_monitor, n_probes: int) -> Dict:
    """Flatness metrics per split ("train", "test"); noise metrics too for PAC-DP."""
    out = {}
    for split, batches in flat_batches.items():
        res = flatness_report(policy, batches, prefix="", n_probes=n_probes)
        if noise_monitor is not None:
            res.update(noise_monitor.report(policy, batches, prefix="noise/", n_probes=n_probes))
        out[split] = res
    return out


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
@click.option("--flatness", is_flag=True, help="Also compute flatness metrics (flatness_log_*.json)")
@click.option("--flatness_only", is_flag=True, help="Only compute flatness metrics (no rollouts, loss or NLL)")
@click.option("--flat_n_batches", type=int, default=None, help="Fixed batches per split (default: cfg or 4)")
@click.option("--flat_batch_size", type=int, default=None, help="Batch size (default: cfg or 64)")
@click.option("--flat_probes", type=int, default=None, help="Hutchinson probes (default: cfg or 32)")
@click.option("--sigma_param", type=click.Choice(["softplus", "exp"]), default="softplus",
              help="sigma(rho) used when the PAC-DP checkpoints were trained "
                   "(exp for runs trained 2026-08-07 .. 2026-09-14)")
@click.option("--last_n", type=int, default=None,
              help="Only evaluate the last N step checkpoints (by step; default: all)")
def main(ckpts_dir: Path, output_dir: Optional[Path], device: str, override: Tuple[str, ...], delete_ckpts: bool = False,
         flatness: bool = False, flatness_only: bool = False, flat_n_batches: Optional[int] = None,
         flat_batch_size: Optional[int] = None, flat_probes: Optional[int] = None,
         sigma_param: str = "softplus", last_n: Optional[int] = None):
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
    dropped = drop_stale_null_model_kwargs(cfg)
    if dropped:
        logger.info("Dropped null model arguments no longer accepted by the current code: %s", dropped)
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

    # Environment runner (not needed when only flatness is computed)
    env_runner = None if flatness_only else hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output_dir))

    # Flatness setup: the same fixed batches as the training-time monitor (same seed and,
    # when the checkpoint's cfg has them, the same training.flat_* settings).
    do_flatness = flatness or flatness_only
    if do_flatness:
        pick = lambda cli, key, default: cli if cli is not None else int(
            OmegaConf.select(cfg, f"training.{key}", default=default))
        flat_settings = dict(n_batches=pick(flat_n_batches, "flat_n_batches", 4),
                             batch_size=pick(flat_batch_size, "flat_batch_size", 64),
                             probes=pick(flat_probes, "flat_probes", 32))
        kw = dict(n_batches=flat_settings["n_batches"], batch_size=flat_settings["batch_size"],
                  seed=0, device=device_obj)
        flat_batches = {"train": make_fixed_batches(dataset, **kw)}
        if len(val_dataset) > 0:
            flat_batches["test"] = make_fixed_batches(val_dataset, **kw)
        else:
            logger.warning("Validation set is empty: flatness is computed on train batches only.")

        # Initial noise for PAC-DP (see module docstring): from the checkpoint's own prior for
        # a data-dependent prior (built per checkpoint below), else from the config's
        # rho_post / post_sigma_scale via an untrained model built from that config.
        sigma_fn = (lambda g: torch.exp(g.rho)) if sigma_param == "exp" else (lambda g: torch.nn.functional.softplus(g.rho))
        ddp_prior = bool(OmegaConf.select(cfg, "training.data_dependent_prior", default=False))
        fresh = instantiate_workspace(cfg, output_dir)
        fresh_policy = fresh.ema_model if cfg.training.use_ema else fresh.model
        noise_monitor = None
        is_bayes = bool(_is_stochastic_policy(fresh_policy) and NoiseContribution._pairs(fresh_policy))
        if is_bayes and not ddp_prior:
            noise_monitor = NoiseContribution(fresh_policy, sigma_fn=sigma_fn)
        del fresh, fresh_policy
        free_cuda_memory()
        if is_bayes:
            initial_sigma_source = (
                "checkpoint prior (data-dependent prior: posterior training starts at Q = P)" if ddp_prior else
                f"config: policy.model.rho_post={OmegaConf.select(cfg, 'policy.model.rho_post', default=None)}, "
                f"post_sigma_scale={OmegaConf.select(cfg, 'policy.model.post_sigma_scale', default=None)} "
                f"(sigma = {sigma_param}(rho); post_sigma_scale x fan-in std overrides rho_post when set)")
        else:
            initial_sigma_source = None

        flat_path = output_dir / f"flatness_log_{timestamp}.json"
        flat_log = {
            "eval_config": {
                "overrides": list(override),
                "model_type": "PAC-DP (Bayesian)" if is_bayes else "DP (deterministic)",
                "sigma_param": sigma_param if is_bayes else None,
                "evaluated_network": "ema_model" if cfg.training.use_ema else "model",
                "splits": list(flat_batches.keys()),
                **flat_settings,
                "initial_sigma_source": initial_sigma_source,
            },
        }

    # Prepare containers for results
    json_log = {
        # Records which eval invocation this log is (e.g. eval.stochastic=True
        # vs. =False, or policy.eta=1.0 vs. =0.0) - the two halves of a
        # comparison pair otherwise look identical except for the output
        # filename's timestamp, which is easy to mix up downstream.
        "eval_config": {
            "overrides": list(override),
            "eval_stochastic": bool(OmegaConf.select(cfg, "eval.stochastic", default=False)),
            "policy_eta": OmegaConf.select(cfg, "policy.eta", default=None),
        },
    }
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
    # Steps whose 0.0 came from a crash with no previous checkpoint to carry
    # forward from - the ONLY case the zero-replacement safety net below
    # should touch. A genuine 0.0 (every episode actually failed) must stay
    # 0.0: replacing it with the mean of the other checkpoints would inflate
    # the reported success rate, sometimes drastically, on tasks/checkpoints
    # that really did fail.
    crash_no_prior_steps: set = set()
    last_success_rate: Optional[float] = None
    last_success_step: Optional[int] = None

    # Iterate over checkpoints (optionally only the last N, ordered by step)
    step_ckpt_files = sorted((p for p in all_ckpt_files if parse_step_from_filename(p.name) is not None),
                             key=lambda p: parse_step_from_filename(p.name))
    if last_n is not None:
        step_ckpt_files = step_ckpt_files[-last_n:]
    keys_by_step: Dict[int, str] = {}
    for ckpt_path in step_ckpt_files:
        step = parse_step_from_filename(ckpt_path.name)
        keys_by_step[step] = ckpt_key(ckpt_path.name, step)

        logger.info("Evaluating checkpoint %s (step %d)", ckpt_path.name, step)

        # Load checkpoint payload
        payload = load_checkpoint_payload(ckpt_path)
        if "cfg" not in payload:
            logger.warning("No 'cfg' in payload for %s, skipping", ckpt_path.name)
            continue

        # Load workspace and model
        workspace = instantiate_workspace(cfg, output_dir)
        # The raw model's state is skipped only when the EMA copy is evaluated - with
        # use_ema=False the evaluated network IS workspace.model and must be loaded.
        exclude = ["optimizer", "model"] if cfg.training.use_ema else ["optimizer"]
        workspace.load_payload(payload, exclude_keys=exclude, include_keys=None)
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.to(device_obj)
        policy.eval()

        if do_flatness:
            flat_start = time.perf_counter()
            if is_bayes and ddp_prior:     # initial noise = this checkpoint's (fixed) prior
                noise_monitor = NoiseContribution(policy, use_prior=True, sigma_fn=sigma_fn)
            flat_log[keys_by_step[step]] = compute_flatness(
                policy, flat_batches, noise_monitor, flat_settings["probes"])
            flat_log[keys_by_step[step]]["time_sec"] = time.perf_counter() - flat_start
            save_json_log(flat_path, flat_log)
            logger.info("Flatness for step %d written to %s", step, flat_path)

        if flatness_only:
            del policy, workspace, payload
            free_cuda_memory()
            continue

        try:
            # Run environment evaluation
            eval_start = time.perf_counter()
            runner_log, success_rate = run_env_runner(env_runner, policy, cfg)
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
            if last_success_rate is None:
                crash_no_prior_steps.add(step)
            json_log[keys_by_step[step]] = {
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
        key = keys_by_step[step]
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
        # Only a checkpoint that crashed with NO prior checkpoint to carry
        # forward from (its 0.0 is purely a "we have no idea" placeholder,
        # not a measurement) gets replaced - never a genuine 0.0, which is a
        # real measurement (every episode failed) and would otherwise get
        # silently inflated by averaging it away.
        success_rates = step_results["success_rates"]
        steps = step_results["steps"]
        replace_idx = [i for i, s in enumerate(steps) if s in crash_no_prior_steps]
        real_rates = [r for i, r in enumerate(success_rates) if i not in replace_idx]
        if replace_idx and real_rates:
            replacement = float(np.mean(real_rates))
            for i in replace_idx:
                success_rates[i] = replacement
                key = keys_by_step[steps[i]]
                if key in json_log:
                    json_log[key]["success_rate"] = replacement
                    json_log[key]["success_rate_zero_replaced_with_mean"] = True
            sum_success_rates = float(np.sum(success_rates))
            logger.warning(
                "%d checkpoint(s) crashed with no prior checkpoint to carry "
                "forward from - replaced their placeholder 0.0 with the mean "
                "of the other %d evaluated checkpoint(s), %.4f.",
                len(replace_idx), len(real_rates), replacement,
            )

        json_log["loss_val"] = np.mean(loss_val)
        json_log["nll_val"] = np.mean(nll_val)
        json_log["mean_scores"] = step_results["success_rates"]
        json_log["num_steps"] = step_results["steps"]
        json_log[f"mean_success_rate_last_{num_evaluated}_checkpoints"] = sum_success_rates / num_evaluated
        if eval_times:
            json_log["eval_times_sec"] = eval_times
            json_log[f"mean_eval_time_sec_last_{len(eval_times)}_checkpoints"] = float(np.mean(eval_times))
    elif not flatness_only:
        logger.warning("No valid checkpoints found.")

    if failed_checkpoints:
        json_log["failed_checkpoints"] = failed_checkpoints
        logger.warning(
            "%d checkpoint(s) skipped due to MuJoCo instability: %s",
            len(failed_checkpoints), [f["ckpt"] for f in failed_checkpoints],
        )

    if not flatness_only:
        save_json_log(out_path, json_log)
        logger.info("Evaluation complete. Log written to %s", out_path)
    if do_flatness:
        logger.info("Flatness log written to %s", flat_path)

    # Delete checkpoints after ALL evaluations are complete
    if delete_ckpts:
        logger.info("All evaluations completed. Deleting checkpoints...")

        for ckpt_path in all_ckpt_files:
            delete_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()
