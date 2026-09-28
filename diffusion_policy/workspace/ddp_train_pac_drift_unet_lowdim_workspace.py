if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import copy
import random
import pathlib
import dill
from contextlib import nullcontext

import hydra
from hydra.utils import get_class, instantiate
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf, DictConfig
import wandb
import tqdm
from diffusers.training_utils import EMAModel

from diffusion_policy.common.pytorch_util import (
    dict_apply, optimizer_to, action_sample_diversity, action_reconstruction_loss)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.pac_drift_unet_lowdim_policy import PacDriftUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.checkpoint_util import LastNCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

# Number of independent predict_action() calls (fresh noise/weight-sample each
# time, no policy change needed) used to measure deployed-time single-sample
# action diversity - see the validation block in run() and
# diffusion_policy.common.pytorch_util.action_sample_diversity. Matches
# train_pac_drift_unet_lowdim_workspace.py/eval_ckpts_drift.py's default so
# single-GPU, DDP, and post-hoc numbers are directly comparable.
N_DEPLOYED_DIVERSITY_SAMPLES = 10


class PacLossWrapper(nn.Module):
    """
    Standard PyTorch Module wrapper routing forward() to model.compute_bound() 
    or model.compute_loss(). Ensures DDP autograd hooks and gradient synchronization 
    buckets execute properly.
    """
    def __init__(self, model: nn.Module, cfg: DictConfig, n_bound: int):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.n_bound = n_bound

    def forward(self, batch):
        if self.cfg.training.kl_penalty > 0.0:
            raw_loss, emp_risk_train, kl_train, metrics, _loss_emp_bounded = self.model.compute_bound(
                batch,
                n_bound=self.n_bound,
                objective=self.cfg.training.pac_objective,
                delta=self.cfg.training.delta,
                kl_penalty=self.cfg.training.kl_penalty,
                stochastic=self.cfg.training.stochastic,
                bounded=self.cfg.training.bounded,
                bound_transform=self.cfg.training.bound_transform,
                loss_scale=self.cfg.training.loss_scale,
            )
            return raw_loss, emp_risk_train.detach(), kl_train.detach(), metrics
        else:
            raw_loss, metrics = self.model.compute_loss(batch, stochastic=self.cfg.training.stochastic)
            emp_risk_train = raw_loss.detach()
            kl_train = torch.zeros(1, device=raw_loss.device, dtype=raw_loss.dtype)
            return raw_loss, emp_risk_train, kl_train, metrics


def setup_ddp():
    """
    Initializes NCCL distributed backend and sets local GPU device.
    Returns:
        rank (int): Global process rank.
        local_rank (int): Local GPU device rank.
        world_size (int): Total distributed workers.
        is_distributed (bool): True if running under multi-GPU DDP.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        local_rank = 0
        world_size = 1

    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    return rank, local_rank, world_size, is_distributed


def cleanup_ddp():
    """Cleanly destroys distributed process group upon script termination."""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalars(scalars: dict, world_size: int, device: torch.device) -> dict:
    """Averages a dict of scalar tensors/floats across all workers in deterministic key
    order, using a single collective call (all values stacked into one tensor) instead
    of one all_reduce per entry - this keeps the per-step communication cost constant
    regardless of how many scalars are logged (loss/emp_risk/kl plus whatever metrics
    the policy returns), which matters once world_size grows or ranks span multiple
    nodes."""
    if not scalars:
        return {}

    keys = sorted(scalars.keys())
    if world_size <= 1 or not dist.is_initialized():
        return {
            k: scalars[k].item() if isinstance(scalars[k], torch.Tensor) else float(scalars[k])
            for k in keys
        }

    values = torch.stack([
        scalars[k].detach().to(device=device, dtype=torch.float32).reshape(())
        if isinstance(scalars[k], torch.Tensor)
        else torch.tensor(float(scalars[k]), device=device, dtype=torch.float32)
        for k in keys
    ])
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= world_size
    return dict(zip(keys, values.tolist()))


class TrainPacDriftUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: DictConfig, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # Global base seed for deterministic initial model parameters across ranks
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Underlying policy model
        self.model: PacDriftUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)

        # Underlying EMA policy model
        self.ema_model: PacDriftUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Initialize data-dependent prior deterministically on all ranks
        if cfg.training.data_dependent_prior:
            checkpoint = cfg.training.init_model_path
            with open(checkpoint, 'rb') as f:
                # map_location='cpu': this runs in __init__, before setup_ddp()/
                # torch.cuda.set_device() has assigned this process its GPU. An
                # unmapped load would deserialize CUDA tensors onto whatever the
                # default device is (typically GPU 0) for every rank, causing
                # redundant allocations / OOM risk on GPU 0 as world_size grows.
                init_payload = torch.load(f, pickle_module=dill, map_location='cpu')
            init_cfg = init_payload['cfg']
            cls = hydra.utils.get_class(init_cfg._target_)
            init_workspace = cls(init_cfg, output_dir=output_dir)
            init_workspace.load_payload(init_payload, exclude_keys=['optimizer'], include_keys=None)

            init_model = init_workspace.model.model
            if cfg.training.use_ema:
                init_ema_model = init_workspace.ema_model.model

            self.model.prior_initialization(init_model, cfg.policy.model.rho_post)
            if cfg.training.use_ema:
                self.ema_model.prior_initialization(init_ema_model, cfg.policy.model.rho_post)
            del init_workspace

        # Optimizer references unwrapped model parameters directly
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        rank, local_rank, world_size, is_distributed = setup_ddp()

        try:
            # --- DDP batch-size / LR / schedule-length scaling ---------------------
            # `dataloader.batch_size`, `training.num_updates`, `training.lr_warmup_steps`,
            # `training.checkpoint_every`, and `optimizer.lr` in the config are all
            # specified as their SINGLE-GPU-equivalent values. Under DDP:
            #   - per-GPU batch size is kept EQUAL to the configured value (unlike an
            #     earlier version of this script, which divided it by world_size to
            #     hold the *global* batch fixed). At this workload's batch size, per-step
            #     wall-clock time is already overhead- rather than compute-bound (see the
            #     Diffusion-vs-Flow training-time analysis elsewhere in this project), so
            #     shrinking the per-GPU batch gives diminishing/near-zero speedup while
            #     still paying gradient all-reduce cost. Keeping per-GPU batch size fixed
            #     instead reproduces single-GPU per-step wall-clock time exactly, and the
            #     speedup comes entirely from needing fewer total steps (below).
            #   - num_updates is divided by world_size, so total data throughput
            #     (num_updates * batch_size * world_size) matches the single-GPU
            #     baseline - this is what actually gives the near-linear wall-clock
            #     speedup.
            #   - lr_warmup_steps and checkpoint_every are divided by world_size too, so
            #     warmup stays the same FRACTION of (now-shorter) training and the
            #     checkpoint_last_N retention window covers the same fraction of
            #     training, rather than a shrunk one.
            # No rollout (environment/simulator evaluation) ever runs during DDP
            # training - unlike train_pac_drift_unet_lowdim_workspace.py, which is
            # the only single-GPU workspace in this project with an ACTIVE
            # (uncommented) rollout block. Checkpoints are evaluated via rollout
            # post-hoc instead, through eval_ckpts_drift.py. Validation-set
            # evaluation (no simulator involved) DOES run, but only on rank 0 - see
            # val_dataloader and the per-step validation block below.
            #   - optimizer.lr is multiplied by world_size**ddp_lr_scale_power (linear
            #     scaling rule at the default power=1.0) to compensate for the larger
            #     effective global batch (batch_size * world_size) and preserve
            #     single-GPU-equivalent optimization dynamics. Set
            #     training.ddp_lr_scale_power=0.5 for sqrt scaling if linear scaling
            #     proves unstable for a given task.
            # This block runs before checkpoint restoration below: on a fresh run it
            # sets the optimizer's initial LR; on a resumed run, load_checkpoint()'s
            # optimizer.load_state_dict() immediately overwrites it with the exact
            # (already-scaled, already-annealed) historical LR, so there is no
            # double-scaling risk either way. If you resume with a DIFFERENT
            # world_size than the original run used, the effective schedule length
            # changes accordingly - this is ordinary DDP behavior, not specific to
            # this codebase.
            if is_distributed:
                lr_scale_power = float(OmegaConf.select(cfg, "training.ddp_lr_scale_power", default=1.0))
                lr_scale = world_size ** lr_scale_power
                for group in self.optimizer.param_groups:
                    group['lr'] = group['lr'] * lr_scale
                if rank == 0:
                    print(f"DDP scaling: world_size={world_size}, per-GPU batch size unchanged, "
                          f"optimizer.lr *= {world_size}**{lr_scale_power} = {lr_scale:.4g}")

            # Synchronize output directory across workers by setting the backing attribute _output_dir
            if is_distributed:
                output_dir_sync = [str(self.output_dir)] if rank == 0 else [None]
                dist.broadcast_object_list(output_dir_sync, src=0)
                self._output_dir = output_dir_sync[0]

            # Checkpoint restoration across all ranks
            if cfg.training.resume:
                if cfg.training.desired_ckpt_path is not None:
                    desired_ckpt_path = cfg.training.desired_ckpt_path
                    if not os.path.isfile(desired_ckpt_path):
                        raise ValueError(f"No such checkpoint file: {desired_ckpt_path}")
                    if rank == 0:
                        print("Resuming from checkpoint:", desired_ckpt_path)
                    self.load_checkpoint(path=desired_ckpt_path)
                else:
                    latest_ckpt_path = self.get_checkpoint_path()
                    if latest_ckpt_path.is_file():
                        if rank == 0:
                            print("Resuming from checkpoint:", latest_ckpt_path)
                        self.load_checkpoint(path=latest_ckpt_path)
                    else:
                        if rank == 0:
                            print("Starting training from scratch.")
            else:
                if rank == 0:
                    print("Starting training from scratch.")

            # Barrier to guarantee all ranks finish loading checkpoints before proceeding
            if is_distributed:
                dist.barrier()

            # Decorrelate the main-process RNG across ranks now that rank is known.
            # Model weights stay synchronized regardless (identical seed at
            # construction, plus checkpoint loading above / DDP's parameter broadcast
            # at wrap time below), so offsetting the seed here only affects stochastic
            # sampling during training (Bayesian posterior weight sampling,
            # reparameterization noise in compute_loss/compute_bound). Without this,
            # every rank starts from the same RNG state and consumes it in lockstep
            # (equal per-GPU batch sizes), so all ranks draw the exact same "random"
            # samples each step - just applied to different data - which quietly
            # throws away the Monte-Carlo diversity extra GPUs should buy you.
            rank_seed = cfg.training.seed + rank
            torch.manual_seed(rank_seed)
            torch.cuda.manual_seed_all(rank_seed)
            np.random.seed(rank_seed)
            random.seed(rank_seed)

            # Dataset configuration
            dataset: BaseLowdimDataset = hydra.utils.instantiate(cfg.task.dataset)
            assert isinstance(dataset, BaseLowdimDataset)
            # Computed once (matches train_pac_drift_unet_lowdim_workspace.py's
            # n_bound = len(train_dataloader.dataset)) - the total PAC-Bayes bound
            # sample count, independent of how DistributedSampler shards the
            # dataset across ranks for actual training.
            n_bound = len(dataset)

            # Worker seed initialization helper
            def worker_init_fn(worker_id):
                worker_seed = cfg.training.seed + rank * 1000 + worker_id
                np.random.seed(worker_seed)
                random.seed(worker_seed)
                torch.manual_seed(worker_seed)

            # DataLoader & DistributedSampler configuration
            train_sampler = None
            if is_distributed:
                dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
                dataloader_cfg.pop('sampler', None)
                dataloader_cfg.pop('shuffle', None)
                # batch_size is used as-is, per GPU (see the DDP scaling comment in
                # run() above) - the effective global batch size is
                # batch_size * world_size.

                train_sampler = DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=True,
                    seed=cfg.training.seed,
                    drop_last=False
                )
                train_dataloader = DataLoader(
                    dataset,
                    sampler=train_sampler,
                    worker_init_fn=worker_init_fn,
                    **dataloader_cfg
                )
            else:
                train_dataloader = DataLoader(
                    dataset,
                    worker_init_fn=worker_init_fn,
                    **cfg.dataloader
                )

            # Normalizer configuration (deterministic across ranks)
            normalizer = dataset.get_normalizer()
            if rank == 0:
                print("Training dataset size: ", len(dataset))
                if is_distributed:
                    print(f"Per-GPU batch size: {dataloader_cfg['batch_size']} "
                          f"(effective global batch size: {dataloader_cfg['batch_size'] * world_size})")

            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema and self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)

            # Validation dataset - built and evaluated on rank 0 only (empty
            # whenever task.dataset.val_ratio=0.0, matching every real DDP
            # comparison run so far). Deliberately NOT sharded via
            # DistributedSampler - self.model's weights are already
            # DDP-synchronized across ranks, so a full, unsharded rank-0-only
            # pass gives the same result any other rank would.
            val_dataloader = None
            if rank == 0:
                val_dataset = dataset.get_validation_dataset()
                val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
                print("Validation dataset size: ", len(val_dataset))

            # Safe conversion for scientific notation string parameters
            lr_warmup_steps = int(float(cfg.training.lr_warmup_steps))
            num_updates = int(float(cfg.training.num_updates))
            checkpoint_every = int(float(cfg.training.checkpoint_every))
            val_every = int(float(cfg.training.val_every))

            # Scale schedule length and checkpoint cadence to preserve single-GPU
            # equivalent total data throughput and checkpoint-window coverage under
            # the larger effective global batch size - see the DDP scaling comment
            # near the top of run().
            if is_distributed:
                num_updates = max(1, round(num_updates / world_size))
                lr_warmup_steps = max(1, round(lr_warmup_steps / world_size))
                checkpoint_every = max(1, round(checkpoint_every / world_size))
                val_every = max(1, round(val_every / world_size))
                if rank == 0:
                    print(f"DDP scaling: num_updates={num_updates}, "
                          f"lr_warmup_steps={lr_warmup_steps}, checkpoint_every={checkpoint_every}, "
                          f"val_every={val_every} "
                          f"(single-GPU-equivalent config values divided by world_size={world_size})")

            # Debug configuration overrides (applied before the LR scheduler is built
            # below, so the schedule length actually matches the shortened debug run)
            if cfg.training.debug:
                num_updates = 1000
                max_train_steps = 100
                checkpoint_every = 100
                val_every = 100
            else:
                max_train_steps = cfg.training.max_train_steps

            # Learning rate scheduler
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=lr_warmup_steps,
                num_training_steps=num_updates,
                last_epoch=self.global_step - 1
            )

            # EMA initialization on rank 0
            ema: EMAModel = None
            if cfg.training.use_ema and rank == 0:
                ema = hydra.utils.instantiate(
                    cfg.ema,
                    model=self.ema_model
                )
                if hasattr(ema, 'optimization_step'):
                    ema.optimization_step = self.global_step

            # Logging and Checkpoint Managers on rank 0
            wandb_run = None
            lastN_manager = None
            if rank == 0:
                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **cfg.logging
                )
                wandb.config.update({"output_dir": self.output_dir})

                lastN_manager = LastNCheckpointManager(
                    save_dir=os.path.join(self.output_dir, "checkpoints"),
                    **cfg.checkpoint_last_N.topk
                )

            # Device placement
            if is_distributed:
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(device)
            else:
                device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
                if device.type == "cuda" and device.index is not None:
                    torch.cuda.set_device(device)

            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)
            optimizer_to(self.optimizer, device)

            # Wrap in LossWrapper (with find_unused_parameters=True to support PAC prior/posterior parameter routing)
            loss_module = PacLossWrapper(self.model, cfg, n_bound=n_bound)
            if is_distributed:
                ddp_loss_model = DDP(
                    loss_module,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    find_unused_parameters=True
                )
            else:
                ddp_loss_model = loss_module

            # Training loop
            log_path = os.path.join(self.output_dir, 'logs.json.txt')
            json_logger_ctx = JsonLogger(log_path) if rank == 0 else nullcontext()

            with json_logger_ctx as json_logger:
                while self.global_step < num_updates:
                    if train_sampler is not None:
                        train_sampler.set_epoch(self.epoch)

                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training step {self.global_step}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                        disable=(rank != 0)
                    ) as tepoch:

                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                            # Forward through DDP wrapper invokes PAC loss / bound computation
                            raw_loss, emp_risk_train, kl_train, metrics = ddp_loss_model(batch)

                            # Backward on local loss initiates gradient all-reduce across all GPUs
                            raw_loss.backward()

                            # Diagnostic only (see train_pac_drift_unet_lowdim_workspace.py for
                            # the max_norm=inf rationale: no clipping is actually applied yet).
                            # Gradients are already all-reduced/averaged by DDP's backward hooks
                            # by the time backward() returns, so this is naturally identical
                            # across ranks without needing reduce_scalars. Added to match
                            # ddp_train_drift_unet_lowdim_workspace.py: this training path has a
                            # documented history of a real vanishing-empirical-risk-gradient bug
                            # (see the git history around the PAC-Bayes bound's gradient-norm
                            # diagnostics), so DDP runs need the same visibility.
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=float('inf'))

                            # Diagnostic only: gradient-norm split between the empirical-risk
                            # path (loss_emp_bounded) and the KL path (kl_train) - matches
                            # train_pac_drift_unet_lowdim_workspace.py, whose comment explains
                            # this exact split is what the vanishing-empirical-risk-gradient bug
                            # referenced above was originally diagnosed with. Commented out for
                            # now (computationally expensive under DDP): unlike single-GPU (which
                            # reuses the same forward pass that also produces raw_loss), this
                            # requires a SEPARATE, non-DDP-tracked forward pass through
                            # self.model.compute_bound() directly (calling torch.autograd.grad()
                            # on tensors from ddp_loss_model's own forward would fire DDP's
                            # gradient-reduction hooks a second time, corrupting or crashing the
                            # real gradient all-reduce) - confirmed via smoke test to roughly
                            # double per-step compute cost (an entire extra forward pass every
                            # step, on rank 0). Uncomment if this specific bug needs
                            # re-diagnosing under DDP.
                            emp_risk_grad_norm_val = 0.0
                            kl_grad_norm_val = 0.0
                            # if rank == 0 and cfg.training.kl_penalty > 0.0:
                            #     _, _, kl_train_diag, _, loss_emp_bounded_diag = self.model.compute_bound(
                            #         batch,
                            #         n_bound=n_bound,
                            #         objective=cfg.training.pac_objective,
                            #         delta=cfg.training.delta,
                            #         kl_penalty=cfg.training.kl_penalty,
                            #         stochastic=cfg.training.stochastic,
                            #         bounded=cfg.training.bounded,
                            #         bound_transform=cfg.training.bound_transform,
                            #         loss_scale=cfg.training.loss_scale,
                            #     )
                            #     diag_params = [p for p in self.model.parameters() if p.requires_grad]
                            #     zero = torch.zeros((), device=device)
                            #     emp_grads = torch.autograd.grad(
                            #         loss_emp_bounded_diag, diag_params, retain_graph=True, allow_unused=True)
                            #     kl_grads = torch.autograd.grad(
                            #         kl_train_diag, diag_params, retain_graph=True, allow_unused=True)
                            #     emp_risk_grad_norm_val = torch.sqrt(sum(
                            #         (g.detach().pow(2).sum() for g in emp_grads if g is not None), zero)).item()
                            #     kl_grad_norm_val = torch.sqrt(sum(
                            #         (g.detach().pow(2).sum() for g in kl_grads if g is not None), zero)).item()

                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                            # Update EMA on rank 0 from synchronized model weights
                            if cfg.training.use_ema and rank == 0 and ema is not None:
                                ema.step(self.model)

                            # Increment global step before checkpointing
                            is_first_step = (self.global_step == 0)
                            self.global_step += 1
                            current_step = self.global_step
                            current_lr = lr_scheduler.get_last_lr()[0]

                            # Reduce loss, empirical risk, KL, and metrics across all ranks
                            # for logging in a single collective call (see reduce_scalars)
                            scalars_to_reduce = {
                                'train_loss (pac_bayes bound)': raw_loss,
                                'emp_risk_train': emp_risk_train,
                                'kl_train': kl_train,
                                **metrics,
                            }
                            reduced_scalars = reduce_scalars(scalars_to_reduce, world_size, device)

                            step_log = {
                                **reduced_scalars,
                                'grad_norm': grad_norm.item(),
                                # Effective parameter-update magnitude this step (grad_norm
                                # alone doesn't say how far the parameters actually moved).
                                'grad_step_size': grad_norm.item() * current_lr,
                                # How much of the total gradient (grad_norm above) comes from
                                # the empirical-risk path vs. the KL path - see the diagnostic
                                # computed above. 0.0 when kl_penalty<=0 (no PAC bound, no split).
                                'emp_risk_grad_norm': emp_risk_grad_norm_val,
                                'kl_grad_norm': kl_grad_norm_val,
                                'global_step': current_step,
                                'lr': current_lr,
                                'epoch': self.epoch,
                            }

                            # Diagnostic only, computed periodically (val_every, scaled like
                            # checkpoint_every above): total parameter norm, and the relative
                            # update size (grad_step_size / param_norm) - a more
                            # scale-invariant instability indicator than grad_norm alone.
                            # Independent of whether a validation set exists, so not gated on
                            # val_dataloader (mirrors train_pac_drift_unet_lowdim_workspace.py).
                            # Cheap (a single sqrt-sum-of-squares pass over parameters, no
                            # forward pass) and rank-invariant (self.model.parameters() are
                            # already DDP-synchronized) - like grad_norm above, no
                            # reduce_scalars needed, safe to compute on rank 0 only.
                            if rank == 0 and ((current_step % val_every) == 0 or is_first_step):
                                param_norm = torch.sqrt(
                                    sum((p.detach() ** 2).sum() for p in self.model.parameters())
                                ).item()
                                step_log['param_norm'] = param_norm
                                step_log['relative_update_size'] = step_log['grad_step_size'] / (param_norm + 1e-12)

                                # Real validation-set evaluation - excluded from DDP training
                                # by default (see the DDP scaling comment near the top of
                                # run()), but active here whenever val_dataloader actually has
                                # data (val_ratio > 0), on rank 0 only. Mirrors
                                # train_pac_drift_unet_lowdim_workspace.py's noise-prediction-loss
                                # and deployed-time diversity/reconstruction diagnostics exactly,
                                # including running both deterministic and stochastic weight-
                                # sampling variants. Costs exactly what single-GPU already pays
                                # for this same computation at this same val_every cadence -
                                # confirmed cheap once decoupled from the (now-removed)
                                # emp_risk_grad_norm/kl_grad_norm per-step diagnostic, which was
                                # what actually made an earlier smoke test intractable.
                                if val_dataloader is not None and len(val_dataloader) > 0:
                                    eval_policy = self.ema_model if cfg.training.use_ema else self.model
                                    eval_policy.eval()
                                    for val_stochastic in (False, True):
                                        suffix = 'stochastic' if val_stochastic else 'deterministic'
                                        with torch.no_grad():
                                            val_losses = []
                                            val_metric_sums = {}
                                            n_samples_total = 0
                                            for v_idx, vbatch in enumerate(val_dataloader):
                                                n_samples = len(vbatch["obs"])
                                                n_samples_total += n_samples
                                                vbatch = dict_apply(vbatch, lambda x: x.to(device, non_blocking=True))
                                                val_loss, val_metrics = eval_policy.compute_loss(vbatch, stochastic=val_stochastic)
                                                val_losses.append(val_loss.item() * n_samples)
                                                for k, v in val_metrics.items():
                                                    val_metric_sums[k] = val_metric_sums.get(k, 0.0) + v * n_samples

                                                if v_idx == 0:
                                                    obs_dict = {"obs": vbatch["obs"]}
                                                    action_samples = torch.stack(
                                                        [eval_policy.predict_action(obs_dict, stochastic=val_stochastic)["action"]
                                                         for _ in range(N_DEPLOYED_DIVERSITY_SAMPLES)],
                                                        dim=1,
                                                    )
                                                    step_log[f"test_deployed_action_diversity_{suffix}"] = \
                                                        action_sample_diversity(action_samples)
                                                    start = eval_policy.n_obs_steps - 1
                                                    end = start + eval_policy.n_action_steps
                                                    reference_action = vbatch["action"][:, start:end]
                                                    step_log[f"test_deployed_reconstruction_loss_{suffix}"] = \
                                                        action_reconstruction_loss(action_samples, reference_action)

                                                if (cfg.training.max_val_steps is not None) and v_idx >= (cfg.training.max_val_steps - 1):
                                                    break
                                            if len(val_losses) > 0:
                                                step_log[f'test_loss_{suffix}'] = np.sum(val_losses) / n_samples_total
                                                for k, total in val_metric_sums.items():
                                                    step_log[f'test_{k}_{suffix}'] = total / n_samples_total
                                    eval_policy.train()

                            if rank == 0:
                                tepoch.set_postfix(loss=reduced_scalars['train_loss (pac_bayes bound)'], refresh=False)

                            # Checkpointing (Last-N)
                            if (current_step % checkpoint_every) == 0:
                                if rank == 0:
                                    if cfg.checkpoint_last_N.get('save_last_ckpt', False):
                                        self.save_checkpoint(use_thread=False)
                                    if cfg.checkpoint_last_N.get('save_last_snapshot', False):
                                        self.save_snapshot()
                                    if lastN_manager is not None:
                                        lastN_ckpt_path = lastN_manager.get_ckpt_path(step_log)
                                        if lastN_ckpt_path is not None:
                                            self.save_checkpoint(path=lastN_ckpt_path, use_thread=False)
                                if is_distributed:
                                    dist.barrier()

                            # Rank 0 telemetry logging
                            if rank == 0:
                                if wandb_run is not None:
                                    wandb_run.log(step_log, step=current_step)
                                if json_logger is not None:
                                    json_logger.log(step_log)

                            # Early stopping limits per epoch or run
                            if (max_train_steps is not None) and batch_idx >= (max_train_steps - 1):
                                break

                            if self.global_step >= num_updates:
                                break

                    self.epoch += 1

            if rank == 0 and wandb_run is not None:
                wandb_run.finish()

        finally:
            cleanup_ddp()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem
)
def main(cfg):
    output_dir = os.environ.get("OUTPUT_DIR", None)
    workspace = TrainPacDriftUnetLowdimWorkspace(cfg, output_dir=output_dir)
    workspace.run()


if __name__ == "__main__":
    main()