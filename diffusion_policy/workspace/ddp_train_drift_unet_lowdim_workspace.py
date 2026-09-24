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
from contextlib import nullcontext

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import wandb
import tqdm
from diffusers.training_utils import EMAModel

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.drift_unet_lowdim_policy import DriftUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.checkpoint_util import LastNCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class DriftLossWrapper(nn.Module):
    """
    Standard PyTorch Module wrapper routing forward() to model.compute_loss().
    Ensures DDP autograd hooks and gradient synchronization buckets execute
    properly (mirrors PacLossWrapper in ddp_train_pac_drift_unet_lowdim_workspace.py).

    DDP is still constructed with find_unused_parameters=True below, even though
    every one of self.model's parameters is genuinely exercised in every forward
    call (no prior/posterior routing like the PAC variant): empirically, without
    it, DDP raises "Expected to have finished reduction in the prior iteration
    before starting a new one" on this ConditionalUnet1D (down/mid/up modules
    iterated via nn.ModuleList in a Python loop) - DDP's default (fast) usage
    detection relies on autograd-graph reachability analysis at trace time and
    can apparently mis-detect reachability for this architecture shape even when
    everything is in fact used, rather than there being an actual dead branch.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.compute_loss(batch)


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
    regardless of how many scalars are logged (train_loss plus whatever drift_loss
    diagnostics the policy returns), which matters once world_size grows or ranks span
    multiple nodes."""
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


class TrainDriftUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # Global base seed for deterministic initial model parameters across ranks
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Underlying policy model
        self.model: DriftUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)

        # Underlying EMA policy model
        self.ema_model: DriftUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

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
            #   - per-GPU batch size is kept EQUAL to the configured value. At this
            #     workload's batch size, per-step wall-clock time is already overhead-
            #     rather than compute-bound (see the Diffusion-vs-Flow training-time
            #     analysis elsewhere in this project), so shrinking the per-GPU batch
            #     gives diminishing/near-zero speedup while still paying gradient
            #     all-reduce cost. Keeping per-GPU batch size fixed instead reproduces
            #     single-GPU per-step wall-clock time exactly, and the speedup comes
            #     entirely from needing fewer total steps (below).
            #   - num_updates is divided by world_size, so total data throughput
            #     (num_updates * batch_size * world_size) matches the single-GPU
            #     baseline - this is what actually gives the near-linear wall-clock
            #     speedup.
            #   - lr_warmup_steps and checkpoint_every are divided by world_size too,
            #     so warmup and checkpoint-retention coverage both stay the same
            #     FRACTION of (now-shorter) training, rather than a shrunk one.
            # No validation/evaluation is performed during DDP training at all (see
            # ddp_train_pac_drift_unet_lowdim_workspace.py for the same design) -
            # checkpoints are evaluated after training completes instead.
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
            # sampling during training (the G generated-candidate noise draws in
            # DriftUnetLowdimPolicy.compute_loss). Without this, every rank starts
            # from the same RNG state and consumes it in lockstep, so all ranks draw
            # the exact same "random" candidates each step - just applied to
            # different data - which quietly throws away the Monte-Carlo diversity
            # extra GPUs should buy you.
            rank_seed = cfg.training.seed + rank
            torch.manual_seed(rank_seed)
            torch.cuda.manual_seed_all(rank_seed)
            np.random.seed(rank_seed)
            random.seed(rank_seed)

            # Dataset configuration
            dataset: BaseLowdimDataset = hydra.utils.instantiate(cfg.task.dataset)
            assert isinstance(dataset, BaseLowdimDataset)

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

            if rank == 0:
                print("Training dataset size: ", len(dataset))
                if is_distributed:
                    print(f"Per-GPU batch size: {dataloader_cfg['batch_size']} "
                          f"(effective global batch size: {dataloader_cfg['batch_size'] * world_size})")

            # Normalizer configuration (deterministic across ranks)
            normalizer = dataset.get_normalizer()
            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema and self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)

            # Safe conversion for scientific notation string parameters
            lr_warmup_steps = int(float(cfg.training.lr_warmup_steps))
            num_updates = int(float(cfg.training.num_updates))
            checkpoint_every = int(float(cfg.training.checkpoint_every))

            # Scale schedule length and checkpoint cadence to preserve single-GPU
            # equivalent total data throughput and checkpoint-window coverage under
            # the larger effective global batch size - see the DDP scaling comment
            # near the top of run().
            if is_distributed:
                num_updates = max(1, round(num_updates / world_size))
                lr_warmup_steps = max(1, round(lr_warmup_steps / world_size))
                checkpoint_every = max(1, round(checkpoint_every / world_size))
                if rank == 0:
                    print(f"DDP scaling: num_updates={num_updates}, "
                          f"lr_warmup_steps={lr_warmup_steps}, checkpoint_every={checkpoint_every} "
                          f"(single-GPU-equivalent config values divided by world_size={world_size})")

            # Debug configuration overrides (applied before the LR scheduler is built
            # below, so the schedule length actually matches the shortened debug run)
            if cfg.training.debug:
                num_updates = 1000
                max_train_steps = 100
                checkpoint_every = 100
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

            # Wrap in LossWrapper. find_unused_parameters=True is required here in
            # practice (see the DriftLossWrapper docstring) even though every
            # parameter is genuinely used every forward call.
            loss_module = DriftLossWrapper(self.model)
            if is_distributed:
                ddp_loss_model = DDP(
                    loss_module,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    find_unused_parameters=True,
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

                            # Forward through DDP wrapper invokes drift_loss computation
                            raw_loss, metrics = ddp_loss_model(batch)

                            # Backward on local loss initiates gradient all-reduce across all GPUs
                            raw_loss.backward()

                            # Diagnostic only (see train_drift_unet_lowdim_workspace.py for
                            # the max_norm=inf rationale: no clipping is actually applied
                            # yet). Gradients are already all-reduced/averaged by DDP's
                            # backward hooks by the time backward() returns, so this is
                            # naturally identical across ranks without needing
                            # reduce_scalars.
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), max_norm=float('inf'))

                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                            # Update EMA on rank 0 from synchronized model weights
                            if cfg.training.use_ema and rank == 0 and ema is not None:
                                ema.step(self.model)

                            # Increment global step before checkpointing
                            self.global_step += 1
                            current_step = self.global_step
                            current_lr = lr_scheduler.get_last_lr()[0]

                            # raw_loss/metrics are per-rank local-batch quantities (unlike
                            # grad_norm/param_norm, which DDP already makes rank-invariant)
                            # - reduce them across ranks for a meaningful global signal, in
                            # a single collective call (see reduce_scalars).
                            reduced_scalars = reduce_scalars(
                                {'train_loss': raw_loss, **metrics}, world_size, device)

                            step_log = {
                                **reduced_scalars,
                                'grad_norm': grad_norm.item(),
                                # Effective parameter-update magnitude this step (grad_norm
                                # alone doesn't say how far the parameters actually moved).
                                'grad_step_size': grad_norm.item() * current_lr,
                                'global_step': current_step,
                                'lr': current_lr,
                                'epoch': self.epoch,
                            }

                            if rank == 0:
                                tepoch.set_postfix(loss=reduced_scalars['train_loss'], refresh=False)

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
    workspace = TrainDriftUnetLowdimWorkspace(cfg, output_dir=output_dir)
    workspace.run()


if __name__ == "__main__":
    main()
