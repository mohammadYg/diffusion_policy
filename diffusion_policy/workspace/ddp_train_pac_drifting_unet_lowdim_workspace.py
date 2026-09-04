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
from omegaconf import OmegaConf
import wandb
import tqdm
from diffusers.training_utils import EMAModel

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.pac_drifting_unet_lowdim_policy import PacDriftingUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager, LastNCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class PacLossWrapper(nn.Module):
    """
    Standard PyTorch Module wrapper routing forward() to model.compute_bound() 
    or model.compute_loss(). Ensures DDP autograd hooks and gradient synchronization 
    buckets execute properly.
    """
    def __init__(self, model: nn.Module, cfg: OmegaConf, n_bound: int):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.n_bound = n_bound

    def forward(self, batch):
        if self.cfg.training.kl_penalty > 0.0:
            raw_loss, emp_risk_train, kl_train, metrics = self.model.compute_bound(
                batch,
                n_bound=self.n_bound,
                objective=self.cfg.training.pac_objective,
                delta=self.cfg.training.delta,
                kl_penalty=self.cfg.training.kl_penalty,
                stochastic=self.cfg.training.stochastic,
                bounded=self.cfg.training.bounded,
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


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Averages a scalar tensor across all distributed processes."""
    if world_size <= 1 or not dist.is_initialized():
        return tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def reduce_metrics(metrics: dict, world_size: int, device: torch.device) -> dict:
    """Averages a dictionary of scalar metrics across all workers in deterministic key order."""
    if world_size <= 1 or not dist.is_initialized():
        return {
            k: v.item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics.items()
        }

    reduced_metrics = {}
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, torch.Tensor):
            t = v.detach().clone().to(device)
        else:
            t = torch.tensor(float(v), device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= world_size
        reduced_metrics[k] = t.item()
    return reduced_metrics


class TrainPacDriftingUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # Global base seed for deterministic initial model parameters across ranks
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Underlying policy model
        self.model: PacDriftingUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)

        # Underlying EMA policy model
        self.ema_model: PacDriftingUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Initialize data-dependent prior deterministically on all ranks
        if cfg.training.data_dependent_prior:
            checkpoint = cfg.training.init_model_path
            with open(checkpoint, 'rb') as f:
                init_payload = torch.load(f, pickle_module=dill)
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

                # Preserve global batch size: divide batch_size across workers
                global_batch_size = dataloader_cfg.get('batch_size', 64)
                per_gpu_batch_size = int(global_batch_size // world_size)
                if per_gpu_batch_size < 1:
                    raise ValueError(
                        f"Configured batch_size ({global_batch_size}) is smaller than world_size ({world_size}). "
                        f"Ensure batch_size >= world_size."
                    )
                if global_batch_size % world_size != 0 and rank == 0:
                    print(
                        f"Warning: Global batch_size ({global_batch_size}) is not evenly divisible by "
                        f"world_size ({world_size}). Using per-GPU batch size of {per_gpu_batch_size}, "
                        f"resulting in effective global batch size of {per_gpu_batch_size * world_size}."
                    )
                dataloader_cfg['batch_size'] = per_gpu_batch_size

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
                    print(f"Per-GPU batch size: {dataloader_cfg['batch_size']} (Global batch size: {dataloader_cfg['batch_size'] * world_size})")

            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema and self.ema_model is not None:
                self.ema_model.set_normalizer(normalizer)

            # Safe conversion for scientific notation string parameters
            lr_warmup_steps = int(float(cfg.training.lr_warmup_steps))
            num_updates = int(float(cfg.training.num_updates))
            checkpoint_every = int(float(cfg.training.checkpoint_every))

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
            topk_manager = None
            lastN_manager = None
            if rank == 0:
                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **cfg.logging
                )
                wandb.config.update({"output_dir": self.output_dir})

                topk_manager = TopKCheckpointManager(
                    save_dir=os.path.join(self.output_dir, 'checkpoints'),
                    **cfg.checkpoint_max_score.topk
                )
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

            # Debug configuration overrides
            if cfg.training.debug:
                num_updates = 1000
                max_train_steps = 100
                checkpoint_every = 100
            else:
                max_train_steps = cfg.training.max_train_steps

            # Wrap in LossWrapper (with find_unused_parameters=True to support PAC prior/posterior parameter routing)
            loss_module = PacLossWrapper(self.model, cfg, n_bound=len(dataset))
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

                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                            # Update EMA on rank 0 from synchronized model weights
                            if cfg.training.use_ema and rank == 0 and ema is not None:
                                ema.step(self.model)

                            # Increment global step before checkpointing
                            self.global_step += 1
                            current_step = self.global_step

                            # Reduce loss, empirical risk, KL, and metrics across all ranks for logging
                            reduced_loss = reduce_tensor(raw_loss, world_size).item()
                            reduced_emp_risk = reduce_tensor(emp_risk_train, world_size).item()
                            reduced_kl = reduce_tensor(kl_train, world_size).item()
                            reduced_metrics = reduce_metrics(metrics, world_size, device)

                            step_log = {
                                'train_loss (pac_bayes bound)': reduced_loss,
                                'emp_risk_train': reduced_emp_risk,
                                'kl_train': reduced_kl,
                                'global_step': current_step,
                                'lr': lr_scheduler.get_last_lr()[0],
                                'epoch': self.epoch
                            }
                            step_log.update(reduced_metrics)

                            if rank == 0:
                                tepoch.set_postfix(loss=reduced_loss, refresh=False)

                            # Checkpointing (Top-K and Last-N)
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
                                    if topk_manager is not None:
                                        metric_dict = {k.replace('/', '_'): v for k, v in step_log.items()}
                                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                                        if topk_ckpt_path is not None:
                                            self.save_checkpoint(path=topk_ckpt_path, use_thread=False)
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
    workspace = TrainPacDriftingUnetLowdimWorkspace(cfg, output_dir=output_dir)
    workspace.run()


if __name__ == "__main__":
    main()