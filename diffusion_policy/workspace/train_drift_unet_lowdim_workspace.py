if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm

from mujoco_py.builder import MujocoException

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.drift_unet_lowdim_policy import DriftUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager, LastNCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDriftUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DriftUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DriftUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Resume training
        if cfg.training.resume:
            if cfg.training.desired_ckpt_path is not None:
                desired_ckpt_path = cfg.training.desired_ckpt_path
                if not os.path.isfile(desired_ckpt_path):
                    raise ValueError(f"No such file: {desired_ckpt_path}")
                print("Resuming from checkpoint:", desired_ckpt_path)
                self.load_checkpoint(path=desired_ckpt_path)
            else:
                latest_ckpt_path = self.get_checkpoint_path()
                if latest_ckpt_path.is_file():
                    print("Resuming from checkpoint:", latest_ckpt_path)
                    self.load_checkpoint(path=latest_ckpt_path)
                else:
                    print("Starting training from scratch.")
        else:
            # Otherwise start fresh
            print("Starting training from scratch.")

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        print ("Training dataset size: ", len(dataset))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print ("validation dataset size: ", len(val_dataset))
        
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
        
        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.num_updates,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # # configure env
        # env_runner: BaseLowdimRunner
        # try:
        #     env_runner = hydra.utils.instantiate(
        #         cfg.task.env_runner,
        #         output_dir=self.output_dir)
        #     assert isinstance(env_runner, BaseLowdimRunner)
        # except Exception as e:
        #     print(f"Warning: env_runner instantiation failed ({e}). Rollouts will be skipped.")
        #     env_runner = None
        #
        # # Carries the last successful rollout's score(s) forward across MuJoCo
        # # instability so wandb's mean_score plot has no gap/jump.
        # last_runner_log: dict = {}
        # if env_runner is not None:
        #     prefixes = sorted(set(getattr(env_runner, 'env_prefixs', ['test/'])))
        #     last_runner_log = {p + 'mean_score': 0.0 for p in prefixes}

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint for topk based on score
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint_max_score.topk
        )

        # configure checkpoint to save last N checkpoints
        lastN_manager = LastNCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint_last_N.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_updates = 1000
            cfg.training.max_train_steps = 100
            cfg.training.max_val_steps = 100
            rollout_every = 100
            checkpoint_every = 100
            val_every = 100
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            # ensure local control vars from cfg
            num_updates = int(cfg.training.num_updates)
            rollout_every = int(cfg.training.rollout_every)
            val_every = int(cfg.training.val_every)
            checkpoint_every = int(cfg.training.checkpoint_every)

            # training: run until we hit num_updates
            while self.global_step < num_updates:
                step_log = dict()

                with tqdm.tqdm(train_dataloader, desc=f"Training step {self.global_step}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:

                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        # compute loss
                        raw_loss, metrics = self.model.compute_loss(batch)
                            
                        loss = raw_loss
                        loss.backward()
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)

                        # Diagnostic only: pre-clip global gradient norm (max_norm=inf means
                        # clip_grad_norm_ never actually rescales anything - the comparison
                        # total_norm > max_norm is always False - it just returns total_norm).
                        # Logged so we can look at its typical value/tail over a real run and
                        # pick a principled max_grad_norm empirically, the same way the
                        # original drifting repo's train.py logs g_norm, rather than
                        # copying its max_grad_norm=2.0 default on faith. No clipping is
                        # actually applied yet.
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))

                        # step optimizer and scheduler
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                        # update ema after optimizer step
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # build step-log (use the upcoming/global step index)
                        current_step = self.global_step + 1
                        current_lr = lr_scheduler.get_last_lr()[0]
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'grad_norm': grad_norm.item(),
                            # Effective parameter-update magnitude this step (grad_norm
                            # alone doesn't say how far the parameters actually moved).
                            'grad_step_size': grad_norm.item() * current_lr,
                            'global_step': current_step,
                            'lr': current_lr
                        }
                        step_log.update(metrics)

                        # evaluation runs after optimizer step
                        policy = self.ema_model if cfg.training.use_ema else self.model
                        policy.eval()

                        # # run rollout
                        # if env_runner is not None and ((current_step % rollout_every) == 0): #or self.global_step==0:
                        #     try:
                        #         runner_log = env_runner.run(policy)
                        #         last_runner_log.update(runner_log)
                        #     except MujocoException as e:
                        #         print(f"Warning: MuJoCo instability during rollout at step "
                        #               f"{current_step} ({e}). Reporting the previous rollout's "
                        #               f"score(s) instead so wandb has no gap.")
                        #         step_log['rollout_mujoco_error'] = str(e)
                        #         # The crashed worker's pipe is permanently closed by
                        #         # AsyncVectorEnv._raise_if_errors, so env_runner can't be
                        #         # reused - rebuild it.
                        #         try:
                        #             env_runner.env.close(terminate=True)
                        #         except Exception:
                        #             pass
                        #         env_runner = hydra.utils.instantiate(
                        #             cfg.task.env_runner,
                        #             output_dir=self.output_dir)
                        #         runner_log = dict(last_runner_log)
                        #     step_log.update(runner_log)

                        # validation: noise prediction loss
                        if ((current_step % val_every) == 0 or self.global_step==0):
                            # Diagnostic only, piggybacked on the validation cadence (cheap,
                            # no need for its own schedule): total parameter norm, and the
                            # relative update size (grad_step_size / param_norm) - a more
                            # scale-invariant instability indicator than grad_norm alone.
                            # Independent of whether a validation set exists (unlike the
                            # test_* block below), so it isn't gated on len(val_dataloader).
                            param_norm = torch.sqrt(
                                sum((p.detach() ** 2).sum() for p in self.model.parameters())
                            ).item()
                            step_log['param_norm'] = param_norm
                            step_log['relative_update_size'] = step_log['grad_step_size'] / (param_norm + 1e-12)

                            if len(val_dataloader) > 0:
                                with torch.no_grad():
                                    val_losses = []
                                    val_metric_sums = {}
                                    with tqdm.tqdm(val_dataloader, desc=f"Validation step {current_step}: Noise Prediction Loss on test set",
                                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as vepoch:
                                        n_samples_total=0
                                        for v_idx, vbatch in enumerate(vepoch):
                                            n_samples = len(vbatch["obs"])
                                            n_samples_total = n_samples_total + n_samples
                                            vbatch = dict_apply(vbatch, lambda x: x.to(device, non_blocking=True))
                                            val_loss, val_metrics = policy.compute_loss(vbatch)
                                            val_losses.append(val_loss.item() * n_samples)
                                            # Same drift_loss diagnostics as train (scale, loss_R,
                                            # mean_dist_to_pos, diversity, entropy_R, ...) but on
                                            # held-out data, so train-vs-val divergence in these
                                            # interpretable quantities is visible directly, not
                                            # just via the (sometimes uninformative) scalar loss.
                                            for k, v in val_metrics.items():
                                                val_metric_sums[k] = val_metric_sums.get(k, 0.0) + v * n_samples
                                            if (cfg.training.max_val_steps is not None) and v_idx >= (cfg.training.max_val_steps - 1):
                                                break
                                    if len(val_losses) > 0:
                                        noise_loss = np.sum(val_losses) / n_samples_total
                                        step_log['test_loss'] = noise_loss
                                        for k, total in val_metric_sums.items():
                                            step_log[f'test_{k}'] = total / n_samples_total

                        policy.train()
                        
                        # # Checkpoint (k-top models)
                        # if (current_step % checkpoint_every) == 0:
                        #     # checkpointing
                        #     if cfg.checkpoint_max_score.save_last_ckpt:
                        #         self.save_checkpoint()
                        #     if cfg.checkpoint_max_score.save_last_snapshot:
                        #         self.save_snapshot()

                        #     # sanitize metric names
                        #     metric_dict = dict()
                        #     for key, value in step_log.items():
                        #         new_key = key.replace('/', '_')
                        #         metric_dict[new_key] = value
                        #     topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                        #     if topk_ckpt_path is not None:
                        #         self.save_checkpoint(path=topk_ckpt_path)


                        # checkpointing (last N)
                        if (current_step % checkpoint_every) == 0:
                            if cfg.checkpoint_last_N.save_last_ckpt:
                                self.save_checkpoint()
                            if cfg.checkpoint_last_N.save_last_snapshot:
                                self.save_snapshot()
                            lastN_ckpt_path = lastN_manager.get_ckpt_path(step_log)
                            if lastN_ckpt_path is not None:
                                self.save_checkpoint(path=lastN_ckpt_path)

                        # log & step
                        wandb_run.log(step_log, step=current_step)
                        json_logger.log(step_log)
                        self.global_step = current_step

                        # optional early stopping per-batch limit
                        if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                        # stop if reached total updates
                        if self.global_step >= num_updates:
                            break

                    # end for batches in dataloader
                # end tepoch
            # end while self.global_step < num_updates
        
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDriftUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

# %%
