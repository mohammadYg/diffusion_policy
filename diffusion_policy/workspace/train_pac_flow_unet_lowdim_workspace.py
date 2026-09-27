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
from omegaconf import OmegaConf, DictConfig
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
from diffusion_policy.policy.pac_flow_unet_lowdim_policy import PacFlowUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager, LastNCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainPacFlowUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: DictConfig, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: PacFlowUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: PacFlowUnetLowdimPolicy = None
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

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # Carries the last successful rollout's score(s) forward across MuJoCo
        # instability so wandb's mean_score plot has no gap/jump.
        prefixes = sorted(set(getattr(env_runner, 'env_prefixs', ['test/'])))
        last_runner_log: dict = {
            **{p + 'mean_score_deterministic': 0.0 for p in prefixes},
            **{p + 'mean_score_stochastic': 0.0 for p in prefixes},
        }

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

                        # compute objective
                        if cfg.training.kl_penalty > 0.0:
                            raw_loss, emp_risk_train, kl_train, loss_emp_bounded = self.model.compute_bound(
                                batch,
                                n_bound=len(train_dataloader.dataset),
                                objective=cfg.training.pac_objective,
                                delta=cfg.training.delta,
                                kl_penalty=cfg.training.kl_penalty,
                                stochastic=cfg.training.stochastic,
                                bounded=cfg.training.bounded,
                                bound_transform=cfg.training.bound_transform,
                                loss_scale=cfg.training.loss_scale,
                            )
                        else:
                            raw_loss = self.model.compute_loss(batch, stochastic=cfg.training.stochastic)
                            emp_risk_train = raw_loss
                            kl_train = torch.tensor([0.0])
                            loss_emp_bounded = None

                        loss = raw_loss

                        # Diagnostic only: gradient-norm split between the empirical-risk
                        # path (loss_emp_bounded) and the KL path (kl_train), computed
                        # BEFORE the real backward via two isolated autograd.grad calls
                        # (retain_graph=True keeps the graph alive for them, and for the
                        # real backward() right after). Neither call writes to .grad, so
                        # they don't affect the actual optimizer step - this only tells
                        # us how much gradient signal each path contributes, e.g. to spot
                        # bound_transform saturating the empirical-risk path to ~0. Adds
                        # two extra backward-equivalent passes per step - non-trivial
                        # overhead. Mirrors TrainPacDriftUnetLowdimWorkspace.run().
                        if loss_emp_bounded is not None:
                            # requires_grad filter: the Bayesian layers' prior mu/rho are
                            # registered as frozen params (requires_grad=False) - autograd.grad
                            # errors if any `inputs` tensor doesn't require grad, unlike
                            # backward()/allow_unused (which only covers "unused", not "frozen").
                            params = [p for p in self.model.parameters() if p.requires_grad]
                            zero = torch.zeros((), device=device)
                            emp_grads = torch.autograd.grad(
                                loss_emp_bounded, params, retain_graph=True, allow_unused=True)
                            kl_grads = torch.autograd.grad(
                                kl_train, params, retain_graph=True, allow_unused=True)
                            emp_risk_grad_norm = torch.sqrt(sum(
                                (g.detach().pow(2).sum() for g in emp_grads if g is not None), zero))
                            kl_grad_norm = torch.sqrt(sum(
                                (g.detach().pow(2).sum() for g in kl_grads if g is not None), zero))
                        else:
                            emp_risk_grad_norm = None
                            kl_grad_norm = None

                        loss.backward()
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)

                        # Diagnostic only: pre-clip global gradient norm (max_norm=inf means
                        # clip_grad_norm_ never actually rescales anything - the comparison
                        # total_norm > max_norm is always False - it just returns total_norm).
                        # This is the norm of the REAL, total gradient (loss_sum's) that the
                        # optimizer step below actually uses - unlike emp_risk_grad_norm/
                        # kl_grad_norm above, which isolate the two paths BEFORE they combine.
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
                            'train_loss (pac_bayes bound)': raw_loss_cpu,
                            'emp_risk_train': emp_risk_train.item(),
                            'kl_train': kl_train.item(),
                            'grad_norm': grad_norm.item(),
                            # Effective parameter-update magnitude this step (grad_norm
                            # alone doesn't say how far the parameters actually moved).
                            'grad_step_size': grad_norm.item() * current_lr,
                            # How much of the total gradient (grad_norm above) comes from
                            # the empirical-risk path vs. the KL path - see the diagnostic
                            # computed above. 0.0 when kl_penalty<=0 (no PAC bound, no split).
                            'emp_risk_grad_norm': emp_risk_grad_norm.item() if emp_risk_grad_norm is not None else 0.0,
                            'kl_grad_norm': kl_grad_norm.item() if kl_grad_norm is not None else 0.0,
                            'global_step': current_step,
                            'lr': current_lr
                        }

                        # evaluation runs after optimizer step
                        policy = self.ema_model if cfg.training.use_ema else self.model
                        policy.eval()

                        # run rollout
                        if (current_step % rollout_every) == 0 or self.global_step==0:
                            try:
                                runner_log = env_runner.run(policy, stochastic=False)
                                last_runner_log.update(runner_log)
                            except MujocoException as e:
                                print(f"Warning: MuJoCo instability during rollout at step "
                                      f"{current_step} ({e}). Reporting the previous rollout's "
                                      f"score(s) instead so wandb has no gap.")
                                step_log['rollout_mujoco_error'] = str(e)
                                # The crashed worker's pipe is permanently closed by
                                # AsyncVectorEnv._raise_if_errors, so env_runner can't be
                                # reused - rebuild it.
                                try:
                                    env_runner.env.close(terminate=True)
                                except Exception:
                                    pass
                                env_runner = hydra.utils.instantiate(
                                    cfg.task.env_runner,
                                    output_dir=self.output_dir)
                                runner_log = dict(last_runner_log)
                            step_log.update(runner_log)

                            try:
                                runner_log = env_runner.run(policy, stochastic=True)
                                last_runner_log.update(runner_log)
                            except MujocoException as e:
                                print(f"Warning: MuJoCo instability during rollout at step "
                                      f"{current_step} ({e}). Reporting the previous rollout's "
                                      f"score(s) instead so wandb has no gap.")
                                step_log['rollout_mujoco_error'] = str(e)
                                try:
                                    env_runner.env.close(terminate=True)
                                except Exception:
                                    pass
                                env_runner = hydra.utils.instantiate(
                                    cfg.task.env_runner,
                                    output_dir=self.output_dir)
                                runner_log = dict(last_runner_log)
                            step_log.update(runner_log)

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

                        # validation: nll computation
                        if ((current_step % val_every) == 0 or self.global_step==0) and (len(val_dataloader) > 0):
                            nlls = []
                            val_losses = []
                            with tqdm.tqdm(val_dataloader, desc=f"Validation step {current_step}: NLL computation on the test set", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as vepoch:
                                n_samples_total=0
                                for v_idx, vbatch in enumerate(vepoch):
                                    n_samples = len(vbatch["obs"])
                                    n_samples_total = n_samples_total + n_samples
                                    vbatch = dict_apply(vbatch, lambda x: x.to(device, non_blocking=True))
                                    
                                    val_loss = policy.compute_loss(vbatch, stochastic=cfg.eval.stochastic)
                                    nll = policy.compute_nll(vbatch, stochastic=cfg.eval.stochastic,
                                                                exact_divergence=cfg.eval.exact_divergence,
                                                            )
                                    
                                    nlls.append(nll.item() * n_samples)
                                    val_losses.append(val_loss.item() * n_samples)
                                    if (cfg.training.max_val_steps is not None) and v_idx >= (cfg.training.max_val_steps - 1):
                                        break
                            if len(nlls) > 0:
                                nll = np.sum(nlls) / n_samples_total
                                step_log['test_NLL'] = nll
                            if len(val_losses) > 0:
                                val_loss = np.sum(val_losses) / n_samples_total
                                step_log['test_loss'] = val_loss

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
    workspace = TrainPacFlowUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

# %%
