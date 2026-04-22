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

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager, CheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel, enable_full_determinism

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        enable_full_determinism(seed)
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

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

        # # configure validation dataset
        # val_dataset = dataset.get_validation_dataset()
        # val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        # print ("validation dataset size: ", len(val_dataset))
        
        # # configure dataset for covariance_spectrum
        # cov_dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=1, pin_memory = True, persistent_workers = False)
        
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * (cfg.training.num_epochs - self.epoch)) \
                    // cfg.training.gradient_accumulate_every,
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

        # # configure env runner
        # env_runner: BaseLowdimRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseLowdimRunner)

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

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # configure checkpoint
        topk_manager_every = CheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint_every.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
        
        # # compute covariance_spectrum of the training data
        # self.model.dataset_info(cov_dataloader, covariance_spectrum=None, diagonal=False)
        # if cfg.training.use_ema:
        #     self.ema_model.dataset_info(cov_dataloader, covariance_spectrum=None, diagonal=False)

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        last_ten_success_rate = []
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch, train=True)
                            
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        raw_loss_cpu = raw_loss.item()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # # ========= eval for this epoch ==========
                # policy = self.model
                # if cfg.training.use_ema:
                #     policy = self.ema_model
                # policy.eval()

                # # # run rollout
                # # if (self.epoch % cfg.training.rollout_every) == 0 and (self.epoch>0):
                # #     env_runner.current_epoch = self.epoch
                # #     runner_log = env_runner.run(policy, cfg)
                # #     # log all
                # #     step_log.update(runner_log)
                # #     if self.epoch>cfg.training.num_epochs-500:
                # #         last_ten_success_rate.append(step_log["test/mean_score"])

                # # run validation
                # if (self.epoch % cfg.training.val_every) == 0:
                #     with torch.no_grad():
                #         # compute test noise prediction loss
                #         val_noise_pred_losses = list()
                #         n_total_samples = 0
                #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}: Noise Prediction Loss on test set", 
                #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 n_samples = len(batch["obs"])
                #                 n_total_samples += n_samples
                                
                #                 # device transfer
                #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                #                 val_noise_pred_loss = policy.compute_loss(batch, train=False)
                #                 val_noise_pred_losses.append(val_noise_pred_loss.item() * n_samples)
                #                 if (cfg.training.max_val_steps is not None) \
                #                     and batch_idx >= (cfg.training.max_val_steps-1):
                #                     break   

                #         if len(val_noise_pred_losses) > 0:
                #             noise_pred_loss = np.sum(val_noise_pred_losses)/n_total_samples
                #             step_log['test_noise_pred_loss'] = noise_pred_loss


                # # Compute upper bound on NLL
                # if (self.epoch % cfg.training.nll_every)==0:
                #     NLL_test = policy.nll_bound(val_dataloader, self.epoch, npoints=100)
                #     step_log['test_nll_bpd'] = NLL_test 

                
                # # # Compute Reconstruction loss
                # if (self.epoch % cfg.training.reconst_loss_every)==0:
                #     reconst_loss = policy.compute_action_reconst_loss(val_dataloader, cfg)
                #     step_log['test_action_reconst_loss'] = reconst_loss.item()

                # # # checkpoint
                # # if (self.epoch % cfg.training.checkpoint_every) == 0 and (self.epoch>0):
                # #     # checkpointing
                # #     if cfg.checkpoint.save_last_ckpt:
                # #         #self.save_checkpoint(exclude_keys=['model', 'optimizer'])
                # #         self.save_checkpoint()
                # #     if cfg.checkpoint.save_last_snapshot:
                # #         self.save_snapshot()

                # #     # sanitize metric names
                # #     metric_dict = dict()
                # #     for key, value in step_log.items():
                # #         new_key = key.replace('/', '_')
                # #         metric_dict[new_key] = value
                    
                # #     # We can't copy the last checkpoint here
                # #     # since save_checkpoint uses threads.
                # #     # therefore at this point the file might have been empty!
                # #     topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                # #     if topk_ckpt_path is not None:
                # #         #self.save_checkpoint(path=topk_ckpt_path,exclude_keys=['model', 'optimizer'])
                # #         self.save_checkpoint(path=topk_ckpt_path)


                # # ========= eval end for this epoch ==========
                # policy.train()

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint_every.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint_every.save_last_snapshot:
                        self.save_snapshot()
                    if self.epoch>cfg.training.num_epochs-500:

                        topk_ckpt_path = topk_manager_every.get_ckpt_path(step_log)
                        if topk_ckpt_path is not None:
                            #self.save_checkpoint(path=topk_ckpt_path,exclude_keys=['model', 'optimizer'])
                            self.save_checkpoint(path=topk_ckpt_path)

                # end of epoch
                # log of last step is combined with validation and rollout
                if local_epoch_idx == cfg.training.num_epochs-1:
                    step_log["test/last_10_mean_score"] = np.mean(last_ten_success_rate)
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

# %%
