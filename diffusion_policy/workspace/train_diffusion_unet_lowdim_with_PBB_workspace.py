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
import shutil

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel
from diffusion_policy.policy.diffusion_model_for_NLL_IT import DiffusionModel_IT
OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainDiffusionUnetLowdimWithPBBWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

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
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print("Resuming from checkpoint:", latest_ckpt_path)
                self.load_checkpoint(path=latest_ckpt_path)

        # Retrain from a specific checkpoint
        elif cfg.training.retrain:
            if cfg.training.desired_ckpt_path is None:
                raise ValueError("desired_ckpt_path must be set when retraining.")
            desired_ckpt_path = cfg.training.desired_ckpt_path
            if not os.path.isfile(desired_ckpt_path):
                raise ValueError(f"No such file: {desired_ckpt_path}")
            print("Retraining from checkpoint:", desired_ckpt_path)
            self.load_checkpoint(path=desired_ckpt_path)
            
        # Otherwise start fresh
        else:
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

        ## configure dataset for covariance_spectrum
        print ("length of the training dataset: ", len(dataset))
        cov_dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=1,   pin_memory = True, persistent_workers = False)
        
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
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

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

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
        
        # set up intial diffusion model and data statistics for approximating NLL (need to be updated)
        diffusion_model = DiffusionModel_IT(DP_model = self.model.model, device = device )
        batch = next(iter(cov_dataloader))
        
        # normalize the data
        nbatch = normalizer.normalize(batch)
        data = nbatch["action"].to(device)
        # compute covariance_spectrum of the training data
        diffusion_model.dataset_info(data, covariance_spectrum=None, diagonal=False)
        
        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                train_lips_const_prod = 0
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    
                    # random batch for computing Lipschitz Constant
                    rand_batch = np.random.randint(0, len(train_dataloader))

                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        
                        losses = []
                        # Sample the noise for reconstruction (refer to the PAC-Bayes paper for the reconstruction loss definition)
                        if cfg.obs_as_local_cond:
                            shape = batch["action"].shape

                        elif cfg.obs_as_global_cond:
                            # condition throught global feature
                            shape = batch["action"].shape
                            if cfg.pred_action_steps_only:
                                shape = (batch["action"].shape[0], cfg.n_action_steps, cfg.action_dim)
                        else:
                            shape = (batch["action"].shape[0], cfg.horizon, cfg.action_dim + cfg.obs_dim)

                        # Sample noise that we'll add to the input
                        noise = torch.randn(shape, device=batch["action"].device)

                        for _ in range (cfg.training.num_expect):
                            raw_loss = self.model.compute_reconst_loss_T(batch, noise, cfg.training.PAC_loss_type) 
                            losses.append(raw_loss.item())
                            loss = raw_loss / (cfg.training.num_expect*cfg.training.gradient_accumulate_every)
                            loss.backward()

                        #loss = torch.stack(losses).mean()
                        #loss.backward()
                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = np.mean(losses)
                        #raw_loss_cpu = loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        # if rand_batch == batch_idx:
                        #     # 2 random samples from the random batch to compute Lipschitz Constant
                        #     rand_sample1 = np.random.randint(0, len(batch))
                        #     #rand_sample2 = np.random.randint(0, len(batch))
                        #     obs1 = batch["obs"][rand_sample1]
                        #     obs2 = batch['obs'][rand_sample1]
                        #     obs = torch.stack((obs1, obs2))
                        #     with torch.no_grad():
                        #         train_lips_const, train_lips_const_prod = self.model.lip_const(obs)
                        #         step_log['train_lips_const_prod'] = train_lips_const_prod.item()

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

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)
                
                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses_noise_pred = 0
                        val_losses_reconstruction = 0
                        val_lips_const_prod = 0
                        n_total_samples = 0

                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            
                            # random batch for computing Lipschitz Constant
                            rand_batch = np.random.randint(0, len(val_dataloader))

                            for batch_idx, batch in enumerate(tepoch):
                                n_samples = len(batch["obs"])
                                n_total_samples += n_samples
                                # device transfer
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                
                                # noise prediction loss
                                loss_noise_pred = policy.compute_loss(batch)
                                val_losses_noise_pred += loss_noise_pred.item() * n_samples
                                
                                # Sample the noise for reconstruction (refer to the PAC-Bayes paper for the reconstruction loss definition)
                                if cfg.obs_as_local_cond:
                                    shape = batch["action"].shape

                                elif cfg.obs_as_global_cond:
                                    # condition throught global feature
                                    shape = batch["action"].shape
                                    if cfg.pred_action_steps_only:
                                        shape = (batch["action"].shape[0], cfg.n_action_steps, cfg.action_dim)
                                else:
                                    shape = (batch["action"].shape[0], cfg.horizon, cfg.action_dim + cfg.obs_dim)

                                # Sample noise that we'll add to the input
                                noise = torch.randn(shape, device=batch["action"].device)
                                loss_reconstruct = policy.compute_reconst_loss_T(batch, noise, cfg.training.PAC_loss_type) 
                                val_losses_reconstruction += loss_reconstruct.item() * n_samples

                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                                
                                if rand_batch == batch_idx:
                                    # 2 random samples from the random batch to compute Lipschitz Constant
                                    rand_sample1 = np.random.randint(0, len(batch))
                                    #rand_sample2 = np.random.randint(0, len(batch))
                                    obs1 = batch["obs"][rand_sample1]
                                    obs2 = batch['obs'][rand_sample1]
                                    obs = torch.stack((obs1, obs2))
                                    val_lips_const, val_lips_const_prod = policy.lip_const(obs)
                                    
                        if (self.epoch % cfg.training.nll_every)==0:
                            diffusion_model.update_model(policy.model)
                            NLL = policy.test_nll(diffusion_model, val_dataloader, self.epoch, npoints=100, xinterval=None)
                            step_log['nll (bpd)'] = NLL

                        #if len(val_losses) > 0:
                        val_loss_noise_pred = val_losses_noise_pred/n_total_samples
                        val_loss_reconstruction = val_losses_reconstruction/n_total_samples
                        # log epoch average validation loss
                        step_log['val_loss_reconstruct'] = val_loss_reconstruction
                        step_log['val_loss_noise_pred'] = val_loss_noise_pred
                        step_log['val_lips_const_prod'] = val_lips_const_prod.item()

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                #TODO compute the PAC-Bayes Bounds
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    # def PAC_Bayes_bounds(self, data_loader):
    #     def reconstruct_loss (data_loader):
    #         for batch_idx, batch in enumerate(data_loader):               
    #             # device transfer
    #             device = cfg.training.device
    #             batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
    #             obs_dict = {"obs": batch["obs"]}
    #             gt_action = batch["action"]
                
    #             loss = 0
    #             for _ in range (cfg.training.num_expect):
    #                 result = self.model.predict_action(obs_dict)
    #                 if cfg.pred_action_steps_only:
    #                     pred_action = result["action"]
    #                     start = cfg.n_obs_steps - 1
    #                     end = start + cfg.n_action_steps
    #                     gt_action = gt_action[:, start:end]
    #                 else:
    #                     pred_action = result["action_pred"]
                    
    #                 # compute loss
    #                 if cfg.training.loss_type == "MSE":
    #                     loss += torch.nn.functional.mse_loss(pred_action, gt_action)
    #                 else:
    #                     loss += torch.linalg.norm(pred_action - gt_action, ord=2, dim=(1, 2)).mean()
    #             val_losses.append(loss/cfg.training.num_expect)
        


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
