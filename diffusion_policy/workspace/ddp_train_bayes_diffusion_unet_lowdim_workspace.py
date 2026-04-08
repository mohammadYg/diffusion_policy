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
from pathlib import Path

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_prob_policy import DiffusionUnetLowdimProbPolicy, DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager, CheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel, enable_full_determinism

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainProbDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, prior_model_path= None, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        enable_full_determinism(seed)
        # torch.manual_seed(seed)
        # np.random.seed(seed)
        # random.seed(seed)

        self.model: DiffusionUnetLowdimProbPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimProbPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure dataset
        self.dataset: BaseLowdimDataset
        self.dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(self.dataset, BaseLowdimDataset)
        normalizer = self.dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        
        self.global_step = 0
        self.epoch = 0
        self.prior_model_path = prior_model_path

    def run(self):
        # load config file
        cfg = copy.deepcopy(self.cfg)

        # DDP setup
        dist.init_process_group(backend="nccl")                 # "nccl" is NVIDIA's collective communication library for communication

        # Get the local and Global rank of the process
        self.local_rank = int(os.environ["LOCAL_RANK"])      
        self.global_rank = int(os.environ["RANK"])       

        # Resume training
        latest_ckpt_path = self.get_checkpoint_path()
        if latest_ckpt_path.is_file():
            print("Resuming from checkpoint:", latest_ckpt_path)
            self.load_checkpoint(path=latest_ckpt_path)
        else:
            if cfg.training.train_prior:
                if cfg.task.dataset.train_episodes_for_posterior <= 0:
                    raise ValueError("No demonstrations for training the posterior")
                prior_dir = Path("data") / "prior_models" / cfg.task.name
                # check directory exists
                prior_dir.mkdir(parents=True, exist_ok=True)
                # Model filename
                prior_model_path = (
                    prior_dir
                    / f"prior_demos={cfg.task.dataset.max_train_episodes-cfg.task.dataset.train_episodes_for_posterior}"
                    f"-post_demos={cfg.task.dataset.train_episodes_for_posterior}.pth"
                )
                if not os.path.isfile(prior_model_path):
                    print("Training and initializing prior network...")
                    # get dataset for training the prior
                    prior_dataset = self.dataset.get_prior_dataset()
                    init_net = self.model.train_prior(prior_dataset, cfg)
                    # Save model
                    torch.save(init_net.state_dict(), prior_model_path)
                    print(f"Prior model saved to: {prior_model_path}")
                else:
                    print("Loading and initializing prior network...")
                    # load model
                    init_net: DiffusionUnetLowdimPolicy
                    prior_policy = hydra.utils.instantiate(cfg.prior_policy)
                    init_net = prior_policy.model
                    init_net.load_state_dict(torch.load(prior_model_path, weights_only=True))
                    init_net.eval()
                                
                self.model.prior_post_initialize(init_net, cfg.policy.model.rho_post, cfg.policy.model.rho_prior, initialize_from_prior=cfg.training.initialize_from_prior)
        
        ## configure dataset for PAC-Bayes training 
        post_dataset = self.dataset.get_post_dataset() if cfg.task.dataset.train_episodes_for_posterior>0 else self.dataset
        train_sampler = DistributedSampler(post_dataset)
        post_dataloader = DataLoader(post_dataset, sampler = train_sampler**cfg.post_dataloader)

        # configure validation dataset
        val_dataset = self.dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        if self.global_rank == 0:
            print ("validation dataset size: ", len(val_dataset))

        ## configure dataset for covariance_spectrum
        cov_dataloader = DataLoader(self.dataset, batch_size=len(self.dataset), 
                                    num_workers=1,   pin_memory = True, 
                                    persistent_workers = False)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(post_dataloader) * (cfg.training.num_epochs - self.epoch)),
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
        if self.global_rank == 0:
            env_runner: BaseLowdimRunner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseLowdimRunner)
        else:
            env_runner = None

        # configure logging (only on global_rank 0)
        if self.global_rank == 0:
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

        # configure checkpoint (only on global_rank 0)
        if self.global_rank == 0:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )
            topk_manager_every = CheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint_every.topk
            )

        # device transfer
        device = torch.device(f"cuda:{self.local_rank}")
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # DDP model wrapping 
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank])

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
        
        # compute covariance_spectrum of the training data
        self.model.module.dataset_info(cov_dataloader, covariance_spectrum=None, diagonal=False)
        if cfg.training.use_ema:
            self.ema_model.dataset_info(cov_dataloader, covariance_spectrum=None, diagonal=False)

        # training loop
        if self.global_rank == 0:
            log_path = os.path.join(self.output_dir, 'logs.json.txt')
            json_logger_cm = JsonLogger(log_path)        
        else:
            json_logger_cm = None
        
        # store the success rate of last 10 epochs to compute their mean at the end of training
        last_ten_success_rate = []
        log_rho = {}
        with json_logger_cm if self.global_rank == 0 else dummy_context_mgr():
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                train_sampler.set_epoch(local_epoch_idx)
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(post_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        if cfg.training.kl_penalty>0.0:
                            raw_loss, emp_risk_train, kl_train = self.model.module.compute_bound(batch, n_bound=len(post_dataloader.dataset), objective=cfg.training.pac_objective,
                                                        delta=cfg.training.delta, 
                                                        kl_penalty=cfg.training.kl_penalty, 
                                                        mc_sampling=cfg.eval.mc_sampling, stochastic=cfg.training.stochastic, bounded=cfg.training.bounded, train=True)
                            
                        else:
                            raw_loss = self.model.module.compute_loss(batch, stochastic=cfg.training.stochastic, train=True)
                            emp_risk_train = raw_loss
                            kl_train = torch.tensor([0.0])
                            
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        raw_loss_cpu = raw_loss.item()

                        # step optimizer
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model.module)

                        # logging (only on global_rank 0)
                        if self.global_rank == 0:
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                'train_loss (pac_bayes bound)': raw_loss_cpu,
                                'emp_risk_train': emp_risk_train.item(),
                                'kl_train': kl_train.item(),
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }

                            is_last_batch = (batch_idx == (len(post_dataloader)-1))
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
                if self.global_rank == 0:
                    train_loss = np.mean(train_losses)
                    step_log['train_loss (pac_bayes bound)'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model.module
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout (only on global_rank 0)
                if self.global_rank == 0 and (self.epoch % cfg.training.rollout_every) == 0:
                    env_runner.current_epoch = self.epoch
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)
                    if self.epoch>cfg.training.num_epochs-500:
                        last_ten_success_rate.append(step_log["test/mean_score"])

                # run validation (only on global_rank 0)
                if self.global_rank == 0 and (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        # compute test noise prediction loss
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}: Noise Prediction Loss on test set", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                n_samples = len(batch["obs"])
                                # device transfer
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                val_loss = policy.compute_loss(batch, stochastic=cfg.eval.stochastic, train=False)
                                val_losses.append(val_loss.item() * n_samples)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break

                        if len(val_losses) > 0:
                            noise_loss = np.sum(val_losses)/len(val_dataset)
                            step_log['test_noise_pred_loss'] = noise_loss
                        
                # Compute upper bound on NLL (only on global_rank 0)
                if self.global_rank == 0 and (self.epoch % cfg.training.nll_every)==0:
                    NLL_test = policy.nll_bound(val_dataloader, self.epoch, npoints=100, stochastic=cfg.eval.stochastic)
                    step_log['test_nll_bpd'] = NLL_test 
                
                # Compute Reconstruction loss (only on global_rank 0)
                if self.global_rank == 0 and (self.epoch % cfg.training.reconst_loss_every)==0:
                    reconst_loss = policy.compute_action_reconst_loss(val_dataloader, cfg)
                    step_log['test_action_reconst_loss'] = reconst_loss.item()

                # # log learned rho values
                # if self.global_rank == 0 and (self.epoch % cfg.training.rho_log_every) == 0:
                #     log_rho[self.epoch] = policy.rho_stats()

                # checkpoint (only on global_rank 0)
                if self.global_rank == 0 and (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(ddp=True)
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path,ddp=True)

               
                #========= eval end for this epoch ==========
                policy.train()
                
                # checkpoint
                if self.global_rank == 0 and (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint_every.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint_every.save_last_snapshot:
                        self.save_snapshot()
                    topk_ckpt_path = topk_manager_every.get_ckpt_path(step_log)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                        
                # end of epoch logging (only on global_rank 0)
                if self.global_rank == 0:
                    if local_epoch_idx == cfg.training.num_epochs-1:
                        step_log["test/last_10_mean_score"] = np.mean(last_ten_success_rate)
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger_cm.log(step_log)
                    self.global_step += 1
                self.epoch += 1
        
        # cleanup DDP
        dist.destroy_process_group()

# Dummy context manager for non-global_rank-0 processes
from contextlib import contextmanager
@contextmanager
def dummy_context_mgr():
    yield

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainProbDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

# %%
