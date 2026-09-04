if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
from typing import Optional
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
import dill

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.pac_drifting_unet_lowdim_policy import PacDriftingUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager, LastNCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class TrainPacDriftingUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: PacDriftingUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: PacDriftingUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0

    def _train_prior(self, cfg, prior_policy: PacDriftingUnetLowdimPolicy, prior_dataset, device):
        """
        Phase 1 of the data-dependent-prior recipe: train `prior_policy`'s Bayesian
        network on the prior-only demos with the plain stochastic (Bayes-by-backprop)
        empirical risk - compute_loss(stochastic=True) - and NO PAC-Bayes KL/bound
        term, since there is no informative prior to measure divergence against yet.
        Weights are sampled from the network's own posterior distribution on every
        forward pass (that's what makes this "Bayesian" rather than a deterministic
        ERM fit), so both the mean and the uncertainty (rho) are shaped by training,
        not just the mean. The resulting network becomes the informative,
        data-dependent prior (and the posterior's initialization) for the PAC-Bayes
        bound training phase that follows in run().
        """
        prior_optimizer = hydra.utils.instantiate(cfg.optimizer, params=prior_policy.parameters())
        prior_num_updates = int(float(cfg.training.prior_num_updates))
        prior_lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=prior_optimizer,
            num_warmup_steps=int(float(cfg.training.lr_warmup_steps)),
            num_training_steps=prior_num_updates,
            last_epoch=-1
        )
        prior_dataloader = DataLoader(prior_dataset, **cfg.dataloader)

        prior_policy.train()
        step = 0
        with tqdm.tqdm(total=prior_num_updates, desc="Training data-dependent prior",
                mininterval=cfg.training.tqdm_interval_sec) as pbar:
            while step < prior_num_updates:
                for batch in prior_dataloader:
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                    raw_loss, _ = prior_policy.compute_loss(batch, stochastic=True)
                    raw_loss.backward()
                    prior_optimizer.step()
                    prior_optimizer.zero_grad()
                    prior_lr_scheduler.step()

                    step += 1
                    pbar.update(1)
                    pbar.set_postfix(loss=raw_loss.item(), refresh=False)
                    if step >= prior_num_updates:
                        break
        prior_policy.eval()

    @staticmethod
    def _resolve_prior_checkpoint_path(cfg, n_prior_demos: int) -> Optional[str]:
        """
        Fills in the {task_name}/{n_prior_demos} placeholders in
        cfg.training.prior_checkpoint_path, e.g. the template
        "prior_{task_name}_demos={n_prior_demos}.ckpt" resolves to
        "prior_pusht_lowdim_demos=102.ckpt" - so each (task, prior split size)
        combination gets its own cache file instead of silently overwriting/
        reusing one trained for a different task or split. Returns None if the
        template itself is None (caching disabled).
        """
        template = cfg.training.prior_checkpoint_path
        if template is None:
            return None
        return template.format(task_name=cfg.task_name, n_prior_demos=n_prior_demos)

    def _load_cached_prior(self, cfg, prior_ckpt_path, normalizer, device) -> Optional[PacDriftingUnetLowdimPolicy]:
        """
        Returns a policy loaded from `prior_ckpt_path`, or None if that path is
        None or the file doesn't exist yet (nothing cached).
        """
        if prior_ckpt_path is None or not os.path.isfile(prior_ckpt_path):
            return None
        print("Found trained prior, loading from:", prior_ckpt_path)
        prior_policy: PacDriftingUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)
        prior_policy.set_normalizer(normalizer)
        prior_policy.to(device)
        prior_state = torch.load(prior_ckpt_path, pickle_module=dill, map_location=device)
        prior_policy.model.load_state_dict(prior_state['model_state_dict'])
        return prior_policy

    def _get_trained_prior(self, cfg, prior_ckpt_path, prior_dataset, n_prior_demos, normalizer, device) -> PacDriftingUnetLowdimPolicy:
        """
        Returns a policy holding the data-dependent prior network: loaded from
        `prior_ckpt_path` if that file already exists (via _load_cached_prior),
        otherwise trained fresh via _train_prior() and cached there for next time.
        """
        prior_policy = self._load_cached_prior(cfg, prior_ckpt_path, normalizer, device)
        if prior_policy is not None:
            return prior_policy

        prior_policy: PacDriftingUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)
        prior_policy.set_normalizer(normalizer)
        prior_policy.to(device)
        print(
            f"No trained prior found (prior_checkpoint_path={prior_ckpt_path}). "
            f"Training prior for {int(float(cfg.training.prior_num_updates))} "
            f"updates on {n_prior_demos} prior demos..."
        )
        self._train_prior(cfg, prior_policy, prior_dataset, device)
        if prior_ckpt_path is not None:
            pathlib.Path(prior_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {'model_state_dict': prior_policy.model.state_dict(), 'cfg': cfg},
                prior_ckpt_path, pickle_module=dill)
            print("Saved trained prior to:", prior_ckpt_path)
        return prior_policy

    def _setup_train_dataset(self, cfg, dataset, normalizer, device):
        """
        Without a data-dependent prior: prior/posterior stay exactly as cfg.policy
        constructed them, and training runs on the full dataset.

        With one: task.dataset.train_episodes_for_posterior splits `dataset` into
        a prior split and a posterior/bound split; training always continues on
        the posterior split. On a fresh start (global_step == 0), self.model/
        self.ema_model get seeded (prior AND posterior) from the trained (or
        cached) prior via prior_initialization(). On resume, self.model's
        posterior already reflects partially completed training and must not be
        reset - but we still refresh the FIXED prior (only, via
        init_posterior=False) from the cached prior checkpoint if one is
        available, so the PAC-Bayes bound keeps measuring KL against the
        intended data-dependent prior. We deliberately never *train* a prior on
        resume: training is stochastic, so a freshly retrained prior would differ
        from the one the checkpointed posterior was actually trained against,
        silently corrupting the bound - only a cache hit refreshes it.
        """
        if not cfg.training.data_dependent_prior:
            return dataset

        prior_dataset = dataset.get_prior_dataset()
        post_dataset = dataset.get_post_dataset()
        if len(prior_dataset) == 0 or len(post_dataset) == 0:
            raise ValueError(
                "cfg.training.data_dependent_prior=True requires "
                "task.dataset.train_episodes_for_posterior to carve out a "
                "nonzero split for both the prior demos and the posterior/bound "
                f"demos (got prior={len(prior_dataset)}, posterior={len(post_dataset)})."
            )
        # Number of DEMO EPISODES (not sequence-sample windows - len(prior_dataset)
        # counts those instead) in the prior split. get_prior_dataset() sets
        # train_mask to the prior episode_mask, so its sum is exactly the prior
        # episode count - equal to task.dataset.max_train_episodes -
        # task.dataset.train_episodes_for_posterior when max_train_episodes is
        # set, but still correct (falls back to the actual non-val episode
        # count) when it's left at its null default.
        n_prior_demos = int(prior_dataset.train_mask.sum())
        # Resolved once here (needs the actual prior split size, only known now
        # that the dataset has been split) and threaded through below, so the
        # cache file name reflects this task and prior-demo-count combination -
        # see _resolve_prior_checkpoint_path().
        prior_ckpt_path = self._resolve_prior_checkpoint_path(cfg, n_prior_demos)

        if self.global_step > 0:
            prior_policy = self._load_cached_prior(cfg, prior_ckpt_path, normalizer, device)
            if prior_policy is None:
                print(
                    f"Resuming at global_step={self.global_step}: no cached prior found at "
                    f"{prior_ckpt_path} to refresh from - keeping the checkpointed "
                    "prior/posterior as-is."
                )
            else:
                print(
                    f"Resuming at global_step={self.global_step}: refreshing the fixed prior "
                    f"from {prior_ckpt_path}; posterior left untouched."
                )
                self.model.prior_initialization(prior_policy.model, init_posterior=False)
                if cfg.training.use_ema:
                    self.ema_model.prior_initialization(prior_policy.model, init_posterior=False)
            return post_dataset

        prior_policy = self._get_trained_prior(cfg, prior_ckpt_path, prior_dataset, n_prior_demos, normalizer, device)

        # Seed self.model's prior AND posterior from the trained prior - the
        # posterior for the PAC-Bayes-bound phase starts out identical to the
        # prior. Same trained network seeds both the online and EMA posterior
        # (phase 1 doesn't keep its own EMA copy).
        self.model.prior_initialization(prior_policy.model)
        if cfg.training.use_ema:
            self.ema_model.prior_initialization(prior_policy.model)
        return post_dataset

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
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print ("validation dataset size: ", len(val_dataset))
        
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # device transfer (moved up: the prior-training/dataset-split logic below
        # both need a device)
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_updates = 1000
            cfg.training.max_train_steps = 100
            cfg.training.max_val_steps = 100
            cfg.training.prior_num_updates = 100
            cfg.training.rollout_every = 100
            cfg.training.checkpoint_every = 100
            cfg.training.val_every = 100

        # Seed self.model/self.ema_model from a data-dependent prior (if
        # requested) and pick which dataset the posterior/PAC-Bayes phase below
        # trains on - see _setup_train_dataset()/_get_trained_prior().
        train_dataset = self._setup_train_dataset(cfg, dataset, normalizer, device)
        train_dataloader = DataLoader(train_dataset, **cfg.dataloader)

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

        env_runner: BaseLowdimRunner
        try:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseLowdimRunner)
        except Exception as e:
            print(f"Warning: env_runner instantiation failed ({e}). Rollouts will be skipped.")
            env_runner = None

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

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            # ensure local control vars from cfg
            num_updates = int(cfg.training.num_updates)
            rollout_every = int(cfg.training.rollout_every)
            val_every = int(cfg.training.val_every)
            checkpoint_every = int(cfg.training.checkpoint_every)
            # train_dataloader.dataset never changes across the loop below, so
            # compute this once rather than re-calling len() every single step.
            n_bound = len(train_dataloader.dataset)

            # training: run until we hit num_updates
            while self.global_step < num_updates:
                with tqdm.tqdm(train_dataloader, desc=f"Training step {self.global_step}",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:

                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        # compute objective
                        if cfg.training.kl_penalty > 0.0:
                            raw_loss, emp_risk_train, kl_train, metrics = self.model.compute_bound(
                                batch,
                                n_bound=n_bound,
                                objective=cfg.training.pac_objective,
                                delta=cfg.training.delta,
                                kl_penalty=cfg.training.kl_penalty,
                                stochastic=cfg.training.stochastic,
                                bounded=cfg.training.bounded,
                            )
                        else:
                            raw_loss, metrics = self.model.compute_loss(batch, stochastic=cfg.training.stochastic)
                            emp_risk_train = raw_loss
                            kl_train = torch.tensor([0.0])

                        loss = raw_loss
                        loss.backward()
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)

                        # step optimizer and scheduler
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()

                        # update ema after optimizer step
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # build step-log (use the upcoming/global step index)
                        current_step = self.global_step + 1
                        step_log = {
                            'train_loss (pac_bayes bound)': raw_loss_cpu,
                            'emp_risk_train': emp_risk_train.item(),
                            'kl_train': kl_train.item(),
                            'global_step': current_step,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        step_log.update(metrics)

                        # evaluation runs after optimizer step
                        policy = self.ema_model if cfg.training.use_ema else self.model
                        policy.eval()

                        # run rollout
                        if env_runner is not None and ((current_step % rollout_every) == 0 or self.global_step==0):
                            runner_log = env_runner.run(policy, stochastic=False)
                            step_log.update(runner_log)
                            # runner_log = env_runner.run(policy, stochastic=True)
                            # step_log.update(runner_log)

                        # validation: noise prediction loss
                        if ((current_step % val_every) == 0 or self.global_step==0) and (len(val_dataloader) > 0):
                            with torch.no_grad():
                                val_losses = []
                                with tqdm.tqdm(val_dataloader, desc=f"Validation step {current_step}: Noise Prediction Loss on test set", 
                                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as vepoch:
                                    n_samples_total=0
                                    for v_idx, vbatch in enumerate(vepoch):
                                        n_samples = len(vbatch["obs"])
                                        n_samples_total = n_samples_total + n_samples
                                        vbatch = dict_apply(vbatch, lambda x: x.to(device, non_blocking=True))
                                        val_loss, _ = policy.compute_loss(vbatch, stochastic=cfg.eval.stochastic)
                                        val_losses.append(val_loss.item() * n_samples)
                                        if (cfg.training.max_val_steps is not None) and v_idx >= (cfg.training.max_val_steps - 1):
                                            break
                                if len(val_losses) > 0:
                                    noise_loss = np.sum(val_losses) / n_samples_total
                                    step_log['test_loss'] = noise_loss

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
                        # if (current_step % checkpoint_every) == 0:
                        #     if cfg.checkpoint_last_N.save_last_ckpt:
                        #         self.save_checkpoint()
                        #     if cfg.checkpoint_last_N.save_last_snapshot:
                        #         self.save_snapshot()
                            # lastN_ckpt_path = lastN_manager.get_ckpt_path(step_log)
                            # if lastN_ckpt_path is not None:
                            #     self.save_checkpoint(path=lastN_ckpt_path)

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
    workspace = TrainPacDriftingUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

# %%
