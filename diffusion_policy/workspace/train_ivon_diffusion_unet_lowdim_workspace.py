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

try:
    import ivon
except ImportError as e:
    raise ImportError(
        "This workspace needs the IVON optimizer package. Install it with "
        "`pip install ivon-opt` (PyPI) or "
        "`pip install git+https://github.com/team-approx-bayes/ivon.git` "
        "(the repo is GPL-3.0 licensed - install it as a dependency rather "
        "than vendoring its source into this repo)."
    ) from e

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.ivon_diffusion_unet_lowdim_policy import IvonDiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager, LastNCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


@torch.no_grad()
def _ivon_hess_stats(optimizer):
    """
    Summary stats of IVON's internal Hessian estimate (`hess`, per
    ivon._ivon.IVON._new_hess) and the posterior sigma it implies
    (sigma^2 = 1/(ess*(hess+weight_decay)), same quantity IVON itself uses
    to sample - see _sample_params). Logged every step: `hess` starting at
    `hess_init` and shrinking over training directly grows the injected
    sampling noise (sigma) - worth watching for exactly the kind of late,
    accelerating train_loss divergence seen in earlier long runs (confirmed
    post-hoc from a checkpoint: hess collapsed to ~150x below hess_init by
    step 500k, sigma ~12x larger than if hess had stayed put).
    """
    hess_parts, sigma_parts = [], []
    for group in optimizer.param_groups:
        hess = group["hess"]
        hess_parts.append(hess)
        sigma_parts.append(1.0 / (group["ess"] * (hess + group["weight_decay"])).sqrt())
    hess_all = torch.cat([h.flatten() for h in hess_parts])
    sigma_all = torch.cat([s.flatten() for s in sigma_parts])
    return {
        'ivon_hess_mean': hess_all.mean().item(),
        'ivon_hess_min': hess_all.min().item(),
        'ivon_hess_max': hess_all.max().item(),
        'ivon_sigma_mean': sigma_all.mean().item(),
        'ivon_sigma_max': sigma_all.max().item(),
    }


# %%
class TrainIvonDiffusionUnetLowdimWorkspace(BaseWorkspace):
    """
    IVON counterpart of TrainDiffusionUnetLowdimWorkspace: same model
    (ConditionalUnet1D), same dataset/dataloader/noise-scheduler/EMA setup,
    same rollout/validation/NLL-bound/reconstruction-loss logic - the only
    real difference is `ivon.IVON` instead of `torch.optim.AdamW` as the
    optimizer, which is what makes this a Bayesian (mean-field Gaussian
    posterior) network with no architectural changes at all - see
    IvonDiffusionUnetLowdimPolicy's docstring.

    What IVON requires that AdamW didn't (the only structural differences
    from TrainDiffusionUnetLowdimWorkspace):
      - IVON needs `ess`/`weight_decay` hyperparameters (they define the
        implicit prior and posterior scale - there is no equivalent for
        AdamW). Set here as `ess = len(dataset)`, `weight_decay =
        1/(ess*prior_std^2)` for a `cfg.training.prior_std`-controlled
        implicit N(0, prior_std^2*I) prior (default 0.3, matching the
        IVON paper's own CIFAR-10/100/TinyImageNet setting - NOT
        prior_std=1, which none of the paper's real experiments actually
        use, and which was found to let `hess` collapse over long runs -
        see IvonDiffusionUnetLowdimPolicy's docstring and this file's
        __init__). Because that needs
        `n_bound = len(dataset)`, and BaseWorkspace.load_checkpoint() needs
        `self.optimizer` to already exist before it can restore its state
        on resume, the dataset is instantiated once here in __init__
        (cheap for lowdim tasks) purely to read off its length, and reused
        (not re-instantiated) in run().
      - IVON caches the device/dtype of its parameters at construction time
        and never re-queries it, so the model must be moved to its target
        device *before* the optimizer is built (AdamW doesn't care about
        this ordering) - see __init__'s comment.
      - The training step samples weights via `optimizer.sampled_params()`
        around the forward/backward pass instead of a plain
        backward()/step()/zero_grad() - see run()'s training loop.
      - `pytorch_util.optimizer_to()` is never called on the IVON optimizer:
        it assumes the standard per-parameter `optimizer.state[param]`
        layout, but IVON keeps its transient MC-sample accumulators as flat
        (non-tensor) entries directly in `self.state` and its persistent
        buffers in `param_groups`, not `state` at all.
    """
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: DictConfig, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: IvonDiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: IvonDiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Instantiated here (not only in run()) purely to read off
        # len(dataset) = ess before the optimizer is built - see class
        # docstring. Reused (not rebuilt) in run().
        self._dataset: BaseLowdimDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(self._dataset, BaseLowdimDataset)
        ess = len(self._dataset)

        # prior precision = ess * weight_decay (confirmed against the IVON
        # paper's own cross-reference: "Laplace... prior precision is set
        # to 10.0 corresponding to the same prior setup as other methods",
        # which is exactly their CIFAR-10 ess=50000 * weight_decay=2e-4=10),
        # so weight_decay = 1/(ess*prior_std^2) gives an implicit
        # N(0, prior_std^2*I) prior. Default here (0.3) matches the
        # authors' own CIFAR-10/100/TinyImageNet setting (delta=2e-4,
        # ess=50000 -> prior_std=sqrt(1/10)=0.316) - notably NOT the
        # N(0,I) (prior_std=1) this workspace used previously: none of the
        # paper's real experiments actually use prior_std=1 (ResNet-50/
        # ImageNet uses ~0.125, GLUE finetuning ~0.5-5.8 depending on task
        # size) - see cfg.training.prior_std's comment in the yaml for the
        # full table. A prior_std=1 gives weight_decay a much smaller floor
        # in sigma^2=1/(ess*(hess+weight_decay)), which was found (by
        # inspecting a completed run's saved optimizer state) to let `hess`
        # collapse ~150x below hess_init over 500k steps, driving sigma
        # ~12x higher and coinciding with the late-training train_loss
        # divergence - a tighter, paper-validated prior gives weight_decay
        # a real floor against exactly that failure mode.
        prior_std = float(cfg.training.get('prior_std', 0.3))
        weight_decay = 1.0 / (ess * prior_std ** 2)
        print(
            f"[IVON] ess=len(dataset)={ess}, prior_std={prior_std} "
            f"-> weight_decay={weight_decay:.6g} "
            f"(implicit prior = N(0, {prior_std}^2 * I))"
        )

        # Move the model(s) to the target device BEFORE constructing IVON -
        # see class docstring.
        self.device = torch.device(cfg.training.device)
        self.model.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)

        # configure training state - scoped to self.model.model.parameters()
        # (just the ConditionalUnet1D), NOT self.model.parameters() (the
        # whole policy): the policy also owns non-network parameters (e.g.
        # the normalizer's) that never appear in the loss graph, and IVON
        # (unlike AdamW, which silently tolerates p.grad is None) raises if
        # any tracked parameter never receives a gradient.
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.model.parameters(),
            ess=ess, weight_decay=weight_decay)
        self.model.set_ivon_optimizer(self.optimizer)
        # Deliberately NOT set on self.ema_model: there is no well-defined
        # "EMA of the posterior variance" (IVON's `hess` lives on
        # self.optimizer, tied to self.model's own parameter tensors, not
        # the EMA copy's) - evaluation stays fully deterministic for both
        # models, matching TrainDiffusionUnetLowdimWorkspace exactly, so
        # this is moot in practice.

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

        # configure dataset (reuse the one built in __init__ for ess)
        dataset: BaseLowdimDataset = self._dataset
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print("validation dataset size: ", len(val_dataset))

        # configure dataset for covariance_spectrum
        cov_dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=1, pin_memory=True, persistent_workers=False)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # LinearNormalizer/DictOfTensorMixin overrides _load_from_state_dict
        # to *replace* its params_dict wholesale with freshly-cloned tensors
        # from whatever device `normalizer` (built straight from the
        # dataset, i.e. CPU) was on - it does not preserve the destination
        # device the way a plain in-place state_dict load would. Since
        # __init__ already moved self.model/self.ema_model to self.device
        # (required for IVON), set_normalizer() just silently put the
        # normalizer back on CPU; re-apply .to() to fix only that
        # (self.model.model's parameters are already correct and this is a
        # no-op for them).
        self.model.to(self.device)
        if cfg.training.use_ema:
            self.ema_model.to(self.device)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.num_updates,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1
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

        # configure checkpoint for topk based on score
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint_max_score.topk
        )

        # configure checkpoint to save last N checkpoints
        lastN_manager = LastNCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint_last_N.topk
        )

        # Model(s) and optimizer are already on the right device - see
        # __init__ (IVON needs to be constructed after that move, not
        # before it; optimizer_to() is never used here - see class
        # docstring).
        device = self.device

        if cfg.training.debug:
            cfg.training.num_updates = 1000
            cfg.training.max_train_steps = 100
            cfg.training.max_val_steps = 100
            cfg.training.rollout_every = 100
            cfg.training.checkpoint_every = 100
            cfg.training.val_every = 100

        # compute covariance_spectrum of the training data
        self.model.dataset_info(cov_dataloader, covariance_spectrum=None, diagonal=False)
        if cfg.training.use_ema:
            self.ema_model.dataset_info(cov_dataloader, covariance_spectrum=None, diagonal=False)

        num_mc_samples = int(cfg.get('num_mc_samples', 1))

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            # ensure local control vars from cfg
            num_updates = int(cfg.training.num_updates)
            rollout_every = int(cfg.training.rollout_every)
            val_every = int(cfg.training.val_every)
            nll_every = int(cfg.training.nll_every)
            reconst_loss_every = int(cfg.training.reconst_loss_every)
            checkpoint_every = int(cfg.training.checkpoint_every)

            # training: run until we hit num_updates
            while self.global_step < num_updates:
                step_log = dict()

                with tqdm.tqdm(train_dataloader, desc=f"Training step {self.global_step}",
                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:

                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        # --- IVON step: sample weights, forward, backward, update ---
                        # (AdamW's plain loss.backward(); optimizer.step();
                        # optimizer.zero_grad() becomes this sampled_params()
                        # bracket, run num_mc_samples times before a single
                        # closure-less optimizer.step() - see IVON's
                        # documented usage pattern)
                        for _ in range(num_mc_samples):
                            with self.optimizer.sampled_params(train=True):
                                self.optimizer.zero_grad()
                                raw_loss = self.model.compute_loss(batch, train=True)
                                raw_loss.backward()
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)

                        # step optimizer and scheduler
                        self.optimizer.step()
                        lr_scheduler.step()

                        # update ema after optimizer step
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # build step-log (use the upcoming/global step index)
                        current_step = self.global_step + 1
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': current_step,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        step_log.update(_ivon_hess_stats(self.optimizer))

                        # evaluation runs after optimizer step
                        policy = self.ema_model if cfg.training.use_ema else self.model
                        policy.eval()

                        # run rollout
                        if (current_step % rollout_every) == 0 or self.global_step == 0:
                            runner_log = env_runner.run(policy)
                            step_log.update(runner_log)

                        # validation: noise prediction loss
                        if ((current_step % val_every) == 0 or self.global_step == 0) and (len(val_dataloader) > 0):
                            with torch.no_grad():
                                val_losses = []
                                with tqdm.tqdm(val_dataloader, desc=f"Validation step {current_step}: Noise Prediction Loss on test set",
                                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as vepoch:
                                    n_samples_total = 0
                                    for v_idx, vbatch in enumerate(vepoch):
                                        n_samples = len(vbatch["obs"])
                                        n_samples_total = n_samples_total + n_samples
                                        vbatch = dict_apply(vbatch, lambda x: x.to(device, non_blocking=True))
                                        val_loss = policy.compute_loss(vbatch, train=False)
                                        val_losses.append(val_loss.item() * n_samples)
                                        if (cfg.training.max_val_steps is not None) and v_idx >= (cfg.training.max_val_steps - 1):
                                            break
                                if len(val_losses) > 0:
                                    noise_loss = np.sum(val_losses) / n_samples_total
                                    step_log['test_noise_pred_loss'] = noise_loss

                        # NLL bound
                        if ((current_step % nll_every) == 0 or self.global_step == 0) and (len(val_dataloader) > 0):
                            NLL_test = policy.nll_bound(val_dataloader, current_step, npoints=100)
                            step_log['test_nll_bpd'] = NLL_test

                        # reconstruction loss
                        if ((current_step % reconst_loss_every) == 0 or self.global_step == 0) and (len(val_dataloader) > 0):
                            reconst_loss = policy.compute_action_reconst_loss(val_dataloader, cfg)
                            step_log['test_action_reconst_loss'] = reconst_loss.item()

                        policy.train()

                        # checkpointing (last N)
                        if (current_step % checkpoint_every) == 0:
                            if cfg.checkpoint_last_N.save_last_ckpt:
                                self.save_checkpoint()
                            if cfg.checkpoint_last_N.save_last_snapshot:
                                self.save_snapshot()

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
    workspace = TrainIvonDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

# %%
