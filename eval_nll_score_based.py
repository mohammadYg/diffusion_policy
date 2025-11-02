"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.policy.base_lowdim_prob_policy import BaseLowdimProbPolicy
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--override', multiple=True,
              help="Hydra-style config overrides, e.g. task.env_runner.n_test=300")
def main(checkpoint, output_dir, device, override):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    # apply overrides (if any)
    if override:
        override_cfg = OmegaConf.from_dotlist(override)
        cfg = OmegaConf.merge(cfg, override_cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # configure dataset
    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseLowdimDataset)

    # configure validation dataset
    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

    if isinstance(policy, BaseLowdimProbPolicy):
        NLL = policy.nll_sde(val_dataloader, stochastic = cfg.eval.stochastic, clamping = cfg.eval.clamping)
    else:
        NLL = policy.nll_sde(val_dataloader)
    print ('nll_bpd', NLL)

if __name__ == '__main__':
    main()
