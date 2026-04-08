"""
Usage:
python eval_compound_error.py --checkpoint data/outputs/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
from pathlib import Path
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import pickle
import os
import pathlib
import click
import hydra
import torch
import dill
import json
import wandb
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.robomimic_lowdim_compound_error import RobomimicLowdimRunner
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import numpy as np

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=False, default=None, help="Where to write eval outputs")
@click.option('-d', '--device', default='cuda:0')
@click.option('--override', multiple=True,
              help="Hydra-style config overrides, e.g. task.env_runner.n_test=300")

def main(checkpoint, output_dir, device, override):
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    print ("payload keys:", payload.keys())
    cfg = payload['cfg']

    # apply overrides (if any)
    if override:
        override_cfg = OmegaConf.from_dotlist(override)
        cfg = OmegaConf.merge(cfg, override_cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=['model', 'optimizer'], include_keys=None)

    # get the epoch on the trained model
    epoch = workspace.epoch - 1
    print (epoch)
    
    # create the output dir if not specified     
    ckpts_dir = Path(os.path.dirname(checkpoint))
    parent_dir = ckpts_dir.parent
    if output_dir is None:
        output_dir = parent_dir / f"eval_output/epoch={epoch}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print (output_dir)
    
    # configure training dataset
    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseLowdimDataset)
    train_dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4)

    dataset_obs = list()
    dataset_actions = list()
    for batch in train_dataloader:
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        dataset_obs.append(batch['obs'])
        dataset_actions.append(batch['action'])
    
    dataset_train = {
        'obs': torch.cat(dataset_obs, dim=0),           # (N, H, Do)
        'action': torch.cat(dataset_actions, dim=0)    # (N, H, Da)
    }

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = RobomimicLowdimRunner(
            output_dir=output_dir,
            dataset_path=cfg.task.dataset_path,
            obs_keys=cfg.task.obs_keys,
            n_test=50,
            n_test_vis=50,
            max_steps=cfg.task.env_runner.max_steps,
            n_obs_steps=cfg.task.env_runner.n_obs_steps,
            n_action_steps=cfg.task.env_runner.n_action_steps,
            abs_action=cfg.task.abs_action,
            n_envs=cfg.task.env_runner.n_envs)

    # perturb action and get observations
    env_runner._get_epoch(epoch)
    runner_log, reconst_loss, manifold_adh, all_pred_actions = env_runner.run(policy, dataset_train, cfg)

    json_log = dict()
    json_log["test/mean_score"] = runner_log["test/mean_score"]
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif isinstance(value, torch.Tensor):
            # completely remove tensor entries
            continue
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

    np.savetxt(os.path.join(output_dir, 'reconstruction_loss.csv'), X=reconst_loss, header="rollout policy, loss", delimiter=",")
    np.savetxt(os.path.join(output_dir, 'manifold_adherence.csv'), X=manifold_adh, header="rollout policy, manifold adherence", delimiter=",")
    np.save(os.path.join(output_dir, 'all_pred_actions.npy'), all_pred_actions)

if __name__ == '__main__':
    main()
