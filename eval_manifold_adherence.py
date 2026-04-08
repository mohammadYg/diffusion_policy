"""
Usage:
python eval_manifold_adherence.py --checkpoint data/outputs/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
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
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.robomimic_lowdim_manifold_adh import ActionPerturbationEvaluator
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import numpy as np

def compute_knn_observations(perturbed_obs, dataset, k):
    """
    perturbed_obs: (M, 2, 23)
    dataset_obs:   (N, 16, 23)

    return:
        knn_obs: (M, k, 2, 23)
    """
    M, n_obs_step, D = perturbed_obs.shape
    # (N, H, Do) → (N, n_obs_step, Do)
    obs = dataset['obs'][:, :n_obs_step, :]  # (N, n_obs_step, Do)
    N = obs.shape[0]
    
    # Flatten for distance computation
    # (M, n_obs_step, Do) → (M, n_obs_step*Do)
    X = perturbed_obs.reshape(M, -1)

    # (N, n_obs_step, Do) → (N,n_obs_step*Do)
    D_flat = obs.reshape(N, -1)

    # Compute pairwise distances
    # (M, N)
    dists = torch.cdist(X, D_flat, p=2.0)

    # Get k nearest neighbors
    knn_dists, knn_idx = torch.topk(dists, k, dim=1, largest=False)

    print ("knn_idx shape:", knn_idx.shape)  # (M, k)
    # Gather actions neighbors
    # (M, k, n_obs_step, D)
    nearest_actions = list()    
    for i in range(M):
        nearest_actions.append(dataset['action'][knn_idx[i]])  # (k, n_action_step, Da)
    nearest_actions = torch.stack(nearest_actions, dim=0)  # (M, k, n_action_step, Da)    
    print ("nearest_actions shape:", nearest_actions.shape)
    return nearest_actions

def compute_manifold_adherence(pred_actions, knn_actions):
    """
    pred_actions: (M, T, Da)
    knn_actions:  (M, k, T, Da)
    """
    M, T, Da = pred_actions.shape
    k = knn_actions.shape[1]
    pred = pred_actions.reshape(M, -1)  # (M, T*Da)
    knn = knn_actions.reshape(M, k, -1)  # (M, k, T*Da)
    errors = []
    for i in range(M):
        a = pred[i]  # (T*Da,)
        A = knn[i].T  # (T*Da, k)
        # Solve least squares: min_c ||a - A @ c||_2
        sol = torch.linalg.lstsq(A, a)  
        c = sol.solution            # (k, 1)
        proj = A @ c  # (T*Da, 1)
        error = torch.norm(a - proj, p=2)
        errors.append(error)
    errors = torch.stack(errors)
    metric = errors.mean()
    return metric, errors

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

    # configure dataset
    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseLowdimDataset)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4)

    
    dataset_obs = list()
    dataset_actions = list()
    for batch in dataloader:
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        dataset_obs.append(batch['obs'])
        dataset_actions.append(batch['action'])
    
    dataset = {
        'obs': torch.cat(dataset_obs, dim=0),           # (N, H, Do)
        'action': torch.cat(dataset_actions, dim=0)    # (N, H, Da)
    }

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = ActionPerturbationEvaluator(dataset_path=cfg.task.dataset_path,
            obs_keys=cfg.task.obs_keys,
            n_test=1000,
            max_steps=cfg.task.env_runner.max_steps,
            n_obs_steps=cfg.task.env_runner.n_obs_steps,
            n_action_steps=cfg.task.env_runner.n_action_steps,
            abs_action=cfg.task.abs_action,
            n_envs=cfg.task.env_runner.n_envs,
            eps=0.6)

    # perturb action and get observations
    peturbed_action_obs = env_runner.run(policy, cfg)
    peturbed_action_obs = dict_apply(peturbed_action_obs, 
                lambda x: torch.from_numpy(x).to(
                    device=device))
    peturbed_obs = peturbed_action_obs['obs']
    
    # compute knn observations
    k_nearest_actions = compute_knn_observations(
        perturbed_obs=peturbed_obs,
        dataset=dataset,
        k=50
    )

    metric, _ = compute_manifold_adherence(peturbed_action_obs['action'], k_nearest_actions)
    print ("Manifold adherence metric:", metric.item())

    # json_log = dict()
    # json_log["test/mean_score_avg"] = avg_success_rate
    # json_log["test/mean_score_var"] = var_success_rate
    # json_log["test/mean_score_sd"] = np.sqrt(var_success_rate)
    # json_log["test/loss_noise_pred_avg"] = avg_loss_pred_noise
    # json_log["test/loss_noise_pred_var"] = var_loss_pred_noise
    # json_log["test/loss_noise_pred_sd"] = np.sqrt(var_loss_pred_noise)
            
    # out_path = os.path.join(output_dir, 'eval_log.json')
    # json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
