"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
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
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_lowdim_pac_policy import BaseLowdimPacPolicy
from omegaconf import OmegaConf
import numpy as np
import csv
from pathlib import Path

def write_player_csv(path: Path, times: np.ndarray, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp"]
    header += [f"panda_joint{i}_pos" for i in range(1, 8)]
    header += [f"panda_joint{i}_vel" for i in range(1, 8)]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t, q in zip(times, positions):
            writer.writerow(
                [f"{t:.9f}"]
                + [f"{value:.10f}" for value in q]
            )

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
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    if isinstance(policy, BaseLowdimPacPolicy):   
        name="PAC_DP"
    else:                 
        name="DP"

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir)

    runner_log, all_generated_actions = env_runner.run(policy, cfg)
    avg_success_rate = np.mean(runner_log["test/mean_score"])
   
    # save the generated actions
    dt = 0.05 
    for traj_idx in range(all_generated_actions.shape[0]):

        positions = all_generated_actions[traj_idx]

        timestamps = np.arange(
            len(positions)
        ) * dt

        write_player_csv(
            Path(f"data/real_exp_models/tool_hang/trajectories/traj_{name}_{traj_idx:03d}.csv"),
            timestamps,
            positions
        )

    json_log = dict()
    json_log["test/mean_score_avg"] = avg_success_rate

    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        elif isinstance(value, torch.Tensor):
            # completely remove tensor entries
            continue
        else:
            json_log[key] = value
            
    out_path = os.path.join(output_dir, f"eval_log_{name}.json")
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
