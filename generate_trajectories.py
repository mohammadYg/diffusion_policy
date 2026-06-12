"""
Usage:
python generate_trajectories.py -c data/model/latest.ckpt -o data/generated_trajectories 
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_lowdim_pac_policy import BaseLowdimPacPolicy
from omegaconf import OmegaConf
import numpy as np
import csv
from pathlib import Path
from diffusion_policy.env_runner.robomimic_lowdim_single_runner import SingleEnvPandaRunner

def write_player_csv(path: Path, times: np.ndarray, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp"]
    header += [f"panda_joint{i}_pos" for i in range(1, 8)]
    header += [f"finger_joint{i}" for i in range(1, 3)]
    
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t, q in zip(times, positions):
            writer.writerow([f"{t:.9f}"] + [f"{value:.10f}" for value in q])


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option(
    "--override",
    multiple=True,
    help="Hydra-style config overrides, e.g. task.env_runner.n_test=300",
)
def main(checkpoint, output_dir, device, override):
    if os.path.exists(output_dir):
        click.confirm(
            f"Output path {output_dir} already exists! Overwrite?", abort=True
        )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    print("payload keys:", payload.keys())
    cfg = payload["cfg"]

    # apply overrides (if any)
    if override:
        override_cfg = OmegaConf.from_dotlist(override)
        cfg = OmegaConf.merge(cfg, override_cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(
        payload, exclude_keys=["model", "optimizer"], include_keys=None
    )

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    if isinstance(policy, BaseLowdimPacPolicy):
        name = "PAC_DP"
    else:
        name = "DP"

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    qpos = []
    qvel = []
    act = []
    for seed in range (10000, 10000 + 5):
        runner = SingleEnvPandaRunner(
            cfg,
            seed=seed,
            demo_idx=None,      # random seed reset
        )

        results = runner.run(policy, cfg)
        qpos.append(results["joint_positions"])      # (T,7)
        qvel.append(results["joint_velocities"])     # (T,7)
        act.append(results["actions"])
        print(results["actions"].shape)

    ## save the generated actions
    qpos = np.stack(qpos, axis=0)    # (N_traj, T, 7)
    qvel = np.stack(qvel, axis=0)    # (N_traj, T, 7)
    act = np.stack(act, axis=0)      # (N_traj, T, 14)
    
    dt = 0.05
    for traj_idx in range(qpos.shape[0]):
        positions = qpos[traj_idx]

        timestamps = np.arange(len(positions)) * dt

        write_player_csv(
            Path(
                output_dir + f"traj_{name}_{traj_idx:03d}.csv"
            ),
            timestamps,
            positions,
        )

if __name__ == "__main__":
    main()
