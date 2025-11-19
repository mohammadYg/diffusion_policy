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
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.pytorch_util import dict_apply
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import numpy as np
import tqdm

@click.command()
@click.option('-c', '--ckpts_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--override', multiple=True,
              help="Hydra-style config overrides, e.g. task.env_runner.n_test=300")

def main(ckpts_dir, device, override):
    
    # Go one folder up (parent of checkpoints)
    parent_dir = os.path.dirname(os.path.normpath(ckpts_dir))

    # Create new eval_output directory inside the parent
    output_dir = os.path.join(parent_dir, "eval_output")

    if os.path.exists(output_dir):
        click.confirm(
            f"Output path {output_dir} already exists! Overwrite?", abort=True
        )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # configure best checkpoint saving
    topk_manager = TopKCheckpointManager(
        save_dir=output_dir,
        monitor_key="test_mean_score_avg",
        mode = "max",
        k=1,
        format_str='epoch={epoch:04d}-test_mean_score_avg={test_mean_score_avg:.3f}.ckpt'
    )
    
    json_log = dict()
    
    # List all .cpk files in the directory
    ckpt_files = sorted([f for f in os.listdir(ckpts_dir) if f.endswith('.ckpt')])
    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(ckpts_dir, ckpt_file)
        
        # load checkpoint
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']

        if ckpt_file == 'latest.ckpt':
            epoch = cfg.training.num_epochs
        else:
            epoch = int(ckpt_file.split("=")[1].split(".")[0])

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
        
        # run eval
        env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=output_dir)
        
        success_rate = list()
        for _ in range (cfg.task.n_repeat_runner):
            if isinstance(policy, BaseLowdimProbPolicy):
                runner_log = env_runner.run_prob(policy, cfg.eval.stochastic, cfg.eval.clamping)
            else:
                runner_log = env_runner.run(policy)
            success_rate.append(runner_log["test/mean_score"])

        avg_success_rate = np.mean(success_rate)
        var_success_rate = np.var(success_rate, ddof=0)

        #save best model
        success_rate = {"test_mean_score_avg": avg_success_rate, "epoch": epoch}
        topk_ckpt_path = topk_manager.get_ckpt_path(success_rate)

        if topk_ckpt_path is not None:
            workspace.save_checkpoint(path=topk_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        val_loss_noise_pred = list()
        with torch.no_grad():
            for _ in range (cfg.task.n_repeat_runner):
                val_losses_noise_pred = list()
                n_total_samples = 0

                with tqdm.tqdm(val_dataloader, desc=f"Noise Prediction Validation", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    
                    for batch in tepoch:
                        n_samples = len(batch["obs"])
                        n_total_samples += n_samples
                        
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        emp_loss_test = policy.compute_loss(batch)
                        val_losses_noise_pred.append(emp_loss_test.item() * n_samples)
                
                val_loss_noise_pred.append(torch.sum(torch.tensor(val_losses_noise_pred)).item()/n_total_samples)
            
            avg_loss_pred_noise = np.mean(val_loss_noise_pred)
            var_loss_pred_noise = np.var(val_loss_noise_pred, ddof=0)
        
            # Create hierarchical entry
            epoch_key = f"model_at_epoch_{epoch:04d}"
            json_log[epoch_key] = {
                "test": {
                    "mean_score_avg": avg_success_rate,
                    "mean_score_var": var_success_rate,
                    "loss_noise_pred_avg": avg_loss_pred_noise,
                    "loss_noise_pred_var": var_loss_pred_noise,
                }
            }
            
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
        del policy
        del workspace
        del env_runner
        del dataset
        del val_dataset
        del val_dataloader
        del payload
        del cfg
        
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
