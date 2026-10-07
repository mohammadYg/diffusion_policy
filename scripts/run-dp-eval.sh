#!/bin/bash -l

#SBATCH --job-name=dp-eval
#SBATCH --time=4:00:00
#SBATCH --account=eu-26-52
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512000
#SBATCH --error=./job_outputs/job.%j.err
#SBATCH --output=./job_outputs/job.%j.out


SEED=${SEED:-100}
TASK=${TASK:-'can_lowdim_abs'}
DEMOS=${DEMOS:-100}
DATE=${DATE:-$(date +"%Y.%m.%d")}
UPDATES=${UPDATES:-5e5}
VAL_RATIO=${VAL_RATIO:-0.0}

echo "Running seed $SEED"

## Conda activation
module load Anaconda3
source activate DP

export WANDB_API_KEY="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"

## Task execution
cd ~/documents/codes/diffusion_policy

## Environments Variables
export HYDRA_FULL_ERROR=1

##OUT_DIR="/ceph/hpc/data/d2026d04-094-users/data/outputs/${TASK}_dp_demos_${DEMOS}_${DATE}_seed_${SEED}"
OUT_DIR="/mnt/proj1/eu-26-52/data/outputs/${TASK}_dp_demos_${DEMOS}_val_ratio_${VAL_RATIO}_updates_${UPDATES}_${DATE}_seed_${SEED}"


# Evaluation
python3 eval_ckpts_dp.py \
        -c "$OUT_DIR/checkpoints/" \
        --override policy.eta=1.0 \
        --override task.env_runner.n_envs=500 \
        --override task.env_runner.n_test=1000

python3 eval_ckpts_dp.py \
        -c "$OUT_DIR/checkpoints/" \
        --override policy.eta=0.0 \
        --override task.env_runner.n_envs=500 \
        --override task.env_runner.n_test=1000


