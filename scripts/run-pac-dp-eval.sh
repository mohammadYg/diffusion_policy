#!/bin/bash -l

#SBATCH --job-name=pac-dp-eval
#SBATCH --time=2:00:00
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
DEMOS=${DEMOS:-100}
TASK=${TASK:-'lift_lowdim_abs'}
DATE=${DATE:-$(date +"%Y.%m.%d")}
UPDATES=${UPDATES:-5e5}
VAL_RATIO=${VAL_RATIO:-0.0}
PAC_OBJ=${PAC_OBJ:-'fquad'}
RHO_INT=${RHO_INT:--3.5}
KL=${KL:-0.0001}

echo "Running seed $SEED"

## Conda activation
module load Anaconda3
source activate DP

export WANDB_API_KEY="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"

## Task execution
cd ~/documents/codes/diffusion_policy

## Environments Variables
export HYDRA_FULL_ERROR=1

# NOTE: this OUT_DIR template must always match run-pac-dp-train.sh's
# exactly (same fields, same order), since this script only re-runs eval
# against an existing training run's checkpoints, it doesn't recreate them.
OUT_DIR="/mnt/proj1/eu-26-52/data/outputs/${TASK}_pac-dp_demos_${DEMOS}_val_ratio_${VAL_RATIO}_updates_${UPDATES}_pac_obj_${PAC_OBJ}_rho_${RHO_INT}_kl_${KL}_${DATE}_seed_${SEED}"

# Evaluation
python3 eval_ckpts_dp.py \
        -c "$OUT_DIR/checkpoints/" \
        --override policy.eta=1.0 \
        --override task.env_runner.n_envs=500 \
        --override task.env_runner.n_test=1000 \
	--override eval.stochastic=False

