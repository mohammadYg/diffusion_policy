#!/bin/bash -l

#SBATCH --job-name=dp-train
#SBATCH --time=15:00:00
#SBATCH --account=eu-26-54
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=96
#SBATCH --mem=128000
#SBATCH --error=./job_outputs/job.%j.err
#SBATCH --output=./job_outputs/job.%j.out
#SBATCH --array=0-2

SEEDS=(100 200 300)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

TASK=${TASK:-'can_lowdim_abs'}
DEMOS=${DEMOS:-100}
DATE=${DATE:-$(date +"%Y.%m.%d")}
UPDATES=${UPDATES:-5e5}
VAL_RATIO=${VAL_RATIO:-0.0}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5e3}

echo "Running seed $SEED"

## Conda activation
module load Anaconda3
source activate DP

export WANDB_API_KEY="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"

## Task execution
cd ~/documents/codes/diffusion_policy

## Environments Variables
export HYDRA_FULL_ERROR=1

OUT_DIR="/mnt/proj1/eu-26-52/data/outputs/${TASK}_dp_demos_${DEMOS}_val_ratio_${VAL_RATIO}_updates_${UPDATES}_${DATE}_seed_${SEED}"
mkdir -p "$OUT_DIR"


# Training
TRAIN_START=$(date +%s)
python3 train.py \
  --config-name=train_diffusion_unet_ddim_lowdim_workspace.yaml \
  task=$TASK \
  training.seed=$SEED \
  training.device=cuda \
  training.num_updates=$UPDATES \
  task.dataset.val_ratio=$VAL_RATIO \
  task.dataset.max_train_episodes=$DEMOS \
  obs_as_global_cond=True \
  policy.eta=0.0 \
  hydra.run.dir="$OUT_DIR" \
  training.resume=True \
  training.checkpoint_every=$CHECKPOINT_EVERY \
  task.env_runner.n_envs=0 \
  task.env_runner.n_test=0 \
  training.rollout_every=5000 \
  training.val_every=5000 \
  training.nll_every=5000
TRAIN_EXIT=$?
TRAIN_END=$(date +%s)

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT - skipping eval." >&2
    exit $TRAIN_EXIT
fi

echo "training_elapsed_sec=$((TRAIN_END - TRAIN_START))" | tee "$OUT_DIR/train_time.log"


# Evaluation
EVAL_START=$(date +%s)
python3 eval_ckpts_dp.py \
        -c "$OUT_DIR/checkpoints/" \
        --override policy.eta=0.0 \
        --override task.env_runner.n_envs=96 \
        --override task.env_runner.n_test=1000
EVAL_END=$(date +%s)
echo "eval_elapsed_sec=$((EVAL_END - EVAL_START))" | tee "$OUT_DIR/eval_time.log"

