#!/bin/bash -l

#SBATCH --job-name=ddp-drift-train
#SBATCH --time=8:00:00
#SBATCH --account=eu-26-54
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=128000
#SBATCH --error=./job_outputs/job.%j.err
#SBATCH --output=./job_outputs/job.%j.out
#SBATCH --array=0-2

SEEDS=(100 200 300)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

NGPUS=${NGPUS:-4}
TASK=${TASK:-'tool_hang_lowdim_abs'}
DEMOS=${DEMOS:-200}
DATE=${DATE:-$(date +"%Y.%m.%d")}
UPDATES=${UPDATES:-2.5e5}
VAL_RATIO=${VAL_RATIO:-0.0}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5e3}
GEN_PER_LABEL=${GEN_PER_LABEL:-3}
PER_TIMESTEP_LOSS=${PER_TIMESTEP_LOSS:-True}
TEMPERATURES=${TEMPERATURES:-"[0.02,0.05,0.2]"}
LR_SCALE_POWER=${LR_SCALE_POWER:-1.0}
# Matches run-drift-train.sh's default. The workspace's own DDP scaling
# divides this by world_size like checkpoint_every - without overriding it
# here, it would fall back to the yaml's val_every=1e3 (250 steps at 4 GPUs),
# validating ~10x more often per sample seen than the single-GPU pipeline
# whenever VAL_RATIO>0 (currently a no-op at the default VAL_RATIO=0.0).
VAL_EVERY=${VAL_EVERY:-1e4}

# Filesystem-safe tag for TEMPERATURES ("[0.02,0.05,0.2]" -> "0.02-0.05-0.2")
# so distinct temperature sets never collide in OUT_DIR - a real collision
# bug in the past: two same-day runs with different TEMPERATURES (not
# otherwise part of the path) resumed into each other's checkpoints.
TEMPERATURES_TAG=$(echo "$TEMPERATURES" | tr -d '[] ' | tr ',' '-')

echo "Running seed $SEED on $NGPUS GPUs"

## Conda activation
module load Anaconda3
source activate DP

export WANDB_API_KEY="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"

## Task execution
cd ~/documents/codes/diffusion_policy
export HYDRA_FULL_ERROR=1

OUT_DIR="/mnt/proj1/eu-26-52/data/outputs/${TASK}_ddp-drift_ngpus_${NGPUS}_demos_${DEMOS}_val_ratio_${VAL_RATIO}_updates_${UPDATES}_gen_per_label_${GEN_PER_LABEL}_per_timestep_loss_${PER_TIMESTEP_LOSS}_temperatures_${TEMPERATURES_TAG}_lr_scale_power_${LR_SCALE_POWER}_${DATE}_seed_${SEED}"
mkdir -p "$OUT_DIR"

# Training (single-GPU-equivalent values below; the script auto-scales
# num_updates/lr_warmup_steps/checkpoint_every by world_size and optimizer.lr
# by world_size**ddp_lr_scale_power - see ddp_train_drift_unet_lowdim_workspace.py)
TRAIN_START=$(date +%s)
torchrun --standalone --nproc_per_node=$NGPUS \
  diffusion_policy/workspace/ddp_train_drift_unet_lowdim_workspace.py \
  --config-name=ddp_train_drift_unet_lowdim_workspace.yaml \
  task=$TASK \
  training.seed=$SEED \
  training.num_updates=$UPDATES \
  task.dataset.val_ratio=$VAL_RATIO \
  task.dataset.max_train_episodes=$DEMOS \
  obs_as_global_cond=True \
  hydra.run.dir="$OUT_DIR" \
  training.resume=True \
  training.checkpoint_every=$CHECKPOINT_EVERY \
  training.val_every=$VAL_EVERY \
  policy.per_timestep_loss=$PER_TIMESTEP_LOSS \
  policy.gen_per_label=$GEN_PER_LABEL \
  policy.temperatures="$TEMPERATURES" \
  training.ddp_lr_scale_power=$LR_SCALE_POWER
TRAIN_EXIT=$?
TRAIN_END=$(date +%s)

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "Training failed with exit code $TRAIN_EXIT - skipping eval." >&2
    exit $TRAIN_EXIT
fi

echo "training_elapsed_sec=$((TRAIN_END - TRAIN_START))" | tee "$OUT_DIR/train_time.log"

# Evaluation (unchanged from the single-GPU drift pipeline)
EVAL_START=$(date +%s)
python3 eval_ckpts_drift.py \
        -c "$OUT_DIR/checkpoints/" \
        --override task.env_runner.n_envs=96 \
        --override task.env_runner.n_test=1000
EVAL_END=$(date +%s)
echo "eval_elapsed_sec=$((EVAL_END - EVAL_START))" | tee "$OUT_DIR/eval_time.log"
