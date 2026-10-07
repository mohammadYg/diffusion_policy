#!/bin/bash -l

#SBATCH --job-name=ddp-drift-eval
#SBATCH --time=3:00:00
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

NGPUS=${NGPUS:-4}
TASK=${TASK:-'tool_hang_lowdim_abs'}
DEMOS=${DEMOS:-200}
DATE=${DATE:-'2026.09.22'}
UPDATES=${UPDATES:-2.5e5}
VAL_RATIO=${VAL_RATIO:-0.0}
GEN_PER_LABEL=${GEN_PER_LABEL:-3}
PER_TIMESTEP_LOSS=${PER_TIMESTEP_LOSS:-True}
TEMPERATURES=${TEMPERATURES:-"[0.02,0.05,0.2]"}
LR_SCALE_POWER=${LR_SCALE_POWER:-1.0}

# Filesystem-safe tag for TEMPERATURES - must match run-ddp-drift-train.sh's
# tag exactly, or this script constructs the wrong OUT_DIR and never finds
# the checkpoints the training run actually produced.
TEMPERATURES_TAG=$(echo "$TEMPERATURES" | tr -d '[] ' | tr ',' '-')

module load Anaconda3
source activate DP

cd ~/documents/codes/diffusion_policy
export HYDRA_FULL_ERROR=1

# NOTE: this OUT_DIR template must always match run-ddp-drift-train.sh's
# exactly (same fields, same order) - this script only re-runs eval against
# an existing training run's checkpoints, it doesn't recreate them. This
# template was just aligned to gain temperatures_${TEMPERATURES_TAG} and
# lr_scale_power_${LR_SCALE_POWER} (previously missing here entirely, so
# this script never matched what run-ddp-drift-train.sh actually wrote) -
# if you need to re-evaluate a specific already-completed run made with the
# OLD template (e.g. the DATE=2026.09.22 default below), pass that run's
# real OUT_DIR by overriding OUT_DIR directly instead of relying on this
# templated construction.
#
# WARNING: an explicit OUT_DIR override is a single literal path with no
# per-seed variation - it only makes sense combined with --array=0 (one
# task). Submitting it with the default --array=0-2 sends all 3 array
# tasks at the SAME directory: they evaluate one run three times at once
# and race on the same eval_log_<timestamp>.json/eval_time.log filenames.
OUT_DIR=${OUT_DIR:-"/mnt/proj1/eu-26-52/data/outputs/${TASK}_ddp-drift_ngpus_${NGPUS}_demos_${DEMOS}_val_ratio_${VAL_RATIO}_updates_${UPDATES}_gen_per_label_${GEN_PER_LABEL}_per_timestep_loss_${PER_TIMESTEP_LOSS}_temperatures_${TEMPERATURES_TAG}_lr_scale_power_${LR_SCALE_POWER}_${DATE}_seed_${SEED}"}

EVAL_START=$(date +%s)
python3 eval_ckpts_drift.py \
        -c "$OUT_DIR/checkpoints/" \
        --override task.env_runner.n_envs=96 \
        --override task.env_runner.n_test=1000
EVAL_END=$(date +%s)
echo "eval_elapsed_sec=$((EVAL_END - EVAL_START))" | tee "$OUT_DIR/eval_time.log"
