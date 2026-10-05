#!/bin/bash -l

#SBATCH --job-name=flat-eval
#SBATCH --time=3:00:00
#SBATCH --account=eu-26-54
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --error=./job_outputs/flat.%A_%a.err
#SBATCH --output=./job_outputs/flat.%A_%a.out

# Flatness metrics (eval_ckpts_dp.py --flatness_only) of the last LAST_N checkpoints of every run
# listed in RUN_LIST (one run name per line, relative to OUT_ROOT); one array task per run.
#   RUN_LIST=flat_runs.txt sbatch --array=0-$(( $(wc -l < flat_runs.txt) - 1 )) run-flat-eval.sh
# Writes <run>/flat_eval/flatness_log_<timestamp>.json; aggregate with aggregate_flatness.py.

RUN_LIST=${RUN_LIST:-flat_runs.txt}
OUT_ROOT=${OUT_ROOT:-/mnt/proj1/eu-26-52/data/outputs}
LAST_N=${LAST_N:-10}
SIGMA_PARAM=${SIGMA_PARAM:-softplus}

## Conda activation
module load Anaconda3
source activate DP

## Task execution
cd ~/documents/codes/diffusion_policy
export HYDRA_FULL_ERROR=1

RUN=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$RUN_LIST")
echo "task $SLURM_ARRAY_TASK_ID: $RUN (code $(git rev-parse --short HEAD), host $(hostname))"

python3 eval_ckpts_dp.py \
        -c "$OUT_ROOT/$RUN/checkpoints/" \
        -o "$OUT_ROOT/$RUN/flat_eval" \
        --flatness_only --last_n "$LAST_N" --sigma_param "$SIGMA_PARAM"
