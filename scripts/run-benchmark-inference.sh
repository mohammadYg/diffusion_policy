#!/bin/bash -l

#SBATCH --job-name=bench-inference
#SBATCH --time=00:30:00
#SBATCH --account=eu-26-54
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --error=./job_outputs/job.%j.err
#SBATCH --output=./job_outputs/job.%j.out

module load Anaconda3
source activate DP

cd ~/documents/codes/diffusion_policy
export HYDRA_FULL_ERROR=1

BENCH_START=$(date +%s)
python3 benchmark_predict_action.py \
  -c bench_ckpts.json \
  -o bench_results.json \
  -d cuda:0 \
  --batch-size 96 \
  --n-warmup 10 \
  --n-iters 50
BENCH_END=$(date +%s)
echo "bench_elapsed_sec=$((BENCH_END - BENCH_START))"
