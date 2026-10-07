#!/bin/bash -l

#SBATCH --job-name=pac-dp-train
#SBATCH --time=15:00:00
#SBATCH --account=eu-26-54
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --mem=512000
#SBATCH --error=./job_outputs/job.%j.err
#SBATCH --output=./job_outputs/job.%j.out
#SBATCH --array=0-2


SEEDS=(100 200 300)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DEMOS=${DEMOS:-100}
TASK=${TASK:-'lift_lowdim_abs'}
DATE=${DATE:-$(date +"%Y.%m.%d")}
UPDATES=${UPDATES:-5e5}
PAC_OBJ=${PAC_OBJ:-'fquad'}
RHO_INT=${RHO_INT:--3.5}
KL=${KL:-0.0001}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5e3}
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

OUT_DIR="/mnt/proj1/eu-26-52/data/outputs/${TASK}_pac-dp_demos_${DEMOS}_val_ratio_${VAL_RATIO}_updates_${UPDATES}_pac_obj_${PAC_OBJ}_rho_${RHO_INT}_kl_${KL}_${DATE}_seed_${SEED}"


TRAIN_START=$(date +%s)
python3 train.py \
        --config-name=train_pac_diffusion_unet_ddim_lowdim_workspace.yaml \
        task=$TASK training.seed=$SEED \
        training.device=cuda \
        training.num_updates=$UPDATES \
        training.kl_penalty=$KL \
        training.pac_objective=$PAC_OBJ \
        policy.model.rho_post=$RHO_INT \
        task.dataset.val_ratio=$VAL_RATIO \
        task.dataset.max_train_episodes=$DEMOS \
        policy.eta=0.0 \
        hydra.run.dir="$OUT_DIR" \
	training.resume=True \
	policy.model.init_prior='random' \
	policy.model.rho_prior=$RHO_INT \
	training.checkpoint_every=$CHECKPOINT_EVERY \
	task.env_runner.n_envs=500 \
        task.env_runner.n_test=1000 \
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
        --override policy.eta=1.0 \
        --override task.env_runner.n_envs=500 \
        --override task.env_runner.n_test=1000 \
        --override eval.stochastic=True

python3 eval_ckpts_dp.py \
        -c "$OUT_DIR/checkpoints/" \
        --override policy.eta=1.0 \
        --override task.env_runner.n_envs=500 \
        --override task.env_runner.n_test=1000 \
        --override eval.stochastic=False
EVAL_END=$(date +%s)
echo "eval_elapsed_sec=$((EVAL_END - EVAL_START))" | tee "$OUT_DIR/eval_time.log"

