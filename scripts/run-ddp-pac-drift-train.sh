#!/bin/bash -l

#SBATCH --job-name=ddp-pac-drift-train
#SBATCH --time=12:00:00
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
PAC_OBJ=${PAC_OBJ:-'friendly'}
# Fan-in-relative posterior init noise (sigma = SIGMA_SCALE * sigma_weights
# per layer, where sigma_weights = 1/sqrt(fan_in)) - replaces the old flat
# rho_post/rho_prior sweep, which applied one absolute sigma to every layer
# regardless of width and left wide/bottleneck layers (down_dims up to 1024)
# proportionally far noisier than narrow ones. prior_sigma_scale/rho_prior are
# deliberately left unset here (falling back to the yaml's own rho_prior=0.54
# default, sigma=1.0) so the prior stays a fixed, broad reference instead of
# shrinking in lockstep with whatever posterior noise level is being swept -
# the old rho_post=rho_prior=$RHO_INT pattern confounded "posterior noise
# level" with "prior sharpness" and made the KL term's meaning drift across
# the sweep.
SIGMA_SCALE=${SIGMA_SCALE:-0.1}
KL=${KL:-0.0001}
# Matches the single-GPU pac-drift yaml's default (7.5) explicitly, since the
# single-GPU launch script never overrides this field itself (relies on that
# yaml default) - kept as an explicit override here (rather than relying on
# the DDP yaml's own default) so a future drift between the two yamls'
# defaults can't silently break single-GPU vs DDP comparability again.
LOSS_SCALE=${LOSS_SCALE:-7.5}
LR_SCALE_POWER=${LR_SCALE_POWER:-0.0}
# Matches run-pac-drift-train.sh's default. See run-ddp-drift-train.sh's
# VAL_EVERY comment for why this needs an explicit override under DDP.
VAL_EVERY=${VAL_EVERY:-1e4}

# Filesystem-safe tag for TEMPERATURES ("[0.02,0.05,0.2]" -> "0.02-0.05-0.2")
# so distinct temperature sets never collide in OUT_DIR - see
# run-drift-train.sh's comment for the collision this is guarding against.
TEMPERATURES_TAG=$(echo "$TEMPERATURES" | tr -d '[] ' | tr ',' '-')

echo "Running seed $SEED on $NGPUS GPUs"

## Conda activation
module load Anaconda3
source activate DP

export WANDB_API_KEY="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"

## Task execution
cd ~/documents/codes/diffusion_policy
export HYDRA_FULL_ERROR=1

OUT_DIR="/mnt/proj1/eu-26-52/data/outputs/${TASK}_ddp-pac-drift_ngpus_${NGPUS}_demos_${DEMOS}_val_ratio_${VAL_RATIO}_updates_${UPDATES}_gen_per_label_${GEN_PER_LABEL}_per_timestep_loss_${PER_TIMESTEP_LOSS}_temperatures_${TEMPERATURES_TAG}_pac_obj_${PAC_OBJ}_sigma_scale_${SIGMA_SCALE}_kl_${KL}_loss_scale_${LOSS_SCALE}_lr_scale_power_${LR_SCALE_POWER}_${DATE}_seed_${SEED}"
mkdir -p "$OUT_DIR"

# Training (single-GPU-equivalent values below; the script auto-scales
# num_updates/lr_warmup_steps/checkpoint_every by world_size and optimizer.lr
# by world_size**ddp_lr_scale_power - see ddp_train_pac_drift_unet_lowdim_workspace.py).
# data_dependent_prior stays at the config's default (False) - init_model_path/
# post_sigma_scale below only seeds the (non-data-dependent) posterior
# directly, it doesn't enable the separate prior-training phase.
# policy.model.rho_prior/prior_sigma_scale are deliberately NOT overridden -
# see the SIGMA_SCALE comment above.
TRAIN_START=$(date +%s)
torchrun --standalone --nproc_per_node=$NGPUS \
  diffusion_policy/workspace/ddp_train_pac_drift_unet_lowdim_workspace.py \
  --config-name=ddp_train_pac_drift_unet_lowdim_workspace.yaml \
  task=$TASK \
  training.seed=$SEED \
  training.num_updates=$UPDATES \
  training.kl_penalty=$KL \
  training.pac_objective=$PAC_OBJ \
  training.loss_scale=$LOSS_SCALE \
  policy.model.post_sigma_scale=$SIGMA_SCALE \
  policy.model.init_prior='random' \
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

# Evaluation (unchanged from the single-GPU pac-drift pipeline) - stochastic
# (sampled Bayesian weights) and deterministic (posterior mean) variants
# separately. eval_ckpts_drift.py's rollout and validation-loss stochasticity
# both come from cfg.eval.stochastic (--override eval.stochastic=...).
EVAL_START=$(date +%s)
python3 eval_ckpts_drift.py \
        -c "$OUT_DIR/checkpoints/" \
        --override task.env_runner.n_envs=96 \
        --override task.env_runner.n_test=1000 \
        --override eval.stochastic=True

python3 eval_ckpts_drift.py \
        -c "$OUT_DIR/checkpoints/" \
        --override task.env_runner.n_envs=96 \
        --override task.env_runner.n_test=1000 \
        --override eval.stochastic=False

EVAL_END=$(date +%s)
echo "eval_elapsed_sec=$((EVAL_END - EVAL_START))" | tee "$OUT_DIR/eval_time.log"
