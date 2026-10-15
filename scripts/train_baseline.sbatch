#!/bin/bash
# Train one baseline, predict the validation centre, score it and record the run in docs/results.xlsx
# and docs/EXPERIMENT_LOG.md, so the results arrive without anyone watching the job. One GPU on ws-ia.
#   sbatch --job-name=<run> --time=<hh:mm:ss> scripts/train_baseline.sbatch configs/<run>.yaml
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
#SBATCH --exclude=ws-l5-032,ws-l1-009
#SBATCH --cpus-per-task=8
#SBATCH --mem=11000M
#SBATCH --time=12:00:00
#SBATCH --output=/l/users/mohamed.abdalla/medai/outputs/slurm-%x-%j.log
set -euo pipefail
MEDAI=/l/users/mohamed.abdalla/medai
REPO=$HOME/MedAI-Project
CONFIG=$(realpath "$1")
# Caches stay on /l/users (the /home quota is small). Weights are downloaded beforehand, so run offline.
export PYTHONPATH=$REPO/src TORCH_HOME=$MEDAI/cache/torch HF_HOME=$MEDAI/cache/hf MPLCONFIGDIR=$MEDAI/cache/mpl
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "job $SLURM_JOB_ID on $(hostname), config $CONFIG, code $(git -C "$REPO" rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
# Stop at once if another process already holds this GPU (on 2026-10-15 busy GPUs on ws-l5-032 and ws-l1-009 failed 22 runs).
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 2000 ]; then
  echo "GPU busy ($used MiB used) on $(hostname), sending the run again on another node"
  # At most 5 tries; each try also skips the busy nodes seen so far.
  tries=${MEDAI_TRIES:-0}
  skip="${MEDAI_SKIP:-ws-l5-032,ws-l1-009},$(hostname -s)"
  if [ "$tries" -lt 5 ]; then
    MEDAI_TRIES=$((tries + 1)) MEDAI_SKIP=$skip sbatch --export=ALL --exclude="$skip" \
      --job-name="$SLURM_JOB_NAME" --time="$(squeue -h -j "$SLURM_JOB_ID" -o %l)" "$REPO/scripts/train_baseline.sbatch" "$CONFIG"
  fi
  exit 1
fi
# Timestamps and Slurm times in the log give the train time when sacct is unavailable.
echo "train start $(date -Is)"
"$MEDAI/venv/bin/python" -m medai.train "$CONFIG"
echo "train end $(date -Is)"
"$MEDAI/venv/bin/python" -m medai.predict "$CONFIG"
# The train time recorded is the Slurm run time up to here: training plus C5 prediction.
scontrol show job "$SLURM_JOB_ID" | grep -o -E "(StartTime|RunTime)=[^ ]+"
"$MEDAI/venv/bin/python" -m medai.evaluate "$CONFIG" --save --logs "$MEDAI/outputs/slurm-$SLURM_JOB_NAME-$SLURM_JOB_ID.log"
