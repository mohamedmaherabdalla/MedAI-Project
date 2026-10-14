#!/bin/bash
# Train one baseline, then predict the validation centre. One GPU on ws-ia.
#   sbatch --job-name=<run> --time=<hh:mm:ss> scripts/train_baseline.sbatch configs/<run>.yaml
#SBATCH --partition=ws-ia
#SBATCH --gres=gpu:1
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
# Timestamps and Slurm times in the log give the train time when sacct is unavailable.
echo "train start $(date -Is)"
"$MEDAI/venv/bin/python" -m medai.train "$CONFIG"
echo "train end $(date -Is)"
"$MEDAI/venv/bin/python" -m medai.predict "$CONFIG"
scontrol show job "$SLURM_JOB_ID" | grep -o -E "(StartTime|RunTime)=[^ ]+"
