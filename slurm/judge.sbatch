#!/bin/bash
#SBATCH --job-name=a2-judge
#SBATCH --partition=gpu_requeue
#SBATCH --account=ullman_lab
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONNOUSERSITE=1  # ignore broken ~/.local packages
source /n/sw/Miniforge3-26.9.4-1/etc/profile.d/conda.sh
conda activate a2
export HF_HOME=/n/netscratch/ullman_lab/Everyone/schen1249/hf
export HF_HUB_OFFLINE=1

MODE="${1:?usage: sbatch slurm/judge.sbatch <correctness|sycophancy|agreement> [--pilot]}"
shift || true
python src/judge.py --mode "$MODE" --dry-run
python src/judge.py --mode "$MODE" "$@"
