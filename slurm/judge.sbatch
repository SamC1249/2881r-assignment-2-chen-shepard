#!/bin/bash
#SBATCH --job-name=a2-judge
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out

export HF_HUB_OFFLINE=1
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MODE="${1:?usage: sbatch slurm/judge.sbatch <correctness|sycophancy|agreement> [--pilot]}"
shift
python src/judge.py --mode "$MODE" --dry-run
python src/judge.py --mode "$MODE" "$@"
