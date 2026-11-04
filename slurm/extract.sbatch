#!/bin/bash
#SBATCH --job-name=a2-extract
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out

export HF_HUB_OFFLINE=1
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Gate: the CPU smoke tests must pass before burning GPU time on extraction.
python -m pytest src/tests/test_extract.py -q
python src/extract_acts.py "$@"
