#!/bin/bash
#SBATCH --job-name=a2-generate
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/%x-%j.out

# Models must be pre-downloaded from a login node (see PIPELINE.md S0):
#   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
#   huggingface-cli download Qwen/Qwen2.5-32B-Instruct
export HF_HUB_OFFLINE=1
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Engine dry-run first: a boot failure costs 2 minutes, not the queue slot.
python src/generate.py --dry-run
python src/generate.py "$@"
