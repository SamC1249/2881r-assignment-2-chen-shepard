#!/bin/bash
#SBATCH --job-name=a2-setup
#SBATCH --partition=shared
#SBATCH --account=ullman_lab
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x-%j.out

# One-time environment build + model pre-download. Runs on a compute node (node-stable,
# survives disconnects, visible via squeue) instead of a login node. Idempotent: skips
# the env create and re-download of already-cached files.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs
export HF_HOME=/n/netscratch/ullman_lab/Everyone/schen1249/hf
export HF_TOKEN=$(cat /n/home09/schen1249/.cache/huggingface/token 2>/dev/null)  # token lives in default HF cache, not our HF_HOME
MAMBA=/n/sw/Miniforge3-26.9.4-1/bin/mamba
export PYTHONNOUSERSITE=1  # ignore broken ~/.local packages
source /n/sw/Miniforge3-26.9.4-1/etc/profile.d/conda.sh

echo "=== [1/4] create env ($(date)) ==="
$MAMBA env list | grep -q "^a2 " || $MAMBA create -y -n a2 python=3.12
conda activate a2

echo "=== [2/4] pip install ($(date)) ==="
pip install -q vllm
pip install -q -r requirements.txt
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"

echo "=== [3/4] gated access check ($(date)) ==="
python - <<'PYEOF'
from huggingface_hub import auth_check
for m in ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]:
    auth_check(m); print("ACCESS OK:", m)
PYEOF

echo "=== [4/4] model downloads ($(date)) ==="
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*" > /dev/null
echo "llama done"
huggingface-cli download Qwen/Qwen2.5-32B-Instruct > /dev/null
echo "qwen done"
echo "=== SETUP COMPLETE ($(date)) ==="
