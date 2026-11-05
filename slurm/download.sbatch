#!/bin/bash
#SBATCH --job-name=a2-download
#SBATCH --partition=shared
#SBATCH --account=ullman_lab
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/%x-%j.out

# Model pre-download only. The conda env is already built by setup.sbatch; this job
# skips the slow pip step and goes straight to auth + download. Idempotent: re-running
# resumes partial downloads from the HF cache. Downloads need no GPU -> shared partition.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs
export PYTHONNOUSERSITE=1  # ignore broken ~/.local packages
source /n/sw/Miniforge3-26.9.4-1/etc/profile.d/conda.sh
conda activate a2
export HF_HOME=/n/netscratch/ullman_lab/Everyone/schen1249/hf
export HF_TOKEN=$(cat /n/home09/schen1249/.cache/huggingface/token 2>/dev/null)
# Belt-and-suspenders: also drop the token in HF_HOME so hub finds it by file, not just env.
mkdir -p "$HF_HOME"
printf '%s' "$HF_TOKEN" > "$HF_HOME/token"

echo "=== [1/3] auth check ($(date)) ==="
# Retry the gated check a few times in case of HF access-grant propagation lag.
python - <<'PYEOF'
import time
from huggingface_hub import auth_check
models = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-32B-Instruct"]
for m in models:
    for attempt in range(1, 6):
        try:
            auth_check(m); print("ACCESS OK:", m); break
        except Exception as e:
            print(f"attempt {attempt} failed for {m}: {e}")
            if attempt == 5: raise
            time.sleep(30)
PYEOF

echo "=== [2/3] download llama ($(date)) ==="
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --exclude "original/*" > /dev/null
echo "llama done ($(date))"

echo "=== [3/3] download qwen ($(date)) ==="
huggingface-cli download Qwen/Qwen2.5-32B-Instruct > /dev/null
echo "qwen done ($(date))"

echo "=== cache size ==="
du -sh "$HF_HOME"
echo "=== DOWNLOAD COMPLETE ($(date)) ==="
