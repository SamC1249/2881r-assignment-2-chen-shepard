# 2881r-assignment-2-chen-shepard

Sycophancy eval + activation probe on Llama-3.1-8B-Instruct.
Design: `PLAN.md` (research) + `PIPELINE.md` (engineering). Both adversarially reviewed Nov 4.

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
# GPU nodes additionally: pip install vllm
# Login node, ONCE (gated model — accept license + set HF token first):
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
huggingface-cli download Qwen/Qwen2.5-32B-Instruct
```

## Run order

```bash
# local (done — artifacts committed)
python src/build_dataset.py                 # 01: scenarios/prompts/splits
python -m pytest src/tests/test_extract.py  # gate for stage 05

# PILOT (first GPU session; go/no-go gates in PIPELINE.md)
sbatch slurm/generate.sbatch --pilot
sbatch slurm/judge.sbatch correctness --pilot
sbatch slurm/judge.sbatch sycophancy --pilot
python src/finalize_dataset.py              # eyeball flip rate / traps

# FULL
sbatch slurm/generate.sbatch
sbatch slurm/judge.sbatch correctness
sbatch slurm/judge.sbatch sycophancy
sbatch slurm/judge.sbatch agreement
python src/finalize_dataset.py              # 04: filter + behavioral_dev.json
python src/finalize_dataset.py --make-human-sample kappa   # blind-label, compute kappa, FREEZE rubric
sbatch slurm/extract.sbatch                 # 05
python src/train_probes.py                  # 06: CV grid + freeze
python src/baselines.py                     # 07
python src/eval_test.py                     # 08: ONE SHOT
python src/analyze.py                       # 09: figures + review sheets
```

Every GPU stage is resumable (config-hash-keyed artifacts, done-ids skipped) and starts
with an engine dry-run. Never run two vLLM stages in one process.
