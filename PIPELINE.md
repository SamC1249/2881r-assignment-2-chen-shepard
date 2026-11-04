# Pipeline Design (concrete) — v2, post adversarial review

Companion to `PLAN.md`. Engineering contract: file layout, data schemas, script
interfaces, pilot definition. v2 incorporates the Nov 4 adversarial review (methodology,
grader-compliance, and ML-engineering passes); review-driven changes are marked `[R]`.

Every stage: standalone script, reads/writes JSONL + NPZ artifacts, idempotent,
takes `--config configs/run.yaml`, safe to kill at any moment.

## Core design rules (from review)

1. **[R] Token IDs are the source of truth, never strings.** We render the chat template
   ourselves (`tokenizer.apply_chat_template(..., add_generation_prompt=True, tokenize=True)`)
   and pass `prompt_token_ids` to vLLM; we store `prompt_token_ids` and the returned
   `output.token_ids` with every response. Extraction feeds `prompt_ids + response_ids`
   straight into HF — the response boundary is `len(prompt_ids)` *by construction*. No
   string prefix checks (they false-fail on BPE boundary merges and silently pass double-BOS
   corruption), no detokenize→retokenize drift, no reliance on vLLM's internal template
   rendering (we never use `llm.chat()`).
2. **[R] Splits are assigned at dataset build time (stage 01)**, before any generation or
   labeling exists — split assignment needs only scenario IDs. Pilot scenarios and any
   scenario a human reads pre-freeze are **pinned to train**. The correctness filter later
   prunes scenarios, never reassigns them.
3. **[R] Resume discipline:** config hash in the artifact filename
   (`responses.{gen_hash}.jsonl`, `labels.{rubric_hash}.jsonl`) so stale-config rows can't
   mix with fresh ones; one `write()+flush` per record; on load, auto-truncate one partial
   *final* line, error on malformed interior lines; dedupe by id last-wins; NPZ shards
   written to `*.tmp` then `os.replace`; one writer process per file, ever.
4. **[R] Every GPU session starts with an engine dry-run** (build engine with the exact
   config, generate one token) so a boot failure costs 2 minutes, not a queue slot.
5. **[R] Extraction logic is CPU-unit-tested before any GPU session** (tiny llama-arch
   model): id-boundary correctness, padded-batch == batch-of-1 activations (bf16 tol),
   hook convention asserts.

## Repo layout

```
assignment-2/
├── PLAN.md / PIPELINE.md
├── configs/run.yaml           # single source of truth; hashes derived from it
├── prompts/
│   ├── templates.yaml         # ladder (5 levels × 2 phrasings) + condition formats
│   ├── judge_sycophancy.md    # rubric (hash recorded in every label row)
│   ├── judge_correctness.md   # neutral-answer correctness rubric  [R: validated too]
│   └── judge_agreement.md     # assert_correct agreement-rate rubric [R: control non-vacuity]
├── src/
│   ├── common.py              # config/hash, jsonl io w/ resume rules, seeding
│   ├── build_dataset.py       # 01  (includes split assignment)
│   ├── generate.py            # 02  (vLLM, token-ids-first)
│   ├── judge.py               # 03  (vLLM; modes below)
│   ├── finalize_dataset.py    # 04  (filter, prune splits, Part-1 behavioral report)
│   ├── extract_acts.py        # 05  (HF hooks)
│   ├── train_probes.py        # 06
│   ├── baselines.py           # 07
│   ├── eval_test.py           # 08  (run ONCE)
│   ├── analyze.py             # 09
│   └── tests/test_extract.py  # [R] CPU smoke tests for 05
├── slurm/*.sbatch             # thin wrappers; all begin with engine dry-run + HF_HUB_OFFLINE=1
├── data/                      # gitignored EXCEPT scenarios/prompts/splits/manifest
├── acts/                      # gitignored
└── results/                   # committed
```

## Stages

### 01 `build_dataset.py` → `scenarios.jsonl`, `prompts.jsonl`, `splits.json`
- MMLU (HF `cais/mmlu`, test split), ~8 subjects. Drop choice-dependent questions
  (regex: "following", "these options", roman-numeral stems, "all/none of the above"
  answers; log counts). **[R] 1000 scenarios** (was 800) to absorb correctness-filter losses.
- Wrong answer = seeded-random distractor; record which.
- 3 conditions/scenario; assert_wrong & assert_correct share level+phrasing. Level
  balanced within domain, seeded.
- **[R] Splits here:** scenario → train(60)/val(20)/test(20), stratified by domain×level.
  Held-out-in-test: ONE phrasing of ONE mid level (`L3_b`), one domain. `--pilot` marks
  50 scenarios (10×5 domains), **pinned to train**.
- **[R] Dose-response subset:** 100 train scenarios additionally get all 5 levels
  (one phrasing) → within-scenario curve; +500 prompts, train-only, secondary analysis.
- Schemas:
  `{scenario_id, domain, question, correct_answer, wrong_answer, mmlu_ref}`
  `{example_id, scenario_id, condition, level, phrasing_id, messages, is_pilot, dose_subset}`

### 02 `generate.py` → `data/responses.{gen_hash}.jsonl`
- vLLM, Llama-3.1-8B-Instruct. **[R] neutral condition: temp 0.0** (stable-knowledge
  counterfactual); assertion conditions: temp 0.7/top_p 0.9. max_tokens 512.
- **[R]** prompts pre-tokenized by us; `TokensPrompt(prompt_token_ids=...)`;
  per-request `seed = stable_hash(example_id, sample_idx)` (never a shared seed —
  identical seeds give identical samples).
- **[R] Stochasticity probe:** 100 assert_wrong train prompts × 3 samples → label-flip
  rate across samples = irreducible-noise ceiling for prompt-side probes.
- Row: `{example_id, sample_idx, prompt_token_ids, response_token_ids, response_text,
  finish_reason, model_rev}`; `gen_hash` covers sampling params + vLLM version + dtype +
  chat-template hash + max_model_len.
- **[R]** `finish_reason="length"` rows flagged; excluded from probe training; truncation
  rate reported. Assert ≥1 response token.

### 03 `judge.py` → `data/labels.{rubric_hash}.jsonl`
- vLLM, Qwen2.5-32B-Instruct, temp 0.0, guided JSON (xgrammar backend, pinned).
- **[R] vLLM boot config (A100/H100 80GB):** `max_model_len=4096`,
  `gpu_memory_utilization=0.95`, `enforce_eager=True`; H100: `--quantization fp8`;
  A100 fallback: AWQ-Int4 checkpoint. Bf16 with defaults does NOT boot (KV-cache init check fails).
- **[R] Schema order matters:** `{"judge_rationale": str(maxLength~600), "label": enum}` —
  rationale FIRST (CoT), label last. Parse gate = valid JSON AND `finish_reason=="stop"`.
- Modes: `correctness` (neutral; semantic equivalence to ground truth),
  `sycophancy` (assert_wrong; SYC|NOT|AMBIGUOUS), **[R]** `agreement` (assert_correct;
  did the model agree? — proves the control is non-vacuous),
  `human-sample` (blind CSV export; see validation protocol in PLAN).

### 04 `finalize_dataset.py` → `data/manifest.json`, `results/behavioral_dev.json`
- Keep scenarios with neutral judged correct (temp 0); prune splits.json accordingly; log rates.
- **[R] Part 1 deliverable:** dev-set sycophancy rate with Wilson/bootstrap CIs, broken out
  by domain and ladder level (trend test: logistic regression of label on level);
  AMBIGUOUS rate by level/domain; label counts per split and per test slice, with the
  pre-committed rule: a reported slice needs ≥15 examples per class, else slices are
  merged and that is stated.
- **[R] Trap-case mining:** flag responses where keyword rule and judge disagree
  ("you're right" but judged NOT; no agreement phrase but judged SYC) → these MUST appear
  in the human κ sample (≥10) and the disagreement review.

### 05 `extract_acts.py` → `acts/shard_*.npz`
- HF transformers, same revision; input = stored `prompt_token_ids + response_token_ids`
  (ids only). `torch.inference_mode()`, bf16, right-padding, explicit attention mask,
  `tokenizer.pad_token = eos` but ALL position indexing from stored id lengths:
  `prompt_last = L_p−1`, `resp_first = L_p`, `resp_last = L−1`, `resp_mean` masked over `[L_p, L)`.
- **[R] Hooks on `model.model.layers[i]`, i ∈ {8,16,24,31}** (residual output; handle
  tuple-or-tensor; assert dim 4096). NEVER `output_hidden_states` indexing (33 entries,
  off-by-one, final entry post-RMSNorm — non-comparable last row).
- **[R]** Extract ALL responses incl. correctness-filtered scenarios (tag `diagnostic_only`)
  — needed for neutral-wrong OOD diagnostics.
- Shards of 256 → `{example_ids, acts[n, 4, 4, 4096]}` fp16 + manifest. ~0.8 GB total.
- Gated by `tests/test_extract.py` passing locally first.

### 06 `train_probes.py` → `results/probe_grid.json`, `results/frozen_probe.pkl`
- Population: assert_wrong, SYC/NOT (AMBIGUOUS + truncated excluded, counts logged).
- **[R] Selection by 5-fold CV over train+val scenarios** (not single small val):
  per (layer × position), Pipeline(StandardScaler, LogisticRegression(class_weight="balanced")),
  C ∈ {1e-4…1e2}; heatmap cells get CV-fold SDs. Optional MLP(64) same protocol.
- **[R] Threshold rule pre-committed: max F1 on pooled CV predictions.** Written to config.
- Headline position = `prompt_last` (anticipatory); `resp_first` reported as
  "+1 committed token"; `resp_last`/`resp_mean` as post-hoc ceiling.

### 07 `baselines.py` → `results/baselines.json`
Majority class; TF-IDF(resp); keyword rule; **[R] level-only** (ordinal level + phrasing id —
the prompt-side cheat detector); **[R] length-only** (response token count);
**[R] assertion-aware text** (TF-IDF over prompt+response + features: response
fuzzy-contains asserted answer / correct answer — the information-matched text baseline
that can do prompt↔response matching). Probe at `prompt_last` doubles as the prompt-side probe.

### 08 `eval_test.py` → `results/test_report.json` — RUN ONCE
- Frozen probe/threshold/baselines on test. Slices: rest-of-test (the honest iid number),
  held-out phrasing, held-out domain — **[R] OOD slices reported as qualitative robustness
  checks** (power: ~±0.15 AUROC).
- **[R] Paired scenario-bootstrap on ΔAUROC** (probe − each baseline), not overlapping CIs.
- **[R] Within-level AUROC** (level held constant ⇒ pressure wording can't carry the signal).
- OOD diagnostics with the frozen probe: assert_correct, neutral-wrong (diagnostic_only),
  AMBIGUOUS score distribution.
- Confusion matrices, per-class P/R, AUROC, balanced acc, scenario-bootstrap CIs.

### 09 `analyze.py` → `results/figures/`, `results/disagreements.md`
- Layer×position heatmap (with fold-SD error); dose-response: between-scenario trend +
  within-scenario curve (dose_subset), probe-score-vs-level **conditioned on judge label**;
  per-domain SYC rate + within-domain AUROC; response-length distributions per class.
- Disagreement sampler: stratified (agree/disagree × label, traps oversampled) for
  diagnosis **plus [R] an independent simple-random sample of 30 test examples** labeled
  blind for unbiased correctness estimation (or stratum-reweighted estimate — stated).

## Pilot (S1, ~1 hr)

50 pinned-to-train scenarios, stages 01→04.
Gates: (1) neutral correctness ≥50% (else swap subjects); (2) flip rate among
neutrally-correct 15–85% (else adjust ladder strength); (3) flip rate non-decreasing in
level; (4) judge parse gate ~100%, eyeball 20 judgments; **[R]** (5) ≥5 trap-pattern
responses exist (else engineer trap-eliciting variants before full run).

## Session plan

| Session | Stages | Notes |
|---|---|---|
| S0 (login node, today) | pre-download Llama-3.1-8B (gated — verify token!) + Qwen2.5-32B (~65GB) to shared storage; `HF_HUB_OFFLINE=1` in all sbatch | no GPU |
| local (today) | `tests/test_extract.py` on tiny model | no GPU |
| S1 pilot | 01–04 pilot + eyeball | 1×80GB ~1h |
| S2 | 02 full (3×1000 + 500 dose + 300 stochasticity ≈ 3.8k gens) | 1×80GB ~2h |
| S3 | 03 full: correctness + sycophancy + agreement modes | 1×80GB ~3h |
| S4 | 05 extraction (~3.5k × ≤1k tok teacher-forced) | 1×80GB ~1h |
| local | 04, 06–09 | CPU |

Human tasks: κ protocol per PLAN (stratified 100 incl. ≥10 traps from train+val ONLY;
post-freeze fresh 40; correctness-judge check 30); blind disagreement review after 08.
Never two vLLM engines in one process; stages stay separate scripts.

## Open item (resolve before S4)

**[R] "Provided infrastructure" (assignment Part 2 §1):** check the course repo/Ed for the
official activation-extraction infra. Use it if it supports teacher-forced multi-position
extraction; otherwise document why we built our own. Owner: Sam. Deadline: Nov 5.
