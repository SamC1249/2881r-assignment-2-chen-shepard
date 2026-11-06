"""Stage 03: judge responses with Qwen2.5-32B via vLLM guided JSON.
Modes: correctness (neutral), sycophancy (assert_wrong), agreement (assert_correct).
Schema puts rationale BEFORE label (CoT); parse gate = valid JSON AND finish_reason=="stop".
Output filename embeds rubric_hash so stale-rubric rows can't mix. Failed rows are
re-judged on resume.

Usage: python src/judge.py --mode sycophancy [--pilot] [--dry-run]
"""

import argparse
import json

from common import (JsonlWriter, gen_hash, load_config, pending_ids, read_jsonl,
                    repo_path, rubric_hash)

MODE_CONDITION = {"correctness": "neutral", "sycophancy": "assert_wrong", "agreement": "assert_correct"}
MODE_LABELS = {
    "correctness": ["CORRECT", "INCORRECT"],
    "sycophancy": ["SYCOPHANTIC", "NOT_SYCOPHANTIC", "AMBIGUOUS"],
    "agreement": ["AGREED", "DID_NOT_AGREE"],
}
MODE_RUBRIC = {"correctness": "judge_correctness.md", "sycophancy": "judge_sycophancy.md",
               "agreement": "judge_agreement.md"}


def build_prompt(rubric: str, scen: dict, prompt_row: dict, response_text: str, mode: str) -> str:
    fields = {"question": scen["question"], "correct_answer": scen["correct_answer"],
              "response": response_text}
    if mode == "sycophancy":
        fields["asserted_answer"] = scen["wrong_answer"]
    elif mode == "agreement":
        fields["asserted_answer"] = scen["correct_answer"]
    return rubric.format(**fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--mode", required=True, choices=list(MODE_CONDITION))
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    j = cfg["judge"]
    data = cfg["paths"]["data"]
    rhash = rubric_hash(cfg, args.mode)
    out_path = repo_path(cfg, data, f"labels_{args.mode}.{rhash}.jsonl")
    print(f"[03] mode={args.mode} rubric_hash={rhash} -> {out_path.name}")

    scenarios = read_jsonl(repo_path(cfg, data, "scenarios.jsonl"), key="scenario_id")
    prompts = read_jsonl(repo_path(cfg, data, "prompts.jsonl"), key="example_id")
    responses = read_jsonl(repo_path(cfg, data, f"responses.{gen_hash(cfg)}.jsonl"), key="row_id")

    work = []
    for rid, r in sorted(responses.items()):
        p = prompts[r["example_id"]]
        if p["condition"] != MODE_CONDITION[args.mode]:
            continue
        if args.pilot and not p["is_pilot"]:
            continue
        work.append((rid, r, p))

    done = read_jsonl(out_path, key="row_id")
    todo_ids = set(pending_ids([rid for rid, _, _ in work], done,
                               extra_ok=lambda row: row.get("parse_ok", False)))
    work = [wk for wk in work if wk[0] in todo_ids]
    print(f"[03] {len(todo_ids)} to judge")
    if not work and not args.dry_run:
        return

    rubric = repo_path(cfg, cfg["paths"]["prompts"], MODE_RUBRIC[args.mode]).read_text()
    schema = {
        "type": "object",
        "properties": {  # rationale FIRST (order preserved by xgrammar), label LAST
            "judge_rationale": {"type": "string", "maxLength": j["rationale_max_chars"]},
            "label": {"type": "string", "enum": MODE_LABELS[args.mode]},
        },
        "required": ["judge_rationale", "label"],
        "additionalProperties": False,
    }

    from vllm import LLM, SamplingParams
    # vLLM 0.33 renamed guided-decoding -> structured-outputs.
    from vllm.sampling_params import StructuredOutputsParams, StructuredOutputsConfig

    llm = LLM(model=j["id"], dtype=j["dtype"], quantization=j["quantization"],
              max_model_len=j["max_model_len"], gpu_memory_utilization=j["gpu_memory_utilization"],
              enforce_eager=j["enforce_eager"],
              structured_outputs_config=StructuredOutputsConfig(backend=j["guided_backend"]))
    sp = SamplingParams(temperature=j["temperature"], max_tokens=j["max_tokens"],
                        structured_outputs=StructuredOutputsParams(json=schema))

    if args.dry_run:
        out = llm.chat([{"role": "user", "content": "Reply in the required JSON. "
                        'Rationale: say "ok". Label: pick any valid value.'}], sp)
        print(f"[03] dry-run OK: {out[0].outputs[0].text[:200]}")
        return

    with JsonlWriter(out_path) as w:
        CHUNK = 200
        for c in range(0, len(work), CHUNK):
            chunk = work[c:c + CHUNK]
            convs = [[{"role": "user", "content": build_prompt(
                rubric, scenarios[p["scenario_id"]], p, r["response_text"], args.mode)}]
                for _, r, p in chunk]
            outs = llm.chat(convs, sp)
            for (rid, r, p), o in zip(chunk, outs):
                gen = o.outputs[0]
                parse_ok, label, rationale = False, None, None
                if gen.finish_reason == "stop":
                    try:
                        obj = json.loads(gen.text)
                        label, rationale = obj["label"], obj["judge_rationale"]
                        parse_ok = label in MODE_LABELS[args.mode]
                    except (json.JSONDecodeError, KeyError):
                        pass
                w.write({"row_id": rid, "example_id": r["example_id"],
                         "sample_idx": r["sample_idx"], "mode": args.mode, "label": label,
                         "judge_rationale": rationale, "parse_ok": parse_ok,
                         "finish_reason": gen.finish_reason, "rubric_hash": rhash})
            print(f"[03] {min(c + CHUNK, len(work))}/{len(work)}")
    bad = sum(1 for row in read_jsonl(out_path) if not row["parse_ok"])
    print(f"[03] done; parse failures in file: {bad} (rerun to retry)")


if __name__ == "__main__":
    main()
