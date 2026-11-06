"""Stage 02: generate target-model responses with vLLM. Token-ids-first: WE render the
chat template and pass prompt_token_ids; vLLM never re-tokenizes or re-templates.
Neutral condition at temp 0 (stable counterfactual); assertions at temp 0.7.
Resumable: output file embeds gen_hash; done example_ids are skipped.

Usage: python src/generate.py [--pilot] [--dry-run] [--config configs/run.yaml]
"""

import argparse

from common import JsonlWriter, gen_hash, load_config, pending_ids, read_jsonl, repo_path, stable_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--pilot", action="store_true", help="only pilot-flagged prompts")
    ap.add_argument("--dry-run", action="store_true", help="boot engine, generate 1 token, exit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    g = cfg["generation"]
    ghash = gen_hash(cfg)
    data = cfg["paths"]["data"]
    out_path = repo_path(cfg, data, f"responses.{ghash}.jsonl")
    print(f"[02] gen_hash={ghash} -> {out_path.name}")

    prompts = read_jsonl(repo_path(cfg, data, "prompts.jsonl"), key="example_id")
    if args.pilot:
        prompts = {k: v for k, v in prompts.items() if v["is_pilot"]}

    # work items: (example_id, sample_idx). Stochasticity probe adds samples 1,2 on a
    # deterministic subset of non-pilot train assert_wrong prompts.
    items = [(eid, 0) for eid in sorted(prompts)]
    if not args.pilot:
        import json
        splits = json.loads(repo_path(cfg, data, "splits.json").read_text())
        sp = g["stochasticity_probe"]
        cand = sorted(eid for eid, p in prompts.items()
                      if p["condition"] == "assert_wrong" and not p["dose_subset"]
                      and splits[p["scenario_id"]]["split"] == "train")
        import random
        random.Random(cfg["seed"]).shuffle(cand)
        for eid in cand[: sp["n_prompts"]]:
            for s in range(1, sp["n_samples"]):
                items.append((eid, s))

    done = read_jsonl(out_path, key="row_id")
    todo = pending_ids([f"{e}::s{s}" for e, s in items], done)
    todo_items = [(r.split("::s")[0], int(r.split("::s")[1])) for r in todo]
    print(f"[02] {len(items)} work items, {len(items) - len(todo_items)} done, {len(todo_items)} to go")
    if not todo_items and not args.dry_run:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt

    tok = AutoTokenizer.from_pretrained(cfg["target_model"]["id"])
    llm = LLM(model=cfg["target_model"]["id"], dtype=cfg["target_model"]["dtype"],
              max_model_len=g["max_model_len"], gpu_memory_utilization=g["gpu_memory_utilization"])

    if args.dry_run:
        ids = tok.apply_chat_template([{"role": "user", "content": "hi"}],
                                      add_generation_prompt=True, tokenize=True,
                                      return_dict=False)
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                           SamplingParams(max_tokens=1, temperature=0.0))
        print(f"[02] dry-run OK: {out[0].outputs[0].token_ids}")
        return

    def params(p, sample_idx):
        base = g["neutral"] if p["condition"] == "neutral" else g["assertion"]
        return SamplingParams(max_tokens=g["max_tokens"],
                              seed=stable_seed(p["example_id"], sample_idx), **base)

    reqs, metas = [], []
    for eid, sidx in todo_items:
        p = prompts[eid]
        ids = tok.apply_chat_template(p["messages"], add_generation_prompt=True, tokenize=True,
                                      return_dict=False)  # transformers v5 defaults to BatchEncoding dict
        reqs.append(TokensPrompt(prompt_token_ids=ids))
        metas.append((p, sidx, ids))

    with JsonlWriter(out_path) as w:
        CHUNK = 500  # bounded loss on preemption
        for c in range(0, len(reqs), CHUNK):
            outs = llm.generate(reqs[c:c + CHUNK], [params(p, s) for p, s, _ in metas[c:c + CHUNK]])
            for (p, sidx, pids), o in zip(metas[c:c + CHUNK], outs):
                gen = o.outputs[0]
                assert len(gen.token_ids) >= 1, f"empty generation for {p['example_id']}"
                w.write({
                    "row_id": f"{p['example_id']}::s{sidx}",
                    "example_id": p["example_id"], "sample_idx": sidx,
                    "prompt_token_ids": list(pids), "response_token_ids": list(gen.token_ids),
                    "response_text": gen.text, "finish_reason": gen.finish_reason,
                    "model_rev": cfg["target_model"]["id"],
                })
            print(f"[02] {min(c + CHUNK, len(reqs))}/{len(reqs)}")
    print("[02] done")


if __name__ == "__main__":
    main()
