"""Stage 05: extract residual-stream activations via HF forward hooks.

Review-driven invariants (see PIPELINE.md):
- Input = stored prompt_token_ids + response_token_ids (ids only, never re-tokenized).
- Hooks on model.model.layers[i] (residual output; tuple-or-tensor safe).
  NEVER output_hidden_states indexing (33 entries, off-by-one, final entry post-RMSNorm).
- Right padding; ALL position indexing from stored id lengths, not tensor shape:
  prompt_last = Lp-1, resp_first = Lp, resp_last = L-1, resp_mean over [Lp, L).
- torch.inference_mode(); extract ALL responses incl. filtered scenarios (diagnostic_only).
- Shards written to .tmp then os.replace; resume skips complete shards via manifest.

Usage: python src/extract_acts.py [--limit N]
Gate:  python -m pytest src/tests/test_extract.py must pass locally first.
"""

import argparse
import json
import os

import numpy as np

from common import gen_hash, load_config, read_jsonl, repo_path


def get_hidden(out):
    """Decoder layer forward output: tuple in older transformers, tensor in newer."""
    h = out[0] if isinstance(out, tuple) else out
    assert h.dim() == 3, f"unexpected hook output shape {h.shape}"
    return h


def extract_batch(model, layers, batch_rows, device, hidden_size):
    """batch_rows: list of dicts with prompt_token_ids/response_token_ids.
    Returns acts[n, n_layers, 4, hidden] float16 (positions: p_last, r_first, r_last, r_mean)."""
    import torch

    seqs = [r["prompt_token_ids"] + r["response_token_ids"] for r in batch_rows]
    lens = [len(s) for s in seqs]
    T = max(lens)
    pad_id = 0  # masked out; value irrelevant
    input_ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):  # RIGHT padding
        input_ids[i, : len(s)] = torch.tensor(s)
        attn[i, : len(s)] = 1

    captured = {}
    hooks = []
    for li in layers:
        def mk(li):
            def hook(_m, _inp, out):
                captured[li] = get_hidden(out).detach()
            return hook
        hooks.append(model.model.layers[li].register_forward_hook(mk(li)))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids.to(device), attention_mask=attn.to(device))
    finally:
        for h in hooks:
            h.remove()

    n = len(seqs)
    acts = np.zeros((n, len(layers), 4, hidden_size), dtype=np.float16)
    for k, li in enumerate(layers):
        H = captured[li].float()  # [n, T, hid]
        assert H.shape[-1] == hidden_size
        for i, r in enumerate(batch_rows):
            Lp, L = len(r["prompt_token_ids"]), lens[i]
            assert L > Lp >= 1, f"bad lengths for {r['row_id']}"
            acts[i, k, 0] = H[i, Lp - 1].cpu().numpy()          # prompt_last
            acts[i, k, 1] = H[i, Lp].cpu().numpy()              # resp_first
            acts[i, k, 2] = H[i, L - 1].cpu().numpy()           # resp_last
            acts[i, k, 3] = H[i, Lp:L].mean(0).cpu().numpy()    # resp_mean (masked by slice)
    return acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ex = cfg["extraction"]
    acts_dir = repo_path(cfg, cfg["paths"]["acts"], "manifest.json").parent
    manifest_path = acts_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"shards": {}}

    rows = list(read_jsonl(repo_path(cfg, cfg["paths"]["data"],
                                     f"responses.{gen_hash(cfg)}.jsonl"), key="row_id").values())
    rows = [r for r in rows if len(r["response_token_ids"]) >= 1]
    rows.sort(key=lambda r: len(r["prompt_token_ids"]) + len(r["response_token_ids"]))  # batch efficiency
    if args.limit:
        rows = rows[: args.limit]

    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"]["id"], torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    hidden_size = model.config.hidden_size
    n_layers_model = model.config.num_hidden_layers
    assert max(ex["layers"]) < n_layers_model

    shard_size = ex["shard_size"]
    done_ids = set()
    for s in manifest["shards"].values():
        done_ids.update(s["row_ids"])
    rows = [r for r in rows if r["row_id"] not in done_ids]
    print(f"[05] {len(rows)} rows to extract ({len(done_ids)} already done)")

    shard_idx = max([int(k) for k in manifest["shards"]], default=-1) + 1
    for s0 in range(0, len(rows), shard_size):
        shard_rows = rows[s0: s0 + shard_size]
        buf = []
        batch, batch_tokens = [], 0
        for r in shard_rows:
            t = len(r["prompt_token_ids"]) + len(r["response_token_ids"])
            if batch and batch_tokens + t > ex["max_batch_tokens"]:
                buf.append(extract_batch(model, ex["layers"], batch, device, hidden_size))
                batch, batch_tokens = [], 0
            batch.append(r)
            batch_tokens += t
        if batch:
            buf.append(extract_batch(model, ex["layers"], batch, device, hidden_size))
        acts = np.concatenate(buf, axis=0)
        shard_file = acts_dir / f"shard_{shard_idx:04d}.npz"
        tmp = str(shard_file) + ".tmp"
        np.savez_compressed(tmp, acts=acts,
                            row_ids=np.array([r["row_id"] for r in shard_rows]),
                            layers=np.array(ex["layers"]),
                            positions=np.array(ex["positions"]))
        os.replace(tmp, shard_file)
        manifest["shards"][str(shard_idx)] = {"file": shard_file.name,
                                              "row_ids": [r["row_id"] for r in shard_rows]}
        manifest_path.write_text(json.dumps(manifest))
        print(f"[05] shard {shard_idx} ({len(shard_rows)} rows)")
        shard_idx += 1
    print("[05] done")


if __name__ == "__main__":
    main()
