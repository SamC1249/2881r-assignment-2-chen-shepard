"""CPU smoke tests for extract_acts — MUST pass before any GPU session.
Uses a tiny random llama-architecture model (no download).

Covers the three HIGH-severity extraction bugs from review:
1. id-boundary position indexing (prompt_last/resp_first/resp_last/resp_mean)
2. padded-batch activations == batch-of-1 activations
3. hook convention: layer hook output != final hidden_states entry (which is post-RMSNorm)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_acts import extract_batch, get_hidden  # noqa: E402

HID = 64
LAYERS = [0, 2, 3]


@pytest.fixture(scope="module")
def model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=128, hidden_size=HID, intermediate_size=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def rows(specs):
    out = []
    for i, (lp, lr) in enumerate(specs):
        g = torch.Generator().manual_seed(i)
        out.append({"row_id": f"r{i}",
                    "prompt_token_ids": torch.randint(1, 127, (lp,), generator=g).tolist(),
                    "response_token_ids": torch.randint(1, 127, (lr,), generator=g).tolist()})
    return out


def test_positions_match_manual_forward(model):
    [r] = rows([(7, 5)])
    acts = extract_batch(model, LAYERS, [r], "cpu", HID)
    seq = torch.tensor([r["prompt_token_ids"] + r["response_token_ids"]])
    with torch.inference_mode():
        hs = model(seq, output_hidden_states=True).hidden_states
    lp, L = 7, 12
    for k, li in enumerate(LAYERS):
        if li == 3:  # last layer: hidden_states[-1] is post-norm, can't compare — skip here
            continue
        ref = hs[li + 1][0]  # hidden_states[k] = output of layer k-1; embedding at [0]
        np.testing.assert_allclose(acts[0, k, 0], ref[lp - 1].numpy(), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(acts[0, k, 1], ref[lp].numpy(), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(acts[0, k, 2], ref[L - 1].numpy(), rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(acts[0, k, 3], ref[lp:L].mean(0).numpy(), rtol=1e-3, atol=1e-3)


def test_padded_batch_equals_single(model):
    rs = rows([(5, 3), (20, 40), (11, 1)])  # wildly different lengths
    batched = extract_batch(model, LAYERS, rs, "cpu", HID)
    for i, r in enumerate(rs):
        single = extract_batch(model, LAYERS, [r], "cpu", HID)
        np.testing.assert_allclose(batched[i], single[0], rtol=1e-3, atol=1e-4)


def test_last_layer_hook_is_pre_final_norm(model):
    [r] = rows([(4, 4)])
    seq = torch.tensor([r["prompt_token_ids"] + r["response_token_ids"]])
    cap = {}
    h = model.model.layers[3].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", get_hidden(o).detach()))
    with torch.inference_mode():
        hs = model(seq, output_hidden_states=True).hidden_states
    h.remove()
    hooked = cap["h"][0]
    final_entry = hs[-1][0]  # post model.norm
    assert not torch.allclose(hooked, final_entry, rtol=1e-3, atol=1e-4), \
        "layer-3 hook equals hidden_states[-1]: indexing is off (that entry is post-RMSNorm)"
    with torch.inference_mode():
        renormed = model.model.norm(hooked)
    np.testing.assert_allclose(renormed.numpy(), final_entry.numpy(), rtol=1e-3, atol=1e-4)


def test_empty_response_rejected(model):
    [r] = rows([(5, 1)])
    r["response_token_ids"] = []
    with pytest.raises(AssertionError):
        extract_batch(model, LAYERS, [r], "cpu", HID)
