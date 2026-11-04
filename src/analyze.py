"""Stage 09 (local, CPU): figures + disagreement review materials.

- layer x position heatmap (val-CV AUROC with fold SD)
- dose-response: between-scenario rate by level + within-scenario curve (dose subset)
- probe-score-vs-level conditioned on judge label
- response-length distributions per class
- disagreement sampler: stratified (agree/disagree x label, traps oversampled) + the
  independent random 30 — blind column first, judge/probe shown only in the answer key
"""

import argparse
import json
import random
from collections import defaultdict

import joblib
import numpy as np

from common import load_config, repo_path
from baselines import AssertionAwareText  # noqa: F401 (unpickle)
from finalize_dataset import load_all, wilson
from train_probes import load_acts, probe_population


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = cfg["paths"]["results"]
    rng = random.Random(cfg["seed"] + 99)
    scen, prompts, responses, splits, labels = load_all(cfg)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = repo_path(cfg, res, "figures", ".keep").parent

    # ---- 1. heatmap from probe_grid.json (dev/CV numbers — exploratory)
    grid = json.loads(repo_path(cfg, res, "probe_grid.json").read_text())["grid"]
    layers, positions = cfg["extraction"]["layers"], cfg["extraction"]["positions"]
    M = np.array([[grid[f"{l}/{p}"]["best_auroc"] for p in positions] for l in layers])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(positions)), positions)
    ax.set_yticks(range(len(layers)), [f"layer {l}" for l in layers])
    for i in range(len(layers)):
        for j in range(len(positions)):
            c = grid[f"{layers[i]}/{positions[j]}"]
            ax.text(j, i, f"{c['best_auroc']:.2f}\n±{c['fold_sd']:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] < 0.8 else "black")
    ax.set_title("Probe CV-AUROC by layer x position (train+val, exploratory)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(figdir / "heatmap_layer_position.png", dpi=150)

    # ---- 2. dose-response
    rows_all = []
    for rid, lab in labels["sycophancy"].items():
        if not lab["parse_ok"] or lab["label"] == "AMBIGUOUS" or lab["sample_idx"] != 0:
            continue
        p = prompts[lab["example_id"]]
        rows_all.append({"level": p["level"], "dose": p["dose_subset"],
                         "scenario_id": p["scenario_id"],
                         "split": splits[p["scenario_id"]]["split"],
                         "y": 1 if lab["label"] == "SYCOPHANTIC" else 0})
    dev = [r for r in rows_all if r["split"] in ("train", "val") and not r["dose"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    lv = sorted({r["level"] for r in dev})
    rates = []
    for l in lv:
        sub = [r["y"] for r in dev if r["level"] == l]
        k, n = sum(sub), len(sub)
        lo, hi = wilson(k, n)
        rates.append((k / n if n else 0, lo, hi))
    ax.errorbar(lv, [r[0] for r in rates],
                yerr=[[r[0] - r[1] for r in rates], [r[2] - r[0] for r in rates]],
                marker="o", capsize=3, label="between-scenario (dev)")
    # within-scenario curve (dose subset: same scenarios, all levels)
    by_s = defaultdict(dict)
    for r in rows_all:
        if r["dose"] or splits[r["scenario_id"]].get("dose_subset"):
            by_s[r["scenario_id"]][r["level"]] = r["y"]
    complete = {s: d for s, d in by_s.items() if len(d) == 5}
    if complete:
        ax.plot(range(1, 6), [np.mean([d[l] for d in complete.values()]) for l in range(1, 6)],
                marker="s", ls="--", label=f"within-scenario (n={len(complete)}, train)")
    ax.set_xlabel("pressure level"); ax.set_ylabel("sycophancy rate"); ax.legend()
    ax.set_title("Dose-response")
    fig.tight_layout(); fig.savefig(figdir / "dose_response.png", dpi=150)

    # ---- 3. probe score vs level conditioned on label; 4. length dists (dev population)
    rows = probe_population(cfg)  # train+val
    y = np.array([r["y"] for r in rows])
    probes = joblib.load(repo_path(cfg, res, "frozen_probe.pkl"))
    X_all, ids = load_acts(cfg, {r["row_id"] for r in rows})
    order = {rid: i for i, rid in enumerate(ids)}
    X_all = X_all[[order[r["row_id"]] for r in rows]].astype(np.float32)
    pb = probes["prompt_last"]
    s = pb["model"].predict_proba(X_all[:, pb["layer_idx"], pb["pos_idx"], :])[:, 1]
    lvl = np.array([r["level"] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4))
    for lab_val, name, style in [(1, "SYC", "-o"), (0, "NOT", "--s")]:
        mask = y == lab_val
        ax.plot(range(1, 6), [s[(lvl == l) & mask].mean() if ((lvl == l) & mask).any() else np.nan
                              for l in range(1, 6)], style, label=f"judge={name}")
    ax.set_xlabel("pressure level"); ax.set_ylabel("prompt_last probe score (dev)")
    ax.set_title("Probe score vs level, conditioned on judge label"); ax.legend()
    fig.tight_layout(); fig.savefig(figdir / "probe_score_vs_level_by_label.png", dpi=150)

    lens = np.array([len(responses[r["row_id"]]["response_token_ids"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist([lens[y == 1], lens[y == 0]], bins=30, label=["SYC", "NOT"], density=True)
    ax.set_xlabel("response tokens"); ax.legend(); ax.set_title("Response length by class (dev)")
    fig.tight_layout(); fig.savefig(figdir / "length_by_class.png", dpi=150)

    # ---- 5. disagreement review materials (TEST; blind sheet + separate answer key)
    test_rows = probe_population(cfg, splits_wanted=("test",))
    if test_rows:
        ty = np.array([r["y"] for r in test_rows])
        Xt, tids = load_acts(cfg, {r["row_id"] for r in test_rows})
        torder = {rid: i for i, rid in enumerate(tids)}
        Xt = Xt[[torder[r["row_id"]] for r in test_rows]].astype(np.float32)
        ts = pb["model"].predict_proba(Xt[:, pb["layer_idx"], pb["pos_idx"], :])[:, 1]
        tpred = (ts >= pb["threshold"]).astype(int)
        strata = defaultdict(list)
        for i, r in enumerate(test_rows):
            strata[("agree" if tpred[i] == ty[i] else "disagree",
                    "SYC" if ty[i] else "NOT")].append(i)
        pick = []
        for key, idxs in strata.items():
            rng.shuffle(idxs)
            pick += [(i, f"{key[0]}/{key[1]}") for i in idxs[:8]]
        rng.shuffle(pick)
        blind, key_rows = [], []
        for i, stratum in pick:
            r = test_rows[i]
            p = prompts[responses[r["row_id"]]["example_id"]]
            s_ = scen[p["scenario_id"]]
            blind.append(f"### {r['row_id']}\n**Q:** {s_['question']}\n"
                         f"**Correct:** {s_['correct_answer']} | **Asserted:** {s_['wrong_answer']}\n\n"
                         f"**Response:**\n{responses[r['row_id']]['response_text']}\n\n"
                         f"**YOUR LABEL:** ______\n")
            key_rows.append(f"{r['row_id']}: stratum={stratum} judge={'SYC' if ty[i] else 'NOT'} "
                            f"probe={'SYC' if tpred[i] else 'NOT'} score={ts[i]:.3f}")
        (figdir.parent / "disagreements_blind.md").write_text(
            "# Blind review sheet (label BEFORE opening the answer key)\n\n" + "\n".join(blind))
        (figdir.parent / "disagreements_key.md").write_text(
            "# Answer key (open after labeling)\n\n" + "\n".join(key_rows))
        print(f"[09] review sheets: {len(pick)} stratified examples "
              f"(+ run finalize_dataset.py --make-human-sample random_review for the unbiased 30)")
    print(f"[09] figures in {figdir}")


if __name__ == "__main__":
    main()
