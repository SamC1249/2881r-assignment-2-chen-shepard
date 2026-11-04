"""Stage 06 (local, CPU): train the probe grid and freeze the winner.

Population: assert_wrong, sample 0, non-dose, usable scenarios, SYC/NOT only (AMBIGUOUS +
truncated excluded, counts logged). Selection: 5-fold stratified CV over train+val (the
small val set alone is a winner's-curse machine across a 100+ config grid).
Threshold rule (pre-committed in config): max F1 on pooled out-of-fold predictions.
Headline position: prompt_last (anticipatory). Preprocessing fit inside each fold.
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_config, repo_path
from finalize_dataset import load_all


def load_acts(cfg, want_row_ids: set):
    acts_dir = repo_path(cfg, cfg["paths"]["acts"], "manifest.json").parent
    manifest = json.loads((acts_dir / "manifest.json").read_text())
    X, ids = [], []
    for s in manifest["shards"].values():
        z = np.load(acts_dir / s["file"], allow_pickle=False)
        mask = np.isin(z["row_ids"], list(want_row_ids))
        if mask.any():
            X.append(z["acts"][mask])
            ids.extend(z["row_ids"][mask].tolist())
    return np.concatenate(X), ids  # [n, n_layers, n_pos, hid]


def probe_population(cfg, splits_wanted=("train", "val")):
    scen, prompts, responses, splits, labels = load_all(cfg)
    manifest = json.loads(repo_path(cfg, cfg["paths"]["data"], "manifest.json").read_text())
    usable = set(manifest["usable_scenarios"])
    rows = []
    for rid, lab in labels["sycophancy"].items():
        if not lab["parse_ok"] or lab["sample_idx"] != 0:
            continue
        p = prompts[lab["example_id"]]
        if p["dose_subset"] or p["scenario_id"] not in usable:
            continue
        if splits[p["scenario_id"]]["split"] not in splits_wanted:
            continue
        if responses[rid]["finish_reason"] == "length":
            continue
        if lab["label"] == "AMBIGUOUS":
            continue
        rows.append({"row_id": rid, "scenario_id": p["scenario_id"], "level": p["level"],
                     "domain": scen[p["scenario_id"]]["domain"],
                     "split": splits[p["scenario_id"]]["split"],
                     "y": 1 if lab["label"] == "SYCOPHANTIC" else 0})
    return rows


def make_lr(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ex = cfg["extraction"]
    pr = cfg["probe"]
    assert pr["threshold_rule"] == "max_f1_pooled_cv"

    rows = probe_population(cfg)
    y = np.array([r["y"] for r in rows])
    print(f"[06] population: {len(rows)} (SYC={y.sum()}, NOT={(1 - y).sum()})")
    X_all, ids = load_acts(cfg, {r["row_id"] for r in rows})
    order = {rid: i for i, rid in enumerate(ids)}
    X_all = X_all[[order[r["row_id"]] for r in rows]].astype(np.float32)

    skf = StratifiedKFold(n_splits=pr["cv_folds"], shuffle=True, random_state=cfg["seed"])
    folds = list(skf.split(np.zeros(len(y)), y))

    grid = {}
    best = {"auroc": -1}
    for li, layer in enumerate(ex["layers"]):
        for pi, pos in enumerate(ex["positions"]):
            X = X_all[:, li, pi, :]
            for C in pr["c_grid"]:
                aucs, oof = [], np.zeros(len(y))
                for tr, va in folds:
                    m = make_lr(C).fit(X[tr], y[tr])
                    p = m.predict_proba(X[va])[:, 1]
                    oof[va] = p
                    aucs.append(roc_auc_score(y[va], p))
                mean, sd = float(np.mean(aucs)), float(np.std(aucs))
                cell = grid.setdefault(f"{layer}/{pos}", {"best_auroc": -1})
                if mean > cell["best_auroc"]:
                    cell.update({"best_auroc": round(mean, 4), "fold_sd": round(sd, 4), "C": C})
                if mean > best["auroc"]:
                    ths = np.unique(oof)
                    f1s = [f1_score(y, oof >= t) for t in ths]
                    best = {"auroc": mean, "layer": layer, "pos": pos, "pos_idx": pi,
                            "layer_idx": li, "C": C,
                            "threshold": float(ths[int(np.argmax(f1s))])}
            print(f"[06] layer {layer} pos {pos}: AUROC {grid[f'{layer}/{pos}']['best_auroc']}"
                  f" ±{grid[f'{layer}/{pos}']['fold_sd']} (C={grid[f'{layer}/{pos}']['C']})")

    # headline-position winner (prompt_last) tracked separately from overall winner
    pl_idx = ex["positions"].index("prompt_last")
    pl_cells = {k: v for k, v in grid.items() if k.endswith("/prompt_last")}
    pl_best_key = max(pl_cells, key=lambda k: pl_cells[k]["best_auroc"])

    # MLP comparison at the overall winner cell (geometry assumption test)
    Xw = X_all[:, best["layer_idx"], best["pos_idx"], :]
    mlp_aucs = []
    for tr, va in folds:
        m = make_pipeline(StandardScaler(),
                          MLPClassifier((pr["mlp_hidden"],), max_iter=800, random_state=cfg["seed"]))
        m.fit(Xw[tr], y[tr])
        mlp_aucs.append(roc_auc_score(y[va], m.predict_proba(Xw[va])[:, 1]))

    frozen = {}
    for name, (layer_idx, pos_idx, C) in {
        "overall": (best["layer_idx"], best["pos_idx"], best["C"]),
        "prompt_last": (ex["layers"].index(int(pl_best_key.split("/")[0])), pl_idx,
                        pl_cells[pl_best_key]["C"]),
    }.items():
        X = X_all[:, layer_idx, pos_idx, :]
        oof = np.zeros(len(y))
        for tr, va in folds:
            oof[va] = make_lr(C).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        ths = np.unique(oof)
        thr = float(ths[int(np.argmax([f1_score(y, oof >= t) for t in ths]))])
        frozen[name] = {"model": make_lr(C).fit(X, y), "layer_idx": layer_idx,
                        "pos_idx": pos_idx, "layer": ex["layers"][layer_idx],
                        "pos": ex["positions"][pos_idx], "C": C, "threshold": thr}

    res = repo_path(cfg, cfg["paths"]["results"], "probe_grid.json")
    res.write_text(json.dumps({
        "grid": grid, "winner_overall": {k: best[k] for k in ["layer", "pos", "C", "auroc", "threshold"]},
        "winner_prompt_last": {"cell": pl_best_key, **pl_cells[pl_best_key]},
        "mlp_at_winner": {"auroc_mean": round(float(np.mean(mlp_aucs)), 4),
                          "auroc_sd": round(float(np.std(mlp_aucs)), 4)},
        "population": {"n": len(rows), "syc": int(y.sum())},
    }, indent=1))
    joblib.dump(frozen, repo_path(cfg, cfg["paths"]["results"], "frozen_probe.pkl"))
    print(f"[06] FROZEN. overall={best['layer']}/{best['pos']} AUROC={best['auroc']:.3f}; "
          f"prompt_last winner={pl_best_key} AUROC={pl_cells[pl_best_key]['best_auroc']}")


if __name__ == "__main__":
    main()
