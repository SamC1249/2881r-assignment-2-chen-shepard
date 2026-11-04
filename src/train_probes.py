"""Stage 06 (local, CPU): train the probe grid and freeze the winner.

Population: assert_wrong, sample 0, non-dose, usable scenarios, SYC/NOT only (AMBIGUOUS +
truncated excluded, counts logged). Selection: 5-fold stratified CV over train+val (the
small val set alone is a winner's-curse machine across a 100+ config grid).
Threshold rule (pre-committed in config): max F1 on pooled out-of-fold predictions.
Headline position: prompt_last (anticipatory). Preprocessing fit inside each fold.

Selectivity control (Hewitt & Liang 2019): labels permuted at the scenario level (each
example = one scenario here), identical CV pipeline, C re-selected adversarially on the
permuted data. Expected AUROC ~0.5; selectivity = real − permuted. Run for the linear
winner cells AND the MLP (whose capacity is exactly what the control exists to catch).
Permuted AUROC meaningfully above 0.5 in ANY probe class = leakage/pipeline-bug alarm.
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


def cv_auroc(model_fn, X, y, n_folds, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(np.zeros(len(y)), y):
        m = model_fn().fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
    return float(np.mean(aucs))


def permuted_control(model_fns: dict, X, y, n_folds, seed, n_perms=5):
    """Selectivity control: scenario-level label permutation, identical CV protocol.
    model_fns: {name: callable giving a fresh fit-able model} — for the linear probe the
    callable should re-select C adversarially (give the probe its best shot on fake labels).
    Returns {name: {permuted_mean, permuted_sd, per_perm}}."""
    rng = np.random.default_rng(seed)
    out = {name: [] for name in model_fns}
    for _ in range(n_perms):
        yp = rng.permutation(y)  # one example per scenario -> this IS scenario-level
        for name, fn in model_fns.items():
            out[name].append(fn(X, yp))
    return {name: {"permuted_mean": round(float(np.mean(v)), 4),
                   "permuted_sd": round(float(np.std(v)), 4),
                   "per_perm": [round(x, 4) for x in v]}
            for name, v in out.items()}


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

    # ---- selectivity control (permuted labels, identical protocol, adversarial C)
    def lr_best_c(X, yy):
        return max(cv_auroc(lambda: make_lr(C), X, yy, pr["cv_folds"], cfg["seed"])
                   for C in pr["c_grid"])

    def mlp_fn(X, yy):
        return cv_auroc(lambda: make_pipeline(
            StandardScaler(), MLPClassifier((pr["mlp_hidden"],), max_iter=800,
                                            random_state=cfg["seed"])),
            X, yy, pr["cv_folds"], cfg["seed"])

    Xpl = X_all[:, ex["layers"].index(int(pl_best_key.split("/")[0])), pl_idx, :]
    sel = {
        "linear_overall": permuted_control({"m": lr_best_c}, Xw, y, pr["cv_folds"], cfg["seed"])["m"],
        "linear_prompt_last": permuted_control({"m": lr_best_c}, Xpl, y, pr["cv_folds"], cfg["seed"] + 1)["m"],
        "mlp_at_winner": permuted_control({"m": mlp_fn}, Xw, y, pr["cv_folds"], cfg["seed"] + 2)["m"],
    }
    sel["linear_overall"]["real"] = round(best["auroc"], 4)
    sel["linear_prompt_last"]["real"] = pl_cells[pl_best_key]["best_auroc"]
    sel["mlp_at_winner"]["real"] = round(float(np.mean(mlp_aucs)), 4)
    for v in sel.values():
        v["selectivity"] = round(v["real"] - v["permuted_mean"], 4)
    alarm = [k for k, v in sel.items() if v["permuted_mean"] > 0.58]
    if alarm:
        print(f"[06] *** LEAKAGE ALARM: permuted-label AUROC > 0.58 for {alarm} — "
              f"investigate before trusting ANY probe number ***")

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
        "selectivity_control": sel,
        "population": {"n": len(rows), "syc": int(y.sum())},
    }, indent=1))
    joblib.dump(frozen, repo_path(cfg, cfg["paths"]["results"], "frozen_probe.pkl"))
    print(f"[06] FROZEN. overall={best['layer']}/{best['pos']} AUROC={best['auroc']:.3f}; "
          f"prompt_last winner={pl_best_key} AUROC={pl_cells[pl_best_key]['best_auroc']}")


if __name__ == "__main__":
    main()
