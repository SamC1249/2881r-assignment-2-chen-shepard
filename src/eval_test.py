"""Stage 08 (local, CPU): THE one-shot test evaluation. Refuses to run twice without
--force-rerun (any rerun must be documented in the report).

Reports, for frozen probes (overall + prompt_last) and frozen baselines:
- slices: rest_of_test (honest iid number), heldout_phrasing, heldout_domain (qualitative)
- confusion matrix, per-class P/R, AUROC, balanced accuracy, scenario-bootstrap CIs
- paired bootstrap ΔAUROC (probe − baseline)
- within-level AUROC (pressure wording held constant)
- OOD diagnostics: assert_correct, neutral-wrong (diagnostic_only), AMBIGUOUS score dists
"""

import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

from common import load_config, repo_path
from baselines import AssertionAwareText, featurize  # noqa: F401 (joblib unpickle needs class)
from finalize_dataset import load_all
from train_probes import load_acts, probe_population


def boot_ci(stat_fn, n, seed, B=2000):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        idx = rng.choice(n, n)
        v = stat_fn(idx)
        if v is not None:
            vals.append(v)
    return [round(float(np.percentile(vals, q)), 4) for q in (2.5, 97.5)]


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else None


def metrics_block(y, prob, thr, seed, min_n):
    out = {"n": len(y), "pos": int(np.sum(y))}
    if len(y) == 0:
        return out
    pred = (prob >= thr).astype(int)
    out["auroc"] = safe_auc(y, prob)
    if out["auroc"] is not None:
        out["auroc_ci95"] = boot_ci(lambda i: safe_auc(y[i], prob[i]), len(y), seed)
    out["balanced_acc"] = round(float(balanced_accuracy_score(y, pred)), 4)
    out["confusion"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    out["per_class"] = {c: {"precision": round(float(pr[i]), 4), "recall": round(float(rc[i]), 4),
                            "f1": round(float(f1[i]), 4), "support": int(sup[i])}
                        for i, c in enumerate(["NOT", "SYC"])}
    out["underpowered"] = bool(min(np.sum(y), np.sum(1 - y)) < min_n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--force-rerun", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out_path = repo_path(cfg, cfg["paths"]["results"], "test_report.json")
    if out_path.exists() and not args.force_rerun:
        raise SystemExit("[08] test_report.json exists — the test set has been used. "
                         "Rerun requires --force-rerun and MUST be documented in the report.")
    seed = cfg["seed"]
    min_n = cfg["reporting"]["min_examples_per_class_per_slice"]
    scen, prompts, responses, splits, labels = load_all(cfg)

    rows = probe_population(cfg, splits_wanted=("test",))
    y = np.array([r["y"] for r in rows])
    probes = joblib.load(repo_path(cfg, cfg["paths"]["results"], "frozen_probe.pkl"))
    bases = joblib.load(repo_path(cfg, cfg["paths"]["results"], "frozen_baselines.pkl"))
    X_all, ids = load_acts(cfg, {r["row_id"] for r in rows})
    order = {rid: i for i, rid in enumerate(ids)}
    X_all = X_all[[order[r["row_id"]] for r in rows]].astype(np.float32)
    F = featurize(rows, prompts, responses, scen)

    scores = {}
    for name, pb in probes.items():
        scores[f"probe_{name}"] = (pb["model"].predict_proba(
            X_all[:, pb["layer_idx"], pb["pos_idx"], :])[:, 1], pb["threshold"])
    scores["tfidf_response"] = (bases["tfidf_response"].predict_proba(F["resp_text"])[:, 1], 0.5)
    scores["assertion_aware_text"] = (
        bases["assertion_aware_text"].predict_proba(F["full_text"], F["extra"])[:, 1], 0.5)
    scores["level_only"] = (bases["level_only"].predict_proba(F["level"])[:, 1], 0.5)
    scores["length_only"] = (bases["length_only"].predict_proba(F["extra"][:, [2]])[:, 1], 0.5)
    scores["majority"] = (np.full(len(y), float(bases["majority_label"])), 0.5)

    slices = {
        "all_test": np.ones(len(rows), bool),
        "heldout_domain": np.array([splits[r["scenario_id"]]["heldout_domain"] for r in rows]),
        "heldout_phrasing": np.array([splits[r["scenario_id"]]["heldout_phrasing"] for r in rows]),
    }
    slices["rest_of_test"] = slices["all_test"] & ~slices["heldout_domain"] & ~slices["heldout_phrasing"]

    report = {"methods": {}, "paired_delta_auroc": {}, "within_level": {}, "ood": {},
              "note": "heldout slices are qualitative robustness checks (power)"}
    for name, (prob, thr) in scores.items():
        report["methods"][name] = {sl: metrics_block(y[m], prob[m], thr, seed, min_n)
                                   for sl, m in slices.items()}

    # paired bootstrap deltas vs the headline probe, on rest_of_test
    m = slices["rest_of_test"]
    hp = scores["probe_prompt_last"][0][m]
    for name in ["level_only", "tfidf_response", "assertion_aware_text", "length_only",
                 "probe_overall"]:
        q = scores[name][0][m]
        yy = y[m]
        def delta(idx):
            a, b = safe_auc(yy[idx], hp[idx]), safe_auc(yy[idx], q[idx])
            return None if a is None or b is None else a - b
        report["paired_delta_auroc"][f"probe_prompt_last_minus_{name}"] = {
            "delta": round(safe_auc(yy, hp) - safe_auc(yy, q), 4),
            "ci95": boot_ci(delta, int(m.sum()), seed)}

    # within-level AUROC (pressure wording constant) for headline + overall probes
    lv = np.array([r["level"] for r in rows])
    for pname in ["probe_prompt_last", "probe_overall", "level_only"]:
        report["within_level"][pname] = {
            int(l): safe_auc(y[(lv == l) & m], scores[pname][0][(lv == l) & m])
            for l in range(1, 6)}

    # ---- OOD diagnostics with the frozen probes (never trained on these)
    manifest = json.loads(repo_path(cfg, cfg["paths"]["data"], "manifest.json").read_text())
    usable = set(manifest["usable_scenarios"])

    def ood_rows(pred):
        out = []
        for rid, r in responses.items():
            p = prompts[r["example_id"]]
            if r["sample_idx"] == 0 and pred(p, rid):
                out.append(rid)
        return out

    neutral_wrong = ood_rows(lambda p, rid: p["condition"] == "neutral"
                             and p["scenario_id"] not in usable
                             and labels["correctness"].get(rid, {}).get("parse_ok"))
    assert_correct = ood_rows(lambda p, rid: p["condition"] == "assert_correct"
                              and p["scenario_id"] in usable)
    ambiguous = [rid for rid, l in labels["sycophancy"].items()
                 if l["parse_ok"] and l["label"] == "AMBIGUOUS" and l["sample_idx"] == 0]
    for tag, rids in [("neutral_wrong", neutral_wrong), ("assert_correct", assert_correct),
                      ("ambiguous", ambiguous)]:
        if not rids:
            report["ood"][tag] = {"n": 0}
            continue
        Xo, oids = load_acts(cfg, set(rids))
        report["ood"][tag] = {"n": len(oids)}
        for name, pb in probes.items():
            s = pb["model"].predict_proba(Xo[:, pb["layer_idx"], pb["pos_idx"], :])[:, 1]
            report["ood"][tag][f"probe_{name}"] = {
                "mean_score": round(float(s.mean()), 4),
                "frac_over_threshold": round(float((s >= pb["threshold"]).mean()), 4)}

    out_path.write_text(json.dumps(report, indent=1))
    print(f"[08] wrote {out_path} — test set is now SPENT.")
    for name in ["probe_prompt_last", "probe_overall", "level_only", "assertion_aware_text"]:
        print(f"  {name}: rest_of_test AUROC = {report['methods'][name]['rest_of_test'].get('auroc')}")


if __name__ == "__main__":
    main()
