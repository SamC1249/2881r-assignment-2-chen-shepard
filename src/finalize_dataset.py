"""Stage 04 (local, CPU): apply the neutral-correctness filter, prune splits, emit the
Part-1 behavioral report (results/behavioral_dev.json), mine surface-cue trap cases, and
export blind human-labeling CSVs.

Usage:
  python src/finalize_dataset.py                       # filter + behavioral report + traps
  python src/finalize_dataset.py --make-human-sample kappa|post_freeze|correctness|random_review
"""

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict

from common import load_config, read_jsonl, repo_path, responses_path, rubric_hash

AGREE_CUE = re.compile(r"you'?re (right|correct)|that'?s (right|correct)|exactly|yes[,.!]", re.IGNORECASE)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def load_all(cfg):
    data = cfg["paths"]["data"]
    scen = read_jsonl(repo_path(cfg, data, "scenarios.jsonl"), key="scenario_id")
    prompts = read_jsonl(repo_path(cfg, data, "prompts.jsonl"), key="example_id")
    responses = read_jsonl(responses_path(cfg), key="row_id")
    splits = json.loads(repo_path(cfg, data, "splits.json").read_text())
    labels = {}
    for mode in ["correctness", "sycophancy", "agreement"]:
        path = repo_path(cfg, data, f"labels_{mode}.{rubric_hash(cfg, mode)}.jsonl")
        labels[mode] = read_jsonl(path, key="row_id") if path.exists() else {}
    return scen, prompts, responses, splits, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    ap.add_argument("--make-human-sample",
                    choices=["kappa", "post_freeze", "correctness", "random_review"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = cfg["paths"]["data"]
    scen, prompts, responses, splits, labels = load_all(cfg)
    rng = random.Random(cfg["seed"] + 17)

    # ---- neutral-correctness filter (temp-0 neutral response judged CORRECT)
    correct_scen = set()
    for rid, lab in labels["correctness"].items():
        if lab["parse_ok"] and lab["label"] == "CORRECT":
            correct_scen.add(prompts[lab["example_id"]]["scenario_id"])
    judged_scen = {prompts[l["example_id"]]["scenario_id"]
                   for l in labels["correctness"].values() if l["parse_ok"]}
    usable = {sid: {**splits[sid], "usable": sid in correct_scen} for sid in splits
              if sid in judged_scen}
    print(f"[04] correctness filter: {len(correct_scen)}/{len(judged_scen)} scenarios usable "
          f"({100 * len(correct_scen) / max(1, len(judged_scen)):.0f}%)")

    # ---- main-table rows: assert_wrong, sample 0, non-dose, usable scenario
    rows = []
    for rid, lab in labels["sycophancy"].items():
        if not lab["parse_ok"] or lab["sample_idx"] != 0:
            continue
        p = prompts[lab["example_id"]]
        if p["dose_subset"] or p["scenario_id"] not in correct_scen:
            continue
        r = responses[rid]
        rows.append({"scenario_id": p["scenario_id"], "label": lab["label"],
                     "level": p["level"], "domain": scen[p["scenario_id"]]["domain"],
                     "split": splits[p["scenario_id"]]["split"],
                     "truncated": r["finish_reason"] == "length",
                     "has_agree_cue": bool(AGREE_CUE.search(r["response_text"])),
                     "row_id": rid})

    # ---- trap mining: surface cue conflicts (the assignment's required hard cases)
    traps = [r for r in rows if (r["has_agree_cue"] and r["label"] == "NOT_SYCOPHANTIC")
             or (not r["has_agree_cue"] and r["label"] == "SYCOPHANTIC")]
    with open(repo_path(cfg, data, "trap_candidates.json"), "w") as f:
        json.dump([t["row_id"] for t in traps], f)
    print(f"[04] trap candidates: {len(traps)}")

    # ---- human-sample exports (blind: no judge output in the CSV)
    if args.make_human_sample:
        hv = cfg["human_validation"]
        mode = args.make_human_sample
        if mode == "correctness":
            ok = [rid for rid, l in labels["correctness"].items() if l["parse_ok"]]
            by = defaultdict(list)
            for rid in ok:
                by[labels["correctness"][rid]["label"]].append(rid)
            n2 = hv["correctness_check_n"] // 2
            pick = rng.sample(by["CORRECT"], min(n2, len(by["CORRECT"]))) + \
                   rng.sample(by["INCORRECT"], min(n2, len(by["INCORRECT"])))
        else:
            pool = [r for r in rows if r["split"] in ("train", "val")]
            if mode == "random_review":
                pool = [r for r in rows if r["split"] == "test"]
                pick = [r["row_id"] for r in rng.sample(pool, min(hv["random_review_n"], len(pool)))]
            else:
                n = hv["kappa_sample_n"] if mode == "kappa" else hv["post_freeze_n"]
                trap_ids = [t["row_id"] for t in traps if t["split"] in ("train", "val")]
                pick = rng.sample(trap_ids, min(hv["kappa_min_traps"], len(trap_ids)))
                syc = [r["row_id"] for r in pool if r["label"] == "SYCOPHANTIC" and r["row_id"] not in pick]
                not_ = [r["row_id"] for r in pool if r["label"] == "NOT_SYCOPHANTIC" and r["row_id"] not in pick]
                half = (n - len(pick)) // 2
                pick += rng.sample(syc, min(half, len(syc))) + rng.sample(not_, min(half, len(not_)))
        out = repo_path(cfg, data, f"human_sample_{mode}.csv")
        with open(out, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["row_id", "question", "correct_answer", "asserted_answer",
                           "response", "YOUR_LABEL"])
            for rid in pick:
                p = prompts[responses[rid]["example_id"]]
                s = scen[p["scenario_id"]]
                asserted = s["wrong_answer"] if p["condition"] == "assert_wrong" else s["correct_answer"]
                wcsv.writerow([rid, s["question"], s["correct_answer"], asserted,
                               responses[rid]["response_text"], ""])
        print(f"[04] wrote {out} ({len(pick)} rows) — label blind, then compare.")
        return

    # ---- Part 1 behavioral report (DEV ONLY: train+val)
    dev = [r for r in rows if r["split"] in ("train", "val")]
    def rate(sub):
        k = sum(1 for r in sub if r["label"] == "SYCOPHANTIC")
        n = sum(1 for r in sub if r["label"] != "AMBIGUOUS")
        return {"n": n, "syc": k, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}

    report = {
        "dev_overall": rate(dev),
        "by_level": {lv: rate([r for r in dev if r["level"] == lv]) for lv in range(1, 6)},
        "by_domain": {d: rate([r for r in dev if r["domain"] == d])
                      for d in sorted({r["domain"] for r in dev})},
        "ambiguous_rate": {
            "overall": round(sum(1 for r in dev if r["label"] == "AMBIGUOUS") / max(1, len(dev)), 4),
            "by_level": {lv: round(sum(1 for r in dev if r["level"] == lv and r["label"] == "AMBIGUOUS")
                                   / max(1, sum(1 for r in dev if r["level"] == lv)), 4)
                         for lv in range(1, 6)}},
        "truncation_rate": round(sum(1 for r in dev if r["truncated"]) / max(1, len(dev)), 4),
        "label_counts_per_split": {sp: dict(Counter(r["label"] for r in rows if r["split"] == sp))
                                   for sp in ["train", "val", "test"]},
        "test_slice_counts": {
            "heldout_domain": dict(Counter(r["label"] for r in rows if r["split"] == "test"
                                           and splits[r["scenario_id"]]["heldout_domain"])),
            "heldout_phrasing": dict(Counter(r["label"] for r in rows if r["split"] == "test"
                                             and splits[r["scenario_id"]]["heldout_phrasing"])),
        },
        "min_per_class_rule": cfg["reporting"]["min_examples_per_class_per_slice"],
    }
    # level trend test (between-scenario): logistic regression of label on level
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        pts = [(r["level"], 1 if r["label"] == "SYCOPHANTIC" else 0)
               for r in dev if r["label"] != "AMBIGUOUS"]
        X = np.array([[p[0]] for p in pts]); y = np.array([p[1] for p in pts])
        if len(set(y)) == 2:
            coef = LogisticRegression().fit(X, y).coef_[0][0]
            boots = []
            idx = np.arange(len(y))
            brng = np.random.default_rng(cfg["seed"])
            for _ in range(1000):
                b = brng.choice(idx, len(idx))
                if len(set(y[b])) == 2:
                    boots.append(LogisticRegression().fit(X[b], y[b]).coef_[0][0])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            report["level_trend"] = {"logit_coef_per_level": round(float(coef), 4),
                                     "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    except Exception as e:
        report["level_trend"] = f"failed: {e}"

    # agreement-rate control non-vacuity (assert_correct)
    ag = [l for l in labels["agreement"].values() if l["parse_ok"]
          and prompts[l["example_id"]]["scenario_id"] in correct_scen
          and splits[prompts[l["example_id"]]["scenario_id"]]["split"] in ("train", "val")]
    k = sum(1 for l in ag if l["label"] == "AGREED")
    report["assert_correct_agreement"] = {"n": len(ag), "agree_rate":
                                          round(k / len(ag), 4) if ag else None}

    out = repo_path(cfg, cfg["paths"]["results"], "behavioral_dev.json")
    out.write_text(json.dumps(report, indent=1))
    repo_path(cfg, data, "manifest.json").write_text(json.dumps(
        {"usable_scenarios": sorted(correct_scen), "splits_pruned": usable}, indent=1))
    print(f"[04] wrote {out}")
    print(json.dumps({k: report[k] for k in ["dev_overall", "by_level", "level_trend",
                                             "assert_correct_agreement"]}, indent=1))


if __name__ == "__main__":
    main()
