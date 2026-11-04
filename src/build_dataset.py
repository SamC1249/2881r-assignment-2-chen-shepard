"""Stage 01: build scenarios, prompts, and splits — splits assigned HERE, before anything
is generated or labeled. Pilot scenarios pinned to train. Held-out domain -> test;
held-out phrasing (L3_b) appears only in test. CPU-only; no GPU needed.

Usage: python src/build_dataset.py [--config configs/run.yaml]
"""

import argparse
import random
import re
from collections import Counter

import yaml

from common import JsonlWriter, load_config, repo_path

CHOICE_DEPENDENT = re.compile(
    r"\bfollowing\b|\bthese (options|statements|choices)\b|\bstatements? (above|below)\b"
    r"|\b(I{1,3}|IV)\s*(,|and|only)\b|\bEXCEPT\b",
    re.IGNORECASE,
)
BAD_ANSWER = re.compile(r"^(all|none|both) of the (above|these)", re.IGNORECASE)


def usable(question: str, choices: list[str], answer_idx: int) -> bool:
    if CHOICE_DEPENDENT.search(question):
        return False
    if BAD_ANSWER.match(choices[answer_idx].strip()):
        return False
    distractors = [c for i, c in enumerate(choices) if i != answer_idx and not BAD_ANSWER.match(c.strip())]
    return len(distractors) >= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]
    rng = random.Random(cfg["seed"])

    with open(repo_path(cfg, cfg["paths"]["prompts"], "templates.yaml")) as f:
        templates = yaml.safe_load(f)
    phrasings = sorted(templates["assertion"].keys())  # L1_a ... L5_b
    heldout_phr = ds_cfg["heldout_phrasing"]

    # ---- load MMLU
    from datasets import load_dataset

    pools = {}
    dropped = Counter()
    for subj in ds_cfg["mmlu_subjects"]:
        rows = load_dataset("cais/mmlu", subj, split="test")
        keep = []
        for i, r in enumerate(rows):
            if usable(r["question"], r["choices"], r["answer"]):
                keep.append((i, r))
            else:
                dropped[subj] += 1
        rng.shuffle(keep)
        pools[subj] = keep
        print(f"[01] {subj}: {len(keep)} usable ({dropped[subj]} dropped)")

    # ---- sample scenarios: held-out domain capped at ~12% of total, rest spread evenly
    n_total = ds_cfg["n_scenarios"]
    ho_domain = ds_cfg["heldout_domain"]
    n_ho = min(len(pools[ho_domain]), max(1, int(0.12 * n_total)))
    others = [s for s in ds_cfg["mmlu_subjects"] if s != ho_domain]
    per_other = (n_total - n_ho) // len(others)
    # redistribute shortfall from small pools to subjects with spare capacity
    takes = {s: min(per_other, len(pools[s])) for s in others}
    shortfall = (n_total - n_ho) - sum(takes.values())
    for s in sorted(others, key=lambda s: len(pools[s]) - takes[s], reverse=True):
        add = min(shortfall, len(pools[s]) - takes[s])
        takes[s] += add
        shortfall -= add
    takes[ho_domain] = n_ho

    scenarios = []
    for subj in ds_cfg["mmlu_subjects"]:
        take = takes[subj]
        for i, r in pools[subj][:take]:
            clean = lambda c: c.strip().rstrip(".").strip()
            correct = clean(r["choices"][r["answer"]])
            distractors = [clean(c) for j, c in enumerate(r["choices"])
                           if j != r["answer"] and not BAD_ANSWER.match(c.strip())]
            scenarios.append({
                "scenario_id": f"{subj}_{i}",
                "domain": subj,
                "question": r["question"].strip(),
                "correct_answer": correct,
                "wrong_answer": rng.choice(distractors),
                "mmlu_ref": {"subject": subj, "index": i},
            })
    rng.shuffle(scenarios)
    print(f"[01] total scenarios: {len(scenarios)}")

    # ---- splits (scenario-level, domain-stratified; held-out domain forced to test)
    splits = {}
    frac = ds_cfg["splits"]
    by_domain = {}
    for s in scenarios:
        by_domain.setdefault(s["domain"], []).append(s["scenario_id"])
    for domain, ids in by_domain.items():
        rng.shuffle(ids)
        if domain == ho_domain:
            for sid in ids:
                splits[sid] = {"split": "test", "heldout_domain": True}
            continue
        n = len(ids)
        n_tr, n_va = int(frac["train"] * n), int(frac["val"] * n)
        for k, sid in enumerate(ids):
            sp = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
            splits[sid] = {"split": sp, "heldout_domain": False}

    # ---- pilot: pinned to train (eyeballed by humans pre-freeze)
    train_ids = [sid for sid, v in splits.items() if v["split"] == "train"]
    rng.shuffle(train_ids)
    pilot_ids = set(train_ids[: ds_cfg["n_pilot"]])
    dose_ids = set(train_ids[ds_cfg["n_pilot"]: ds_cfg["n_pilot"] + ds_cfg["dose_subset_n"]])

    # ---- ladder assignment: balanced within (domain x split); L3_b only in test
    levels = [1, 2, 3, 4, 5]
    scen_by_id = {s["scenario_id"]: s for s in scenarios}
    assign = {}
    groups = {}
    for sid, v in splits.items():
        groups.setdefault((scen_by_id[sid]["domain"], v["split"]), []).append(sid)
    for (_, split), ids in sorted(groups.items()):
        ids.sort()
        rng.shuffle(ids)
        for k, sid in enumerate(ids):
            lvl = levels[k % 5]
            options = [p for p in phrasings if p.startswith(f"L{lvl}_")]
            if split != "test":
                options = [p for p in options if p != heldout_phr]
            assign[sid] = (lvl, rng.choice(options))
    n_hp = sum(1 for sid, (_, p) in assign.items() if p == heldout_phr)
    print(f"[01] held-out phrasing {heldout_phr}: {n_hp} scenarios (all test)")

    # ---- write artifacts
    data = cfg["paths"]["data"]
    with JsonlWriter(repo_path(cfg, data, "scenarios.jsonl")) as w:
        for s in scenarios:
            w.write(s)

    with JsonlWriter(repo_path(cfg, data, "prompts.jsonl")) as w:
        def emit(sid, condition, level, phrasing_id, dose=False):
            s = scen_by_id[sid]
            if condition == "neutral":
                text = templates["neutral"].format(question=s["question"])
            else:
                ans = s["wrong_answer"] if condition == "assert_wrong" else s["correct_answer"]
                text = templates["assertion"][phrasing_id].format(question=s["question"], answer=ans)
            w.write({
                "example_id": f"{sid}__{condition}" + (f"__dose{level}" if dose else ""),
                "scenario_id": sid, "condition": condition, "level": level,
                "phrasing_id": phrasing_id, "is_pilot": sid in pilot_ids,
                "dose_subset": dose, "messages": [{"role": "user", "content": text}],
            })

        for sid, (lvl, phr) in assign.items():
            emit(sid, "neutral", None, None)
            emit(sid, "assert_wrong", lvl, phr)
            emit(sid, "assert_correct", lvl, phr)
            if sid in dose_ids:
                for extra_lvl in levels:
                    if extra_lvl != lvl:
                        emit(sid, "assert_wrong", extra_lvl, f"L{extra_lvl}_a", dose=True)

    import json
    split_rows = {sid: {**v, "is_pilot": sid in pilot_ids, "dose_subset": sid in dose_ids,
                        "level": assign[sid][0], "phrasing_id": assign[sid][1],
                        "heldout_phrasing": assign[sid][1] == heldout_phr}
                  for sid, v in splits.items()}
    repo_path(cfg, data, "splits.json").write_text(json.dumps(split_rows, indent=1))
    counts = Counter(v["split"] for v in splits.values())
    print(f"[01] splits: {dict(counts)}; pilot={len(pilot_ids)} (train), dose={len(dose_ids)} (train)")


if __name__ == "__main__":
    main()
