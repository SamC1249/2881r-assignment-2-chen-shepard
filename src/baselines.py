"""Stage 07 (local, CPU): fit the baseline bank on train+val (same CV protocol as probes).

Baselines (see PLAN.md "Baselines"): majority class; TF-IDF(response); keyword rule;
level-only (prompt-side cheat detector); length-only; assertion-aware text (TF-IDF over
prompt+response + fuzzy answer-match features — the information-matched text baseline).
"""

import argparse
import json
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import load_config, repo_path
from finalize_dataset import AGREE_CUE, load_all
from train_probes import probe_population

PHRASING_IDS = [f"L{l}_{p}" for l in range(1, 6) for p in "ab"]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def featurize(rows, prompts, responses, scen):
    """Returns dict of feature blocks per baseline."""
    resp_text, full_text, levels, extra = [], [], [], []
    for r in rows:
        resp = responses[r["row_id"]]
        p = prompts[resp["example_id"]]
        s = scen[p["scenario_id"]]
        rt = resp["response_text"]
        resp_text.append(rt)
        full_text.append(p["messages"][0]["content"] + "\n" + rt)
        onehot = [1.0 if p["phrasing_id"] == ph else 0.0 for ph in PHRASING_IDS]
        levels.append([float(p["level"])] + onehot)
        nrt = norm(rt)
        extra.append([
            float(norm(s["wrong_answer"]) in nrt),     # response contains asserted (wrong) answer
            float(norm(s["correct_answer"]) in nrt),   # response contains correct answer
            float(len(resp["response_token_ids"])),    # length
            float(bool(AGREE_CUE.search(rt))),         # agreement cue
        ])
    return {"resp_text": resp_text, "full_text": full_text,
            "level": np.array(levels), "extra": np.array(extra)}


def build_models():
    lr = dict(class_weight="balanced", max_iter=2000)
    return {
        "tfidf_response": lambda: make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000),
            LogisticRegression(**lr)),
        "assertion_aware_text": None,  # custom: tfidf(full) + extra features, built below
        "level_only": lambda: make_pipeline(StandardScaler(), LogisticRegression(**lr)),
        "length_only": lambda: make_pipeline(StandardScaler(), LogisticRegression(**lr)),
    }


class AssertionAwareText:
    """TF-IDF over prompt+response concatenated with answer-match/length features."""

    def __init__(self):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(class_weight="balanced", max_iter=2000)

    def _stack(self, texts, extra, fit):
        from scipy.sparse import hstack, csr_matrix
        T = self.vec.fit_transform(texts) if fit else self.vec.transform(texts)
        E = self.scaler.fit_transform(extra) if fit else self.scaler.transform(extra)
        return hstack([T, csr_matrix(E)])

    def fit(self, texts, extra, y):
        self.clf.fit(self._stack(texts, extra, fit=True), y)
        return self

    def predict_proba(self, texts, extra):
        return self.clf.predict_proba(self._stack(texts, extra, fit=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    scen, prompts, responses, splits, labels = load_all(cfg)

    rows = probe_population(cfg)  # same train+val population as the probe
    y = np.array([r["y"] for r in rows])
    F = featurize(rows, prompts, responses, scen)
    skf = StratifiedKFold(cfg["probe"]["cv_folds"], shuffle=True, random_state=cfg["seed"])
    folds = list(skf.split(np.zeros(len(y)), y))

    def cv_auc(fit_predict):
        aucs = []
        for tr, va in folds:
            p = fit_predict(tr, va)
            aucs.append(roc_auc_score(y[va], p))
        return round(float(np.mean(aucs)), 4), round(float(np.std(aucs)), 4)

    report = {"population": {"n": len(rows), "syc": int(y.sum()),
                             "majority_class": "NOT" if y.mean() < 0.5 else "SYC"}}

    report["tfidf_response"] = cv_auc(lambda tr, va: build_models()["tfidf_response"]()
                                      .fit([F["resp_text"][i] for i in tr], y[tr])
                                      .predict_proba([F["resp_text"][i] for i in va])[:, 1])
    report["assertion_aware_text"] = cv_auc(lambda tr, va: AssertionAwareText()
                                            .fit([F["full_text"][i] for i in tr], F["extra"][tr], y[tr])
                                            .predict_proba([F["full_text"][i] for i in va], F["extra"][va])[:, 1])
    report["level_only"] = cv_auc(lambda tr, va: build_models()["level_only"]()
                                  .fit(F["level"][tr], y[tr]).predict_proba(F["level"][va])[:, 1])
    report["length_only"] = cv_auc(lambda tr, va: build_models()["length_only"]()
                                   .fit(F["extra"][tr][:, [2]], y[tr])
                                   .predict_proba(F["extra"][va][:, [2]])[:, 1])
    report["keyword_rule"] = {"auroc_as_score": round(float(roc_auc_score(y, F["extra"][:, 3])), 4)}

    # freeze fitted-on-all-train+val versions for test evaluation
    frozen = {
        "tfidf_response": build_models()["tfidf_response"]().fit(F["resp_text"], y),
        "assertion_aware_text": AssertionAwareText().fit(F["full_text"], F["extra"], y),
        "level_only": build_models()["level_only"]().fit(F["level"], y),
        "length_only": build_models()["length_only"]().fit(F["extra"][:, [2]], y),
        "majority_label": int(round(y.mean())),
    }
    joblib.dump(frozen, repo_path(cfg, cfg["paths"]["results"], "frozen_baselines.pkl"))
    out = repo_path(cfg, cfg["paths"]["results"], "baselines.json")
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
