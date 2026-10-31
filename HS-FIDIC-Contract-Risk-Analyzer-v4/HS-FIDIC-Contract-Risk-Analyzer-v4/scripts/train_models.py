"""Train and evaluate every individual and hybrid model for HS-FIDIC.

Run from project root:
    python scripts/train_models.py

Protocol (leakage-conscious)
1. Clean the workbook: English, non-OCR-noise sentences, de-duplicated.
2. Fixed stratified 80/20 split -> DEV / TEST. TEST is touched only once, at the end.
3. 5-fold stratified CV on DEV for every individual model -> out-of-fold (OOF) probabilities,
   CV mean ± std macro-F1.
4. Hybrids (top-k selection, weights, stacking meta-learners) are fitted ONLY on DEV OOF
   probabilities.
5. Every individual model is refit on all of DEV and scored on TEST, then each hybrid is
   applied to the TEST probabilities.  95% bootstrap CIs are reported.
6. The deployed default model is the hybrid with the best DEV-CV macro-F1 (never chosen on TEST).
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import rule_features, RULE_NAMES  # noqa: E402
from src.model_zoo import (HYBRID_INFO, SEED, hybrid_competitive, hybrid_confidence, hybrid_majority,  # noqa: E402
                           hybrid_weighted, individual_models, tune_weights)
from src import hierarchy  # noqa: E402

TYPE_LABELS = ["Definition", "Heading", "Obligation", "Right", "Risk"]
PARTY_LABELS = ["Contractor", "Employer", "Shared"]
DATA = ROOT / "data" / "FIDIC_Thesis_Style_Labeled_Dataset.xlsx"


# ----------------------------------------------------------------------------- data
def _tok(s):
    return re.findall(r"[a-z]+", str(s).lower())


from src.text_utils import is_heading_like, is_fragment  # noqa: E402


def load_clean_dataset():
    raw = pd.read_excel(DATA, sheet_name="Combined_Unique")
    raw["Sentence"] = raw["Sentence"].astype(str).str.strip()
    n_raw = len(raw)
    ref = raw[(raw.Language == "English") & (raw.Extraction_Method == "PDF text")]
    cnt = Counter(w for s in ref.Sentence for w in _tok(s))
    vocab = {w for w, n in cnt.items() if n >= 2}

    def quality(s):
        t = _tok(s)
        return sum(w in vocab for w in t) / max(len(t), 1)

    raw["quality"] = raw.Sentence.map(quality)
    df = raw[(raw.Language == "English") & (raw.quality >= 0.8) & raw.Sentence_Type.isin(TYPE_LABELS)].copy()
    df = df[df.Sentence.str.len() >= 3]
    df["norm"] = df.Sentence.str.lower().str.replace(r"[^a-z]", "", regex=True)
    df = df[df.norm.str.len() > 0]
    # --- label correction & fragment removal (documented in metrics.json)
    s = df.Sentence.str.strip()
    is_head = s.map(is_heading_like)
    relabelled = int((is_head & (df.Sentence_Type != "Heading")).sum())
    df.loc[is_head, "Sentence_Type"] = "Heading"
    df.loc[is_head, "Related_Party"] = np.nan
    frag = (~is_head) & s.map(is_fragment)
    n_frag = int(frag.sum())
    df = df[~frag]
    # resolve duplicates by majority label
    df = (df.groupby("norm", sort=False)
            .agg(Sentence=("Sentence", "first"), Book=("Book", "first"),
                 Sentence_Type=("Sentence_Type", lambda s: s.mode().iloc[0]),
                 Related_Party=("Related_Party", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan))
            .reset_index())
    audit = {"raw_rows": int(n_raw), "kept_rows": int(len(df)),
             "dropped_total": int(n_raw - len(df)),
             "dropped_line_fragments": n_frag,
             "relabelled_as_heading": relabelled,
             "type_distribution": df.Sentence_Type.value_counts().to_dict(),
             "party_distribution": df.Related_Party.value_counts().to_dict()}
    return df, audit


def heading_augmentation(texts, exclude_norm: set[str]):
    """Sub-clause titles harvested from cross references such as 'Sub-Clause 8.4 [Extension of Time]'."""
    rx = re.compile(r"Sub-Clause\s+(\d+(?:\.\d+)*)\s*\[([^\]]{3,80})\]")
    out = set()
    for t in texts:
        for num, title in rx.findall(str(t)):
            title = title.strip()
            out.add(f"{num} {title}")
            out.add(title)
    rows = [h for h in sorted(out) if re.sub(r"[^a-z]", "", h.lower()) not in exclude_norm]
    return rows


# ----------------------------------------------------------------------------- metrics
def _ece(P, y_idx, bins=10):
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    curve = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            acc, c = (pred[m] == y_idx[m]).mean(), conf[m].mean()
            e += m.mean() * abs(acc - c)
            curve.append({"bin": round(float((lo + hi) / 2), 2), "confidence": round(float(c), 4),
                          "accuracy": round(float(acc), 4), "count": int(m.sum())})
    return float(e), curve


def evaluate(P, y_idx, labels, seconds=None, boot=600):
    pred = P.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y_idx, pred, average="macro", zero_division=0,
                                                    labels=list(range(len(labels))))
    pcp, pcr, pcf, pcs = precision_recall_fscore_support(y_idx, pred, average=None, zero_division=0,
                                                         labels=list(range(len(labels))))
    Y = np.eye(len(labels))[y_idx]
    Pc = np.clip(P, 1e-9, 1)
    Pc = Pc / Pc.sum(1, keepdims=True)
    try:
        auc = roc_auc_score(Y, Pc, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")
    ece, curve = _ece(Pc, y_idx)
    rng = np.random.default_rng(SEED)
    bs = []
    n = len(y_idx)
    for _ in range(boot):
        i = rng.integers(0, n, n)
        bs.append(f1_score(y_idx[i], pred[i], average="macro", labels=list(range(len(labels))), zero_division=0))
    res = {
        "accuracy": round(float(accuracy_score(y_idx, pred)), 5),
        "macro_precision": round(float(pr), 5),
        "macro_recall": round(float(rc), 5),
        "macro_f1": round(float(f1), 5),
        "macro_f1_ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
        "weighted_f1": round(float(f1_score(y_idx, pred, average="weighted", zero_division=0)), 5),
        "roc_auc_ovr": round(float(auc), 5),
        "log_loss": round(float(log_loss(y_idx, Pc, labels=list(range(len(labels))))), 5),
        "brier": round(float(np.mean(np.sum((Pc - Y) ** 2, 1))), 5),
        "ece": round(ece, 5),
        "per_class": {lab: {"precision": round(float(a), 4), "recall": round(float(b), 4), "f1": round(float(c), 4),
                            "support": int(d)} for lab, a, b, c, d in zip(labels, pcp, pcr, pcf, pcs)},
        "confusion_matrix": confusion_matrix(y_idx, pred, labels=list(range(len(labels)))).tolist(),
        "calibration_curve": curve,
    }
    if seconds is not None:
        res["inference_ms_per_1000"] = round(1000 * seconds / max(n, 1) * 1000, 2)
    return res


def proba_aligned(model, X, labels):
    p = model.predict_proba(X)
    cls = [str(c) for c in model.classes_]
    out = np.zeros((len(X), len(labels)))
    for j, lab in enumerate(labels):
        if lab in cls:
            out[:, j] = p[:, cls.index(lab)]
    s = out.sum(1, keepdims=True)
    s[s == 0] = 1
    return out / s


# ----------------------------------------------------------------------------- task runner
def run_task(task, X_dev, y_dev, X_test, y_test, labels, extra_dev=None, folds=5):
    """extra_dev: (texts, labels) appended to training folds only (augmentation)."""
    zoo = individual_models()
    lab_idx = {l: i for i, l in enumerate(labels)}
    yd = np.array([lab_idx[v] for v in y_dev])
    yt = np.array([lab_idx[v] for v in y_test])
    X_dev, y_dev = list(X_dev), list(y_dev)
    X_test, y_test = list(X_test), list(y_test)
    ex_X, ex_y = (list(extra_dev[0]), list(extra_dev[1])) if extra_dev else ([], [])
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof, cv, fitted, test_P, records = {}, {}, {}, {}, []

    for name, spec in zoo.items():
        t0 = time.perf_counter()
        P = np.zeros((len(X_dev), len(labels)))
        fold_f1 = []
        for tr, va in skf.split(X_dev, yd):
            m = spec["factory"]()
            m.fit([X_dev[i] for i in tr] + ex_X, [y_dev[i] for i in tr] + ex_y)
            P[va] = proba_aligned(m, [X_dev[i] for i in va], labels)
            fold_f1.append(f1_score(yd[va], P[va].argmax(1), average="macro"))
        oof[name] = P
        cv[name] = (float(np.mean(fold_f1)), float(np.std(fold_f1)))
        m = spec["factory"]()
        m.fit(X_dev + ex_X, list(y_dev) + ex_y)
        fitted[name] = m
        t1 = time.perf_counter()
        test_P[name] = proba_aligned(m, X_test, labels)
        t_inf = time.perf_counter() - t1
        ev = evaluate(test_P[name], yt, labels, t_inf)
        records.append({"model": name, "category": "Individual", "vectorizer": spec["vectorizer"],
                        "algorithm": spec["algorithm"], "family": spec["family"],
                        "cv_macro_f1": round(cv[name][0], 5), "cv_macro_f1_std": round(cv[name][1], 5),
                        "train_seconds": round(t1 - t0, 2), **ev})
        print(f"[{task}] {name:45s} CV-F1 {cv[name][0]:.3f}±{cv[name][1]:.3f}  test-F1 {ev['macro_f1']:.3f}  acc {ev['accuracy']:.3f}", flush=True)

    ranked = sorted(cv, key=lambda n: cv[n][0], reverse=True)
    top3, top5 = ranked[:3], ranked[:5]
    hybrids = {}

    # --- voting family
    rank_w = [3, 2, 1]
    opt_w, opt_f = tune_weights([oof[n] for n in top5], yd)
    defs = {
        "Hybrid: Majority Vote (Top-3)": {"members": top3, "kind": "majority"},
        "Hybrid: Competitive Voting (Thesis, Top-3)": {"members": top3, "kind": "competitive"},
        "Hybrid: Confidence-Weighted Vote (Top-3)": {"members": top3, "kind": "confidence"},
        "Hybrid: Rank-Weighted Vote (Top-3)": {"members": top3, "kind": "weighted", "weights": rank_w},
        "Hybrid: Optimized Soft Voting (Top-5)": {"members": top5, "kind": "weighted", "weights": [float(x) for x in opt_w]},
    }
    # --- stacking (all models)
    all_names = list(zoo)
    Z_dev = np.hstack([oof[n] for n in all_names])
    meta = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced")
    meta.fit(Z_dev, yd)
    defs["Hybrid: Stacking Meta-Learner (All models)"] = {"members": all_names, "kind": "stacking", "meta": meta}
    # --- ML + rules stacking
    R_dev = rule_features(X_dev)
    scaler = StandardScaler().fit(R_dev)
    Zr_dev = np.hstack([np.hstack([oof[n] for n in top5]), scaler.transform(R_dev)])
    meta_r = LogisticRegression(C=0.5, max_iter=4000, class_weight="balanced").fit(Zr_dev, yd)
    defs["Hybrid: ML + Rule-Based Stacking"] = {"members": top5, "kind": "rule_stacking", "meta": meta_r, "scaler": scaler}

    def apply(defn, Pd, texts):
        P = [Pd[n] for n in defn["members"]]
        k = defn["kind"]
        if k == "majority":
            return hybrid_majority(P)
        if k == "competitive":
            return hybrid_competitive(P)
        if k == "confidence":
            return hybrid_confidence(P)
        if k == "weighted":
            return hybrid_weighted(P, defn["weights"])
        if k == "stacking":
            return defn["meta"].predict_proba(np.hstack(P))
        if k == "rule_stacking":
            return defn["meta"].predict_proba(np.hstack([np.hstack(P), defn["scaler"].transform(rule_features(texts))]))
        raise ValueError(k)

    for hname, d in defs.items():
        # dev score from OOF (stacking metas are in-sample on OOF -> nested 5-fold estimate)
        if d["kind"] in ("stacking", "rule_stacking"):
            Pd_oof = np.zeros((len(X_dev), len(labels)))
            Zfull = Z_dev if d["kind"] == "stacking" else Zr_dev
            for tr, va in skf.split(X_dev, yd):
                mm = LogisticRegression(C=d["meta"].C, max_iter=4000, class_weight="balanced").fit(Zfull[tr], yd[tr])
                Pd_oof[va] = mm.predict_proba(Zfull[va])
        else:
            Pd_oof = apply(d, oof, X_dev)
        dev_f1 = float(f1_score(yd, Pd_oof.argmax(1), average="macro"))
        t1 = time.perf_counter()
        Pt = apply(d, test_P, X_test)
        t_inf = time.perf_counter() - t1 + sum(
            next(r["inference_ms_per_1000"] for r in records if r["model"] == m) for m in d["members"]) * len(X_test) / 1e6
        ev = evaluate(Pt, yt, labels, t_inf)
        test_P[hname] = Pt
        hybrids[hname] = {k: v for k, v in d.items()}
        records.append({"model": hname, "category": "Hybrid", "vectorizer": "—", "algorithm": d["kind"],
                        "family": "Hybrid", "members": d["members"] if len(d["members"]) <= 5 else f"all {len(d['members'])} models",
                        "weights": d.get("weights"), "cv_macro_f1": round(dev_f1, 5), "cv_macro_f1_std": None, **ev})
        print(f"[{task}] {hname:45s} CV-F1 {dev_f1:.3f}         test-F1 {ev['macro_f1']:.3f}  acc {ev['accuracy']:.3f}", flush=True)

    # --- thesis hierarchical (sentence type only)
    if task == "sentence_type":
        t0 = time.perf_counter()
        hier = hierarchy.HierarchicalTypeModel().fit(X_dev + ex_X, list(y_dev) + ex_y)
        # dev CV estimate
        Pd = np.zeros((len(X_dev), len(labels)))
        for tr, va in skf.split(X_dev, yd):
            hm = hierarchy.HierarchicalTypeModel().fit([X_dev[i] for i in tr] + ex_X, [y_dev[i] for i in tr] + ex_y)
            Pd[va] = hm.predict_proba_aligned([X_dev[i] for i in va], labels)
        dev_f1 = float(f1_score(yd, Pd.argmax(1), average="macro"))
        t1 = time.perf_counter()
        Pt = hier.predict_proba_aligned(X_test, labels)
        ev = evaluate(Pt, yt, labels, time.perf_counter() - t1)
        hname = "Hybrid: Thesis Hierarchical Binary Decomposition"
        test_P[hname] = Pt
        hybrids[hname] = {"members": hierarchy.STAGE_MEMBERS, "kind": "hierarchical", "model": hier}
        records.append({"model": hname, "category": "Hybrid", "vectorizer": "TF-IDF / char", "algorithm": "hierarchical",
                        "family": "Hybrid", "members": hierarchy.STAGE_MEMBERS, "weights": None,
                        "cv_macro_f1": round(dev_f1, 5), "cv_macro_f1_std": None,
                        "train_seconds": round(t1 - t0, 2), **ev})
        print(f"[{task}] {hname:45s} CV-F1 {dev_f1:.3f}         test-F1 {ev['macro_f1']:.3f}  acc {ev['accuracy']:.3f}", flush=True)

    hyb_records = [r for r in records if r["category"] == "Hybrid"]
    default = max(hyb_records, key=lambda r: r["cv_macro_f1"])["model"]
    best_individual = ranked[0]
    return {
        "labels": labels, "individual": fitted, "hybrids": hybrids, "ranking": ranked,
        "default_model": default, "best_individual": best_individual,
        "records": records, "test_P": {k: v.tolist() for k, v in test_P.items()}, "y_test": yt.tolist(),
    }


def main():
    df, audit = load_clean_dataset()
    print("Data audit:", json.dumps(audit, indent=1), flush=True)

    # ---------------- sentence type
    dev, test = train_test_split(df, test_size=0.2, stratify=df.Sentence_Type, random_state=SEED)
    aug = heading_augmentation(df.Sentence, set(test.norm) | set(dev.norm))
    print(f"Heading augmentation rows (training only): {len(aug)}", flush=True)
    type_res = run_task("sentence_type", dev.Sentence, dev.Sentence_Type, test.Sentence, test.Sentence_Type,
                        TYPE_LABELS, extra_dev=(aug, ["Heading"] * len(aug)))

    # ---------------- related party (only Obligation / Risk / Right sentences)
    pdf = df[df.Sentence_Type.isin(["Obligation", "Risk", "Right"]) & df.Related_Party.isin(PARTY_LABELS)]
    pdev, ptest = train_test_split(pdf, test_size=0.2, stratify=pdf.Related_Party, random_state=SEED)
    party_res = run_task("related_party", pdev.Sentence, pdev.Related_Party, ptest.Sentence, ptest.Related_Party,
                         PARTY_LABELS)

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    bundle = {
        "version": "2.0",
        "type": {k: type_res[k] for k in ("labels", "individual", "hybrids", "ranking", "default_model", "best_individual")},
        "party": {k: party_res[k] for k in ("labels", "individual", "hybrids", "ranking", "default_model", "best_individual")},
        "rule_names": RULE_NAMES,
    }
    # per-model per-class precision on TEST -> used for "historical reliability" in predictability
    for key, res in (("type", type_res), ("party", party_res)):
        bundle[key]["class_precision"] = {r["model"]: {c: v["precision"] for c, v in r["per_class"].items()}
                                          for r in res["records"]}
        bundle[key]["model_accuracy"] = {r["model"]: r["accuracy"] for r in res["records"]}
    common = {"version": bundle["version"], "rule_names": RULE_NAMES}
    joblib.dump({**common, "type": bundle["type"]}, out / "hs_fidic_type_models.joblib", compress=("xz", 6))
    joblib.dump({**common, "party": bundle["party"]}, out / "hs_fidic_party_models.joblib", compress=("xz", 6))

    def strip(res):
        return [{k: v for k, v in r.items()} for r in res["records"]]

    metrics = {
        "evaluation_protocol": ("Stratified 80/20 DEV/TEST split (seed 42). 5-fold stratified CV on DEV gives out-of-fold "
                                "probabilities used to rank models, tune hybrid weights and fit stacking meta-learners. "
                                "TEST is used once for the reported scores; 95% CIs by 600 bootstrap resamples. "
                                "Default model = hybrid with best DEV-CV macro-F1 (never selected on TEST)."),
        "caveats": ["Workbook labels were generated automatically, not by FIDIC experts: scores measure agreement with those labels, not legal correctness.",
                    "OCR noise (ORANGE/SILVER books) and Indonesian rows were removed; RED and YELLOW books share many near-identical clauses, which can inflate scores.",
                    "Heading augmentation (sub-clause titles harvested from cross-references) is used for training only, never in TEST."],
        "data_audit": audit,
        "heading_augmentation_rows": len(aug),
        "sentence_type": {"labels": TYPE_LABELS, "n_dev": len(dev), "n_test": len(test),
                          "default_model": type_res["default_model"], "best_individual": type_res["best_individual"],
                          "results": strip(type_res), "y_test": type_res["y_test"]},
        "related_party": {"labels": PARTY_LABELS, "n_dev": len(pdev), "n_test": len(ptest),
                          "default_model": party_res["default_model"], "best_individual": party_res["best_individual"],
                          "results": strip(party_res), "y_test": party_res["y_test"]},
        "hybrid_descriptions": HYBRID_INFO,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    for f in ("hs_fidic_type_models.joblib", "hs_fidic_party_models.joblib"):
        print("Saved", f, round((out / f).stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
