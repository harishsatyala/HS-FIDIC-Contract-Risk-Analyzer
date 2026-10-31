"""Risk engine: turns sentence-level ML predictions into a scored, mitigated risk register.

For every sentence:
  P_risk   = P(Sentence Type = Risk) from the selected model (individual or hybrid)
  P_rule   = 1 - Π(1 - w_t) over matched onerous-term triggers t        (rule layer)
  Risk Probability L = 1 - (1 - P_risk)(1 - P_rule)(1 - 0.15·P_obligation)
  Allocation A       = P(party = perspective) + 0.5·P(Shared)
  Impact I (1-5)     = category base impact ⊕ risk-library impact dimensions ⊕ trigger severity
  Risk Score (%)     = 100 · L · (0.4 + 0.6·A) · (0.6·I/5 + 0.4·p_lib/0.75)
  Predictability (%) = 100 · (0.35·model agreement + 0.25·confidence + 0.15·(1-entropy) + 0.25·historical class precision)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .features import TextCleaner
from .playbook import CATEGORIES, PROB_LEVEL, TRIGGERS

CAT_NAMES = list(CATEGORIES)
_CAT_RX = {c: [re.compile(t, re.I) for t in v["terms"]] for c, v in CATEGORIES.items()}
# multi-word (more specific) lexicon terms weigh more than single words
_W = {c: [2.0 if (" " in t or "'" in t) else 1.0 for t in v["terms"]] for c, v in CATEGORIES.items()}
_TRIG_RX = [(re.compile(rx, re.I), lab, sev, w, mit) for rx, lab, sev, w, mit in TRIGGERS]
RATING_BANDS = [(50, "Critical"), (30, "High"), (15, "Medium"), (0, "Low")]


# ----------------------------------------------------------------------------- risk library
def load_library(path: str | Path) -> pd.DataFrame:
    """Prefer the quantified library (gross exposure % TCV, probability factor, mitigation effectiveness)."""
    path = Path(path)
    q = path.with_name("Global_Energy_Project_Risk_Library_3000_Quantified.xlsx")
    if q.exists():
        path = q
    lib = pd.read_excel(path, sheet_name="Risk Library")
    lib = lib.rename(columns={"Employer / Contractor / Shared allocation": "Allocation",
                              "Probability (L/M/H)": "Probability", "Impact (Cost / Time / Quality / HSE)": "Impact"})
    lib["impact_dims"] = lib["Impact"].astype(str).str.count("/") + 1
    lib["p_lib"] = lib["Probability"].map(PROB_LEVEL).fillna(0.5)
    lib["quantified"] = "Gross Exposure (% TCV)" in lib.columns
    return lib


# ----------------------------------------------------------------------------- categories
def lexicon_scores(texts) -> np.ndarray:
    S = np.zeros((len(texts), len(CAT_NAMES)))
    for i, t in enumerate(texts):
        for j, c in enumerate(CAT_NAMES):
            S[i, j] = sum(_W[c][k] for k, rx in enumerate(_CAT_RX[c]) if rx.search(t))
    return S


class CategoryModel:
    """Weakly-supervised category model: lexicon labels -> TF-IDF logistic regression, blended with the lexicon."""

    def __init__(self, corpus=None):
        self.ml = None
        if corpus is not None and len(corpus) > 50:
            S = lexicon_scores(corpus)
            srt = np.sort(S, 1)
            keep = (srt[:, -1] > 0) & (srt[:, -1] > srt[:, -2])  # unambiguous lexicon labels only
            X = [t for t, k in zip(corpus, keep) if k]
            y = [CAT_NAMES[j] for j in S[keep].argmax(1)]
            if len(set(y)) >= 5:
                self.ml = make_pipeline(TextCleaner(), TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                                        LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"))
                self.ml.fit(X, y)

    def predict(self, texts, headings=None):
        S = lexicon_scores(texts)
        if headings is not None:
            H = lexicon_scores([h or "" for h in headings])
            S = S + 1.5 * H  # the clause heading is a strong category signal
        lex = S / np.maximum(S.sum(1, keepdims=True), 1)
        if self.ml is not None:
            pm = self.ml.predict_proba(texts)
            P = np.zeros_like(lex)
            for j, c in enumerate(self.ml.classes_):
                P[:, CAT_NAMES.index(c)] = pm[:, j]
            has = (S.sum(1) > 0)[:, None]
            blended = np.where(has, 0.8 * lex + 0.2 * P, P)
        else:
            blended = np.where(lex.sum(1, keepdims=True) > 0, lex, 1.0 / len(CAT_NAMES))
        idx = blended.argmax(1)
        return [CAT_NAMES[i] for i in idx], blended.max(1), S.sum(1)


# ----------------------------------------------------------------------------- triggers
def match_triggers(text: str):
    hits = [(lab, sev, w, mit) for rx, lab, sev, w, mit in _TRIG_RX if rx.search(text)]
    p_rule = 1 - np.prod([1 - w for _, _, w, _ in hits]) if hits else 0.0
    return hits, float(min(p_rule, 0.9))


# ----------------------------------------------------------------------------- library match
def best_library_record(lib: pd.DataFrame, category: str, sentence: str, context: dict, vec_cache: dict):
    cand = lib[lib["Risk Category"] == category]
    for col, key in (("Project Type", "project_type"), ("Energy Segment", "segment"),
                     ("Geographic Context", "region"), ("Lifecycle Phase", "phase")):
        v = context.get(key)
        if v and v != "Any":
            sub = cand[cand[col] == v]
            if len(sub):
                cand = sub
    if cand.empty:
        return None
    if len(cand) == 1:
        return cand.iloc[0]
    key = category
    if key not in vec_cache:
        v = TfidfVectorizer(stop_words="english").fit(lib.loc[lib["Risk Category"] == category, "Risk Statement"].tolist() + [sentence])
        vec_cache[key] = v
    v = vec_cache[key]
    sims = (v.transform(cand["Risk Statement"]) @ v.transform([sentence]).T).toarray().ravel()
    return cand.iloc[int(sims.argmax())]


_DRIVER = re.compile(r"that (.+?), combined with")


def _driver(statement: str) -> str:
    m = _DRIVER.search(str(statement))
    return m.group(1) if m else str(statement)[:90]


def rating(score: float) -> str:
    for th, lab in RATING_BANDS:
        if score >= th:
            return lab
    return "Low"


# ----------------------------------------------------------------------------- main
def analyse(rows, preds, suite, lib, cat_model, *, type_model, party_model, perspective="Contractor",
            context=None, contract_value=100_000_000.0, review_threshold=60.0, mitigation_effectiveness=None):
    """rows: segmentation output; preds: ModelSuite.predict output. Returns sentence-level DataFrame."""
    context = context or {}
    texts = [r["sentence"] for r in rows]
    tl, pl = suite.type.labels, suite.party.labels
    Pt, Pp = preds["type_P"], preds["party_P"]
    type_pred = [tl[i] for i in Pt.argmax(1)]
    # structural heading override from the segmenter (high-precision rule)
    for i, r in enumerate(rows):
        if r.get("is_heading") and Pt[i, tl.index("Heading")] < 0.5:
            type_pred[i] = "Heading"

    # predictability
    all_t = np.stack([preds["type_all"][n].argmax(1) for n in suite.type.individual], 1)
    sel_idx = np.array([tl.index(t) for t in type_pred])
    agreement = (all_t == sel_idx[:, None]).mean(1)
    conf = Pt.max(1)
    ent = -(np.clip(Pt, 1e-12, 1) * np.log(np.clip(Pt, 1e-12, 1))).sum(1) / np.log(len(tl))
    hist_prec = suite.type.class_precision.get(type_model, {})
    hist = np.array([hist_prec.get(t, 0.7) for t in type_pred])
    predictability = 100 * (0.35 * agreement + 0.25 * conf + 0.15 * (1 - ent) + 0.25 * hist)

    all_p = np.stack([preds["party_all"][n].argmax(1) for n in suite.party.individual], 1)
    party_idx = Pp.argmax(1)
    party_agree = (all_p == party_idx[:, None]).mean(1)

    cats, cat_conf, lex_hits = cat_model.predict(texts, [r.get("heading", "") for r in rows])
    vec_cache = {}
    out = []
    for i, r in enumerate(rows):
        t = texts[i]
        typ = type_pred[i]
        p = dict(zip(tl, Pt[i]))
        party = pl[party_idx[i]] if typ in ("Obligation", "Risk", "Right") else "—"
        pp = dict(zip(pl, Pp[i]))
        hits, p_rule = match_triggers(t) if typ not in ("Heading", "Definition") else ([], 0.0)
        if typ in ("Heading", "Definition"):
            L = 0.0
        else:
            L = 1 - (1 - p["Risk"]) * (1 - p_rule) * (1 - 0.15 * p["Obligation"])
        if perspective in ("Contractor", "Employer"):
            A = pp[perspective] + 0.5 * pp["Shared"]
        else:
            A = 1.0
        cat = cats[i]
        meta = CATEGORIES[cat]
        rec = best_library_record(lib, cat, t, context, vec_cache) if (L > 0 and lib is not None) else None
        p_lib = float(rec["p_lib"]) if rec is not None else 0.5
        lib_imp = int(rec["impact_dims"]) + 1 if rec is not None else meta["impact"]
        max_sev = max([h[1] for h in hits], default=0)
        impact = int(np.clip(round((meta["impact"] + lib_imp) / 2 + (1 if max_sev >= 4 else 0) + (1 if max_sev >= 5 else 0)), 1, 5))
        score = 100 * L * (0.4 + 0.6 * A) * (0.6 * impact / 5 + 0.4 * min(p_lib / 0.75, 1.0))
        eff = mitigation_effectiveness
        if rec is not None and "Gross Exposure (% TCV)" in rec.index:
            # quantified library: gross = TCV x exposure %, library EMV = gross x probability factor
            gross = contract_value * float(rec["Gross Exposure (% TCV)"])
            pf = float(rec["Probability Factor"])
            if eff is None:
                eff = float(rec["Mitigation Effectiveness"])
        else:
            gross = contract_value * meta["exposure"] * impact / 3
            pf = {"L": 0.15, "M": 0.40, "H": 0.70}.get(rec["Probability"] if rec is not None else "M", 0.4)
            if eff is None:
                eff = meta["eff"]
        lib_emv = gross * pf
        emv = lib_emv * L            # clause-weighted: library EMV x probability the clause carries the risk
        residual = emv * (1 - eff)
        mit_parts = [h[3] for h in sorted(hits, key=lambda h: -h[1])[:2]] + [meta["mitigation"]]
        out.append({
            "Page": r.get("page"), "Clause": r.get("clause") or "", "Sentence": t,
            "Sentence Type": typ, "Type Confidence %": round(100 * conf[i], 1),
            "P(Risk) %": round(100 * p["Risk"], 1), "P(Obligation) %": round(100 * p["Obligation"], 1),
            "P(Right) %": round(100 * p["Right"], 1),
            "Related Party": party, "Party Confidence %": round(100 * Pp[i].max(), 1) if party != "—" else None,
            "Allocation to you %": round(100 * A, 1) if party != "—" else 0.0,
            "Risk Category": cat if L > 0 else "—", "Category Confidence %": round(100 * cat_conf[i], 1),
            "FIDIC Reference": meta["ref"] if L > 0 else "",
            "Triggers": "; ".join(h[0] for h in hits), "Rule Signal %": round(100 * p_rule, 1),
            "Risk Probability %": round(100 * L, 1), "Impact (1-5)": impact if L > 0 else 0,
            "Library Likelihood": rec["Probability"] if rec is not None else "",
            "Risk Score %": round(score, 1), "Rating": rating(score) if L > 0 else "None",
            "Predictability %": round(float(predictability[i]), 1),
            "Model Agreement %": round(100 * agreement[i], 1),
            "Party Agreement %": round(100 * party_agree[i], 1) if party != "—" else None,
            "Needs Review": bool(predictability[i] < review_threshold),
            "Mitigation": " ".join(dict.fromkeys(mit_parts)),
            "Library Risk ID": rec["Risk ID"] if rec is not None else "",
            "Library Mitigation": rec["Mitigation Actions"] if rec is not None else "",
            "Early Warning Indicators": rec["Early Warning Indicators"] if rec is not None else "",
            "Risk Owner": rec["Risk Owner"] if rec is not None else "",
            "Library Risk Statement": rec["Risk Statement"] if rec is not None else "",
            "Risk Statement": _driver(rec["Risk Statement"]) if rec is not None else "",
            "Responsible Party (library)": rec["Allocation"] if rec is not None else "",
            "Probability (L/M/H)": rec["Probability"] if rec is not None else "",
            "Impact Areas": rec["Impact"] if rec is not None else "",
            "Project Type": rec["Project Type"] if rec is not None else "",
            "Probability Factor": round(pf, 2) if L > 0 else None,
            "Gross Exposure": round(gross, 0) if L > 0 else 0, "Library EMV": round(lib_emv, 0) if L > 0 else 0,
            "EMV": round(emv, 0), "Residual EMV": round(residual, 0),
            "Mitigation Effectiveness %": round(100 * eff),
        })
    return pd.DataFrame(out)


def risk_register(df: pd.DataFrame, min_probability=35.0) -> pd.DataFrame:
    reg = df[(df["Risk Probability %"] >= min_probability) & (df["Risk Category"] != "—")].copy()
    return reg.sort_values(["Risk Score %", "Risk Probability %"], ascending=False).reset_index(drop=True)


def contract_summary(df: pd.DataFrame, reg: pd.DataFrame) -> dict:
    top = reg["Risk Score %"].head(10)
    cat_emv = reg.groupby("Risk Category")["EMV"].max() if len(reg) else pd.Series(dtype=float)
    cat_res = reg.groupby("Risk Category")["Residual EMV"].max() if len(reg) else pd.Series(dtype=float)
    return {
        "sentences": len(df),
        "risks": len(reg),
        "critical": int((reg["Rating"] == "Critical").sum()),
        "high": int((reg["Rating"] == "High").sum()),
        "risk_index": round(float(top.mean()), 1) if len(top) else 0.0,
        "mean_probability": round(float(reg["Risk Probability %"].mean()), 1) if len(reg) else 0.0,
        "predictability": round(float(reg["Predictability %"].mean()), 1) if len(reg) else round(float(df["Predictability %"].mean()), 1),
        "needs_review": int(reg["Needs Review"].sum()) if len(reg) else 0,
        "emv": float(cat_emv.sum()), "residual": float(cat_res.sum()),
        "categories": int(reg["Risk Category"].nunique()) if len(reg) else 0,
    }
