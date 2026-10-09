from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

RANDOM_STATE = 42

def components():
    return [
        Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode",
                ngram_range=(1, 2), sublinear_tf=True, max_features=30000
            )),
            ("clf", LogisticRegression(
                max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
            )),
        ]),
        Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode", analyzer="char_wb",
                ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=45000
            )),
            ("clf", SGDClassifier(
                loss="log_loss", class_weight="balanced",
                max_iter=3000, tol=1e-4, random_state=RANDOM_STATE
            )),
        ]),
    ]

def fit_ensemble(texts, labels):
    models = []
    for base in components():
        model = clone(base)
        model.fit(texts, labels)
        models.append(model)
    return models

def train_bundle(df):
    rows = df.to_dict("records")
    texts = df["Sentence"].astype(str).tolist()
    types = df["Sentence_Type"].astype(str).tolist()

    stage1 = fit_ensemble(texts, ["Heading" if y == "Heading" else "Clause" for y in types])

    s2 = df[df["Sentence_Type"] != "Heading"]
    stage2 = fit_ensemble(
        s2["Sentence"].astype(str).tolist(),
        ["Definition" if y == "Definition" else "Other" for y in s2["Sentence_Type"]],
    )

    s3 = df[df["Sentence_Type"].isin(["Obligation", "Risk", "Right"])]
    stage3 = fit_ensemble(
        s3["Sentence"].astype(str).tolist(),
        ["Obligation" if y == "Obligation" else "Other" for y in s3["Sentence_Type"]],
    )

    s4 = df[df["Sentence_Type"].isin(["Risk", "Right"])]
    stage4 = fit_ensemble(
        s4["Sentence"].astype(str).tolist(),
        s4["Sentence_Type"].astype(str).tolist(),
    )

    p = df[
        df["Sentence_Type"].isin(["Obligation", "Risk", "Right"])
        & df["Related_Party"].isin(["Contractor", "Employer", "Shared"])
    ]
    party = fit_ensemble(
        p["Sentence"].astype(str).tolist(),
        p["Related_Party"].astype(str).tolist(),
    )

    return {
        "version": "1.0.0",
        "taxonomy": {
            "sentence_types": ["Heading", "Definition", "Obligation", "Risk", "Right"],
            "related_parties": ["Contractor", "Employer", "Shared"],
        },
        "type_stages": {
            "heading_vs_clause": stage1,
            "definition_vs_other": stage2,
            "obligation_vs_other": stage3,
            "risk_vs_right": stage4,
        },
        "party_models": party,
        "training_metadata": {
            "n_rows": int(len(df)),
            "source": "user-supplied thesis-style labeled workbook",
            "label_note": "Check whether labels have been expert-validated before treating metrics as ground truth.",
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/FIDIC_Thesis_Style_Labeled_Dataset.xlsx")
    ap.add_argument("--sheet", default="Combined_Unique")
    ap.add_argument("--out", default="models/fidic_thesis_model.joblib")
    args = ap.parse_args()

    df = pd.read_excel(args.dataset, sheet_name=args.sheet)
    required = {"Sentence", "Sentence_Type", "Related_Party"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["Sentence", "Sentence_Type"]).copy()
    df["Sentence"] = df["Sentence"].astype(str).str.strip()
    df["Sentence_Type"] = df["Sentence_Type"].astype(str).str.strip()
    df["Related_Party"] = df["Related_Party"].fillna("").astype(str).str.strip()

    valid_types = {"Heading", "Definition", "Obligation", "Risk", "Right"}
    df = df[df["Sentence_Type"].isin(valid_types)].copy()

    model = train_bundle(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out, compress=3)
    print(f"Saved {args.out} using {len(df)} rows.")

if __name__ == "__main__":
    main()
