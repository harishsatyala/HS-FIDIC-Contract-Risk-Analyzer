from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

from .modeling import predict_sentence_types, predict_related_parties

def load_model(model_path: str | Path):
    return joblib.load(model_path)

def analyze_rows(bundle, rows, review_threshold=0.60):
    texts = [r["sentence"] for r in rows]
    type_pred = predict_sentence_types(bundle, texts)

    result = []
    party_positions = []
    party_texts = []

    for i, (row, pred) in enumerate(zip(rows, type_pred)):
        item = {
            "Page": row.get("page", ""),
            "Clause": row.get("clause", ""),
            "Sentence": row["sentence"],
            "Sentence Type": pred["label"],
            "Type Confidence": pred["confidence"],
            "Related Party": "",
            "Party Confidence": None,
            "Needs Review": pred["confidence"] < review_threshold,
        }
        result.append(item)
        if pred["label"] in {"Obligation", "Risk", "Right"}:
            party_positions.append(i)
            party_texts.append(row["sentence"])

    party_pred = predict_related_parties(bundle, party_texts)
    for pos, pp in zip(party_positions, party_pred):
        result[pos]["Related Party"] = pp["label"]
        result[pos]["Party Confidence"] = pp["confidence"]
        if pp["confidence"] < review_threshold:
            result[pos]["Needs Review"] = True

    return pd.DataFrame(result)
