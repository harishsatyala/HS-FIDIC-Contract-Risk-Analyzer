"""Run any individual or hybrid model on new contract sentences."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .features import rule_features
from .model_zoo import hybrid_competitive, hybrid_confidence, hybrid_majority, hybrid_weighted


def _aligned(model, X, labels):
    p = model.predict_proba(X)
    cls = [str(c) for c in model.classes_]
    out = np.zeros((len(X), len(labels)))
    for j, lab in enumerate(labels):
        if lab in cls:
            out[:, j] = p[:, cls.index(lab)]
    s = out.sum(1, keepdims=True)
    s[s == 0] = 1
    return out / s


class TaskModels:
    """Holds every trained model for one task (sentence type or related party)."""

    def __init__(self, spec: dict):
        self.labels = spec["labels"]
        self.individual = spec["individual"]
        self.hybrids = spec["hybrids"]
        self.ranking = spec["ranking"]
        self.default_model = spec["default_model"]
        self.best_individual = spec["best_individual"]
        self.class_precision = spec.get("class_precision", {})
        self.model_accuracy = spec.get("model_accuracy", {})

    @property
    def model_names(self):
        return list(self.hybrids) + list(self.individual)

    def all_individual_proba(self, texts):
        return {n: _aligned(m, texts, self.labels) for n, m in self.individual.items()}

    def proba(self, name, texts, cache=None):
        """Return (selected-model probabilities, dict of every individual model's probabilities)."""
        texts = list(texts)
        P = cache if cache is not None else self.all_individual_proba(texts)
        if name in self.individual:
            return P[name], P
        d = self.hybrids[name]
        kind = d["kind"]
        members = [P[m] for m in d["members"]] if kind != "hierarchical" else None
        if kind == "majority":
            out = hybrid_majority(members)
        elif kind == "competitive":
            out = hybrid_competitive(members)
        elif kind == "confidence":
            out = hybrid_confidence(members)
        elif kind == "weighted":
            out = hybrid_weighted(members, d["weights"])
        elif kind == "stacking":
            out = d["meta"].predict_proba(np.hstack(members))
        elif kind == "rule_stacking":
            out = d["meta"].predict_proba(np.hstack([np.hstack(members), d["scaler"].transform(rule_features(texts))]))
        elif kind == "hierarchical":
            out = d["model"].predict_proba_aligned(texts, self.labels)
        else:
            raise ValueError(kind)
        return out, P


class ModelSuite:
    def __init__(self, models_dir: str | Path):
        models_dir = Path(models_dir)
        t = joblib.load(models_dir / "hs_fidic_type_models.joblib")
        p = joblib.load(models_dir / "hs_fidic_party_models.joblib")
        self.type = TaskModels(t["type"])
        self.party = TaskModels(p["party"])

    def predict(self, texts, type_model: str, party_model: str):
        texts = list(texts)
        if not texts:
            return None
        Pt, Pt_all = self.type.proba(type_model, texts)
        Pp, Pp_all = self.party.proba(party_model, texts)
        return {"type_P": Pt, "type_all": Pt_all, "party_P": Pp, "party_all": Pp_all}
