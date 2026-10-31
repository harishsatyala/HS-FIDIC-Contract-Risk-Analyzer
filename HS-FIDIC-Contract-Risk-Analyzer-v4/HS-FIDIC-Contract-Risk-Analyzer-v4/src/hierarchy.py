"""Thesis Step-2 hierarchical binary decomposition for Sentence Type (Eken 2022, §5.3, Table 5.8).

Label Group 1: Heading vs Clause
Label Group 2: Definition vs Other
Label Group 3: Obligation vs Other
Label Group 4: Risk vs Right
Each stage is a soft vote of three complementary classifiers.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import TextCleaner

SEED = 42
STAGES = [
    ("Heading vs Clause", "Heading", None),
    ("Definition vs Other", "Definition", {"Heading"}),
    ("Obligation vs Other", "Obligation", {"Heading", "Definition"}),
    ("Risk vs Right", "Risk", {"Heading", "Definition", "Obligation"}),
]
STAGE_MEMBERS = ["TF-IDF + Logistic Regression", "TF-IDF + Linear SVM", "Char n-gram + Linear SVM"]


def _members():
    w = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return [
        Pipeline([("clean", TextCleaner()), ("v", w()), ("c", LogisticRegression(C=8, max_iter=3000, class_weight="balanced"))]),
        Pipeline([("clean", TextCleaner()), ("v", w()),
                  ("c", CalibratedClassifierCV(LinearSVC(C=0.6, class_weight="balanced", random_state=SEED), cv=3))]),
        Pipeline([("v", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=40000)),
                  ("c", CalibratedClassifierCV(LinearSVC(C=0.4, class_weight="balanced", random_state=SEED), cv=3))]),
    ]


class HierarchicalTypeModel:
    def fit(self, X, y):
        X = list(X)
        y = np.asarray(list(y))
        self.stages_ = []
        for name, positive, excluded in STAGES:
            mask = np.ones(len(y), bool) if excluded is None else ~np.isin(y, list(excluded))
            Xs = [X[i] for i in np.where(mask)[0]]
            ys = np.where(y[mask] == positive, 1, 0)
            models = []
            if len(set(ys)) == 2:
                for m in _members():
                    models.append(m.fit(Xs, ys))
            self.stages_.append((name, positive, models, float(ys.mean()) if len(ys) else 0.0))
        return self

    def stage_probabilities(self, X):
        X = list(X)
        out = []
        for name, positive, models, prior in self.stages_:
            if models:
                p = np.mean([m.predict_proba(X)[:, list(m.classes_).index(1)] for m in models], axis=0)
            else:
                p = np.full(len(X), prior)
            out.append(p)
        return np.stack(out, 1)  # n x 4

    def predict_proba_aligned(self, X, labels):
        S = self.stage_probabilities(X)
        h, d, o, r = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
        probs = {
            "Heading": h,
            "Definition": (1 - h) * d,
            "Obligation": (1 - h) * (1 - d) * o,
            "Risk": (1 - h) * (1 - d) * (1 - o) * r,
            "Right": (1 - h) * (1 - d) * (1 - o) * (1 - r),
        }
        P = np.stack([probs[l] for l in labels], 1)
        return P / P.sum(1, keepdims=True)
