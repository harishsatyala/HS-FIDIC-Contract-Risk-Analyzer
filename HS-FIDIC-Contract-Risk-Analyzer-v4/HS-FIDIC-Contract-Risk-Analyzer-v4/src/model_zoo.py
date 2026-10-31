"""Individual models (vectorizer x algorithm, as in thesis Table 4.5) and hybrid combiners."""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from .features import TextCleaner

SEED = 42


# ----------------------------------------------------------------------------- vectorizers
def _bow():
    return CountVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)


def _tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)


def _char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=40000)


def _lsa(dim=120):
    """Dense semantic embedding (Latent Semantic Analysis on TF-IDF)."""
    return [("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=6000)), ("svd", TruncatedSVD(n_components=dim, random_state=SEED)), ("norm", Normalizer())]


def _word_char():
    return FeatureUnion([("w", _tfidf()), ("c", _char())])


def _pipe(vec, clf, clean=True):
    steps = [("clean", TextCleaner())] if clean else []
    if isinstance(vec, list):
        steps += vec
    else:
        steps.append(("vec", vec))
    steps.append(("clf", clf))
    return Pipeline(steps)


# ----------------------------------------------------------------------------- catalogue
def individual_models(include_optional: bool = True) -> dict[str, dict]:
    """Return {name: {"factory": callable, "vectorizer": str, "algorithm": str, "family": str}}."""
    M = {}

    def add(name, vec_name, algo, family, factory):
        M[name] = {"factory": factory, "vectorizer": vec_name, "algorithm": algo, "family": family}

    # Thesis models 1-9 (BoW / TF-IDF / embedding  x  LR / SVM / DT)
    add("BoW + Logistic Regression", "Bag of Words", "Logistic Regression", "Linear",
        lambda: _pipe(_bow(), LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")))
    add("BoW + Linear SVM", "Bag of Words", "Support Vector Machine", "Margin",
        lambda: _pipe(_bow(), CalibratedClassifierCV(LinearSVC(C=0.3, class_weight="balanced", random_state=SEED), cv=3)))
    add("BoW + Decision Tree", "Bag of Words", "Decision Tree", "Tree",
        lambda: _pipe(_bow(), DecisionTreeClassifier(max_depth=40, min_samples_leaf=2, class_weight="balanced", random_state=SEED)))
    add("TF-IDF + Logistic Regression", "TF-IDF", "Logistic Regression", "Linear",
        lambda: _pipe(_tfidf(), LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")))
    add("TF-IDF + Linear SVM", "TF-IDF", "Support Vector Machine", "Margin",
        lambda: _pipe(_tfidf(), CalibratedClassifierCV(LinearSVC(C=0.6, class_weight="balanced", random_state=SEED), cv=3)))
    add("TF-IDF + Decision Tree", "TF-IDF", "Decision Tree", "Tree",
        lambda: _pipe(_tfidf(), DecisionTreeClassifier(max_depth=40, min_samples_leaf=2, class_weight="balanced", random_state=SEED)))
    add("LSA Embedding + Logistic Regression", "LSA semantic embedding", "Logistic Regression", "Linear",
        lambda: _pipe(_lsa(), LogisticRegression(C=10.0, max_iter=3000, class_weight="balanced")))
    add("LSA Embedding + RBF SVM", "LSA semantic embedding", "Support Vector Machine (RBF)", "Margin",
        lambda: _pipe(_lsa(), CalibratedClassifierCV(SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced"), cv=3, ensemble=False)))
    add("LSA Embedding + Decision Tree", "LSA semantic embedding", "Decision Tree", "Tree",
        lambda: _pipe(_lsa(), DecisionTreeClassifier(max_depth=20, min_samples_leaf=3, class_weight="balanced", random_state=SEED)))

    # Additional algorithms (prototype documentation: NB, RF, boosting, MLP/DNN ...)
    add("BoW + Multinomial Naive Bayes", "Bag of Words", "Multinomial Naive Bayes", "Probabilistic",
        lambda: _pipe(_bow(), MultinomialNB(alpha=0.3)))
    add("TF-IDF + Complement Naive Bayes", "TF-IDF", "Complement Naive Bayes", "Probabilistic",
        lambda: _pipe(_tfidf(), ComplementNB(alpha=0.3)))
    add("TF-IDF + SGD (log-loss)", "TF-IDF", "Stochastic Gradient Descent", "Linear",
        lambda: _pipe(_tfidf(), SGDClassifier(loss="log_loss", alpha=5e-5, class_weight="balanced", max_iter=50, random_state=SEED)))
    add("TF-IDF + Ridge Classifier", "TF-IDF", "Ridge Classifier", "Linear",
        lambda: _pipe(_tfidf(), CalibratedClassifierCV(RidgeClassifier(alpha=1.0, class_weight="balanced"), cv=3)))
    add("Char n-gram + Linear SVM", "Character n-gram TF-IDF", "Support Vector Machine", "Margin",
        lambda: _pipe(_char(), CalibratedClassifierCV(LinearSVC(C=0.4, class_weight="balanced", random_state=SEED), cv=3), clean=False))
    add("Word+Char TF-IDF + Logistic Regression", "Word + character TF-IDF", "Logistic Regression", "Linear",
        lambda: _pipe(_word_char(), LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced"), clean=False))
    add("TF-IDF + Random Forest", "TF-IDF", "Random Forest", "Ensemble tree",
        lambda: _pipe(_tfidf(), RandomForestClassifier(n_estimators=120, min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)))
    add("TF-IDF + Extra Trees", "TF-IDF", "Extra Trees", "Ensemble tree",
        lambda: _pipe(_tfidf(), ExtraTreesClassifier(n_estimators=120, min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)))
    add("LSA Embedding + Gradient Boosting (Hist)", "LSA semantic embedding", "Histogram Gradient Boosting", "Boosting",
        lambda: _pipe(_lsa(), HistGradientBoostingClassifier(max_iter=160, learning_rate=0.1, max_leaf_nodes=15, class_weight="balanced", random_state=SEED)))
    add("LSA Embedding + KNN", "LSA semantic embedding", "k-Nearest Neighbours", "Instance-based",
        lambda: _pipe(_lsa(), KNeighborsClassifier(n_neighbors=7, weights="distance", metric="cosine")))
    add("LSA Embedding + MLP (DNN)", "LSA semantic embedding", "Multilayer Perceptron (DNN)", "Neural network",
        lambda: _pipe(_lsa(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300, random_state=SEED)))

    if include_optional:
        try:
            from xgboost import XGBClassifier  # noqa: F401

            add("LSA Embedding + XGBoost", "LSA semantic embedding", "XGBoost", "Boosting",
                lambda: _pipe(_lsa(), _LabelEncodedXGB()))
        except Exception:
            pass
        try:
            from lightgbm import LGBMClassifier

            add("LSA Embedding + LightGBM", "LSA semantic embedding", "LightGBM", "Boosting",
                lambda: _pipe(_lsa(), LGBMClassifier(n_estimators=200, learning_rate=0.08, num_leaves=15,
                                                    class_weight="balanced", random_state=SEED, verbose=-1)))
        except Exception:
            pass
    return M


class _LabelEncodedXGB:
    """XGBoost wrapper that accepts string labels (only used when xgboost is installed)."""

    def __init__(self):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(n_estimators=220, max_depth=5, learning_rate=0.1, subsample=0.9,
                                   colsample_bytree=0.8, random_state=SEED, n_jobs=2, eval_metric="mlogloss")

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        idx = {c: i for i, c in enumerate(self.classes_)}
        self.model.fit(X, np.array([idx[v] for v in y]))
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def get_params(self, deep=True):
        return {}

    def set_params(self, **p):
        return self


# ----------------------------------------------------------------------------- hybrids
HYBRID_INFO = {
    "Hybrid: Majority Vote (Top-3)": "Each of the three best individual models (by cross-validated macro-F1) casts one vote; ties broken by summed probability.",
    "Hybrid: Competitive Voting (Thesis, Top-3)": "Eken (2022) §5.6: if ≥2 of the top-3 models agree, take the majority; if all three differ, take the prediction of the most accurate model.",
    "Hybrid: Confidence-Weighted Vote (Top-3)": "Each top-3 model adds its confidence to the class it predicts; the class with the largest confidence mass wins.",
    "Hybrid: Rank-Weighted Vote (Top-3)": "Top-3 models receive weights 3, 2, 1 by cross-validated rank; weighted soft vote of their probabilities.",
    "Hybrid: Optimized Soft Voting (Top-5)": "Non-negative weights for the five best models chosen by Dirichlet search to maximise out-of-fold macro-F1.",
    "Hybrid: Stacking Meta-Learner (All models)": "A multinomial logistic-regression meta-learner trained on the out-of-fold probabilities of every individual model.",
    "Hybrid: ML + Rule-Based Stacking": "Stacking meta-learner fed with the top-5 models' probabilities plus 28 linguistic rule cues (modal verbs, liability, discretion, party mentions …) — the rule integration recommended in thesis §6.2.",
    "Hybrid: Thesis Hierarchical Binary Decomposition": "Eken (2022) §5.3 Step 2: Heading vs Clause → Definition vs Other → Obligation vs Other → Risk vs Right, each stage a soft vote of TF-IDF LR, TF-IDF SVM and char-SVM.",
}


def _onehot(idx, k):
    out = np.zeros((len(idx), k))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def hybrid_majority(P: list[np.ndarray]) -> np.ndarray:
    k = P[0].shape[1]
    votes = sum(_onehot(p.argmax(1), k) for p in P)
    tie_break = sum(P) * 1e-3
    s = votes + tie_break
    return s / s.sum(1, keepdims=True)


def hybrid_competitive(P: list[np.ndarray]) -> np.ndarray:
    """P ordered best-first. Returns pseudo-probabilities whose argmax implements competitive voting."""
    preds = np.stack([p.argmax(1) for p in P], 1)
    k = P[0].shape[1]
    avg = sum(P) / len(P)
    out = np.zeros_like(avg)
    for i, row in enumerate(preds):
        vals, cnt = np.unique(row, return_counts=True)
        if cnt.max() >= 2:
            win = vals[cnt.argmax()]
        else:
            win = row[0]  # most accurate model
        out[i] = avg[i] * 0.5
        out[i, win] = max(out[i].max(), avg[i, win]) + 0.5
    return out / out.sum(1, keepdims=True)


def hybrid_confidence(P: list[np.ndarray]) -> np.ndarray:
    k = P[0].shape[1]
    s = sum(_onehot(p.argmax(1), k) * p.max(1, keepdims=True) for p in P)
    s = s + 1e-6
    return s / s.sum(1, keepdims=True)


def hybrid_weighted(P: list[np.ndarray], w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    s = sum(wi * p for wi, p in zip(w, P))
    return s / s.sum(1, keepdims=True)


def tune_weights(P: list[np.ndarray], y_idx: np.ndarray, n=400, seed=SEED):
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    m = len(P)
    cands = [np.ones(m) / m] + list(np.eye(m)) + list(rng.dirichlet(np.ones(m), size=n))
    best_w, best_f = None, -1
    for w in cands:
        f = f1_score(y_idx, hybrid_weighted(P, w).argmax(1), average="macro")
        if f > best_f + 1e-12:
            best_f, best_w = f, w
    return best_w, best_f
