"""Text preprocessing and rule-based cue features.

Preprocessing follows the thesis (Eken, 2022, §4.1.1): lower-casing, removal of
digits / punctuation / stop words and extra spaces. Modal and negation words
("shall", "may", "not", "entitled" ...) are *kept* because they carry the
obligation / right / risk meaning of a contract sentence.
"""
from __future__ import annotations

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_STOP = set("""
a an the and or of to in on at by for from with as is are was were be been being this that these those
it its which who whom whose there their them they he his she her we our you your i me my into than then
such any each other same also about above below under over between within without upon after before
during because while where when what how all both either neither whether so very can just only own more most
""".split())
# Domain words that must survive stop-word removal.
_KEEP = {"shall", "may", "not", "no", "must", "will", "entitled", "liable", "responsible",
         "risk", "own", "cost", "costs", "unless", "except", "if", "deemed", "means", "bear"}
_STOP -= _KEEP

_NON_ALPHA = re.compile(r"[^a-z\s]")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    t = str(text).lower().replace("’", "'")
    t = t.replace("'s", " ")
    t = _NON_ALPHA.sub(" ", t)
    toks = [w for w in t.split() if len(w) > 1 and w not in _STOP]
    return _WS.sub(" ", " ".join(toks)).strip()


def clean_many(texts):
    return [clean_text(t) for t in texts]


# ----------------------------------------------------------------------------
# Rule cues (thesis §6.2 recommends integrating a rule-based approach)
# ----------------------------------------------------------------------------
RULE_PATTERNS: dict[str, str] = {
    "shall": r"\bshall\b",
    "shall_not": r"\bshall not\b|\bmust not\b",
    "may": r"\bmay\b",
    "entitled": r"\bentitled\b|\bright to\b|\bat (?:its|his|their) (?:sole )?discretion\b",
    "liable": r"\bliab(?:le|ility)\b|\bindemnif",
    "risk_word": r"\brisks?\b|\bhazard",
    "own_cost": r"\bat (?:the )?(?:contractor|employer)?'?s? ?(?:own )?(?:cost|risk|expense)\b|\bat (?:its|his) own\b",
    "bear": r"\bbear\b|\bborne\b",
    "means": r"\bmeans?\b|\bshall mean\b|\bdefined as\b",
    "quoted_term": r"[\"“][A-Z][^\"”]{1,60}[\"”]",
    "deemed": r"\bdeemed\b",
    "subject_to": r"\bsubject to\b",
    "notice": r"\bnotice\b|\bnotif",
    "damages": r"\bdamages\b|\bpenalt",
    "if_cond": r"^\s*if\b|\bin the event\b|\bprovided that\b",
    "however": r"^\s*however\b|\bnotwithstanding\b",
    "contractor": r"\bcontractor\b",
    "employer": r"\bemployer\b|\bowner\b|\bclient\b",
    "engineer": r"\bengineer\b",
    "both_parties": r"\beach party\b|\bboth parties\b|\beither party\b|\bthe parties\b|\bother party\b",
    "contractor_shall": r"\bcontractor shall\b",
    "employer_shall": r"\b(?:employer|owner) shall\b",
    "contractor_entitled": r"\bcontractor (?:shall be|is) entitled\b",
    "employer_entitled": r"\b(?:employer|owner) (?:shall be|is) entitled\b",
    "clause_number": r"^\s*\d+(?:\.\d+)*\s",
}
_RULES_C = {k: re.compile(v, re.IGNORECASE if k != "quoted_term" else 0) for k, v in RULE_PATTERNS.items()}
RULE_NAMES = list(RULE_PATTERNS) + ["n_words_log", "short_no_period", "title_case_ratio"]


def rule_features(texts) -> np.ndarray:
    rows = []
    for t in texts:
        s = str(t)
        r = [1.0 if rx.search(s) else 0.0 for rx in _RULES_C.values()]
        words = s.split()
        n = len(words)
        r.append(np.log1p(n) / 5.0)
        r.append(1.0 if (n <= 12 and not s.rstrip().endswith((".", ";", ":"))) else 0.0)
        caps = sum(1 for w in words if w[:1].isupper())
        r.append(caps / max(n, 1))
        rows.append(r)
    return np.asarray(rows, dtype=float)


class TextCleaner(BaseEstimator, TransformerMixin):
    """sklearn-compatible thesis preprocessing step (picklable)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return clean_many(X)


class RuleFeaturizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return rule_features(X)
