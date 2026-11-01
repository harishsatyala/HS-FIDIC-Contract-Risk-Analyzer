"""Labelled-dataset reference matching: the closest labelled FIDIC sentence for every contract clause.

Gives each ML prediction human-checkable evidence from the labelled dataset (book, clause, its Sentence_Type and
Related_Party, cosine similarity). It does not change the ML prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .features import clean_many

REF_COLS = ["Reference Sentence", "Reference Book", "Reference Clause", "Reference Type", "Reference Party",
            "Similarity %", "Dataset Agreement"]


class ReferenceIndex:
    def __init__(self, clean_df: pd.DataFrame):
        # headings are not useful evidence for a clause: index only Definition / Obligation / Risk / Right rows
        keep = (clean_df["Sentence_Type"] != "Heading") & (clean_df["Sentence"].astype(str).str.split().str.len() >= 5)
        self.df = clean_df[keep].reset_index(drop=True)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.M = self.vec.fit_transform(clean_many(self.df["Sentence"]))

    def match(self, texts, predicted_types=None, min_sim=0.3) -> pd.DataFrame:
        if not len(texts):
            return pd.DataFrame(columns=REF_COLS)
        Q = self.vec.transform(clean_many(texts))
        out = []
        step = 500
        for a in range(0, Q.shape[0], step):
            S = (Q[a:a + step] @ self.M.T).toarray()
            idx, sim = S.argmax(1), S.max(1)
            for j, (i, s) in enumerate(zip(idx, sim)):
                r = self.df.iloc[int(i)]
                out.append({"Reference Sentence": r["Sentence"], "Reference Book": r.get("Book", ""),
                            "Reference Clause": "" if pd.isna(r.get("Clause", "")) else str(r.get("Clause", "")),
                            "Reference Type": r["Sentence_Type"],
                            "Reference Party": "" if pd.isna(r.get("Related_Party")) else r.get("Related_Party"),
                            "Similarity %": round(100 * float(s), 1)})
        res = pd.DataFrame(out)
        if predicted_types is not None:
            pt = list(predicted_types)
            res["Dataset Agreement"] = [("Agrees" if t == ref else "Differs") if sim >= 100 * min_sim else "No close match"
                                        for t, ref, sim in zip(pt, res["Reference Type"], res["Similarity %"])]
        else:
            res["Dataset Agreement"] = ""
        return res
