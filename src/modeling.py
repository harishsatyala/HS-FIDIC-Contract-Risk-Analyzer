from __future__ import annotations
import numpy as np

def _ensemble_proba(models, texts, target_classes):
    classes = list(target_classes)
    acc = np.zeros((len(texts), len(classes)), dtype=float)
    for model in models:
        probs = model.predict_proba(texts)
        cmap = {str(c): i for i, c in enumerate(model.classes_)}
        for j, c in enumerate(classes):
            if c in cmap:
                acc[:, j] += probs[:, cmap[c]]
    acc /= max(len(models), 1)
    return acc, classes

def predict_sentence_types(bundle, texts):
    """Return [{"label": ..., "confidence": ...}, ...] using the thesis-inspired hierarchy."""
    if not texts:
        return []

    stages = bundle["type_stages"]
    p1, c1 = _ensemble_proba(stages["heading_vs_clause"], texts, ["Clause", "Heading"])
    p2, c2 = _ensemble_proba(stages["definition_vs_other"], texts, ["Definition", "Other"])
    p3, c3 = _ensemble_proba(stages["obligation_vs_other"], texts, ["Obligation", "Other"])
    p4, c4 = _ensemble_proba(stages["risk_vs_right"], texts, ["Right", "Risk"])

    i1 = {c: i for i, c in enumerate(c1)}
    i2 = {c: i for i, c in enumerate(c2)}
    i3 = {c: i for i, c in enumerate(c3)}
    i4 = {c: i for i, c in enumerate(c4)}

    results = []
    for i in range(len(texts)):
        p_heading = float(p1[i, i1["Heading"]])
        p_definition = float(p2[i, i2["Definition"]])
        p_obligation = float(p3[i, i3["Obligation"]])
        p_risk = float(p4[i, i4["Risk"]])

        if p_heading >= 0.5:
            label, confidence = "Heading", p_heading
        elif p_definition >= 0.5:
            label, confidence = "Definition", min(1 - p_heading, p_definition)
        elif p_obligation >= 0.5:
            label, confidence = "Obligation", min(1 - p_heading, 1 - p_definition, p_obligation)
        elif p_risk >= 0.5:
            label, confidence = "Risk", min(1 - p_heading, 1 - p_definition, 1 - p_obligation, p_risk)
        else:
            label, confidence = "Right", min(1 - p_heading, 1 - p_definition, 1 - p_obligation, 1 - p_risk)

        results.append({
            "label": label,
            "confidence": round(float(confidence), 4),
            "stage_probabilities": {
                "heading": round(p_heading, 4),
                "definition": round(p_definition, 4),
                "obligation": round(p_obligation, 4),
                "risk_vs_right_risk": round(p_risk, 4),
            },
        })
    return results

def predict_related_parties(bundle, texts):
    if not texts:
        return []
    probs, classes = _ensemble_proba(
        bundle["party_models"],
        texts,
        ["Contractor", "Employer", "Shared"],
    )
    results = []
    for row in probs:
        j = int(np.argmax(row))
        results.append({
            "label": classes[j],
            "confidence": round(float(row[j]), 4),
            "probabilities": {classes[k]: round(float(row[k]), 4) for k in range(len(classes))},
        })
    return results
