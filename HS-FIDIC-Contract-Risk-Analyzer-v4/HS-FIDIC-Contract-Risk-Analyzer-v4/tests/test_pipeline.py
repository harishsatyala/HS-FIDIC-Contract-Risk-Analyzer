"""Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.text_utils import is_fragment, is_heading_like, segment_pages, extract_uploaded_file  # noqa: E402
from src.model_zoo import hybrid_competitive, hybrid_majority  # noqa: E402
from src import risk_engine as engine  # noqa: E402


def test_heading_rules():
    assert is_heading_like("8.8 Delay Damages")
    assert is_heading_like("8 Commencement, Delays and Suspension")
    assert not is_heading_like("The Contractor shall pay delay damages.")
    assert is_fragment("expressions shall have the meanings stated and the")
    assert not is_fragment("The Engineer shall respond within 28 days.")


def test_segmentation_rejoins_lines_and_tracks_clause():
    text = "4.12 Unforeseeable Physical Conditions\nThe Contractor shall bear all risks of\nunforeseeable ground conditions at its own cost.\nThe Employer shall pay."
    rows = segment_pages([(1, text)])
    sents = [r["sentence"] for r in rows]
    assert sents[0] == "4.12 Unforeseeable Physical Conditions"
    assert "The Contractor shall bear all risks of unforeseeable ground conditions at its own cost." in sents
    assert all(r["clause"] == "4.12" for r in rows)
    assert rows[1]["heading"].endswith("Unforeseeable Physical Conditions")


def test_competitive_voting_logic():
    a = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1]])
    b = np.array([[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
    c = np.array([[0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
    out = hybrid_competitive([a, b, c]).argmax(1)
    assert out[0] == 1  # two of three agree
    assert out[1] == 1  # all differ -> best (first) model
    assert hybrid_majority([a, b, c]).argmax(1)[0] == 1


def test_triggers_and_rating():
    hits, p = engine.match_triggers("Liquidated damages shall not be limited and are payable at its own cost.")
    labels = {h[0] for h in hits}
    assert "Delay damages / penalties" in labels and "Uncapped liability" in labels
    assert 0.5 < p <= 0.9
    assert engine.rating(55) == "Critical" and engine.rating(10) == "Low"


@pytest.mark.skipif(not (ROOT / "models" / "hs_fidic_type_models.joblib").exists(), reason="models not trained")
def test_end_to_end_sample_contract():
    from src.predictor import ModelSuite

    pages, info = extract_uploaded_file("s.pdf", (ROOT / "data" / "sample_epc_contract.pdf").read_bytes())
    rows = segment_pages(pages)
    suite = ModelSuite(ROOT / "models")
    preds = suite.predict([r["sentence"] for r in rows], suite.type.default_model, suite.party.default_model)
    lib = engine.load_library(ROOT / "data" / "Global_Energy_Project_Risk_Library_3000.xlsx")
    df = engine.analyse(rows, preds, suite, lib, engine.CategoryModel(None),
                        type_model=suite.type.default_model, party_model=suite.party.default_model)
    reg = engine.risk_register(df)
    assert len(reg) >= 10
    top = reg.iloc[0]
    assert top["Rating"] == "Critical" and top["Risk Probability %"] > 80
    assert df["Predictability %"].between(0, 100).all()
    # every model in the suite produces valid probabilities
    for name in suite.type.model_names:
        P, _ = suite.type.proba(name, [r["sentence"] for r in rows[:8]])
        assert np.allclose(P.sum(1), 1, atol=1e-6)


def test_read_dataset_accepts_workbook_and_csv_aliases():
    import pandas as pd
    from src.training import read_dataset, load_clean_dataset

    raw, desc = read_dataset((ROOT / "data" / "FIDIC_Thesis_Style_Labeled_Dataset.xlsx").read_bytes(), "x.xlsx")
    assert "Combined_Unique" in desc and len(raw) == 3345
    clean, audit = load_clean_dataset(raw)
    assert audit["kept_rows"] > 1000 and set(clean["Sentence_Type"]) == {"Definition", "Heading", "Obligation", "Right", "Risk"}
    csv = pd.DataFrame({"sentence": ["The Contractor shall pay delay damages to the Employer."],
                        "Sentence Type": ["risk"], "Party": ["contractor"]}).to_csv(index=False).encode()
    d, _ = read_dataset(csv, "mine.csv")
    assert list(d.loc[0, ["Sentence_Type", "Related_Party"]]) == ["Risk", "Contractor"]


def test_reference_matching_returns_labelled_evidence():
    from src.training import read_dataset, load_clean_dataset
    from src.reference import ReferenceIndex

    raw, _ = read_dataset((ROOT / "data" / "FIDIC_Thesis_Style_Labeled_Dataset.xlsx").read_bytes(), "x.xlsx")
    clean, _ = load_clean_dataset(raw)
    r = ReferenceIndex(clean).match(["If the Contractor fails to comply with the Time for Completion, the Contractor "
                                     "shall pay delay damages to the Employer."], ["Risk"])
    assert r.loc[0, "Reference Type"] in {"Risk", "Obligation", "Right", "Definition"}
    assert r.loc[0, "Similarity %"] > 30 and r.loc[0, "Dataset Agreement"] in {"Agrees", "Differs"}
