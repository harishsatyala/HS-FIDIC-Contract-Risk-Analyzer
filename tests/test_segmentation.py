from src.text_utils import segment_pages

def test_clause_and_sentence_segmentation():
    pages = [(1, "4.1 General Obligations\nThe Contractor shall execute the Works. The Employer may inspect the Site.")]
    rows = segment_pages(pages)
    assert len(rows) >= 3
    assert any(r["clause"] == "4.1" for r in rows)
    assert any("Contractor shall execute" in r["sentence"] for r in rows)
