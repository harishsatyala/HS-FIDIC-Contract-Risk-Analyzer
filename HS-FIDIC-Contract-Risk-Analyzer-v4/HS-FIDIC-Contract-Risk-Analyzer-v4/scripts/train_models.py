"""Train and evaluate every individual and hybrid model for HS-FIDIC, then save the bundles.

Run from project root:
    python scripts/train_models.py [path/to/labelled_dataset.xlsx]
"""
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training import read_dataset, train_all  # noqa: E402

if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "FIDIC_Thesis_Style_Labeled_Dataset.xlsx"
    raw, desc = read_dataset(path.read_bytes(), path.name)
    print("Dataset:", desc)
    type_b, party_b, metrics = train_all(raw, folds=5, boot=600, source=path.name)
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(type_b, out / "hs_fidic_type_models.joblib", compress=("xz", 6))
    joblib.dump(party_b, out / "hs_fidic_party_models.joblib", compress=("xz", 6))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    print("Saved models and metrics to", out)
