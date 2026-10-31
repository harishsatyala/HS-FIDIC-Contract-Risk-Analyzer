# Data

- `FIDIC_Thesis_Style_Labeled_Dataset.xlsx` – sentence-level training data (sheet `Combined_Unique`).
  Labels were generated automatically; the training script removes OCR noise / non-English rows and
  line fragments and applies a documented heading-correction rule (see `models/metrics.json → data_audit`).
- `Global_Energy_Project_Risk_Library_3000.xlsx` – 3,000-record risk library used for probability, owner,
  early-warning indicators and library mitigation.
- `sample_epc_contract.pdf` – synthetic FIDIC-style contract (original wording, not FIDIC text) for the demo toggle.

The workbook contains sentences extracted from licensed FIDIC books. If your repository is public,
consider keeping it private or removing it; the app still runs from the trained models (only the
risk-category ML helper falls back to lexicon-only).
