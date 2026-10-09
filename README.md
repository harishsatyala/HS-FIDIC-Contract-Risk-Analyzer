# FIDIC Thesis-Aligned Contract Risk & Responsibility Analyzer

A Streamlit application built around the contract-classification framework described in the Dikmen / Eken research on automated construction contract review.

## What is different from a generic High / Medium / Low risk classifier?

This project does **not** reduce each clause to a generic risk severity. It uses the research taxonomy:

- **Heading**
- **Definition**
- **Obligation**
- **Risk**
- **Right**

For **Obligation, Risk and Right** sentences, it also predicts the related party:

- **Contractor**
- **Employer**
- **Shared**

The dashboard therefore acts as a FIDIC-oriented **risk and responsibility allocation analyzer**.

## Research-aligned architecture

The published work experimented with multiple NLP vectorizations and ML algorithms, BERT, binary decomposition and competitive voting. This GitHub-ready implementation keeps the same taxonomy and the thesis-inspired binary decomposition while using lightweight models that are practical on Streamlit Community Cloud:

1. Heading vs Clause
2. Definition vs Other
3. Obligation vs Other
4. Risk vs Right

Each binary stage uses an ensemble of:

- word-level TF-IDF + Logistic Regression
- character-level TF-IDF + log-loss SGD classifier

A separate ensemble predicts **Contractor / Employer / Shared**.

> This is a **research-aligned implementation**, not an exact reproduction of all 12 models or the BERT configuration in the publication.

## Bundled model

The included model was trained on **3,345 unique rows** from the locally created `FIDIC_Thesis_Style_Labeled_Dataset.xlsx`.

Internal validation on the automatically generated labels:

- Sentence-type accuracy: **92.7%**
- Sentence-type macro F1: **67.6%**
- Related-party accuracy: **96.6%**
- Related-party macro F1: **92.9%**

These scores are **not independent legal validation** because the workbook labels were automatically generated rather than manually validated by the thesis authors.

## Important dataset / licensing note

The full extracted FIDIC sentence dataset is intentionally **not included in this repository template**. FIDIC contract publications are licensed/copyrighted works. Keep your legally obtained workbook locally under:

```text
data/FIDIC_Thesis_Style_Labeled_Dataset.xlsx
```

The `.gitignore` prevents that workbook from being pushed accidentally.

The bundled `joblib` model allows the Streamlit app to run without publishing the source contract text.

## Run locally

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Re-train from your workbook

Place your workbook in `data/` and run:

```bash
python scripts/train_models.py --dataset data/FIDIC_Thesis_Style_Labeled_Dataset.xlsx
```

Expected columns in the `Combined_Unique` sheet:

```text
Sentence
Sentence_Type
Related_Party
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, create a new app.
3. Select the repository.
4. Set the main file path to `app.py`.
5. Deploy.

No API keys are required.

## Input formats

- PDF
- DOCX
- TXT

For image-only/scanned PDFs, run OCR before upload. The app detects unusually low extracted text and shows a warning.

## Outputs

- sentence-level classification
- risk register
- obligation register
- rights register
- Contractor / Employer / Shared allocation
- manual-review flags using a confidence threshold
- CSV and JSON export

## Project structure

```text
fidic-thesis-risk-analyzer/
├── app.py
├── requirements.txt
├── models/
│   ├── fidic_thesis_model.joblib
│   └── metrics.json
├── src/
│   ├── analyzer.py
│   ├── modeling.py
│   └── text_utils.py
├── scripts/
│   └── train_models.py
├── tests/
│   └── test_segmentation.py
├── data/
│   └── README.md
└── .streamlit/
    └── config.toml
```

## Limitations

- The supplied training labels are machine-generated and should be expert-reviewed.
- The app does not claim to reproduce the thesis authors' unpublished labeled dataset.
- A sentence-level classifier can miss cross-clause context and dependencies.
- "Risk" means the thesis taxonomy's contractual risk category; it does not mean a quantified High/Medium/Low severity.
- This application is decision support and research software, not legal advice.

## Suggested next research step

Replace the auto-generated labels with a manually reviewed training set and evaluate on an **external project contract**, matching the research design more closely.
