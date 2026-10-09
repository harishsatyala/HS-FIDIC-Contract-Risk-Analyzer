# HS FIDIC Contract Risk Analyzer

Complete updated version based on the uploaded original repository.

## Run
`pip install -r requirements.txt`
`streamlit run app.py`

## Features
Multiple contract uploads, per-contract dashboards and downloads, combined summary, ZIP export, review flags.

## Important research limitations
The original project uses hierarchical two-model ensembles (word TF-IDF Logistic Regression and character TF-IDF SGD), **not** the separate nine-model/three-hybrid system shown in another application. The original repository does not include either requested Excel workbook. The bundled pretrained model uses automatically generated thesis-style labels; do not present its validation metrics as verified generalization to unseen contracts. To implement the full nine-model benchmark and risk-library retrieval, the corresponding source code and both Excel datasets are still needed.

## GitHub
Upload the extracted folder CONTENTS (not the zip) to your HS repository, preserving paths. Deploy main/app.py on Streamlit Cloud. Never commit secrets.
