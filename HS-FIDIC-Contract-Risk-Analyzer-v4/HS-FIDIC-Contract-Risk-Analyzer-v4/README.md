# HS-FIDIC Contract Risk Analyzer (v6)

Streamlit app that reads contract PDFs, classifies every clause with **20 individual ML models and 8 hybrid models**,
turns the predictions into a **risk probability (%)**, a **risk score**, a **mitigation plan** and a
**predictability score**, and presents them in an interactive dashboard.

Research basis: Eken, G. (2022). *Using Natural Language Processing for Automated Construction Contract Review
During Risk Assessment at the Bidding Stage*. PhD thesis, Middle East Technical University. The app keeps the
thesis taxonomy (Heading / Definition / Obligation / Risk / Right × Contractor / Employer / Shared), the thesis
model grid (vectorizer × algorithm, Table 4.5), the Step-2 hierarchical binary decomposition (§5.3), Step-3
competitive voting (§5.6) and the recommended rule-based integration (§6.2).

## What's new in v6 — the labelled dataset as input

- **📚 Labelled dataset panel (sidebar):** shows the active dataset (default `data/FIDIC_Thesis_Style_Labeled_Dataset.xlsx`)
  and accepts an upload (.xlsx or .csv with Sentence, Sentence_Type, Related_Party; the `Combined_Unique` sheet is used when present).
- **🔁 Retrain in the app:** one click rebuilds all 20 individual + 8 hybrid sentence-type models and 20 + 7 party models on
  the uploaded dataset (3-fold CV, ≈3–8 min on Streamlit Cloud, cached), and every tab then uses them.
- **Reference evidence:** every contract clause shows its closest labelled sentence (book, clause, labelled type and party,
  similarity %, agrees/differs) in the Top 20 table, mitigation cards, clause explorer, predictability and the report.
- **📚 Dataset tab:** cleaning waterfall, sentences by book and type, label distributions, searchable clean rows, CSV download.
- Training logic moved to `src/training.py` (shared by the app and `scripts/train_models.py [dataset]`).

## What's new in v5

- **📑 Generate full report**: one self-contained HTML file (interactive charts, works offline) covering every tab —
  executive summary, risk dashboard, mitigation plans, model execution, predictability, model performance (both tasks),
  clause explorer, contract comparison and method — with explanations. Print it to PDF from the browser if needed.
- Cosmetic fixes: chart legends moved away from the Plotly toolbar; heatmap title no longer overlaps column labels;
  KPI sub-labels without arrows.

## What's new in v4

- **Risk dashboard laid out like the earlier Contract Risk Analyzer**: 📊 Risk Analysis Summary → 📈 Risk Distribution
  (category, L/M/H probability, impact, responsible party, rating, probability × impact) → 🎯 Risk-ID Relevance Ranking →
  🚨 Top 20 Contract Risks with "What the risk-analysis numbers mean" → 🛡️ Mitigation Plans for the top risks.
- Optional **one risk per category** mode (earlier-version style) in the sidebar.
- **🏆 Model Performance Dashboard**: every individual and hybrid model compared on Accuracy, Macro Precision, Macro Recall,
  Macro-F1 and ROC-AUC, for both tasks: individual-vs-hybrid summary, model × metric heatmap, ranking by any metric,
  side-by-side grouped bars, accuracy-vs-F1 scatter, full table with CIs, CSV export, confusion matrices.
- Table-of-contents lines (dot leaders) are no longer analysed as contract sentences.

## What's new in v3 (merged with the best of the earlier Contract Risk Analyzer)

- **Quantified risk library**: gross exposure (% of TCV), probability factor (L 0.15 / M 0.40 / H 0.70) and
  mitigation effectiveness (35% default, adjustable) now drive EMV and residual exposure.
- **Dashboard in the familiar order**: summary metrics → risk distribution grid (category, rating, L/M/H probability,
  impact areas, responsible party, probability × impact matrix) with rich hovers listing clauses and risk IDs →
  risk ranking → top-20 risk table with full library details → "What the numbers mean" guide → CSV download.
- **🤖 Model execution tab**: all 20 individual and 8 hybrid models run automatically; status table per model and a
  detail tab per hybrid (base models and weights, dominant type, average decision strength, held-out accuracy,
  distribution and predictions).
- **Evaluation kept separate from predictions**, with metric definitions.
- Model files recompressed (14 MB + 10 MB) so the repo can be uploaded through the GitHub website.

## What the app shows

| Tab | Content |
|---|---|
| 📊 Risk dashboard | Summary metrics and gauges; risk distribution grid with rich hovers; probability × impact matrix; risk ranking; top-20 risk table with library details; numbers guide; CSV |
| 🛡️ Mitigation plan | One card per risk: clause text, risk probability, impact, allocation, predictability, detected triggers, recommended mitigation (trigger-specific + category playbook), FIDIC sub-clause orientation, risk-library match (owner, early-warning indicators, library mitigation), EMV → residual; CSV export |
| 🤖 Model execution | Every individual and hybrid model run on the contract: status, counts, decision strength, agreement; per-hybrid detail tabs |
| 🎯 Predictability | Predictability distribution, model agreement vs confidence, per-clause vote of all 20 models, calibration curve and per-class historical precision of the selected model, clauses flagged for expert review |
| 🏆 Model performance | Held-out metrics with 95% bootstrap CIs for every model, CV scores, confusion matrices, hybrid descriptions and members, data audit, how each model reads the uploaded contract |
| 📄 Clause explorer | Every sentence with all predictions; Excel / CSV / JSON export |
| ⚖️ Compare contracts | Risk Index and category-level comparison across several uploaded contracts |
| ℹ️ Method | Formulas and limitations |

Sidebar: review perspective (Contractor / Employer / Neutral), model selector for each task (any individual or hybrid
model; ★ marks the CV-selected default), risk-probability threshold, review threshold, contract value & currency,
project context (project type, energy segment, region, lifecycle phase) for the risk-library match.

## Models

**Individual (20):** BoW / TF-IDF / LSA-embedding × Logistic Regression, Linear SVM, RBF SVM, Decision Tree;
Multinomial NB, Complement NB, SGD, Ridge, char n-gram SVM, word+char LR, Random Forest, Extra Trees,
Histogram Gradient Boosting (LightGBM-style), k-NN, MLP (DNN). XGBoost and LightGBM are added automatically if
they are installed when you run the training script.

**Hybrid (8):** Majority vote, Competitive Voting (thesis), Confidence-weighted vote, Rank-weighted vote,
Optimized soft voting (Dirichlet-searched weights), Stacking meta-learner (all models), **ML + Rule-based stacking**,
Thesis hierarchical binary decomposition (sentence type only).

### Held-out results (sentence type, 261 test sentences)

| Model | CV macro-F1 | Test accuracy | Test macro-F1 (95% CI) |
|---|---|---|---|
| Hybrid: ML + Rule-Based Stacking | 0.828 | **0.923** | **0.895** (0.84–0.93) |
| **Hybrid: Stacking Meta-Learner (All models) ★ default** | **0.851** | 0.897 | 0.887 (0.83–0.93) |
| Hybrid: Rank-Weighted Vote (Top-3) | 0.829 | 0.874 | 0.878 (0.82–0.92) |
| Hybrid: Competitive Voting (Thesis) | 0.816 | 0.854 | 0.851 (0.79–0.90) |
| Best individual – BoW + Logistic Regression | 0.817 | 0.854 | 0.847 (0.79–0.89) |
| Word+Char TF-IDF + Logistic Regression | 0.817 | 0.877 | 0.847 (0.77–0.90) |

Related party (196 test sentences): default **Optimized Soft Voting** – accuracy 0.974, macro-F1 0.981;
ML + Rule-based stacking 0.979 / 0.985. The full table for all 28 + 27 models is in the Model lab tab and
`models/metrics.json`.

For reference, the thesis reports 0.89 accuracy / 0.86 F1 for sentence type after its process improvements.

### Evaluation protocol (leakage-conscious)

1. Stratified 80/20 DEV/TEST split (seed 42). TEST is used once.
2. 5-fold stratified CV on DEV → out-of-fold probabilities for every model.
3. Model ranking, top-k selection, hybrid weights and stacking meta-learners use **only** DEV out-of-fold probabilities.
4. Models are refit on DEV and scored on TEST; 600-resample bootstrap CIs.
5. The default model is the hybrid with the best **CV** score — never chosen on TEST.

### Data cleaning (why v2 scores differ from v1)

The original workbook (3,345 rows) contained ~1,350 OCR-noise rows (ORANGE/SILVER books, including Indonesian OCR),
584 PDF line fragments and many short titles labelled "Obligation". The training script now:
keeps English rows whose vocabulary is ≥80% valid, removes mid-sentence fragments, relabels strict structural titles as
Heading (254 rows), and adds 100 sub-clause titles harvested from cross-references (training only).
Result: 1,304 clean sentences. Because the test set is cleaner, v2 numbers are **not** directly comparable to v1's.

## Risk scoring

```
P_rule      = 1 − Π(1 − w_t)                over detected onerous-term triggers t (18 trigger types)
Risk Prob L = 1 − (1 − P(Risk)) · (1 − P_rule) · (1 − 0.15·P(Obligation))
Allocation A= P(party = perspective) + 0.5·P(Shared)
Impact I    = category base impact ⊕ risk-library impact dimensions ⊕ trigger severity   (1–5)
Risk Score  = 100 · L · (0.4 + 0.6·A) · (0.6·I/5 + 0.4·library likelihood/0.75)
Rating      = Critical ≥ 50 · High ≥ 30 · Medium ≥ 15 · Low
Predictability = 35% model agreement + 25% confidence + 15% (1 − entropy) + 25% historical class precision
Gross = TCV × library exposure % ; Library EMV = Gross × probability factor ; EMV = Library EMV × L ; residual = EMV × (1 − mitigation effectiveness)
```

Risk category (30 risk-library categories) = FIDIC-language lexicon on the sentence + its clause heading, blended with a
weakly-supervised TF-IDF classifier. The 3,000-record library is templated (one risk driver per category), so it is used
to enrich risks (probability level, impact dimensions, owner, early-warning indicators, mitigation), not for text matching.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install pytest && python -m pytest -q`

## Deploy on Streamlit Community Cloud (no command line needed)

1. On github.com create a repository (or open your existing app's repository).
2. **Add file → Upload files**, drag in *the contents* of this folder (all files are under 25 MB), commit.
3. On share.streamlit.io → **Create app** → choose the repo, branch `main`, main file `app.py`,
   Advanced settings → Python **3.12** → Deploy. (If you updated an existing app's repo, it redeploys itself.)

`scikit-learn` is pinned to **1.9.1** because the models are pickled with that version. If you change it,
re-train: `python scripts/train_models.py` (≈6 min on 2 CPUs).

## Re-train / extend

```bash
python scripts/train_models.py          # writes models/hs_fidic_type_models.joblib, hs_fidic_party_models.joblib, metrics.json
python scripts/make_sample_contract.py  # regenerates data/sample_epc_contract.pdf
```

Edit `src/model_zoo.py` to add algorithms, `src/playbook.py` to tune the lexicon, triggers, exposure factors and
mitigation playbook.

## Project structure

```
app.py                     Streamlit dashboard
src/text_utils.py          PDF/DOCX/TXT extraction, line re-joining, sentence + heading segmentation
src/features.py            thesis preprocessing, 28 rule-cue features
src/model_zoo.py           20 individual models + hybrid combiners
src/hierarchy.py           thesis Step-2 hierarchical binary decomposition
src/predictor.py           runs any individual/hybrid model on new text
src/risk_engine.py         categories, triggers, library match, scores, predictability, EMV
src/playbook.py            30-category lexicon, triggers, FIDIC orientation, mitigation playbook
src/charts.py              Plotly figures
scripts/train_models.py    training + evaluation
scripts/make_sample_contract.py
models/                    trained bundles + metrics.json
data/                      training workbook, risk library, sample contract
tests/test_pipeline.py
```

## Limitations

- Training labels were generated automatically, not by FIDIC experts; scores measure agreement with those labels.
  RED and YELLOW books share near-identical clauses, which can inflate scores. For thesis-grade claims, evaluate on an
  expert-labelled, document-disjoint contract (as the thesis did with its Test Dataset).
- Risk categories are weakly supervised; the scoring weights are transparent expert heuristics, not fitted to loss data.
- EMV figures are screening estimates from category factors, not amounts read from the contract.
- Scanned PDFs need OCR before upload.
- Decision support for bid-stage review; not legal advice.
