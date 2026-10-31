# Optimized hybrid comparison — HS FIDIC

This package contains already-trained nine individual and three weighted hybrid model comparisons. No command prompt is needed to **view** the results: upload the extracted files into the same paths in your GitHub repository and refresh Streamlit.

## Optimization method

`python scripts/optimize_hybrids.py` trains nine classifiers on a stratified 60% training partition, chooses each hybrid’s three nonnegative soft-voting weights on a disjoint 20% validation partition using macro-F1, and reports performance on the untouched 20% test partition. The model definitions and group membership are in the script. Fixed random seed: 42. The supplied workbook is strongly imbalanced. The comparison is not legal validation.

The output artifacts are `models/comparison_metrics.json` and `models/comparison_models.joblib`. Streamlit reads the JSON for the performance table and chart. `comparison_models.joblib` contains the trained comparison classifiers and weights, but the original hierarchical model continues to process uploaded contracts.

## Important research caveats

Do not compare these figures directly to the previous evaluation as a claim of improvement: the new experiment trains on 60% of the data, reserving 20% for validation and 20% for testing, while the old experiment trained on 80%. Even with the same random test partition, training-set sizes differ. This is a row-level split; near-duplicate language and documents may appear across splits. For thesis-grade results, evaluate on document-disjoint, expert-labeled contracts, report per-class metrics and confidence intervals, and measure repeated-run timing on the deployment hardware. Optimization does not guarantee hybrid superiority.

## Rerun (optional)

`python scripts/optimize_hybrids.py`
