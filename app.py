from __future__ import annotations
import io
import json
import zipfile
from pathlib import Path
import pandas as pd
import streamlit as st
from src.analyzer import load_model, analyze_rows
from src.text_utils import extract_uploaded_file, segment_pages

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="HS FIDIC Contract Risk Analyzer", page_icon="📑", layout="wide")

@st.cache_resource
def get_model():
    return load_model(ROOT / "models" / "fidic_thesis_model.joblib")

@st.cache_data
def get_metrics():
    p = ROOT / "models" / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

@st.cache_data(show_spinner=False, max_entries=20)
def analyze_file(data: bytes, name: str, threshold: float):
    uploaded = io.BytesIO(data)
    uploaded.name = name
    pages, info = extract_uploaded_file(uploaded)
    rows = segment_pages(pages)
    if not rows:
        return pd.DataFrame(), info
    return analyze_rows(get_model(), rows, review_threshold=threshold), info

st.title("HS FIDIC Contract Risk Analyzer")
st.caption("Batch contract screening: sentence types and responsible-party allocation")
st.warning("The supplied repository uses a hierarchical ensemble of two text classifiers per stage, not nine standalone models and three voting hybrids. Its bundled validation scores are not evidence of accuracy on newly uploaded contracts. The included Excel workbooks contain research data; evaluation scores require careful interpretation.")
with st.expander("ML model comparison: 9 individual + 3 hybrid", expanded=True):
    comparison_path = ROOT / "models" / "comparison_metrics.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        st.caption(comparison.get("evaluation_note", "Workbook-label evaluation only."))
        scores = pd.DataFrame(comparison.get("results", []))
        if not scores.empty:
            st.dataframe(scores.sort_values("macro_f1", ascending=False), use_container_width=True, hide_index=True)
            st.bar_chart(scores.set_index("model")[["accuracy", "macro_f1"]])
            best = scores.sort_values("macro_f1", ascending=False).iloc[0]
            st.info(f"Highest held-out macro-F1: {best['model']} ({best['macro_f1']:.3f}). This does not prove superiority on new contracts.")
    else:
        st.info("Comparison models have not been trained yet. Run `python scripts/compare_models.py` locally, then upload `models/comparison_metrics.json` to display measured results. For model selection, also upload `models/comparison_models.joblib`.")

threshold = st.sidebar.slider("Manual-review confidence threshold", 0.40, 0.90, 0.60, 0.05)
files = st.file_uploader("Upload multiple contract PDFs (DOCX and TXT also supported)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if not files:
    st.info("Upload one or more contracts to begin. Text-based PDFs are supported; scanned PDFs need OCR.")
    st.stop()

all_results = []
archive = io.BytesIO()
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for idx, f in enumerate(files):
        st.divider()
        st.subheader(f"Contract {idx + 1}: {f.name}")
        try:
            with st.spinner(f"Analyzing {f.name}..."):
                df, info = analyze_file(f.getvalue(), f.name, threshold)
        except Exception as exc:
            st.error(f"Could not process {f.name}: {exc}")
            continue
        if info.get("ocr_warning"):
            st.warning("Very little extractable text detected; OCR may be required.")
        if df.empty:
            st.warning("No readable sentences found.")
            continue
        df.insert(0, "Contract", f.name)
        all_results.append(df)
        a,b,c,d = st.columns(4)
        a.metric("Sentences",len(df))
        b.metric("Risk",int((df["Sentence Type"] == "Risk").sum()))
        c.metric("Obligations",int((df["Sentence Type"] == "Obligation").sum()))
        d.metric("Needs review",int(df["Needs Review"].sum()))
        left,right = st.columns(2)
        with left:
            st.write("Sentence type distribution")
            st.bar_chart(df["Sentence Type"].value_counts())
        with right:
            st.write("Related-party distribution")
            st.bar_chart(df["Related Party"].replace("", pd.NA).dropna().value_counts())
        tabs=st.tabs(["Risk register", "Obligations", "Rights", "All sentences"])
        for tab,typ in zip(tabs,["Risk","Obligation","Right",None]):
            with tab:
                selected = df if typ is None else df[df["Sentence Type"] == typ]
                st.dataframe(selected, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        safe = Path(f.name).stem.replace("/", "_").replace("\\", "_")
        zf.writestr(f"{idx+1:02d}_{safe}_analysis.csv",csv)
        st.download_button("Download this contract CSV",csv,file_name=f"{safe}_analysis.csv",mime="text/csv",key=f"download_{idx}")

if all_results:
    st.divider()
    st.subheader("Combined contract comparison")
    combined = pd.concat(all_results, ignore_index=True)
    summary = combined.groupby("Contract",sort=False).agg(
        Sentences=("Sentence","size"),
        Risks=("Sentence Type",lambda s:int((s=="Risk").sum())),
        Obligations=("Sentence Type",lambda s:int((s=="Obligation").sum())),
        Rights=("Sentence Type",lambda s:int((s=="Right").sum())),
        Review_Flags=("Needs Review","sum")
    ).reset_index()
    st.dataframe(summary,use_container_width=True,hide_index=True)
    st.download_button("Download all contract reports (ZIP)",archive.getvalue(),"HS_contract_reports.zip",mime="application/zip")
    st.download_button("Download combined CSV",combined.to_csv(index=False).encode("utf-8-sig"),"HS_all_contracts.csv",mime="text/csv")

with st.expander("Model validation and limitations"):
    m=get_metrics()
    if m:
        st.json(m)
    st.info("Accuracy, Macro-Precision, Macro-Recall, Macro-F1 and ROC-AUC require held-out expert-labelled ground truth. Uploaded contracts without labels cannot be used to calculate these metrics. Ensemble performance is not guaranteed to exceed its components.")
st.caption("Research decision-support prototype; not legal advice. Review commercially significant clauses manually.")
