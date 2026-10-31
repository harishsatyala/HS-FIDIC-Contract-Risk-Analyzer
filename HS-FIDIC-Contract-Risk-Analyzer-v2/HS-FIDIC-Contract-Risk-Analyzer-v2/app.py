"""HS-FIDIC Contract Risk Analyzer — Streamlit dashboard.

Thesis basis: Eken, G. (2022) "Using Natural Language Processing for Automated Construction Contract
Review During Risk Assessment at the Bidding Stage", PhD thesis, METU.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import charts
from src import risk_engine as engine
from src.model_zoo import HYBRID_INFO
from src.predictor import ModelSuite
from src.text_utils import extract_uploaded_file, segment_pages

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "data"
LIB_PATH = DATA / "Global_Energy_Project_Risk_Library_3000.xlsx"
DATASET_PATH = DATA / "FIDIC_Thesis_Style_Labeled_Dataset.xlsx"
SAMPLE = DATA / "sample_epc_contract.pdf"

st.set_page_config(page_title="HS-FIDIC Contract Risk Analyzer", page_icon="📑", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
.badge {display:inline-block;padding:2px 10px;border-radius:12px;font-weight:600;font-size:0.8rem;color:#0b0b0b;}
.b-Critical{background:#f8d7d7;border:1px solid #d03b3b}.b-High{background:#fde4d8;border:1px solid #ec835a}
.b-Medium{background:#fff4d6;border:1px solid #fab219}.b-Low{background:#e9f5e9;border:1px solid #0ca30c}
.clause{border-left:3px solid #2a78d6;padding:6px 12px;background:#f6f8fb;border-radius:4px;font-size:0.92rem}
</style>""", unsafe_allow_html=True)
ICON = {"Critical": "⛔", "High": "🔶", "Medium": "🟡", "Low": "🟢", "None": "·"}


# ----------------------------------------------------------------------------- resources
@st.cache_resource(show_spinner="Loading the 20 individual + 8 hybrid models…")
def get_suite():
    return ModelSuite(MODELS)


@st.cache_resource(show_spinner="Loading 3,000-record risk library…")
def get_library():
    return engine.load_library(LIB_PATH) if LIB_PATH.exists() else None


@st.cache_resource(show_spinner="Building risk-category model…")
def get_category_model():
    corpus = None
    if DATASET_PATH.exists():
        d = pd.read_excel(DATASET_PATH, sheet_name="Combined_Unique")
        corpus = d.loc[d["Language"] == "English", "Sentence"].astype(str).tolist()
    return engine.CategoryModel(corpus)


@st.cache_data
def get_metrics():
    p = MODELS / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False, max_entries=32)
def run_analysis(data: bytes, name: str, type_model: str, party_model: str, perspective: str,
                 context: tuple, contract_value: float, review_threshold: float):
    pages, info = extract_uploaded_file(name, data)
    rows = segment_pages(pages)
    if not rows:
        return pd.DataFrame(), info, {}
    suite = get_suite()
    preds = suite.predict([r["sentence"] for r in rows], type_model, party_model)
    df = engine.analyse(rows, preds, suite, get_library(), get_category_model(), type_model=type_model,
                        party_model=party_model, perspective=perspective, context=dict(context),
                        contract_value=contract_value, review_threshold=review_threshold)
    votes = {"type": {k: v.tolist() for k, v in preds["type_all"].items()},
             "party": {k: v.tolist() for k, v in preds["party_all"].items()}}
    return df, info, votes


def badge(r):
    return f'<span class="badge b-{r}">{ICON.get(r, "")} {r}</span>'


def money(x, cur):
    return f"{cur} {x:,.0f}"


def chart(container, fig):
    """Render a Plotly figure full-width on both new (width='stretch') and older Streamlit APIs."""
    try:
        container.plotly_chart(fig, width="stretch")
    except Exception:
        container.plotly_chart(fig, use_container_width=True)


def table(container, data, **kw):
    try:
        container.dataframe(data, width="stretch", **kw)
    except Exception:
        container.dataframe(data, use_container_width=True, **kw)


# ----------------------------------------------------------------------------- load models
try:
    suite = get_suite()
except Exception as exc:  # version mismatch or missing files
    st.error(f"Could not load the trained models ({exc}).")
    st.info("Re-train locally with `python scripts/train_models.py` (≈6 min) and commit the two files in `models/`. "
            "Pin the same scikit-learn version as in requirements.txt.")
    st.stop()
library = get_library()
metrics = get_metrics()


def model_label(name, task):
    star = " ★ default" if name == task.default_model else ""
    acc = task.model_accuracy.get(name)
    return f"{name}{star}" + (f"  · acc {acc:.0%}" if acc is not None else "")


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Analysis settings")
    perspective = st.radio("Review perspective (whose risk?)", ["Contractor", "Employer", "Neutral"], horizontal=True,
                           help="Risk Score weights each clause by how much of it is allocated to this party.")
    st.subheader("Machine-learning models")
    tnames = suite.type.model_names
    type_model = st.selectbox("Sentence-type model", tnames, index=tnames.index(suite.type.default_model),
                              format_func=lambda n: model_label(n, suite.type))
    pnames = suite.party.model_names
    party_model = st.selectbox("Related-party model", pnames, index=pnames.index(suite.party.default_model),
                               format_func=lambda n: model_label(n, suite.party))
    with st.expander("What does this model do?"):
        st.caption(HYBRID_INFO.get(type_model, "Individual model: vectorizer + algorithm pipeline (see Model Lab)."))
    st.subheader("Risk register")
    min_prob = st.slider("Minimum risk probability to register (%)", 10, 90, 35, 5)
    review_thr = st.slider("Predictability below which a clause needs expert review (%)", 40, 90, 60, 5)
    st.subheader("Project context")
    cur = st.selectbox("Currency", ["USD", "EUR", "GBP", "INR", "AED", "SAR", "TRY"], index=0)
    value = st.number_input("Contract value", min_value=0.0, value=25_000_000.0, step=1_000_000.0, format="%.0f")

    def opts(col):
        return ["Any"] + (sorted(library[col].dropna().unique().tolist()) if library is not None else [])

    ctx = {"project_type": st.selectbox("Project type", opts("Project Type")),
           "segment": st.selectbox("Energy segment", opts("Energy Segment")),
           "region": st.selectbox("Region", opts("Geographic Context")),
           "phase": st.selectbox("Lifecycle phase", opts("Lifecycle Phase"))}
    st.caption("Context filters pick the closest record from the 3,000-risk library for owner, early-warning indicators and library mitigation.")

# ----------------------------------------------------------------------------- header + upload
st.title("📑 HS-FIDIC Contract Risk Analyzer")
st.caption("Upload contract PDFs → every clause is classified by individual and hybrid ML models (Eken 2022 taxonomy: "
           "Heading / Definition / Obligation / Risk / Right × Contractor / Employer / Shared), scored as a risk "
           "probability, linked to the risk library and given a mitigation plan.")

c1, c2 = st.columns([3, 1])
with c1:
    files = st.file_uploader("Upload one or more contracts (PDF; DOCX/TXT also accepted)", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True)
with c2:
    st.write("")
    st.write("")
    use_sample = st.toggle("Use sample EPC contract", value=not files, disabled=not SAMPLE.exists(),
                           help="Synthetic FIDIC-style contract written for demonstration (not FIDIC text).")

docs = [(f.name, f.getvalue()) for f in files] if files else []
if use_sample and SAMPLE.exists() and not docs:
    docs = [(SAMPLE.name, SAMPLE.read_bytes())]
if not docs:
    st.info("Upload a contract PDF or switch on the sample contract to begin.")
    st.stop()

results = {}
prog = st.progress(0.0, text="Analysing…")
for i, (name, data) in enumerate(docs):
    try:
        df, info, votes = run_analysis(data, name, type_model, party_model, perspective, tuple(sorted(ctx.items())),
                                       float(value), float(review_thr))
    except Exception as exc:
        st.error(f"Could not process {name}: {exc}")
        continue
    if info.get("ocr_warning"):
        st.warning(f"{name}: very little extractable text" + (" — it may be a scanned PDF that needs OCR first." if name.lower().endswith(".pdf") else "."))
    if df.empty:
        st.warning(f"{name}: no readable sentences found.")
        continue
    results[name] = (df, info, votes)
    prog.progress((i + 1) / len(docs), text=f"Analysed {name}")
prog.empty()
if not results:
    st.stop()

names = list(results)
sel = st.selectbox("Contract", names) if len(names) > 1 else names[0]
df, info, votes = results[sel]
reg = engine.risk_register(df, min_prob)
summ = engine.contract_summary(df, reg)

tabs = st.tabs(["📊 Risk dashboard", "🛡️ Mitigation plan", "🎯 Predictability", "🤖 Model lab",
                "📄 Clause explorer", "⚖️ Compare contracts", "ℹ️ Method"])

# ============================================================================= 1 DASHBOARD
with tabs[0]:
    st.subheader(f"{sel}")
    st.caption(f"{info.get('pages', '?')} page(s) · {summ['sentences']} sentences · model: **{type_model}** · perspective: **{perspective}**")
    g1, g2, g3 = st.columns(3)
    chart(g1, charts.gauge(summ["risk_index"], "Contract Risk Index (mean of top-10 scores)"))
    chart(g2, charts.gauge(summ["mean_probability"], "Mean risk probability of registered risks"))
    chart(g3, charts.gauge(summ["predictability"], "Prediction predictability",
                                 bands=((0, 50, "#f8d7d7"), (50, 70, "#fff4d6"), (70, 100, "#e9f5e9"))))
    k = st.columns(6)
    k[0].metric("Registered risks", summ["risks"])
    k[1].metric("⛔ Critical", summ["critical"])
    k[2].metric("🔶 High", summ["high"])
    k[3].metric("Risk categories", summ["categories"])
    k[4].metric("Screening EMV", money(summ["emv"], cur))
    k[5].metric("Residual after mitigation", money(summ["residual"], cur),
                delta=f"-{(1 - summ['residual'] / summ['emv']) * 100:.0f}%" if summ["emv"] else None, delta_color="inverse")
    if reg.empty:
        st.success("No clause exceeds the risk-probability threshold. Lower the threshold in the sidebar to see weaker signals.")
    else:
        a, b = st.columns(2)
        chart(a, charts.heat_matrix(reg))
        chart(b, charts.category_bars(reg))
        a, b = st.columns(2)
        chart(a, charts.clause_profile(reg))
        chart(b, charts.allocation_donut(reg))
        st.markdown("#### Top 10 risks")
        top = reg.head(10)[["Rating", "Risk Score %", "Risk Probability %", "Impact (1-5)", "Risk Category", "Clause",
                            "Related Party", "Triggers", "Predictability %", "Sentence"]].copy()
        top["Rating"] = top["Rating"].map(lambda r: f"{ICON[r]} {r}")
        table(st, top, hide_index=True,
                     column_config={"Risk Score %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                                    "Risk Probability %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                                    "Predictability %": st.column_config.NumberColumn(format="%.0f"),
                                    "Sentence": st.column_config.TextColumn(width="large")})

# ============================================================================= 2 MITIGATION
with tabs[1]:
    st.subheader("Risk mitigation plan")
    if reg.empty:
        st.info("No registered risks at the current threshold.")
    else:
        chart(st, charts.emv_bars(reg, cur))
        f1, f2, f3 = st.columns(3)
        r_f = f1.multiselect("Rating", charts.RATING_ORDER, default=["Critical", "High", "Medium"])
        c_f = f2.multiselect("Category", sorted(reg["Risk Category"].unique()))
        n_show = f3.slider("Show top N", 5, max(5, len(reg)), min(15, max(5, len(reg))))
        view = reg[reg["Rating"].isin(r_f)]
        if c_f:
            view = view[view["Risk Category"].isin(c_f)]
        for i, r in view.head(n_show).iterrows():
            title = (f"{ICON[r['Rating']]} {r['Rating']} · {r['Risk Score %']:.0f}% score · {r['Risk Category']} · "
                     f"Clause {r['Clause'] or '—'} (p.{r['Page']})")
            with st.expander(title, expanded=i < 3):
                st.markdown(f'<div class="clause">{r["Sentence"]}</div>', unsafe_allow_html=True)
                m = st.columns(5)
                m[0].metric("Risk probability", f"{r['Risk Probability %']:.0f}%")
                m[1].metric("Impact", f"{r['Impact (1-5)']}/5")
                m[2].metric("Allocated to you", f"{r['Allocation to you %']:.0f}%")
                m[3].metric("Predictability", f"{r['Predictability %']:.0f}%")
                m[4].metric("EMV → residual", money(r["EMV"], cur), delta=f"-{money(r['EMV'] - r['Residual EMV'], cur)}",
                            delta_color="inverse")
                left, right = st.columns([3, 2])
                with left:
                    st.markdown(f"**Recommended mitigation** &nbsp; {badge(r['Rating'])}", unsafe_allow_html=True)
                    for line in [x.strip() for x in r["Mitigation"].replace("; ", ";\n").split(". ") if x.strip()]:
                        st.markdown(f"- {line.rstrip('.')}.")
                    if r["Triggers"]:
                        st.markdown(f"**Risk triggers detected:** {r['Triggers']}")
                    st.markdown(f"**FIDIC orientation:** {r['FIDIC Reference']}")
                with right:
                    st.markdown(f"**Risk library match:** `{r['Library Risk ID']}` · likelihood **{r['Library Likelihood']}**")
                    st.markdown(f"**Risk owner:** {r['Risk Owner']}")
                    st.markdown(f"**Early-warning indicators:** {r['Early Warning Indicators']}")
                    st.caption(f"Library mitigation: {r['Library Mitigation']}")
                    st.caption(f"Sentence type {r['Sentence Type']} · party {r['Related Party']} · "
                               f"P(Risk) {r['P(Risk) %']:.0f}% · rule signal {r['Rule Signal %']:.0f}% · "
                               f"model agreement {r['Model Agreement %']:.0f}%")
        plan = reg[["Rating", "Risk Score %", "Risk Probability %", "Impact (1-5)", "Risk Category", "Clause", "Page",
                    "Related Party", "Sentence", "Triggers", "Mitigation", "FIDIC Reference", "Risk Owner",
                    "Early Warning Indicators", "Library Risk ID", "EMV", "Residual EMV", "Predictability %", "Needs Review"]]
        st.download_button("⬇️ Download mitigation plan (CSV)", plan.to_csv(index=False).encode("utf-8-sig"),
                           f"{Path(sel).stem}_mitigation_plan.csv", "text/csv")

# ============================================================================= 3 PREDICTABILITY
with tabs[2]:
    st.subheader("How predictable are these predictions?")
    st.caption("Predictability % = 35% model agreement (share of the 20 individual models voting with the selected model) "
               "+ 25% confidence + 15% (1 − normalised entropy) + 25% historical precision of the selected model for that "
               "class on the held-out test set.")
    clauses = df[~df["Sentence Type"].isin(["Heading"])]
    k = st.columns(4)
    k[0].metric("Mean predictability (all clauses)", f"{clauses['Predictability %'].mean():.0f}%")
    k[1].metric("Mean model agreement", f"{clauses['Model Agreement %'].mean():.0f}%")
    k[2].metric("Clauses needing review", int(clauses["Needs Review"].sum()))
    hist_acc = suite.type.model_accuracy.get(type_model)
    k[3].metric("Selected model held-out accuracy", f"{hist_acc:.1%}" if hist_acc else "—")
    a, b = st.columns(2)
    chart(a, charts.predictability_hist(clauses, review_thr))
    chart(b, charts.agreement_scatter(clauses))

    st.markdown("#### Inspect one clause: how every model votes")
    cand = clauses.sort_values("Risk Score %", ascending=False)
    pick = st.selectbox("Clause", cand.index.tolist(),
                        format_func=lambda i: f"[{cand.loc[i, 'Clause'] or '—'}] {cand.loc[i, 'Sentence'][:120]}")
    tv = {n: np.asarray(v)[pick] for n, v in votes["type"].items()}
    chart(st, charts.model_votes(tv, suite.type.labels, "Sentence-type probability from each individual model"))
    row = df.loc[pick]
    st.info(f"Selected model **{type_model}** → **{row['Sentence Type']}** ({row['Type Confidence %']:.0f}% confidence); "
            f"{row['Model Agreement %']:.0f}% of individual models agree; predictability {row['Predictability %']:.0f}%.")

    res = pd.DataFrame(metrics.get("sentence_type", {}).get("results", []))
    if not res.empty and type_model in set(res["model"]):
        r = res.set_index("model").loc[type_model]
        a, b = st.columns(2)
        chart(a, charts.calibration(r["calibration_curve"], f"Calibration on held-out test — {type_model}"))
        pc = pd.DataFrame(r["per_class"]).T.reset_index().rename(columns={"index": "Class"})
        b.markdown("**Historical reliability per predicted class (held-out test)**")
        table(b, pc, hide_index=True)
        b.caption(f"ECE {r['ece']:.3f} · Brier {r['brier']:.3f} · log-loss {r['log_loss']:.3f} · ROC-AUC {r['roc_auc_ovr']:.3f}")
    low = clauses[clauses["Needs Review"]].sort_values("Predictability %")
    st.markdown(f"#### Clauses flagged for expert review ({len(low)})")
    table(st, low[["Predictability %", "Model Agreement %", "Type Confidence %", "Sentence Type", "Risk Probability %",
                      "Clause", "Sentence"]], hide_index=True)

# ============================================================================= 4 MODEL LAB
with tabs[3]:
    st.subheader("Model lab: 20 individual algorithms + 8 hybrid models")
    if not metrics:
        st.warning("metrics.json not found — run `python scripts/train_models.py`.")
    else:
        st.caption(metrics["evaluation_protocol"])
        for c in metrics.get("caveats", []):
            st.caption("⚠️ " + c)
        task = st.radio("Task", ["Sentence type", "Related party"], horizontal=True)
        key = "sentence_type" if task == "Sentence type" else "related_party"
        m = metrics[key]
        res = pd.DataFrame(m["results"])
        b1, b2, b3, b4 = st.columns(4)
        best = res.sort_values("macro_f1", ascending=False).iloc[0]
        b1.metric("Default (CV-selected)", m["default_model"].replace("Hybrid: ", ""))
        b2.metric("Best held-out macro-F1", f"{best['macro_f1']:.3f}", best["model"].replace("Hybrid: ", "")[:28])
        bi = res[res.category == "Individual"].sort_values("macro_f1", ascending=False).iloc[0]
        b3.metric("Best individual macro-F1", f"{bi['macro_f1']:.3f}", bi["model"][:28])
        b4.metric("DEV / TEST sentences", f"{m['n_dev']} / {m['n_test']}")
        chart(st, charts.model_scores(res, f"{task}: held-out macro-F1"))
        show = res[["model", "category", "vectorizer", "algorithm", "cv_macro_f1", "cv_macro_f1_std", "accuracy",
                    "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "roc_auc_ovr", "ece", "brier",
                    "inference_ms_per_1000"]].sort_values("macro_f1", ascending=False)
        table(st, show, hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                    ["cv_macro_f1", "cv_macro_f1_std", "accuracy", "macro_precision", "macro_recall",
                                     "macro_f1", "weighted_f1", "roc_auc_ovr", "ece", "brier"]})
        a, b = st.columns(2)
        which = a.selectbox("Confusion matrix for", res["model"].tolist(), index=res["model"].tolist().index(m["default_model"]))
        rr = res.set_index("model").loc[which]
        chart(a, charts.confusion(rr["confusion_matrix"], m["labels"], which))
        with b:
            st.markdown("**Hybrid strategies**")
            for hn, desc in metrics["hybrid_descriptions"].items():
                if hn in set(res["model"]):
                    members = res.set_index("model").loc[hn].get("members")
                    st.markdown(f"**{hn.replace('Hybrid: ', '')}** — {desc}")
                    if isinstance(members, list):
                        st.caption("Members: " + ", ".join(members))
        with st.expander("Data audit and cleaning"):
            st.json(metrics.get("data_audit", {}))
            st.caption(f"Heading augmentation rows (training only): {metrics.get('heading_augmentation_rows')}")
        st.markdown("#### How each model reads *this* contract")
        lab = suite.type.labels
        dist = pd.DataFrame({n: pd.Series(np.asarray(v).argmax(1)).map(dict(enumerate(lab))).value_counts()
                             for n, v in votes["type"].items()}).T.fillna(0).astype(int)
        table(st, dist.reindex(columns=lab, fill_value=0))

# ============================================================================= 5 CLAUSE EXPLORER
with tabs[4]:
    st.subheader("Every sentence, every prediction")
    f1, f2, f3 = st.columns(3)
    t_f = f1.multiselect("Sentence type", suite.type.labels, default=["Obligation", "Risk", "Right"])
    p_f = f2.multiselect("Related party", ["Contractor", "Employer", "Shared", "—"])
    q = f3.text_input("Search text")
    v = df[df["Sentence Type"].isin(t_f)] if t_f else df
    if p_f:
        v = v[v["Related Party"].isin(p_f)]
    if q:
        v = v[v["Sentence"].str.contains(q, case=False, regex=False)]
    table(st, v, hide_index=True, height=520,
                 column_config={"Risk Probability %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                                "Sentence": st.column_config.TextColumn(width="large")})
    xbuf = io.BytesIO()
    with pd.ExcelWriter(xbuf, engine="openpyxl") as xw:
        pd.DataFrame([summ]).to_excel(xw, sheet_name="Summary", index=False)
        reg.to_excel(xw, sheet_name="Risk register", index=False)
        df.to_excel(xw, sheet_name="All sentences", index=False)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Excel report", xbuf.getvalue(), f"{Path(sel).stem}_HS_FIDIC_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("⬇️ All sentences (CSV)", df.to_csv(index=False).encode("utf-8-sig"), f"{Path(sel).stem}_sentences.csv", "text/csv")
    d3.download_button("⬇️ JSON", json.dumps({"summary": summ, "risks": reg.to_dict("records")}, default=str, indent=1),
                       f"{Path(sel).stem}_risks.json", "application/json")

# ============================================================================= 6 COMPARE
with tabs[5]:
    st.subheader("Contract comparison")
    rows = []
    for n, (d, _, _) in results.items():
        rg = engine.risk_register(d, min_prob)
        s = engine.contract_summary(d, rg)
        rows.append({"Contract": n, "Risk Index": s["risk_index"], "Risks": s["risks"], "Critical": s["critical"],
                     "High": s["high"], "Mean probability %": s["mean_probability"], "Predictability %": s["predictability"],
                     "EMV": s["emv"], "Residual EMV": s["residual"], "Needs review": s["needs_review"]})
    comp = pd.DataFrame(rows)
    if len(comp) > 1:
        chart(st, charts.contract_compare(comp))
    else:
        st.caption("Upload several contracts to compare them side by side.")
    table(st, comp, hide_index=True)
    if len(results) > 1:
        allreg = pd.concat([engine.risk_register(d, min_prob).assign(Contract=n) for n, (d, _, _) in results.items()])
        piv = allreg.pivot_table(index="Risk Category", columns="Contract", values="Risk Score %", aggfunc="max").fillna(0)
        st.markdown("**Highest risk score per category and contract**")
        table(st, piv.round(0))

# ============================================================================= 7 METHOD
with tabs[6]:
    st.subheader("Method")
    st.markdown("""
**Pipeline.** PDF text extraction → line re-joining, header/footer removal, sentence and heading segmentation with clause
tracking → thesis preprocessing (lower-case, punctuation/digit/stop-word removal keeping modal verbs) → **sentence-type**
and **related-party** classification by the selected individual or hybrid model → rule layer (onerous-term triggers)
→ risk-category assignment (FIDIC-language lexicon + clause heading + weakly-supervised ML) → risk-library match →
scoring, mitigation and predictability.

**Individual models (20).** The thesis grid (Bag-of-Words / TF-IDF / dense embedding × Logistic Regression / SVM /
Decision Tree), extended with Multinomial & Complement Naive Bayes, SGD, Ridge, character n-gram SVM, word+char LR,
Random Forest, Extra Trees, Histogram Gradient Boosting (LightGBM-style), k-NN and an MLP neural network.
XGBoost / LightGBM are added automatically when installed at training time.

**Hybrid models (8).** Majority vote, the thesis **Competitive Voting** (§5.6), confidence-weighted vote, rank-weighted
vote, optimised soft voting, stacking meta-learner, **ML + rule-based stacking** (thesis §6.2 recommendation) and the
thesis **hierarchical binary decomposition** (§5.3, Step 2).

**Scores.**
- Risk Probability L = 1 − (1 − P(Risk))·(1 − P_rule)·(1 − 0.15·P(Obligation)), where P_rule = 1 − Π(1 − wₜ) over detected triggers.
- Allocation A = P(party = perspective) + 0.5·P(Shared).
- Risk Score = 100·L·(0.4 + 0.6·A)·(0.6·Impact/5 + 0.4·library likelihood/0.75). Bands: ≥50 Critical, ≥30 High, ≥15 Medium.
- Screening EMV = contract value × category exposure factor × Impact/3 × L; residual = EMV × (1 − mitigation effectiveness).
  The Dashboard totals take the largest clause per category so overlapping clauses are not double-counted.

**Limitations.** Training labels were auto-generated (not expert-validated); scores measure agreement with those labels.
EMV values are screening estimates from category factors, not losses read from the contract. Risk categories are
weakly supervised. This is decision support for bid-stage review, not legal advice.
""")
st.caption("HS-FIDIC research prototype · decision support only, not legal advice · thesis basis: Eken (2022), METU.")
