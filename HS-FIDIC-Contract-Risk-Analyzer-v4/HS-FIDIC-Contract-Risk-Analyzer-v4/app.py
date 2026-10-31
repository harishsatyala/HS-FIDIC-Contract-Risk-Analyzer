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
                 context: tuple, contract_value: float, review_threshold: float, mit_eff):
    pages, info = extract_uploaded_file(name, data)
    rows = segment_pages(pages)
    if not rows:
        return pd.DataFrame(), info, {}
    suite = get_suite()
    preds = suite.predict([r["sentence"] for r in rows], type_model, party_model)
    df = engine.analyse(rows, preds, suite, get_library(), get_category_model(), type_model=type_model,
                        party_model=party_model, perspective=perspective, context=dict(context),
                        contract_value=contract_value, review_threshold=review_threshold,
                        mitigation_effectiveness=mit_eff)
    texts = [r["sentence"] for r in rows]
    votes = {"type": {k: v.tolist() for k, v in preds["type_all"].items()},
             "party": {k: v.tolist() for k, v in preds["party_all"].items()},
             "type_hybrid": {h: suite.type.proba(h, texts, cache=preds["type_all"])[0].tolist() for h in suite.type.hybrids},
             "party_hybrid": {h: suite.party.proba(h, texts, cache=preds["party_all"])[0].tolist() for h in suite.party.hybrids}}
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
    one_per_cat = st.toggle("One risk per category (earlier-version style)", value=False,
                            help="Keep only the highest-scoring clause in each risk category.")
    review_thr = st.slider("Predictability below which a clause needs expert review (%)", 40, 90, 60, 5)
    st.subheader("Project context")
    cur = st.selectbox("Currency", ["EUR", "USD", "GBP", "INR", "AED", "SAR", "TRY"], index=0)
    value = st.number_input("Total contract value (TCV)", min_value=0.0, value=100_000_000.0, step=1_000_000.0, format="%.0f")
    mit_pct = st.slider("Mitigation effectiveness (%)", 0, 90, 35, 5,
                        help="Expected reduction of EMV after mitigation. 35% is the risk library's screening assumption.")
    mit_eff = mit_pct / 100

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

st.info("All 20 individual machine-learning models and all 8 hybrid models run automatically when a contract is "
        "uploaded. The selected default hybrid drives the risk register; every model's output is shown in **🤖 Model execution**.")
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
                                       float(value), float(review_thr), mit_eff)
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
st.success("Analysed: " + ", ".join(results))

names = list(results)
sel = st.selectbox("Contract", names) if len(names) > 1 else names[0]
df, info, votes = results[sel]
def make_register(d):
    r = engine.risk_register(d, min_prob)
    if one_per_cat and len(r):
        r = r.drop_duplicates("Risk Category").reset_index(drop=True)
    return r


reg = make_register(df)
summ = engine.contract_summary(df, reg)

tabs = st.tabs(["📊 Risk dashboard", "🛡️ Mitigation plans", "🤖 Model execution", "🎯 Predictability",
                "🏆 Model performance", "📄 Clause explorer", "⚖️ Compare contracts", "ℹ️ Method"])

# ============================================================================= 1 DASHBOARD
NUMBERS_GUIDE = """
**Risk Probability %** — probability that the clause carries a contractual risk: the ML model's P(Risk) combined with
onerous-wording triggers (e.g. *uncapped liability*, *sole discretion*, *time-bar*). It **is** a probability-style score
(0–100%), but it is calibrated against auto-generated labels, not against real claims.

**Risk Score %** — priority for review: Risk Probability × how much of the risk falls on your side (perspective) ×
impact (1–5) and library likelihood. Bands: ≥50 ⛔ Critical, ≥30 🔶 High, ≥15 🟡 Medium, otherwise 🟢 Low.

**Relevance score** (Risk-ID ranking) — the Risk Score expressed on a 0–1 scale; it ranks which clause / library risk
deserves attention first.

**Decision Strength %** — the selected model's probability for its predicted sentence type (Heading, Definition,
Obligation, Risk, Right). It is **not** model accuracy; accuracy is in 🏆 Model performance.

**Responsible Party** — allocation in the matched risk-library record. **Predicted Responsible Party** — what the
ML party model reads in this clause (Contractor, Employer, Shared). **Party Confidence %** — that model's probability.

**Predictability %** — how far the prediction can be trusted for this clause: agreement among the 20 individual models,
confidence, entropy and the model's historical precision. Below the review threshold → expert review.

**Probability (L/M/H) & Probability Factor** — qualitative likelihood from the matched risk-library record and its
screening factor: **Low = 0.15, Medium = 0.40, High = 0.70**.

**Gross Exposure** — TCV × the record's exposure % of TCV. **Library EMV** = Gross Exposure × Probability Factor.
**EMV (clause-weighted)** = Library EMV × Risk Probability. **Residual EMV** = EMV × (1 − mitigation effectiveness).
Totals take the largest clause per category to avoid double counting; they are screening estimates, not contract losses.
"""

with tabs[0]:
    st.subheader("📊 Risk Analysis Summary")
    st.caption(f"**{sel}** · {info.get('pages', '?')} page(s) · {summ['sentences']} sentences analysed · risk register "
               f"driven by **{type_model}** · perspective **{perspective}** · TCV {money(value, cur)}")
    k = st.columns(6)
    k[0].metric("Risks Identified", summ["risks"])
    k[1].metric("Risk Categories", summ["categories"])
    k[2].metric("⛔ Critical", summ["critical"])
    k[3].metric("🔶 High", summ["high"])
    k[4].metric("Expected Monetary Value", money(summ["emv"], cur))
    k[5].metric("Residual after mitigation", money(summ["residual"], cur),
                delta=f"-{(1 - summ['residual'] / summ['emv']) * 100:.0f}%" if summ["emv"] else None, delta_color="inverse")
    g1, g2, g3 = st.columns(3)
    chart(g1, charts.gauge(summ["risk_index"], "Contract Risk Index (mean of top-10 risk scores)"))
    chart(g2, charts.gauge(summ["mean_probability"], "Mean risk probability of identified risks"))
    chart(g3, charts.gauge(summ["predictability"], "Prediction predictability",
                           bands=((0, 50, "#f8d7d7"), (50, 70, "#fff4d6"), (70, 100, "#e9f5e9"))))
    st.caption(f"All {len(votes['type'])} individual ML models and all {len(votes['type_hybrid'])} hybrid models were "
               f"executed automatically (see 🤖 Model execution). **{type_model}** — the hybrid selected by cross-validation — "
               "drives risk identification and prioritisation.")
    if reg.empty:
        st.success("No clause exceeds the risk-probability threshold. Lower the threshold in the sidebar to see weaker signals.")
    else:
        st.subheader("📈 Risk Distribution")
        st.caption("Hover over any bar to see the clauses, library Risk IDs, highest risk score and the top clause.")
        a, b = st.columns(2)
        chart(a, charts.distribution_bar(reg, "Risk Category", "Risk Category Distribution"))
        chart(b, charts.distribution_bar(reg, "Probability (L/M/H)", "Probability Distribution (L / M / H)",
                                         order=["H", "M", "L"]))
        a, b = st.columns(2)
        chart(a, charts.distribution_bar(reg, "Impact Areas", "Impact Distribution (cost / time / quality / HSE)"))
        chart(b, charts.distribution_bar(reg, "Related Party", "Responsible Party Distribution (ML-predicted)",
                                         order=["Contractor", "Employer", "Shared"], colors=charts.PARTY_COLORS))
        a, b = st.columns(2)
        chart(a, charts.distribution_bar(reg, "Rating", "Risk Rating Distribution", order=charts.RATING_ORDER,
                                         colors=charts.RATING_COLORS))
        chart(b, charts.heat_matrix(reg))

        st.subheader("🎯 Risk-ID Relevance Ranking")
        st.caption("Hover over any bar to see the Risk ID, risk category, relevance score, risk statement and the contract clause.")
        chart(st, charts.risk_id_ranking(reg, 20))

        n_top = min(20, len(reg))
        st.subheader(f"🚨 Top {n_top} Contract Risks")
        with st.expander("ℹ️ What the risk-analysis numbers mean", expanded=True):
            st.markdown(NUMBERS_GUIDE)
        cols = ["Rating", "Sentence", "Clause", "Page", "Sentence Type", "Library Risk ID", "Risk Statement", "Risk Category",
                "Type Confidence %", "Responsible Party (library)", "Related Party", "Party Confidence %",
                "Risk Probability %", "Risk Score %", "Probability (L/M/H)", "Impact Areas", "Project Type",
                "Probability Factor", "Mitigation Effectiveness %", "Gross Exposure", "EMV", "Residual EMV",
                "Risk Owner", "Early Warning Indicators", "Mitigation", "Library Mitigation", "Triggers",
                "FIDIC Reference", "Predictability %", "Needs Review"]
        top = reg.head(n_top)[cols].rename(columns={
            "Sentence": "Contract Sentence", "Library Risk ID": "Risk ID", "Type Confidence %": "Decision Strength %",
            "Responsible Party (library)": "Responsible Party", "Related Party": "Predicted Responsible Party",
            "Probability (L/M/H)": "Probability", "Impact Areas": "Impact", "Mitigation": "Recommended Mitigation",
            "Library Mitigation": "Mitigation Actions (library)"})
        top["Rating"] = top["Rating"].map(lambda r: f"{ICON[r]} {r}")
        table(st, top, hide_index=True, height=620,
              column_config={"Risk Score %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                             "Risk Probability %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
                             "Decision Strength %": st.column_config.NumberColumn(format="%.1f"),
                             "Party Confidence %": st.column_config.NumberColumn(format="%.1f"),
                             "Predictability %": st.column_config.NumberColumn(format="%.0f"),
                             "Gross Exposure": st.column_config.NumberColumn(format="%.0f"),
                             "EMV": st.column_config.NumberColumn(format="%.0f"),
                             "Residual EMV": st.column_config.NumberColumn(format="%.0f"),
                             "Contract Sentence": st.column_config.TextColumn(width="large")})
        st.download_button("📥 Download Risk Analysis CSV", top.to_csv(index=False).encode("utf-8-sig"),
                           f"{Path(sel).stem}_top_{n_top}_risks.csv", "text/csv")

        st.subheader("🛡️ Mitigation Plans — top risks")
        st.caption("Full mitigation cards with triggers, FIDIC orientation and early-warning indicators are in the 🛡️ Mitigation plans tab.")
        mp = reg.head(n_top)[["Rating", "Library Risk ID", "Clause", "Risk Category", "Risk Probability %", "Mitigation",
                              "Risk Owner", "Early Warning Indicators", "EMV", "Residual EMV"]].rename(
            columns={"Library Risk ID": "Risk ID", "Mitigation": "Recommended Mitigation"})
        mp["Rating"] = mp["Rating"].map(lambda r: f"{ICON[r]} {r}")
        table(st, mp, hide_index=True,
              column_config={"Recommended Mitigation": st.column_config.TextColumn(width="large"),
                             "Risk Probability %": st.column_config.NumberColumn(format="%.1f"),
                             "EMV": st.column_config.NumberColumn(format="%.0f"),
                             "Residual EMV": st.column_config.NumberColumn(format="%.0f")})

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
                    st.markdown(f"**Risk library match:** `{r['Library Risk ID']}` · likelihood **{r['Library Likelihood']}** "
                                f"(factor {r['Probability Factor']}) · impact {r['Impact Areas']}")
                    st.caption(r["Library Risk Statement"])
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

# ============================================================================= 2b MODEL EXECUTION
with tabs[2]:
    st.subheader("🤖 Automatic model execution")
    st.caption("Every individual and hybrid model classified every sentence of this contract. "
               "Decision strength = mean probability of each model's predicted class; it is not accuracy.")
    labs = suite.type.labels
    sel_idx = np.asarray(votes["type_hybrid"].get(type_model, votes["type"].get(type_model))).argmax(1)

    def status_rows(prob_dict, kind):
        out = []
        for n, P in prob_dict.items():
            P = np.asarray(P)
            pred = P.argmax(1)
            dist = pd.Series(pred).map(dict(enumerate(labs))).value_counts()
            out.append({"Model": n, "Kind": kind,
                        "Algorithm": (metrics_algo.get(n) if kind == "Individual" else HYBRID_INFO.get(n, "")[:70]),
                        "Sentences classified": len(P), "Status": "✅ Completed" if len(P) == len(df) else "⚠️ Incomplete",
                        "Risk sentences": int((pred == labs.index("Risk")).sum()),
                        "Obligations": int((pred == labs.index("Obligation")).sum()),
                        "Rights": int((pred == labs.index("Right")).sum()),
                        "Dominant type": dist.index[0] if len(dist) else "—",
                        "Avg decision strength %": round(100 * float(P.max(1).mean()), 1),
                        "Agreement with selected %": round(100 * float((pred == sel_idx).mean()), 1),
                        "Held-out accuracy": suite.type.model_accuracy.get(n)})
        return out

    metrics_algo = {r["model"]: f"{r['vectorizer']} + {r['algorithm']}"
                    for r in metrics.get("sentence_type", {}).get("results", []) if r.get("category") == "Individual"}
    k = st.columns(4)
    k[0].metric("Individual models executed", len(votes["type"]))
    k[1].metric("Hybrid models executed", len(votes["type_hybrid"]))
    k[2].metric("Sentences per model", len(df))
    k[3].metric("Mean agreement with selected", f"{df['Model Agreement %'].mean():.0f}%")
    st.markdown("#### Individual models")
    table(st, pd.DataFrame(status_rows(votes["type"], "Individual")), hide_index=True,
          column_config={"Held-out accuracy": st.column_config.NumberColumn(format="%.3f")})
    st.markdown("#### Hybrid models")
    table(st, pd.DataFrame(status_rows(votes["type_hybrid"], "Hybrid")), hide_index=True,
          column_config={"Held-out accuracy": st.column_config.NumberColumn(format="%.3f")})

    st.markdown("#### 🔀 Hybrid model details")
    htabs = st.tabs([h.replace("Hybrid: ", "") for h in votes["type_hybrid"]])
    hres = {r["model"]: r for r in metrics.get("sentence_type", {}).get("results", [])}
    for ht, (hname, P) in zip(htabs, votes["type_hybrid"].items()):
        with ht:
            P = np.asarray(P)
            pred = pd.Series(P.argmax(1)).map(dict(enumerate(labs)))
            st.markdown(f"**{hname}** — {HYBRID_INFO.get(hname, '')}")
            members = hres.get(hname, {}).get("members")
            if isinstance(members, list):
                w = hres[hname].get("weights")
                st.caption("Base models: " + ", ".join(f"{m} (w={x:.2f})" if w else m
                                                         for m, x in zip(members, w or [0] * len(members))))
            elif members:
                st.caption(f"Base models: {members}")
            m = st.columns(4)
            m[0].metric("Sentences classified", len(P))
            m[1].metric("Dominant type", pred.value_counts().index[0])
            m[2].metric("Avg decision strength", f"{100 * P.max(1).mean():.0f}%")
            ho = hres.get(hname, {})
            m[3].metric("Held-out accuracy / macro-F1", f"{ho.get('accuracy', 0):.1%} / {ho.get('macro_f1', 0):.3f}")
            dist = pred.value_counts().reindex(labs, fill_value=0)
            chart(st, charts.distribution_bar(
                pd.DataFrame({"Sentence Type": pred, "Sentence": df["Sentence"], "Clause": df["Clause"],
                              "Library Risk ID": df["Library Risk ID"], "Risk Score %": df["Risk Score %"]}),
                "Sentence Type", f"Sentence-type distribution — {hname.replace('Hybrid: ', '')}", order=labs))
            view = df[["Clause", "Sentence"]].assign(**{"Predicted type": pred.values,
                                                        "Decision strength %": (100 * P.max(1)).round(1)})
            table(st, view[view["Predicted type"].isin(["Risk", "Right", "Obligation"])], hide_index=True, height=300)
    counts = pd.DataFrame({n: pd.Series(np.asarray(v).argmax(1)).map(dict(enumerate(labs))).value_counts()
                           for n, v in {**votes["type"], **votes["type_hybrid"]}.items()}).T.fillna(0).astype(int)
    chart(st, charts.model_distribution(counts.reindex(columns=labs, fill_value=0),
                                        "How every model classifies this contract (sentences per type)"))

# ============================================================================= 3 PREDICTABILITY
with tabs[3]:
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

# ============================================================================= 4 MODEL PERFORMANCE DASHBOARD
with tabs[4]:
    st.subheader("🏆 Model Performance Dashboard — individual vs hybrid models")
    if not metrics:
        st.warning("metrics.json not found — run `python scripts/train_models.py`.")
    else:
        st.caption("Held-out test-set performance of every individual algorithm and every hybrid model on Accuracy, "
                   "Macro Precision, Macro Recall, Macro-F1 and ROC-AUC (one-vs-rest), shown as %. "
                   "Uploaded contracts have no sentence-level ground truth, so these scores come from the labelled workbook only.")
        task = st.radio("Classification task", ["Sentence type (Heading / Definition / Obligation / Risk / Right)",
                                                "Related party (Contractor / Employer / Shared)"], horizontal=True)
        key = "sentence_type" if task.startswith("Sentence") else "related_party"
        m = metrics[key]
        pf = charts.perf_frame(m["results"])
        ind, hyb = pf[pf["Type"] == "Individual"], pf[pf["Type"] == "Hybrid"]
        bi, bh = ind.iloc[0], hyb.iloc[0]
        k = st.columns(5)
        k[0].metric("Models compared", f"{len(ind)} + {len(hyb)}", "individual + hybrid", delta_color="off")
        k[1].metric("Best individual (Macro-F1)", f"{bi['Macro-F1']:.1f}%", bi["Model"][:30], delta_color="off")
        k[2].metric("Best hybrid (Macro-F1)", f"{bh['Macro-F1']:.1f}%", bh["Model"].replace("Hybrid: ", "")[:30], delta_color="off")
        k[3].metric("Hybrid gain over best individual", f"{bh['Macro-F1'] - bi['Macro-F1']:+.1f} pts", "Macro-F1", delta_color="off")
        dm = pf.set_index("Model").loc[m["default_model"]]
        k[4].metric("Deployed default", f"{dm['Accuracy']:.1f}% acc", m["default_model"].replace("Hybrid: ", "")[:30], delta_color="off")
        st.caption(f"Test set: {m['n_test']} sentences · development set: {m['n_dev']} sentences (5-fold CV) · "
                   "the default model is chosen by cross-validated Macro-F1, not by the test scores below.")

        chart(st, charts.family_summary(pf, "Individual vs hybrid models — best and mean score per metric"))
        a, b = st.columns([3, 2])
        chart(a, charts.perf_heatmap(pf, "All models × all five metrics (%)"))
        with b:
            metric = st.selectbox("Rank models by", list(charts.PERF_METRICS.values()), index=3)
            chart(st, charts.metric_ranking(pf, metric, "Model ranking"))
        default_sel = list(dict.fromkeys(list(hyb["Model"].head(4)) + list(ind["Model"].head(3))))
        chosen = st.multiselect("Compare models side by side", pf["Model"].tolist(), default=default_sel)
        if chosen:
            chart(st, charts.perf_grouped(pf, chosen, "Accuracy · Macro Precision · Macro Recall · Macro-F1 · ROC-AUC"))
        chart(st, charts.perf_scatter(pf, "Accuracy vs Macro-F1 (each point = one model)"))

        st.markdown("#### 📋 Full comparison table")
        res = pd.DataFrame(m["results"]).set_index("model")
        full = pf.copy()
        full["Macro-F1 95% CI"] = full["Model"].map(lambda n: "{:.1f}–{:.1f}".format(*[100 * x for x in res.loc[n, "macro_f1_ci95"]]))
        full["CV Macro-F1 (dev)"] = full["Model"].map(lambda n: round(100 * res.loc[n, "cv_macro_f1"], 2))
        full["Weighted-F1"] = full["Model"].map(lambda n: round(100 * res.loc[n, "weighted_f1"], 2))
        full["ECE"] = full["Model"].map(lambda n: res.loc[n, "ece"])
        full["Vectorizer / method"] = full["Model"].map(lambda n: res.loc[n, "vectorizer"] if res.loc[n, "category"] == "Individual"
                                                        else res.loc[n, "algorithm"])
        full.insert(0, "Rank", range(1, len(full) + 1))
        pc = {c: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.2f") for c in charts.PERF_METRICS.values()}
        table(st, full, hide_index=True, height=560, column_config=pc)
        st.download_button("📥 Download model comparison (CSV)", full.to_csv(index=False).encode("utf-8-sig"),
                           f"model_performance_{key}.csv", "text/csv")

        with st.expander("ℹ️ Metric definitions"):
            st.markdown("**Accuracy** — share of sentences classified correctly.  \n"
                        "**Macro Precision** — for each class, the share of predictions of that class that are correct; averaged over classes.  \n"
                        "**Macro Recall** — for each class, the share of its sentences that are found; averaged over classes.  \n"
                        "**Macro-F1** — harmonic mean of precision and recall per class, averaged so rare classes (Risk, Right) count as much as Obligation.  \n"
                        "**ROC-AUC** — one-vs-rest area under the ROC curve: how well the probabilities rank the true class first (50% = chance).  \n"
                        "**CV Macro-F1** — 5-fold cross-validated score on the development set, used to choose the default model.  \n"
                        "**ECE** — calibration error (0 = confidence matches accuracy).")
        with st.expander("🔍 Confusion matrix and hybrid strategies"):
            c1, c2 = st.columns(2)
            which = c1.selectbox("Confusion matrix for", pf["Model"].tolist(), index=pf["Model"].tolist().index(m["default_model"]))
            chart(c1, charts.confusion(res.loc[which, "confusion_matrix"], m["labels"], which))
            with c2:
                for hn, desc in metrics["hybrid_descriptions"].items():
                    if hn in res.index:
                        st.markdown(f"**{hn.replace('Hybrid: ', '')}** — {desc}")
                        mem = res.loc[hn].get("members")
                        if isinstance(mem, list):
                            st.caption("Members: " + ", ".join(mem))
        with st.expander("🧹 Data audit and evaluation caveats"):
            st.caption(metrics["evaluation_protocol"])
            for c in metrics.get("caveats", []):
                st.caption("⚠️ " + c)
            st.json(metrics.get("data_audit", {}))

# ============================================================================= 5 CLAUSE EXPLORER
with tabs[5]:
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
with tabs[6]:
    st.subheader("Contract comparison")
    rows = []
    for n, (d, _, _) in results.items():
        rg = make_register(d)
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
        allreg = pd.concat([make_register(d).assign(Contract=n) for n, (d, _, _) in results.items()])
        piv = allreg.pivot_table(index="Risk Category", columns="Contract", values="Risk Score %", aggfunc="max").fillna(0)
        st.markdown("**Highest risk score per category and contract**")
        table(st, piv.round(0))

# ============================================================================= 7 METHOD
with tabs[7]:
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
- Quantified risk library: Gross Exposure = TCV × exposure % of TCV; Library EMV = Gross × probability factor
  (L 0.15 / M 0.40 / H 0.70); clause-weighted EMV = Library EMV × L; residual = EMV × (1 − mitigation effectiveness, default 35%).
  Dashboard totals take the largest clause per category so overlapping clauses are not double-counted.

**Limitations.** Training labels were auto-generated (not expert-validated); scores measure agreement with those labels.
EMV values are screening estimates from category factors, not losses read from the contract. Risk categories are
weakly supervised. This is decision support for bid-stage review, not legal advice.
""")
st.caption("HS-FIDIC research prototype · decision support only, not legal advice · thesis basis: Eken (2022), METU.")
