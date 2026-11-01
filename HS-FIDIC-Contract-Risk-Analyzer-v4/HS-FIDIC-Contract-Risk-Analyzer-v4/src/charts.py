"""Plotly figures for the dashboard (colors from the validated dataviz reference palette)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RATING_COLORS = {"Critical": "#d03b3b", "High": "#ec835a", "Medium": "#fab219", "Low": "#0ca30c"}
RATING_ORDER = ["Critical", "High", "Medium", "Low"]
PARTY_COLORS = {"Contractor": "#2a78d6", "Employer": "#eb6834", "Shared": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
TEXT = "#52514e"
GRID = "#e6e5e1"


def _style(fig, height=360, legend=True):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=78 if legend else 48, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color=TEXT),
        title_font=dict(size=14, color="#0b0b0b"), showlegend=legend, bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text=""),
        title=dict(y=0.985, yanchor="top", x=0.01, xanchor="left"),
        modebar=dict(orientation="v"),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def gauge(value, title, suffix="%", bands=((0, 15, "#e9f5e9"), (15, 30, "#fff4d6"), (30, 50, "#fde4d8"), (50, 100, "#f8d7d7"))):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), number={"suffix": suffix, "font": {"size": 34, "color": "#0b0b0b"}},
        title={"text": title, "font": {"size": 13, "color": TEXT}},
        gauge={"axis": {"range": [0, 100], "tickcolor": TEXT}, "bar": {"color": "#0d366b", "thickness": 0.28},
               "steps": [{"range": [a, b], "color": c} for a, b, c in bands], "borderwidth": 0}))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=0), template="plotly_white")
    return fig


def heat_matrix(reg: pd.DataFrame):
    """Probability band x impact matrix, cell = number of risks (sequential blue)."""
    bands = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    b = pd.cut(reg["Risk Probability %"], [0, 20, 40, 60, 80, 100.01], labels=bands, right=False)
    tab = pd.crosstab(b, reg["Impact (1-5)"]).reindex(index=bands, columns=[1, 2, 3, 4, 5], fill_value=0)
    z = tab.values[::-1]
    fig = go.Figure(go.Heatmap(
        z=z, x=["1 Minor", "2 Moderate", "3 Significant", "4 Major", "5 Severe"], y=bands[::-1],
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)], xgap=2, ygap=2,
        text=z, texttemplate="%{text}", hovertemplate="Probability %{y}%<br>Impact %{x}<br>%{z} risk(s)<extra></extra>",
        colorbar=dict(title="Risks", thickness=10)))
    fig.update_xaxes(title="Impact", showgrid=False)
    fig.update_yaxes(title="Risk probability (%)", showgrid=False)
    return _style(fig, 360, legend=False).update_layout(title="Probability × impact matrix")


def category_bars(reg: pd.DataFrame):
    d = reg.groupby(["Risk Category", "Rating"]).size().reset_index(name="Risks")
    order = reg.groupby("Risk Category")["Risk Score %"].max().sort_values().index.tolist()
    fig = px.bar(d, y="Risk Category", x="Risks", color="Rating", orientation="h",
                 category_orders={"Rating": RATING_ORDER, "Risk Category": order},
                 color_discrete_map=RATING_COLORS, title="Risks by category and rating")
    fig.update_traces(marker_line_color="white", marker_line_width=2,
                      hovertemplate="%{y}<br>%{fullData.name}: %{x}<extra></extra>")
    return _style(fig, max(320, 26 * len(order) + 80)).update_yaxes(title="")


def allocation_donut(reg: pd.DataFrame):
    d = reg["Related Party"].value_counts().reset_index()
    d.columns = ["Party", "Risks"]
    fig = go.Figure(go.Pie(labels=d["Party"], values=d["Risks"], hole=0.62, sort=False,
                           marker=dict(colors=[PARTY_COLORS.get(p, "#9a9993") for p in d["Party"]],
                                       line=dict(color="white", width=2)),
                           textinfo="label+percent", hovertemplate="%{label}: %{value} risks (%{percent})<extra></extra>"))
    return _style(fig, 320).update_layout(title="Risk allocation (predicted related party)")


def clause_profile(reg: pd.DataFrame):
    d = reg.copy()
    d["Clause label"] = d["Clause"].replace("", "—")
    d["Short"] = d["Sentence"].str.slice(0, 110) + "…"
    fig = px.scatter(d, x="Risk Probability %", y="Risk Score %", color="Rating", size="Impact (1-5)", size_max=16,
                     color_discrete_map=RATING_COLORS, category_orders={"Rating": RATING_ORDER},
                     hover_data={"Clause label": True, "Risk Category": True, "Short": True, "Impact (1-5)": True,
                                 "Rating": False}, title="Risk probability vs. risk score")
    fig.update_traces(marker=dict(line=dict(color="white", width=2), opacity=0.9))
    return _style(fig, 360).update_xaxes(range=[0, 102]).update_yaxes(range=[0, 102])


def emv_bars(reg: pd.DataFrame, currency: str):
    d = reg.groupby("Risk Category")[["EMV", "Residual EMV"]].max().sort_values("EMV")
    fig = go.Figure()
    fig.add_bar(y=d.index, x=d["EMV"], name="EMV before mitigation", orientation="h", marker_color=SERIES[1],
                hovertemplate="%{y}<br>EMV: %{x:,.0f} " + currency + "<extra></extra>")
    fig.add_bar(y=d.index, x=d["Residual EMV"], name="Residual after mitigation", orientation="h", marker_color=SERIES[0],
                hovertemplate="%{y}<br>Residual: %{x:,.0f} " + currency + "<extra></extra>")
    fig.update_layout(barmode="group", title=f"Screening exposure by category ({currency}, largest clause per category)")
    return _style(fig, max(320, 30 * len(d) + 80)).update_xaxes(title=currency)


def predictability_hist(df: pd.DataFrame, threshold: float):
    fig = px.histogram(df, x="Predictability %", nbins=20, title="Predictability of every risk prediction",
                       color_discrete_sequence=[SERIES[0]])
    fig.add_vline(x=threshold, line_dash="dash", line_color="#d03b3b",
                  annotation_text="review threshold", annotation_position="top")
    fig.update_traces(marker_line_color="white", marker_line_width=2)
    return _style(fig, 320, legend=False).update_yaxes(title="Sentences")


def agreement_scatter(df: pd.DataFrame):
    d = df.copy()
    d["Short"] = d["Sentence"].str.slice(0, 100) + "…"
    fig = px.scatter(d, x="Model Agreement %", y="Type Confidence %", color="Sentence Type",
                     color_discrete_sequence=SERIES, hover_data={"Short": True, "Predictability %": True},
                     title="Model agreement vs. confidence (each dot = one sentence)")
    fig.update_traces(marker=dict(size=9, line=dict(color="white", width=1.5)))
    return _style(fig, 340)


def model_votes(probs: dict[str, np.ndarray], labels, title):
    rows = []
    for name, p in probs.items():
        rows.append({"Model": name, "Predicted": labels[int(np.argmax(p))], "Confidence %": 100 * float(np.max(p)),
                     **{l: 100 * float(v) for l, v in zip(labels, p)}})
    d = pd.DataFrame(rows).sort_values("Confidence %")
    long = d.melt(id_vars=["Model"], value_vars=labels, var_name="Class", value_name="Probability %")
    fig = px.bar(long, y="Model", x="Probability %", color="Class", orientation="h",
                 color_discrete_sequence=SERIES, category_orders={"Model": d["Model"].tolist(), "Class": labels},
                 title=title)
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    return _style(fig, max(380, 22 * len(d) + 90)).update_layout(barmode="stack").update_yaxes(title="")


def model_scores(results: pd.DataFrame, title):
    d = results.sort_values("macro_f1")
    lo = d["macro_f1"] - d["macro_f1_ci95"].map(lambda x: x[0])
    hi = d["macro_f1_ci95"].map(lambda x: x[1]) - d["macro_f1"]
    colors = [SERIES[1] if c == "Hybrid" else SERIES[0] for c in d["category"]]
    fig = go.Figure(go.Bar(y=d["model"], x=d["macro_f1"], orientation="h", marker_color=colors,
                           error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color="#9a9993", thickness=1),
                           customdata=np.stack([d["accuracy"], d["category"]], 1),
                           hovertemplate="%{y}<br>Macro-F1 %{x:.3f}<br>Accuracy %{customdata[0]:.3f}<br>%{customdata[1]}<extra></extra>"))
    fig.update_layout(title=title + "  (blue = individual, orange = hybrid; whiskers = 95% bootstrap CI)")
    return _style(fig, max(420, 22 * len(d) + 90), legend=False).update_xaxes(title="Held-out macro-F1", range=[0, 1])


def confusion(cm, labels, title):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=norm, x=labels, y=labels, colorscale=[[i / 6, c] for i, c in enumerate(SEQ_BLUE)],
                               text=cm, texttemplate="%{text}", xgap=2, ygap=2, zmin=0, zmax=1,
                               hovertemplate="True %{y} → predicted %{x}<br>%{text} sentences (%{z:.0%} of row)<extra></extra>",
                               colorbar=dict(title="Row %", thickness=10, tickformat=".0%")))
    fig.update_yaxes(autorange="reversed", title="True label", showgrid=False)
    fig.update_xaxes(title="Predicted label", showgrid=False)
    return _style(fig, 360, legend=False).update_layout(title=title)


def calibration(curve, title):
    d = pd.DataFrame(curve)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration", line=dict(color="#9a9993", dash="dash", width=2))
    if len(d):
        fig.add_scatter(x=d["confidence"], y=d["accuracy"], mode="lines+markers", name="Selected model",
                        line=dict(color=SERIES[0], width=2), marker=dict(size=9, line=dict(color="white", width=2)),
                        customdata=d["count"], hovertemplate="Confidence %{x:.2f}<br>Accuracy %{y:.2f}<br>%{customdata} sentences<extra></extra>")
    fig.update_xaxes(title="Predicted confidence", range=[0, 1.02])
    fig.update_yaxes(title="Observed accuracy", range=[0, 1.02])
    return _style(fig, 340).update_layout(title=title)


def contract_compare(summary: pd.DataFrame):
    fig = px.bar(summary, x="Contract", y="Risk Index", color_discrete_sequence=[SERIES[0]],
                 hover_data=["Risks", "Critical", "High", "Predictability %"], title="Contract Risk Index by contract")
    fig.update_traces(marker_line_color="white", marker_line_width=2)
    return _style(fig, 340, legend=False).update_yaxes(range=[0, 100])


def distribution_bar(reg: pd.DataFrame, col: str, title: str, order=None, colors=None):
    """Old-version style distribution chart: count per value; hover lists clauses, risk IDs and top clause text."""
    d = reg.copy()
    d[col] = d[col].replace("", "Not available").fillna("Not available")
    agg = (d.groupby(col, sort=False)
             .agg(Risks=("Sentence", "size"),
                  Clauses=("Clause", lambda x: ", ".join(sorted({str(v) for v in x if v}))[:120]),
                  IDs=("Library Risk ID", lambda x: ", ".join(str(v) for v in list(x)[:8] if v)),
                  MaxScore=("Risk Score %", "max"),
                  Top=("Sentence", lambda x: str(x.iloc[0])[:160] + "…"))
             .reset_index())
    if order:
        agg[col] = pd.Categorical(agg[col], categories=[o for o in order if o in set(agg[col])] +
                                  [v for v in agg[col] if v not in order], ordered=True)
        agg = agg.sort_values(col)
    else:
        agg = agg.sort_values("Risks", ascending=False)
    if colors:
        mc = [colors.get(str(v), SERIES[0]) for v in agg[col]]
    else:
        mc = SERIES[0]
    fig = go.Figure(go.Bar(
        x=agg[col].astype(str), y=agg["Risks"], marker_color=mc, marker_line_color="white", marker_line_width=2,
        customdata=np.stack([agg["Clauses"], agg["IDs"], agg["MaxScore"].round(1), agg["Top"]], 1),
        hovertemplate=(f"<b>{col}:</b> %{{x}}<br><b>Risks:</b> %{{y}}<br><b>Clauses:</b> %{{customdata[0]}}<br>"
                       "<b>Library risk IDs:</b> %{customdata[1]}<br><b>Highest risk score:</b> %{customdata[2]}%<br>"
                       "<b>Top clause:</b> %{customdata[3]}<extra></extra>")))
    fig.update_xaxes(title="", tickangle=-30 if len(agg) > 5 else 0)
    fig.update_yaxes(title="Number of risks")
    return _style(fig, 340, legend=False).update_layout(title=title)


def risk_ranking(reg: pd.DataFrame, n=20):
    """Old-version 'relevance ranking', now ranked by Risk Score % and coloured by rating."""
    d = reg.head(n).copy().iloc[::-1]
    d["Label"] = [f"#{len(d) - i} · Cl. {c or '—'} · {cat}" for i, (c, cat) in enumerate(zip(d["Clause"], d["Risk Category"]))]
    fig = go.Figure(go.Bar(
        y=d["Label"], x=d["Risk Score %"], orientation="h",
        marker_color=[RATING_COLORS.get(r, SERIES[0]) for r in d["Rating"]], marker_line_color="white", marker_line_width=1,
        customdata=np.stack([d["Rating"], d["Risk Probability %"], d["Library Risk ID"], d["Sentence"].str.slice(0, 200)], 1),
        hovertemplate=("<b>%{y}</b><br>Rating: %{customdata[0]}<br>Risk score: %{x:.1f}%<br>"
                       "Risk probability: %{customdata[1]:.1f}%<br>Library risk: %{customdata[2]}<br>"
                       "%{customdata[3]}<extra></extra>")))
    fig.update_xaxes(title="Risk score (%)", range=[0, 100])
    fig.update_yaxes(title="")
    return _style(fig, max(360, 24 * len(d) + 90), legend=False).update_layout(
        title=f"Risk ranking — top {len(d)} clauses (colour = rating: red Critical, orange High, yellow Medium, green Low)")


def model_distribution(counts: pd.DataFrame, title):
    long = counts.reset_index().melt(id_vars="index", var_name="Class", value_name="Sentences").rename(columns={"index": "Model"})
    fig = px.bar(long, y="Model", x="Sentences", color="Class", orientation="h", color_discrete_sequence=SERIES, title=title)
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    return _style(fig, max(380, 22 * len(counts) + 90)).update_layout(barmode="stack").update_yaxes(title="")


def risk_id_ranking(reg: pd.DataFrame, n=20):
    """Old-version 'Risk-ID Relevance Ranking': one bar per risk, labelled by library Risk ID."""
    d = reg.head(n).copy()
    d["Label"] = [f"{rid or '—'} · Cl. {c or '—'}" for rid, c in zip(d["Library Risk ID"], d["Clause"])]
    d = d.iloc[::-1]
    fig = go.Figure(go.Bar(
        y=d["Label"], x=d["Risk Score %"] / 100, orientation="h",
        marker_color=[RATING_COLORS.get(r, SERIES[0]) for r in d["Rating"]], marker_line_color="white", marker_line_width=1,
        customdata=np.stack([d["Risk Category"], d["Rating"], d["Risk Probability %"], d["Risk Statement"],
                             d["Sentence"].str.slice(0, 220)], 1),
        hovertemplate=("<b>Risk ID:</b> %{y}<br><b>Category:</b> %{customdata[0]}<br><b>Rating:</b> %{customdata[1]}<br>"
                       "<b>Relevance score:</b> %{x:.3f}<br><b>Risk probability:</b> %{customdata[2]:.1f}%<br>"
                       "<b>Risk statement:</b> %{customdata[3]}<br><b>Contract clause:</b> %{customdata[4]}<extra></extra>")))
    fig.update_xaxes(title="Relevance score (risk score ÷ 100)", range=[0, 1])
    fig.update_yaxes(title="Risk ID")
    return _style(fig, max(380, 24 * len(d) + 90), legend=False).update_layout(
        title="Risk-ID relevance ranking (colour = rating)")


# ----------------------------------------------------------------------------- model performance dashboard
PERF_METRICS = {"accuracy": "Accuracy", "macro_precision": "Macro Precision", "macro_recall": "Macro Recall",
                "macro_f1": "Macro-F1", "roc_auc_ovr": "ROC-AUC"}
METRIC_COLORS = dict(zip(PERF_METRICS.values(), SERIES[:5]))


def perf_frame(results: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(results)
    out = d[["model", "category"] + list(PERF_METRICS)].rename(columns={**PERF_METRICS, "model": "Model", "category": "Type"})
    for c in PERF_METRICS.values():
        out[c] = (100 * out[c]).round(2)
    out["Mean of 5 metrics"] = out[list(PERF_METRICS.values())].mean(1).round(2)
    return out.sort_values("Macro-F1", ascending=False).reset_index(drop=True)


def perf_heatmap(pf: pd.DataFrame, title):
    d = pf.iloc[::-1]
    cols = list(PERF_METRICS.values())
    z = d[cols].values
    lo = float(np.nanmin(z))
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=[f"{'◆ ' if t == 'Hybrid' else ''}{m}" for m, t in zip(d["Model"], d["Type"])],
        colorscale=[[i / 6, c] for i, c in enumerate(SEQ_BLUE)], zmin=max(0, lo - 2), zmax=100, xgap=2, ygap=2,
        text=np.round(z, 1), texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>", colorbar=dict(title="%", thickness=10)))
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig = _style(fig, max(480, 24 * len(d) + 140), legend=False)
    return fig.update_layout(title=dict(text=title + "  (◆ = hybrid; darker = better)", y=0.995, yanchor="top"),
                             margin=dict(t=110))


def perf_grouped(pf: pd.DataFrame, models, title):
    d = pf[pf["Model"].isin(models)]
    long = d.melt(id_vars=["Model", "Type"], value_vars=list(PERF_METRICS.values()), var_name="Metric", value_name="Score %")
    fig = px.bar(long, x="Model", y="Score %", color="Metric", barmode="group", color_discrete_map=METRIC_COLORS,
                 category_orders={"Model": list(d["Model"]), "Metric": list(PERF_METRICS.values())}, title=title,
                 hover_data={"Type": True})
    fig.update_traces(marker_line_color="white", marker_line_width=1.5,
                      hovertemplate="%{x}<br>%{fullData.name}: %{y:.2f}%<extra></extra>")
    lo = float(long["Score %"].min())
    fig.update_yaxes(range=[max(0, lo - 10), 100], title="Score (%)")
    fig.update_xaxes(title="", tickangle=-25)
    return _style(fig, 440)


def metric_ranking(pf: pd.DataFrame, metric, title):
    d = pf.sort_values(metric)
    fig = go.Figure(go.Bar(
        y=d["Model"], x=d[metric], orientation="h",
        marker_color=[SERIES[1] if t == "Hybrid" else SERIES[0] for t in d["Type"]],
        marker_line_color="white", marker_line_width=1,
        customdata=d["Type"], hovertemplate="%{y}<br>" + metric + ": %{x:.2f}%<br>%{customdata}<extra></extra>",
        text=d[metric].map(lambda v: f"{v:.1f}"), textposition="outside", cliponaxis=False))
    lo = float(d[metric].min())
    fig.update_xaxes(range=[max(0, lo - 10), 101], title=f"{metric} (%)")
    fig.update_yaxes(title="")
    return _style(fig, max(440, 22 * len(d) + 90), legend=False).update_layout(
        title=f"{title} — ranked by {metric} (blue = individual, orange = hybrid)")


def family_summary(pf: pd.DataFrame, title):
    cols = list(PERF_METRICS.values())
    rows = []
    for t in ["Individual", "Hybrid"]:
        sub = pf[pf["Type"] == t]
        rows += [{"Group": f"Best {t.lower()}", "Metric": c, "Score %": sub[c].max()} for c in cols]
        rows += [{"Group": f"Mean {t.lower()}", "Metric": c, "Score %": sub[c].mean()} for c in cols]
    d = pd.DataFrame(rows)
    colors = {"Best individual": "#184f95", "Mean individual": "#86b6ef", "Best hybrid": "#c2491d", "Mean hybrid": "#f2a27f"}
    fig = px.bar(d, x="Metric", y="Score %", color="Group", barmode="group", color_discrete_map=colors, title=title,
                 category_orders={"Group": list(colors), "Metric": cols})
    fig.update_traces(marker_line_color="white", marker_line_width=1.5,
                      hovertemplate="%{fullData.name}<br>%{x}: %{y:.2f}%<extra></extra>")
    fig.update_yaxes(range=[max(0, float(d["Score %"].min()) - 10), 100], title="Score (%)")
    return _style(fig, 380)


def perf_scatter(pf: pd.DataFrame, title):
    fig = px.scatter(pf, x="Accuracy", y="Macro-F1", color="Type", symbol="Type", hover_data={"Model": True,
                     "ROC-AUC": ":.2f", "Macro Precision": ":.2f", "Macro Recall": ":.2f"},
                     color_discrete_map={"Individual": SERIES[0], "Hybrid": SERIES[1]}, title=title)
    fig.update_traces(marker=dict(size=12, line=dict(color="white", width=2)))
    return _style(fig, 380).update_xaxes(title="Accuracy (%)").update_yaxes(title="Macro-F1 (%)")


TYPE_COLORS = {"Obligation": SERIES[0], "Risk": SERIES[1], "Right": SERIES[2], "Definition": SERIES[3], "Heading": SERIES[4]}


def dataset_counts(counts: pd.Series, title: str, colors=None, axis="Sentences"):
    d = counts.reset_index()
    d.columns = ["Label", "Count"]
    fig = go.Figure(go.Bar(x=d["Label"].astype(str), y=d["Count"],
                           marker_color=[(colors or {}).get(str(l), SERIES[0]) for l in d["Label"]],
                           marker_line_color="white", marker_line_width=2, text=d["Count"], textposition="outside",
                           cliponaxis=False, hovertemplate="%{x}: %{y} sentences<extra></extra>"))
    fig.update_yaxes(title=axis)
    return _style(fig, 320, legend=False).update_layout(title=title)


def dataset_by_book(clean: pd.DataFrame, title: str):
    d = clean.groupby(["Book", "Sentence_Type"]).size().reset_index(name="Sentences")
    fig = px.bar(d, x="Book", y="Sentences", color="Sentence_Type", color_discrete_map=TYPE_COLORS,
                 category_orders={"Sentence_Type": list(TYPE_COLORS)}, title=title)
    fig.update_traces(marker_line_color="white", marker_line_width=1.5,
                      hovertemplate="%{x}<br>%{fullData.name}: %{y}<extra></extra>")
    return _style(fig, 340).update_layout(barmode="stack")


def cleaning_funnel(audit: dict, title: str):
    raw, kept = audit["raw_rows"], audit["kept_rows"]
    frag, other = audit.get("dropped_line_fragments", 0), audit["dropped_total"] - audit.get("dropped_line_fragments", 0)
    steps = [("Raw rows", raw), ("OCR / non-English / duplicates removed", -other), ("Line fragments removed", -frag),
             ("Clean training rows", kept)]
    fig = go.Figure(go.Waterfall(x=[s[0] for s in steps], y=[s[1] for s in steps],
                                 measure=["absolute", "relative", "relative", "total"],
                                 decreasing=dict(marker=dict(color="#e34948")), totals=dict(marker=dict(color=SERIES[0])),
                                 increasing=dict(marker=dict(color=SERIES[0])), text=[f"{abs(s[1]):,}" for s in steps],
                                 textposition="outside", cliponaxis=False, connector=dict(line=dict(color=GRID))))
    fig.update_yaxes(title="Rows")
    return _style(fig, 340, legend=False).update_layout(title=title)
