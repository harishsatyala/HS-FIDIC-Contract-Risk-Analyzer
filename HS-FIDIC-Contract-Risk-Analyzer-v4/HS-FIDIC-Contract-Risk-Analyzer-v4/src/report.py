"""Full analysis report: one self-contained, interactive HTML file covering every dashboard tab.

The report embeds Plotly once (offline-capable), then every chart, table and explanation. Open it in any browser;
use the browser's Print → Save as PDF for a PDF copy (charts print as shown).
"""
from __future__ import annotations

import datetime as dt
import html

import numpy as np
import pandas as pd

from . import charts
from . import risk_engine as engine
from .model_zoo import HYBRID_INFO

CSS = """
body{font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;color:#0b0b0b;margin:0;background:#fcfcfb}
.wrap{max-width:1180px;margin:0 auto;padding:28px 34px 60px}
h1{font-size:30px;margin:0 0 4px} h2{font-size:22px;margin:42px 0 6px;padding-top:14px;border-top:2px solid #e6e5e1}
h3{font-size:16px;margin:24px 0 6px} p,li{font-size:14px;line-height:1.55;color:#2b2b2b}
.sub{color:#52514e;font-size:13px} .note{background:#eef4fc;border-left:4px solid #2a78d6;padding:10px 14px;margin:12px 0;font-size:13.5px}
.warn{background:#fff6e0;border-left:4px solid #eda100;padding:10px 14px;margin:12px 0;font-size:13.5px}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:14px 0}
.kpi{background:#fff;border:1px solid #e6e5e1;border-radius:8px;padding:10px 12px}
.kpi .l{font-size:11.5px;color:#52514e} .kpi .v{font-size:21px;font-weight:600;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0 14px;background:#fff}
th{background:#f3f4f6;text-align:left;padding:6px 7px;border:1px solid #e6e5e1;position:sticky;top:0}
td{padding:5px 7px;border:1px solid #e6e5e1;vertical-align:top}
.tbl{max-height:560px;overflow:auto;border:1px solid #e6e5e1;border-radius:6px}
.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-weight:600;font-size:11.5px}
.Critical{background:#f8d7d7;border:1px solid #d03b3b}.High{background:#fde4d8;border:1px solid #ec835a}
.Medium{background:#fff4d6;border:1px solid #fab219}.Low{background:#e9f5e9;border:1px solid #0ca30c}
.card{background:#fff;border:1px solid #e6e5e1;border-radius:8px;padding:12px 16px;margin:10px 0;page-break-inside:avoid}
.clause{border-left:3px solid #2a78d6;background:#f6f8fb;padding:6px 10px;font-size:13px;margin:6px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px} .toc a{color:#256abf;text-decoration:none}
.toc li{font-size:14px} .fig{background:#fff;border:1px solid #e6e5e1;border-radius:8px;margin:8px 0;padding:4px}
@media print{.tbl{max-height:none;overflow:visible} h2{page-break-before:always} .grid2{grid-template-columns:1fr}}
"""


class _Builder:
    def __init__(self):
        self.parts = []
        self.first_fig = True

    def add(self, s):
        self.parts.append(s)

    def h(self, level, text, anchor=None):
        a = f' id="{anchor}"' if anchor else ""
        self.add(f"<h{level}{a}>{html.escape(text)}</h{level}>")

    def p(self, text, cls=None):
        c = f' class="{cls}"' if cls else ""
        self.add(f"<p{c}>{text}</p>")

    def note(self, text):
        self.add(f'<div class="note">{text}</div>')

    def warn(self, text):
        self.add(f'<div class="warn">{text}</div>')

    def fig(self, fig):
        try:
            div = fig.to_html(full_html=False, include_plotlyjs=True if self.first_fig else False,
                              config={"displaylogo": False, "responsive": True})
            self.first_fig = False
            return f'<div class="fig">{div}</div>'
        except Exception as exc:  # never let one chart break the report
            return f'<p class="sub">Chart unavailable: {html.escape(str(exc))}</p>'

    def figs(self, *figs):
        if len(figs) == 1:
            self.add(self.fig(figs[0]))
        else:
            self.add('<div class="grid2">' + "".join(self.fig(f) for f in figs) + "</div>")

    def table(self, df: pd.DataFrame, max_rows=None, fmt=None):
        d = df.copy()
        if max_rows and len(d) > max_rows:
            d = d.head(max_rows)
            extra = f'<p class="sub">Showing first {max_rows} of {len(df)} rows (full data in the Excel/CSV exports).</p>'
        else:
            extra = ""
        for c, f in (fmt or {}).items():
            if c in d:
                d[c] = d[c].map(lambda v, f=f: f.format(v) if pd.notna(v) and v != "" else "")
        if "Rating" in d:
            d["Rating"] = d["Rating"].map(lambda r: f'<span class="badge {r}">{r}</span>' if r in charts.RATING_COLORS else r)
        body = d.to_html(index=False, escape=False, border=0, na_rep="")
        self.add(f'<div class="tbl">{body}</div>{extra}')

    def kpis(self, items):
        self.add('<div class="kpis">' + "".join(
            f'<div class="kpi"><div class="l">{html.escape(l)}</div><div class="v">{html.escape(str(v))}</div></div>'
            for l, v in items) + "</div>")

    def html(self, title):
        return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
                f"<style>{CSS}</style></head><body><div class='wrap'>" + "\n".join(self.parts) + "</div></body></html>")


def _esc(x):
    return html.escape(str(x))


def model_status(prob_dict, labs, sel_idx, n, accuracy, describe):
    rows = []
    for name, P in prob_dict.items():
        P = np.asarray(P)
        pred = P.argmax(1)
        dist = pd.Series(pred).map(dict(enumerate(labs))).value_counts()
        rows.append({"Model": name, "Method": describe(name), "Sentences classified": len(P),
                     "Status": "Completed" if len(P) == n else "Incomplete",
                     "Risk": int((pred == labs.index("Risk")).sum()), "Obligation": int((pred == labs.index("Obligation")).sum()),
                     "Right": int((pred == labs.index("Right")).sum()), "Dominant type": dist.index[0] if len(dist) else "—",
                     "Avg decision strength %": round(100 * float(P.max(1).mean()), 1),
                     "Agreement with selected %": round(100 * float((pred == sel_idx).mean()), 1),
                     "Held-out accuracy %": round(100 * accuracy[name], 1) if accuracy.get(name) is not None else None})
    return pd.DataFrame(rows)


def build_report(*, sel, df, reg, summ, info, votes, suite, metrics, settings, results, make_register,
                 numbers_guide, method_md, currency="EUR") -> str:
    b = _Builder()
    now = dt.datetime.now().strftime("%d %B %Y, %H:%M")
    type_model, party_model = settings["type_model"], settings["party_model"]

    def money(x):
        return f"{currency} {x:,.0f}"

    # ------------------------------------------------------------------ cover
    b.add(f"<h1>HS-FIDIC Contract Risk Analysis Report</h1>")
    b.p(f"<b>{_esc(sel)}</b> · {info.get('pages', '?')} page(s) · {summ['sentences']} sentences analysed · generated {now}", "sub")
    b.p(f"Perspective <b>{_esc(settings['perspective'])}</b> · sentence-type model <b>{_esc(type_model)}</b> · "
        f"related-party model <b>{_esc(party_model)}</b> · risk threshold {settings['min_prob']}% · "
        f"review threshold {settings['review_thr']}% · TCV {money(settings['value'])} · mitigation effectiveness "
        f"{settings['mit_pct']}% · context: {_esc(', '.join(f'{k}={v}' for k, v in settings['context'].items()))}", "sub")
    b.add('<ol class="toc">' + "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in [
        ("summary", "Executive summary"), ("dashboard", "Risk dashboard"), ("mitigation", "Mitigation plans"),
        ("execution", "Model execution"), ("predictability", "Predictability"), ("performance", "Model performance"),
        ("clauses", "Clause explorer"), ("compare", "Compare contracts"), ("method", "Method")]) + "</ol>")

    # ------------------------------------------------------------------ 0 executive summary
    b.h(2, "1. Executive summary", "summary")
    if len(reg):
        cats = reg.groupby("Risk Category")["Risk Score %"].max().sort_values(ascending=False)
        party = reg["Related Party"].value_counts()
        b.p(f"The analysis identified <b>{summ['risks']} risk-bearing clauses</b> across <b>{summ['categories']} risk "
            f"categories</b>: <b>{summ['critical']} Critical</b> and <b>{summ['high']} High</b>. The Contract Risk Index "
            f"(mean of the ten highest risk scores) is <b>{summ['risk_index']:.0f}/100</b> and the mean risk probability of "
            f"identified risks is <b>{summ['mean_probability']:.0f}%</b>. The most exposed categories are "
            f"{', '.join(f'<b>{_esc(c)}</b> ({v:.0f})' for c, v in cats.head(4).items())}. "
            f"By the ML party model, {', '.join((f'{v} are shared' if k == 'Shared' else f'{v} fall on the {_esc(k)}') for k, v in party.items())}.")
        b.p(f"Screening exposure is <b>{money(summ['emv'])}</b> expected monetary value, falling to <b>{money(summ['residual'])}</b> "
            f"after mitigation (largest clause per category, so overlapping clauses are not double-counted). Mean prediction "
            f"predictability is <b>{summ['predictability']:.0f}%</b>; <b>{summ['needs_review']}</b> registered risk(s) fall below "
            f"the review threshold and should be checked by a contract specialist.")
        b.h(3, "Five highest-priority risks")
        b.add("<ol>" + "".join(
            f"<li><span class='badge {r['Rating']}'>{r['Rating']}</span> <b>{r['Risk Score %']:.0f}%</b> score · "
            f"{_esc(r['Risk Category'])} · clause {_esc(r['Clause'] or '—')} — <i>{_esc(r['Sentence'] if len(r['Sentence']) <= 260 else r['Sentence'][:260].rsplit(' ', 1)[0] + '…')}</i></li>"
            for _, r in reg.head(5).iterrows()) + "</ol>")
    else:
        b.p("No clause exceeded the risk-probability threshold.")

    # ------------------------------------------------------------------ 1 dashboard
    b.h(2, "2. Risk dashboard", "dashboard")
    b.h(3, "Risk analysis summary")
    b.kpis([("Risks identified", summ["risks"]), ("Risk categories", summ["categories"]), ("Critical", summ["critical"]),
            ("High", summ["high"]), ("Expected monetary value", money(summ["emv"])), ("Residual after mitigation", money(summ["residual"]))])
    b.figs(charts.gauge(summ["risk_index"], "Contract Risk Index"), charts.gauge(summ["mean_probability"], "Mean risk probability"))
    b.p("The <b>Contract Risk Index</b> averages the ten highest risk scores; above 50 means several clauses would be rated "
        "Critical. The <b>mean risk probability</b> is the average ML + rule probability that a registered clause carries risk.")
    if len(reg):
        b.h(3, "Risk distribution")
        b.p("How the identified risks spread over the risk-library categories, the library's likelihood level (L/M/H), the "
            "impact areas, the responsible party predicted by the ML party model, and the rating bands. The matrix crosses "
            "risk probability with impact (1–5).")
        b.figs(charts.distribution_bar(reg, "Risk Category", "Risk Category Distribution"),
               charts.distribution_bar(reg, "Probability (L/M/H)", "Probability Distribution (L / M / H)", order=["H", "M", "L"]))
        b.figs(charts.distribution_bar(reg, "Impact Areas", "Impact Distribution"),
               charts.distribution_bar(reg, "Related Party", "Responsible Party Distribution (ML-predicted)",
                                       order=["Contractor", "Employer", "Shared"], colors=charts.PARTY_COLORS))
        b.figs(charts.distribution_bar(reg, "Rating", "Risk Rating Distribution", order=charts.RATING_ORDER,
                                       colors=charts.RATING_COLORS), charts.heat_matrix(reg))
        b.h(3, "Risk-ID relevance ranking")
        b.p("Top 20 clauses ranked by relevance score (risk score ÷ 100), labelled with the matched risk-library ID; colour = rating.")
        b.figs(charts.risk_id_ranking(reg, 20))
        b.h(3, f"Top {min(20, len(reg))} contract risks")
        b.add(f'<div class="note">{_md(numbers_guide)}</div>')
        b.table(reg.head(20)[["Rating", "Risk Score %", "Risk Probability %", "Clause", "Page", "Sentence", "Sentence Type",
                              "Library Risk ID", "Risk Statement", "Risk Category", "Type Confidence %", "Responsible Party (library)",
                              "Related Party", "Party Confidence %", "Probability (L/M/H)", "Probability Factor", "Impact Areas",
                              "Impact (1-5)", "Gross Exposure", "EMV", "Residual EMV", "Risk Owner", "Predictability %"]]
                .rename(columns={"Sentence": "Contract Sentence", "Library Risk ID": "Risk ID", "Type Confidence %": "Decision Strength %",
                                 "Related Party": "Predicted Party", "Responsible Party (library)": "Library Party"}),
                fmt={"Gross Exposure": "{:,.0f}", "EMV": "{:,.0f}", "Residual EMV": "{:,.0f}"})

    # ------------------------------------------------------------------ 2 mitigation
    b.h(2, "3. Mitigation plans", "mitigation")
    if len(reg):
        b.p("Each card combines trigger-specific advice (from the onerous wording detected in the clause), the category "
            "playbook with its FIDIC orientation, and the matched risk-library record (owner, early-warning indicators, "
            "library mitigation). EMV is reduced by the mitigation-effectiveness assumption to give the residual.")
        b.figs(charts.emv_bars(reg, currency))
        for i, r in reg.head(20).iterrows():
            mit = "".join(f"<li>{_esc(x.strip().rstrip('.'))}.</li>" for x in r["Mitigation"].split(". ") if x.strip())
            b.add(f"""<div class="card"><b>#{i + 1} <span class="badge {r['Rating']}">{r['Rating']}</span>
            {r['Risk Score %']:.0f}% score · {_esc(r['Risk Category'])} · clause {_esc(r['Clause'] or '—')} (p.{_esc(r['Page'])})</b>
            <div class="clause">{_esc(r['Sentence'])}</div>
            <p class="sub">Risk probability {r['Risk Probability %']:.0f}% · impact {r['Impact (1-5)']}/5 · allocated to you
            {r['Allocation to you %']:.0f}% · predictability {r['Predictability %']:.0f}% · EMV {money(r['EMV'])} → residual
            {money(r['Residual EMV'])}</p>
            <b>Recommended mitigation</b><ul>{mit}</ul>
            <p class="sub"><b>Triggers:</b> {_esc(r['Triggers'] or '—')} · <b>FIDIC orientation:</b> {_esc(r['FIDIC Reference'])}<br>
            <b>Library match:</b> {_esc(r['Library Risk ID'])} ({_esc(r['Probability (L/M/H)'])}, factor {_esc(r['Probability Factor'])})
            · <b>Owner:</b> {_esc(r['Risk Owner'])}<br><b>Early-warning indicators:</b> {_esc(r['Early Warning Indicators'])}<br>
            <b>Library mitigation:</b> {_esc(r['Library Mitigation'])}</p></div>""")
    else:
        b.p("No registered risks at the current threshold.")

    # ------------------------------------------------------------------ 3 model execution
    b.h(2, "4. Model execution", "execution")
    labs = suite.type.labels
    sel_P = votes["type_hybrid"].get(type_model, votes["type"].get(type_model))
    sel_idx = np.asarray(sel_P).argmax(1)
    algo = {r["model"]: f"{r['vectorizer']} + {r['algorithm']}" for r in metrics.get("sentence_type", {}).get("results", [])
            if r.get("category") == "Individual"}
    b.p(f"All {len(votes['type'])} individual models and all {len(votes['type_hybrid'])} hybrid models classified every one "
        f"of the {len(df)} sentences. <b>Decision strength</b> is the mean probability of each model's predicted class (not "
        f"accuracy); <b>agreement</b> is the share of sentences where the model agrees with the selected model "
        f"(<b>{_esc(type_model)}</b>).")
    b.h(3, "Individual models")
    b.table(model_status(votes["type"], labs, sel_idx, len(df), suite.type.model_accuracy, lambda n: algo.get(n, "")))
    b.h(3, "Hybrid models")
    b.table(model_status(votes["type_hybrid"], labs, sel_idx, len(df), suite.type.model_accuracy,
                         lambda n: HYBRID_INFO.get(n, "")))
    counts = pd.DataFrame({n: pd.Series(np.asarray(v).argmax(1)).map(dict(enumerate(labs))).value_counts()
                           for n, v in {**votes["type"], **votes["type_hybrid"]}.items()}).T.fillna(0).astype(int)
    b.figs(charts.model_distribution(counts.reindex(columns=labs, fill_value=0),
                                     "How every model classifies this contract (sentences per type)"))
    b.h(3, "Hybrid strategies")
    b.add("<ul>" + "".join(f"<li><b>{_esc(k.replace('Hybrid: ', ''))}</b> — {_esc(v)}</li>" for k, v in HYBRID_INFO.items()) + "</ul>")

    # ------------------------------------------------------------------ 4 predictability
    b.h(2, "5. Predictability", "predictability")
    clauses = df[df["Sentence Type"] != "Heading"]
    b.p("Predictability % = 35% model agreement + 25% confidence + 15% (1 − normalised entropy) + 25% historical precision "
        "of the selected model for the predicted class on the held-out test set. Clauses below the review threshold "
        f"({settings['review_thr']}%) are flagged for expert review.")
    b.kpis([("Mean predictability", f"{clauses['Predictability %'].mean():.0f}%"),
            ("Mean model agreement", f"{clauses['Model Agreement %'].mean():.0f}%"),
            ("Clauses needing review", int(clauses["Needs Review"].sum())),
            ("Selected model accuracy", f"{100 * suite.type.model_accuracy.get(type_model, 0):.1f}%"),
            ("Clauses analysed", len(clauses)), ("Risk clauses", len(reg))])
    b.figs(charts.predictability_hist(clauses, settings["review_thr"]), charts.agreement_scatter(clauses))
    res_t = {r["model"]: r for r in metrics.get("sentence_type", {}).get("results", [])}
    if type_model in res_t:
        r = res_t[type_model]
        b.figs(charts.calibration(r["calibration_curve"], f"Calibration on held-out test — {type_model}"))
        b.table(pd.DataFrame(r["per_class"]).T.reset_index().rename(columns={"index": "Class"}))
    low = clauses[clauses["Needs Review"]].sort_values("Predictability %")
    b.h(3, f"Clauses flagged for expert review ({len(low)})")
    b.table(low[["Predictability %", "Model Agreement %", "Type Confidence %", "Sentence Type", "Risk Probability %",
                 "Clause", "Sentence"]], max_rows=150)

    # ------------------------------------------------------------------ 5 model performance
    b.h(2, "6. Model performance", "performance")
    b.p("Held-out test-set scores (%) for every individual and hybrid model: Accuracy, Macro Precision, Macro Recall, "
        "Macro-F1 and ROC-AUC (one-vs-rest). The default model is chosen by 5-fold cross-validated Macro-F1 on the "
        "development set, never by these test scores. Uploaded contracts have no sentence-level ground truth.")
    for key, title in (("sentence_type", "Sentence type"), ("related_party", "Related party")):
        if key not in metrics:
            continue
        m = metrics[key]
        pf = charts.perf_frame(m["results"])
        ind, hyb = pf[pf["Type"] == "Individual"], pf[pf["Type"] == "Hybrid"]
        b.h(3, f"{title} — {len(ind)} individual + {len(hyb)} hybrid models (test n={m['n_test']})")
        b.p(f"Best individual: <b>{_esc(ind.iloc[0]['Model'])}</b> ({ind.iloc[0]['Macro-F1']:.1f}% Macro-F1). Best hybrid: "
            f"<b>{_esc(hyb.iloc[0]['Model'])}</b> ({hyb.iloc[0]['Macro-F1']:.1f}%). Deployed default: "
            f"<b>{_esc(m['default_model'])}</b>.")
        b.figs(charts.family_summary(pf, f"{title}: individual vs hybrid — best and mean per metric"))
        b.figs(charts.perf_heatmap(pf, f"{title}: all models × all five metrics (%)"))
        b.figs(charts.metric_ranking(pf, "Macro-F1", title),
               charts.perf_scatter(pf, f"{title}: accuracy vs Macro-F1"))
        top = list(dict.fromkeys(list(hyb["Model"].head(4)) + list(ind["Model"].head(3))))
        b.figs(charts.perf_grouped(pf, top, f"{title}: top hybrids vs top individual models"))
        b.table(pf)
    b.warn("<br>".join(_esc(c) for c in metrics.get("caveats", [])))

    # ------------------------------------------------------------------ 6 clause explorer
    b.h(2, "7. Clause explorer", "clauses")
    tc = df["Sentence Type"].value_counts().reindex(labs, fill_value=0)
    pc = df.loc[df["Related Party"] != "—", "Related Party"].value_counts()
    b.p(f"Sentence types: {', '.join(f'{k} {v}' for k, v in tc.items())}. Related parties (obligations, risks, rights): "
        f"{', '.join(f'{k} {v}' for k, v in pc.items())}.")
    cl = df[df["Sentence Type"].isin(["Risk", "Right", "Obligation"])]
    b.table(cl[["Clause", "Page", "Sentence Type", "Type Confidence %", "Related Party", "Party Confidence %",
                "Risk Probability %", "Rating", "Risk Category", "Triggers", "Predictability %", "Sentence"]]
            .sort_values("Risk Probability %", ascending=False), max_rows=400)

    # ------------------------------------------------------------------ 7 compare
    b.h(2, "8. Compare contracts", "compare")
    rows = []
    for n, (d, _, _) in results.items():
        rg = make_register(d)
        s = engine.contract_summary(d, rg)
        rows.append({"Contract": n, "Risk Index": s["risk_index"], "Risks": s["risks"], "Critical": s["critical"],
                     "High": s["high"], "Mean probability %": s["mean_probability"], "Predictability %": s["predictability"],
                     "EMV": s["emv"], "Residual EMV": s["residual"], "Needs review": s["needs_review"]})
    comp = pd.DataFrame(rows)
    if len(comp) > 1:
        b.figs(charts.contract_compare(comp))
        allreg = pd.concat([make_register(d).assign(Contract=n) for n, (d, _, _) in results.items()])
        if len(allreg):
            piv = allreg.pivot_table(index="Risk Category", columns="Contract", values="Risk Score %", aggfunc="max").fillna(0).round(0)
            b.table(comp, fmt={"EMV": "{:,.0f}", "Residual EMV": "{:,.0f}"})
            b.h(3, "Highest risk score per category and contract")
            b.table(piv.reset_index())
    else:
        b.p("Only one contract was analysed; upload several contracts to compare them.")
        b.table(comp, fmt={"EMV": "{:,.0f}", "Residual EMV": "{:,.0f}"})

    # ------------------------------------------------------------------ 8 method
    b.h(2, "9. Method", "method")
    b.add(_md(method_md))
    b.p("HS-FIDIC research prototype · decision support only, not legal advice · thesis basis: Eken (2022), METU.", "sub")
    return b.html(f"HS-FIDIC report — {sel}")


def _md(text: str) -> str:
    """Tiny markdown → HTML for the guide / method texts (bold, italics, bullets, paragraphs)."""
    import re

    out, in_list = [], False
    for block in text.strip().split("\n\n"):
        lines = [l.rstrip() for l in block.strip().splitlines()]
        if all(l.lstrip().startswith("- ") or l.startswith("  ") for l in lines if l):
            items, cur = [], ""
            for l in lines:
                if l.lstrip().startswith("- "):
                    if cur:
                        items.append(cur)
                    cur = l.lstrip()[2:]
                else:
                    cur += " " + l.strip()
            items.append(cur)
            out.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in items) + "</ul>")
        else:
            out.append(f"<p>{_inline(' '.join(lines))}</p>")
    return "\n".join(out)


def _inline(s):
    import re

    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    return s
