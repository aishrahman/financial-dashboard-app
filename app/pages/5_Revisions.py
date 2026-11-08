"""Revisions: what companies first reported, against what they report now.

A revision is not an error. Most are recasts — a new accounting standard
adopted retrospectively, a business moved to discontinued operations, a change
of pension method — and the page says so before it shows anything, because a
ranking of "largest restatements" reads as an accusation whatever its caption.

The figure that separates a recast from something worth a closer look is net
income. A reclassification moves amounts between lines and leaves the bottom
line where it was, so the page leads with revisions to net income and keeps
every other line one click away.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import charts, data, filters, theme
from app.format import currency

st.set_page_config(page_title="Revisions", page_icon="✏️", layout="wide")

if not filters.guard_dataset():
    st.stop()

revisions = data.revisions()
universe = data.companies()
names = dict(zip(universe["ticker"], universe["name"]))

st.title("Revisions")
st.markdown(
    "What each company **first reported** for a period, against what it "
    "reports for that period now. Most revisions are recasts rather than "
    "corrections — a new accounting standard adopted retrospectively, a "
    "business moved to discontinued operations, a change of pension method. "
    "Net income is the line that tells them apart: a reclassification moves "
    "amounts between lines and leaves the bottom line where it was."
)

if revisions.empty:
    st.info("No revision history in this build. It needs the extractor's "
            "restatement file; see the README.")
    st.stop()

revisions = revisions.assign(period_end=pd.to_datetime(revisions["period_end"]),
                             revised_filed=pd.to_datetime(revisions["revised_filed"]),
                             first_filed=pd.to_datetime(revisions["first_filed"]))
annual = revisions[revisions["frequency"] == "annual"]
bottom = annual[annual["item"] == "net_income"].copy()
bottom["year"] = bottom["period_end"].dt.year

cols = st.columns(4)
cols[0].metric("Material revisions", "{:,}".format(len(revisions)),
               help="Changes of at least 0.1% to a value the filer reported. "
                    "Smaller changes are rounding.")
cols[1].metric("Companies revised", "{} of {}".format(
    revisions["ticker"].nunique(), universe["ticker"].nunique()))
cols[2].metric("Annual net income revised", "{:,}".format(len(bottom)),
               help="Company-years whose bottom line changed after it was "
                    "first filed.")
cols[3].metric("Companies affected", "{:,}".format(bottom["ticker"].nunique()))

# ---- The bottom line -----------------------------------------------------
st.subheader("Revisions to annual net income")
st.caption(
    # Dollar signs escaped: Streamlit reads a pair of them as LaTeX, and
    # rendered everything between two of them as a maths expression.
    "The ring is net income as first filed, the dot as reported now, so the "
    "line between them is the revision; red where it lowered profit. Ranked by "
    "size against the first figure. Verizon's 2008 was recast from \\$6.4bn to "
    "a \\$2.2bn loss when it changed pension accounting — a real recast, in the "
    "rules, and the largest kind there is."
)

top_n = st.slider("Show the largest", 10, max(10, min(40, len(bottom))),
                  min(20, len(bottom)), step=5, key="revisions_top_n")
ranked = bottom.reindex(bottom["pct_change"].abs().sort_values(ascending=False).index).head(top_n)
ranked = ranked.assign(case=ranked["ticker"] + " " + ranked["year"].astype(str) + "  "
                       + ranked["pct_change"].map(lambda v: "{:+.0%}".format(v)))
down, neutral = theme.tone_colour("negative"), theme.palette(charts.mode())[0]
figure = charts.dumbbell(
    ranked, label="case", start="first_value", end="value", kind="currency",
    height=max(340, 26 * len(ranked)),
    colours=[down if v < 0 else neutral for v in ranked["change"]],
    start_name="As first filed", end_name="As reported now",
    title="Annual net income, first filed and now",
)
# A reverse merger rewrites a registrant's past with another company's: after
# MetroPCS took T-Mobile USA in 2013, the registrant's earlier years became
# T-Mobile's. That is a change of entity, not of numbers, and the largest
# "revisions" in the data are this. Named rather than removed, because
# removing them would need a rule for spotting one, and there is not a safe one.
if ranked["ticker"].eq("TMUS").any():
    st.caption(
        "T-Mobile's years before 2013 are MetroPCS's as first filed and T-Mobile "
        "USA's as reported now: the 2013 merger was a reverse acquisition, so "
        "the registrant's history was replaced with the accounting acquirer's. "
        "A change of company, not a correction of its numbers.")
charts.render(figure, ranked.assign(
    Company=ranked["ticker"].map(lambda t: "{} · {}".format(t, names.get(t, t))),
    **{"Fiscal year": ranked["fiscal_label"],
       "First filed": ranked["first_value"].map(currency),
       "As revised": ranked["value"].map(currency),
       "Change": (ranked["pct_change"] * 100).round(1).astype(str) + "%",
       "Revised in": ranked["revised_filed"].dt.strftime("%b %Y")},
)[["Company", "Fiscal year", "First filed", "As revised", "Change", "Revised in"]],
    key="revisions_ranked")

# ---- When ----------------------------------------------------------------
st.subheader("When revisions happen")
# Measured, not assumed: an earlier draft of this caption said revisions
# "cluster where standards changed", and the chart beside it showed a flat
# run of 400-1,000 a year. The clustering is real only for net income. The
# years are those of the filing that CHANGED the value (revised_filed); dated
# by the last filing to repeat it, an insurer recast in 2024 also showed up
# in 2025 and the 2013 one was hidden entirely.
dated = revisions.dropna(subset=["revised_filed"]).assign(
    year=lambda d: d["revised_filed"].dt.year.astype(str))
per_year = dated.groupby("year")["ticker"].nunique()
typical = int(per_year.median()) if len(per_year) else 0
st.caption(
    "By the year of the filing that made the change. Revising *something* is "
    "routine: in a typical year {} of {} companies change at least one figure "
    "they reported before, almost all of it reclassification. Revisions to net "
    "income are rare, and those do bunch, each time in the shape a standard "
    "adopted retrospectively leaves — several companies recasting the same "
    "earlier years in the same year's filings. In 2011, Verizon and AT&T "
    "changing pension accounting. In 2013, five insurers recasting 2010 and "
    "2011, which matches the change to how insurers defer acquisition costs. "
    "In 2019, companies adopting the new revenue standard. In 2024, six "
    "insurers recasting 2021 and 2022 under the standard for long-duration "
    "insurance contracts."
    .format(typical, universe["ticker"].nunique()))
left, right = st.columns(2)
with left:
    every = dated.groupby("year").size().rename("n").reset_index()
    charts.render(
        charts.bars(every, x="year", y="n", kind="count", colour_by_entity=False,
                    height=260, title="Every line"),
        every.rename(columns={"year": "Year", "n": "Revisions"}),
        key="revisions_by_year")
with right:
    ni_years = (dated[(dated["item"] == "net_income") & (dated["frequency"] == "annual")]
                .groupby("year").size().reindex(every["year"], fill_value=0)
                .rename("n").reset_index())
    charts.render(
        charts.bars(ni_years, x="year", y="n", kind="count", colour_by_entity=False,
                    height=260, title="Annual net income only"),
        ni_years.rename(columns={"year": "Year", "n": "Revisions"}),
        key="revisions_ni_by_year")

# ---- One company ---------------------------------------------------------
st.divider()
st.subheader("One company")

with st.sidebar:
    st.header("Filters")
    frequency = filters.frequency_control("revision_frequency",
                                          allowed=["annual", "quarterly"])

revised = sorted(revisions["ticker"].unique())
default = (bottom["ticker"].value_counts().index[0] if not bottom.empty else revised[0])
ticker = st.selectbox(
    "Company", revised, index=revised.index(default), key="revision_ticker",
    format_func=lambda t: "{} · {}".format(t, names.get(t, t)))

mine = revisions[(revisions["ticker"] == ticker) & (revisions["frequency"] == frequency)]
if mine.empty:
    st.info("No {} revisions for {}.".format(frequency, ticker))
    st.stop()

# Default to the line revised most often, net income first where it was.
counts = mine["item"].value_counts()
order = (["net_income"] if "net_income" in counts else []) + [
    i for i in counts.index if i != "net_income"]
labels = dict(zip(mine["item"], mine["label"]))
item = st.selectbox("Line item", order, key="revision_item_{}_{}".format(ticker, frequency),
                    format_func=lambda i: "{} ({} revised)".format(labels.get(i, i), counts[i]))

# The line is the history as it reads now; a ring marks what was first filed,
# only where that differed — the same mark as the ranking above, so the gap
# between ring and line IS the revision. An earlier version drew a second full
# line for "first filed", identical to the first everywhere nothing changed and
# drawn on top of it, so the line that mattered was hidden.
current = data.items()
current = current[(current["ticker"] == ticker) & (current["frequency"] == frequency)
                  & (current["item"] == item)][["period_end", "value"]]
current = current.assign(period_end=pd.to_datetime(current["period_end"]))
line = mine[mine["item"] == item]
figure = charts.time_series(
    current.assign(version="As reported now"), series_col="version",
    kind="currency", height=380, namespace="revision", label_last=False,
    title="{} · {}".format(ticker, labels.get(item, item)))
figure.add_trace(go.Scatter(
    x=line["period_end"], y=line["first_value"], mode="markers", name="As first filed",
    marker=dict(size=10, color="rgba(0,0,0,0)",
                line=dict(color=charts.muted(), width=2)),
    customdata=list(zip(line["first_value"].map(currency),
                        line["first_filed"].dt.strftime("%b %Y"))),
    hovertemplate="First filed %{customdata[0]} (%{customdata[1]})<extra></extra>"))
figure.update_layout(showlegend=True)
charts.render(
    figure,
    line.assign(**{
        "Period": line["fiscal_label"],
        "First filed": line["first_value"].map(currency),
        "On": line["first_filed"].dt.strftime("%d %b %Y"),
        "As revised": line["value"].map(currency),
        "In": line["revised_filed"].dt.strftime("%d %b %Y"),
        "Change": (line["pct_change"] * 100).round(1).astype(str) + "%",
        "Versions": line["versions"],
    })[["Period", "First filed", "On", "As revised", "In", "Change", "Versions"]],
    key="revisions_company")

with st.expander("Every revision for {}".format(ticker)):
    st.dataframe(
        mine.assign(**{
            "Line item": mine["label"], "Period": mine["fiscal_label"],
            "First filed": mine["first_value"].map(currency),
            "As revised": mine["value"].map(currency),
            "Change %": (mine["pct_change"] * 100).round(2),
            "Revised in": mine["revised_filed"].dt.strftime("%b %Y"),
        })[["Line item", "Period", "First filed", "As revised", "Change %", "Revised in"]]
        .sort_values(["Period", "Line item"]),
        hide_index=True, width="stretch")

# ---- What was left out ---------------------------------------------------
report = data.revision_report()
if not report.empty:
    parts = ", ".join("{:,} {}".format(r["count"], r["reason"])
                      for _, r in report.iterrows() if r["count"])
    st.caption(
        "Set aside before anything above was counted: {}. A quarter the "
        "extractor computed from year-to-date figures was never filed, so it "
        "cannot have been revised. First-filed figures come from the "
        "extractor's restatement history, matched by the length of the period "
        "as well as its end date — a fiscal year and its fourth quarter end on "
        "the same day, and matching on the date alone mistakes one for the "
        "other.".format(parts))
