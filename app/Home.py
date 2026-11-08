"""Entry page: what is in the dataset, and how much of it you can trust."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from app import charts, data, filters
from findash.provenance import SOURCE_LABELS, SOURCE_ORDER
from app.format import currency

st.set_page_config(
    page_title="Financial Statement Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial Statement Analysis")
st.caption(
    "SEC 10-K and 10-Q fundamentals for a watchlist of US filers, with "
    "comparison, screening and forensic-accounting scores."
)

if not filters.guard_dataset():
    st.stop()

items = data.items()
metrics = data.metrics()
universe = data.companies()
built = data.last_built()

# ---- Headline figures ----------------------------------------------------
# Stat tiles rather than a chart: these are single numbers, and a bar chart of
# four unrelated counts would be decoration.
cols = st.columns(4)
cols[0].metric("Companies", "{:,}".format(universe["ticker"].nunique()))
cols[1].metric("Line items", "{:,}".format(items["item"].nunique()))
cols[2].metric("Metrics", "{:,}".format(metrics["metric"].nunique()) if not metrics.empty else "—")
cols[3].metric(
    "Annual periods",
    "{:,}".format(len(items[items["frequency"] == "annual"]
                      .groupby(["ticker", "period_end"]))),
)

annual = items[items["frequency"] == "annual"]
span = "{}–{}".format(annual["period_end"].min().year, annual["period_end"].max().year)
freshness = built.strftime("%d %b %Y") if built is not None else "unknown"
st.caption(
    "Fiscal years {} · built {} · {}".format(
        span, freshness,
        "market data available" if data.has_market_data()
        # A published snapshot withholds the price layer deliberately, so the
        # old "run fetch_prices" line sent a reader after a file that is
        # absent on purpose and that they may have no way to produce.
        else "valuation metrics withheld — see the README",
    )
)

st.divider()

# ---- Provenance ----------------------------------------------------------
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Where the numbers come from")
    st.markdown(
        "Nothing inferred is presented as reported. Every value carries its "
        "source, and the Data Quality page breaks this down per company."
    )
    source = items["provenance"].str.extract(r"^([a-z_]+)", expand=False)
    # Vocabulary from the model layer, so this list cannot drift from what
    # the data actually contains. It used to be written out here and on the
    # Data Quality page, and both missed the extractor's own source labels.
    counts = source.value_counts().reindex(list(SOURCE_ORDER)).dropna()
    explain = SOURCE_LABELS
    frame = pd.DataFrame({
        "Source": [s.title() for s in counts.index],
        "Values": counts.to_numpy(),
        "Share": counts.to_numpy() / len(items),
        "What it means": [explain.get(s, "") for s in counts.index],
    })
    figure = charts.bars(
        pd.DataFrame({"source": [s.title() for s in counts.index],
                      "n": counts.to_numpy()}),
        x="source", y="n", kind="count", height=240,
        colour_by_entity=False, title="Values by source",
    )
    charts.render(figure, frame.assign(
        Share=lambda d: (d["Share"] * 100).round(1).astype(str) + "%",
        Values=lambda d: d["Values"].map("{:,}".format)), key="home_provenance")

with right:
    st.subheader("Coverage by company type")
    by_template = (
        universe.groupby("template")
        .agg(companies=("ticker", "nunique"),
             avg_fill=("fill_rate", "mean") if "fill_rate" in universe else ("ticker", "size"))
        .reset_index()
    )
    by_template["template"] = by_template["template"].map(
        {"industrial": "Industrial", "bank": "Bank",
         "insurer": "Insurer", "reit": "REIT"}).fillna(by_template["template"])
    by_template = by_template.rename(columns={
        "template": "Type", "companies": "Companies", "avg_fill": "Avg fill rate"})
    if "Avg fill rate" in by_template:
        by_template["Avg fill rate"] = (by_template["Avg fill rate"] * 100).round(1)
    st.dataframe(by_template, width="stretch", hide_index=True)

    st.markdown(
        "Metrics are tiered by type. A universal core applies to all "
        "companies; banks, insurers and REITs each add their own. The Altman "
        "and Beneish models run on industrials only — neither was fitted on a "
        "balance sheet of loans or policy reserves."
    )

st.divider()

# ---- The universe --------------------------------------------------------
st.subheader("The watchlist")
columns = [c for c in ("ticker", "name", "sector", "template", "exchange",
                       "span", "fill_rate") if c in universe.columns]
table = universe[columns].copy()
if "fill_rate" in table:
    table["fill_rate"] = (table["fill_rate"] * 100).round(1)
table = table.rename(columns={
    "ticker": "Ticker", "name": "Company", "sector": "Sector",
    "template": "Type", "exchange": "Exchange", "span": "Filings span",
    "fill_rate": "Fill rate %",
})
table["Type"] = table["Type"].map(
    {"industrial": "Industrial", "bank": "Bank",
     "insurer": "Insurer", "reit": "REIT"}).fillna(table["Type"])

st.dataframe(
    table, width="stretch", hide_index=True,
    column_config={
        "Fill rate %": st.column_config.ProgressColumn(
            "Fill rate %", min_value=0, max_value=100, format="%.1f",
            help="How much of the canonical schema the extractor filled for this filer.",
        ),
    },
)

st.caption(
    "Start with **Company** for one filer over time, **Compare** for several "
    "against each other, **Screener** to rank the whole universe, or "
    "**Forensics** for the distress and manipulation models."
)
