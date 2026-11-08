"""How much of this can you trust, and where does it come from?

Every other page shows numbers. This one shows how they were arrived at —
which values came off a filing, which were recovered by an accounting
identity, which were pulled from an XBRL concept the extractor does not map,
and where the source data is thin enough that a chart deserves a second look.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app import charts, data, filters
from findash.provenance import (INFERRED_SOURCES, SOURCE_LABELS,
                                 SOURCE_ORDER)

st.set_page_config(page_title="Data Quality", page_icon="🩺", layout="wide")

if not filters.guard_dataset():
    st.stop()

items = data.items()
universe = data.companies()
coverage = data.coverage()
registry = data.item_registry()

st.title("Data quality")
st.caption(
    "The extractor measures its own work, and so does this layer. Nothing here "
    "is hidden behind a chart: a company with a 71% fill rate and 48 "
    "reconciliation gaps is telling you something before you read a single ratio."
)

SOURCE_MEANING = SOURCE_LABELS

source = items["provenance"].str.extract(r"^([a-z_]+)", expand=False)
normalised = items["provenance"].str.contains("sign-normalized", na=False)

cols = st.columns(4)
cols[0].metric("Values", "{:,}".format(len(items)))
cols[1].metric("Reported", "{:.1%}".format((source == "reported").mean()))
cols[2].metric(
    "Inferred", "{:.1%}".format(source.isin(INFERRED_SOURCES).mean()),
    help="Anything the filer did not tag for this exact period: recovered by "
         "identity, read from the fact store, summed from components, or a "
         "quarter backed out of year-to-date figures.")
cols[3].metric("Sign-normalised", "{:,}".format(int(normalised.sum())),
               help="Magnitudes a filer tagged with the opposite sign, flipped "
                    "at ingest so arithmetic downstream is correct.")

st.divider()

tabs = st.tabs(["By company", "By line item", "Extractor report", "Metric availability"])

# ---- By company ----------------------------------------------------------
with tabs[0]:
    by_company = (
        items.assign(source=source)
        .groupby("ticker")["source"]
        .value_counts(normalize=True).unstack(fill_value=0).reset_index()
    )
    for column in SOURCE_ORDER:
        if column not in by_company:
            by_company[column] = 0.0

    merged = universe[["ticker", "name", "template", "sector"]].merge(
        by_company, on="ticker", how="left")
    if "fill_rate" in universe:
        merged = merged.merge(universe[["ticker", "fill_rate"]], on="ticker", how="left")

    table = merged.copy()
    for column in SOURCE_ORDER:
        table[column] = (table[column] * 100).round(1)
    if "fill_rate" in table:
        table["fill_rate"] = (table["fill_rate"] * 100).round(1)
    table["template"] = table["template"].map(
        {"industrial": "Industrial", "bank": "Bank",
         "insurer": "Insurer", "reit": "REIT"}).fillna(table["template"])
    table = table.rename(columns={
        "ticker": "Ticker", "name": "Company", "template": "Type",
        "sector": "Sector", "reported": "Reported %", "derived": "Derived %",
        "differenced": "Differenced %", "summed": "Summed %",
        "store": "From store %", "ttm": "TTM %", "fill_rate": "Fill rate %",
    })
    # Differenced is 7.3% of all values and summed 0.7%, so leaving them out
    # of the per-company view would hide most of what "inferred" now counts.
    order = [c for c in ["Ticker", "Company", "Type", "Sector", "Fill rate %",
                         "Reported %", "Differenced %", "Derived %",
                         "Summed %", "From store %", "TTM %"]
             if c in table.columns]
    st.dataframe(
        table[order].sort_values("Reported %"),
        width="stretch", hide_index=True, height=440,
        column_config={
            "Reported %": st.column_config.ProgressColumn(
                "Reported %", min_value=0, max_value=100, format="%.1f"),
            "Fill rate %": st.column_config.ProgressColumn(
                "Fill rate %", min_value=0, max_value=100, format="%.1f"),
        },
    )
    st.caption(
        "Sorted by the share of values taken directly from filings. A low "
        "figure is not necessarily wrong — it usually means the filer tags "
        "sparsely and more had to be recovered by identity — but it is where "
        "to look first if a number surprises you."
    )

# ---- By line item --------------------------------------------------------
with tabs[1]:
    if coverage.empty:
        st.info("No coverage data. Rebuild with `python -m findash.build`.")
    else:
        annual = coverage[coverage["frequency"] == "annual"]
        summary = (
            annual.groupby(["item", "label"])
            .agg(companies=("ticker", "nunique"), periods=("periods", "sum"),
                 reported=("reported", "sum"), derived=("derived", "sum"),
                 from_store=("from_store", "sum"))
            .reset_index()
        )
        summary["reported_share"] = summary["reported"] / summary["periods"]
        summary = summary.sort_values("reported_share")

        display = summary.rename(columns={
            "label": "Line item", "companies": "Companies",
            "periods": "Annual periods", "reported": "Reported",
            "derived": "Derived", "from_store": "From store"})
        display["Reported %"] = (summary["reported_share"] * 100).round(1)
        st.dataframe(
            display[["Line item", "Companies", "Annual periods", "Reported",
                     "Derived", "From store", "Reported %"]],
            width="stretch", hide_index=True, height=420,
            column_config={"Reported %": st.column_config.ProgressColumn(
                "Reported %", min_value=0, max_value=100, format="%.1f")},
        )

        inferred = summary[summary["reported_share"] < 1].head(12)
        if not inferred.empty:
            st.subheader("Most-inferred line items")
            # Plot on `label` directly. Renaming it to "item" collided with the
            # groupby's own `item` column, and a frame with two columns of the
            # same name is rejected outright — the page crashed rather than
            # mislabelled, which at least made it obvious once opened.
            figure = charts.bars(
                inferred.assign(pct=inferred["reported_share"] * 100),
                x="label", y="pct", kind="ratio",
                height=max(320, 26 * len(inferred)),
                horizontal=True, colour_by_entity=False,
            )
            figure.update_xaxes(ticksuffix="%", tickformat=".0f")
            charts.render(figure, display.head(12), key="quality_inferred")
            st.caption(
                "A 0% reading here is expected for the derived aggregates — "
                "EBIT, EBITDA, free cash flow, net debt and the rest are "
                "analytical constructs, not lines in a filing, so nothing "
                "reports them directly. The ones worth attention are the "
                "statement lines partway down: cost of revenue, pre-tax "
                "income, total liabilities and gross profit are real "
                "disclosures that many filers simply omit, leaving an "
                "accounting identity to recover them."
            )

# ---- Extractor report ----------------------------------------------------
with tabs[2]:
    quality = data.quality()
    if quality.empty:
        st.info("The extractor's run reports were not found.")
    else:
        columns = [c for c in ("ticker", "name", "template", "fill_rate", "facts",
                               "unmapped_concepts", "identity_failures",
                               "reconciliation_gaps", "quarters_derived",
                               "duplicate_concepts", "warning")
                   if c in quality.columns]
        report = quality[columns].copy()
        if "fill_rate" in report:
            report["fill_rate"] = (report["fill_rate"] * 100).round(1)
        report = report.rename(columns={
            "ticker": "Ticker", "name": "Company", "template": "Type",
            "fill_rate": "Fill rate %", "facts": "Facts",
            "unmapped_concepts": "Unmapped concepts",
            "identity_failures": "Identity failures",
            "reconciliation_gaps": "Reconciliation gaps",
            "quarters_derived": "Quarters derived",
            "duplicate_concepts": "Duplicate concepts", "warning": "Warning"})
        sort_col = "Reconciliation gaps" if "Reconciliation gaps" in report else "Ticker"
        st.dataframe(report.sort_values(sort_col, ascending=False),
                     width="stretch", hide_index=True, height=440)
        st.caption(
            "Produced by the extractor, not by this app. **Identity failures** "
            "are periods where the statements do not add up as filed; "
            "**reconciliation gaps** are quarters that could not be reconciled "
            "against their annual total. GE leads on both, which is consistent "
            "with the GE Capital wind-down — and it is why five of its derived "
            "cost-of-revenue values were refused as sign-inconsistent rather "
            "than published."
        )

# ---- Metric availability -------------------------------------------------
with tabs[3]:
    report = data.metric_report()
    if report.empty:
        st.info("No metric report. Rebuild with `python -m findash.build`.")
    else:
        display = report.rename(columns={
            "metric": "Metric", "label": "Label", "family": "Family",
            "computed": "Values", "blocked_template": "Blocked · wrong type",
            "blocked_frequency": "Blocked · frequency",
            "blocked_guard": "Blocked · guard", "skipped": "Not computed"})
        keep = [c for c in ["Label", "Family", "Values", "Blocked · wrong type",
                            "Blocked · frequency", "Blocked · guard", "Not computed"]
                if c in display.columns]
        st.dataframe(display[keep].sort_values("Values"),
                     width="stretch", hide_index=True, height=440)
        st.caption(
            "**Blocked · wrong type** counts values the arithmetic produced but "
            "the metric is not valid for — a gross margin for a bank. "
            "**Blocked · guard** counts those refused as meaningless: a P/E on "
            "a loss, a return on negative equity, growth from a negative base. "
            "Both are deliberate. A metric showing a large blocked count is "
            "working correctly, not failing."
        )
