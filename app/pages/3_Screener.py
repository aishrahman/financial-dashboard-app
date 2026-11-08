"""Rank and filter the whole universe on any metric."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app import charts, data, filters, selection
from app.format import metric_value

st.set_page_config(page_title="Screener", page_icon="🔎", layout="wide")

if not filters.guard_dataset():
    st.stop()

registry = data.metric_registry()
universe = data.companies()

with st.sidebar:
    st.header("Filters")
    sectors = filters.sector_control("screen_sectors")
    templates = filters.template_control("screen_templates")
    frequency = filters.frequency_control("screen_frequency", ["annual", "ttm"])
    as_of = st.selectbox(
        "Period", ["Latest available", "Specific year"], key="screen_period_mode")
    year = None
    if as_of == "Specific year":
        year = st.slider("Year", 2007, pd.Timestamp.now().year, 2024, key="screen_year")

st.title("Screener")
st.caption(
    "One row per company at its most recent period, ranked on the metrics you "
    "choose. Companies whose type does not support a metric are absent from "
    "that column rather than shown as zero."
)

pool = filters.apply_universe(sectors=sectors, templates=templates)
if pool.empty:
    st.warning("No companies match those filters.")
    st.stop()

present = sorted(pool["template"].unique())
available = data.available_metrics(present, frequency).sort_values(["family", "label"])
if available.empty:
    st.warning("No metrics available for this selection.")
    st.stop()

available = filters.group_metrics(available)
labels = dict(zip(available["metric"], available["label"]))
render = filters.metric_option_label(available)
default = [m for m in ["revenue_growth", "net_margin", "roe", "net_debt_to_ebitda",
                       "pe_ratio", "altman_z_double_prime"] if m in labels][:5]

chosen = st.multiselect(
    "Columns", list(available["metric"]),
    default=default or list(available["metric"])[:4],
    format_func=render,
    key="screen_columns",
)
if not chosen:
    st.info("Choose at least one metric.")
    st.stop()

frame = data.metrics()
frame = frame[frame["ticker"].isin(pool["ticker"])
              & frame["metric"].isin(chosen)
              & (frame["frequency"] == frequency)]
if year is not None:
    frame = frame[frame["period_end"].dt.year == year]
if frame.empty:
    st.warning("No values for that selection.")
    st.stop()

latest = (frame.sort_values("period_end")
          .groupby(["ticker", "metric"], as_index=False).tail(1))
wide = latest.pivot_table(index="ticker", columns="metric",
                          values="value", aggfunc="first").reset_index()
period_of = (latest.groupby("ticker")["period_end"].max()
             .dt.date.rename("period").reset_index())

table = (pool[["ticker", "name", "sector", "template"]]
         .merge(wide, on="ticker", how="inner")
         .merge(period_of, on="ticker", how="left"))

# ---- Threshold filters ---------------------------------------------------
with st.expander("Narrow by value"):
    active = {}
    columns = st.columns(min(3, len(chosen)))
    for i, metric_id in enumerate(chosen):
        if metric_id not in table:
            continue
        series = table[metric_id].dropna()
        if series.empty:
            continue
        spec = registry.get(metric_id)
        low, high = float(series.min()), float(series.max())
        if low == high:
            continue
        with columns[i % len(columns)]:
            active[metric_id] = st.slider(
                spec.label, low, high, (low, high),
                key="screen_range_" + metric_id,
                help=spec.description or None,
            )
    for metric_id, (low, high) in active.items():
        table = table[table[metric_id].isna()
                      | table[metric_id].between(low, high)]

sortable = [m for m in chosen if m in table]
sort_by = filters.grouped_selectbox(
    "Sort by", available[available["metric"].isin(sortable)], key="screen_sort")
spec = registry.get(sort_by)
ascending = spec.direction == "lower_better"
table = table.sort_values(sort_by, ascending=ascending, na_position="last")

# Narrow BEFORE the count is written, or the caption reports the unfiltered
# universe while the table below shows two rows.
picked = selection.banner(["screen_rank"], list(pool["ticker"]))
if picked:
    table = table[table["ticker"].isin(picked)]

st.caption("{} compan{} · sorted by {} ({})".format(
    len(table), "y" if len(table) == 1 else "ies", spec.label,
    "lower is better" if ascending else "higher is better"
    if spec.direction == "higher_better" else "no inherent direction"))

# ---- The table -----------------------------------------------------------
display = table.copy()
for metric_id in chosen:
    if metric_id in display:
        kind = registry.get(metric_id).format
        display[metric_id] = display[metric_id].map(lambda v: metric_value(v, kind))
display["template"] = display["template"].map(
    {"industrial": "Industrial", "bank": "Bank",
     "insurer": "Insurer", "reit": "REIT"}).fillna(display["template"])
display = display.rename(columns={
    "ticker": "Ticker", "name": "Company", "sector": "Sector",
    "template": "Type", "period": "Period", **labels})

order = ["Ticker", "Company", "Sector", "Type", "Period"] + \
        [labels[m] for m in chosen if labels[m] in display.columns]
st.dataframe(display[order], width="stretch", hide_index=True, height=420)

st.download_button(
    "Download as CSV", display[order].to_csv(index=False).encode("utf-8"),
    file_name="screener_{}_{}.csv".format(frequency, sort_by),
    mime="text/csv",
)

st.divider()

# ---- Charts over the chosen columns --------------------------------------
st.subheader("Chart")

# The chart follows its OWN metric, not whichever column the table happens to
# be sorted by. Both are drawn from the columns picked at the top, so the chart
# stays inside the same slice of the data.
chart_left, chart_right = st.columns([2, 3])
with chart_left:
    chart_form = st.radio(
        "Show", ["Ranking", "Heatmap"], horizontal=True, key="screen_chart_form",
        help=("Ranking compares companies at one period. Heatmap shows every "
              "company over time at once, which is where a whole sector "
              "turning together becomes visible."),
    )
with chart_right:
    chart_metric = filters.grouped_selectbox(
        "Metric to chart", available[available["metric"].isin(sortable)],
        default=sort_by, key="screen_chart_metric")

chart_spec = registry.get(chart_metric)
chart_ascending = chart_spec.direction == "lower_better"

if chart_form == "Ranking":
    st.caption(
        "Click a bar, or drag a box across several, to narrow the table above "
        "to those companies."
    )
    # Ranked from the UNFILTERED set, so a selection never hides the bars you
    # would need in order to change it.
    ranked = (pool[["ticker", "name"]]
              .merge(wide, on="ticker", how="inner")
              .dropna(subset=[chart_metric])
              .sort_values(chart_metric, ascending=chart_ascending)
              .head(20))
    if ranked.empty:
        st.info("Nothing to chart after filtering.")
    else:
        figure = charts.bars(
            ranked.rename(columns={"ticker": "company", chart_metric: "value"}),
            x="company", y="value", kind=chart_spec.format,
            height=max(320, 24 * len(ranked)), horizontal=True,
            colour_by_entity=False,   # one series, one hue
            highlight=picked, selectable=True,
            title="{} · top 20".format(chart_spec.label),
        )
        charts.render(
            figure,
            ranked[["ticker", "name", chart_metric]].assign(
                **{chart_metric: ranked[chart_metric].map(
                    lambda v: metric_value(v, chart_spec.format))}
            ).rename(columns={"ticker": "Ticker", "name": "Company",
                              chart_metric: chart_spec.label}),
            key=selection.chart_key("screen_rank"), on_select=True,
        )

else:
    # Every company over time in one grid. This is the view that scales past
    # small multiples — a ranking shows one period and a line chart runs out of
    # distinguishable colours long before 72 companies.
    grid = data.metrics()
    grid = grid[(grid["metric"] == chart_metric)
                & (grid["frequency"] == frequency)
                & grid["ticker"].isin(pool["ticker"])]
    if picked:
        grid = grid[grid["ticker"].isin(picked)]

    if grid.empty:
        st.info("No values for that metric.")
    else:
        periods = sorted(grid["calendar_period"].dropna().unique())[-12:]
        grid = grid[grid["calendar_period"].isin(periods)]
        matrix = grid.pivot_table(index="ticker", columns="calendar_period",
                                  values="value", aggfunc="first")
        # Order rows by the most recent period that most companies have
        # actually reported. Sorting on the very latest column ranks almost
        # nothing — at any moment only a handful of filers have reported the
        # current year, so the rest tie on null and fall back to alphabetical.
        coverage = matrix.notna().sum()
        dense = [c for c in matrix.columns if coverage[c] >= len(matrix) * 0.5]
        order_on = dense[-1] if dense else (periods[-1] if periods else None)
        if order_on is not None:
            matrix = matrix.sort_values(
                order_on, ascending=chart_ascending, na_position="last")
        matrix = matrix.head(30)

        # Diverging around zero only where the metric genuinely has a midpoint
        # that means something; otherwise a single-hue magnitude ramp.
        centre = 0.0 if chart_spec.format == "percent" else None
        charts.render(
            charts.heatmap(
                matrix, kind=chart_spec.format, centre=centre,
                higher_is_better=chart_spec.direction != "lower_better",
                height=max(320, 22 * len(matrix) + 90),
                title="{} · {} by company and period".format(
                    chart_spec.label, data.FREQUENCY_LABELS[frequency]),
            ),
            matrix.reset_index().rename(columns={"ticker": "Ticker"}),
            key="screen_heatmap",
        )
        st.caption(
            "Up to 30 companies over the last 12 periods, ordered by {}. "
            "Reading across a row shows one company's trend; reading down a "
            "column shows whether a move was company-specific or something the "
            "whole set did at once. The colour scale is set to the 5th–95th "
            "percentile so a single outlier does not flatten everything "
            "else — values beyond it take the end colour.".format(order_on)
        )

if chart_spec.description:
    st.caption(chart_spec.description)
