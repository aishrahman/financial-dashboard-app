"""Segment, geographic and footnote detail.

Everything here comes straight from the filings as the company labelled it —
its own segment names, its own schedule line items. None of it is normalised
onto the canonical schema, so nothing on this page feeds a ratio or a score.
That is the point: it is the underlying detail you go to when a headline
number raises a question the statements cannot answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app import charts, data, filters
from app.format import currency

st.set_page_config(page_title="Detail", page_icon="🔍", layout="wide")

if not filters.guard_dataset():
    st.stop()

segments = data.segments()
footnotes = data.footnotes()

st.title("Detail")

if segments.empty and footnotes.empty:
    st.info(
        "No segment or footnote detail in the curated data. The extractor "
        "writes `_segments.parquet` and `_footnotes.parquet`; rebuild with "
        "`python -m findash.build` once they exist."
    )
    st.stop()

universe = data.companies()
names = dict(zip(universe["ticker"], universe["name"]))

with st.sidebar:
    st.header("Filters")
    ticker = st.selectbox(
        "Company", list(universe["ticker"]),
        format_func=lambda t: "{} · {}".format(t, names.get(t, t)),
        key="detail_ticker",
    )
    frequency = filters.frequency_control("detail_frequency", ["annual", "quarterly"])

st.caption(
    "As reported, using the company's own labels. Not normalised onto the "
    "canonical schema, so none of it feeds a ratio or a score — this is the "
    "detail behind the headline numbers, not another version of them."
)

tab_seg, tab_fn = st.tabs(["Segments", "Footnotes"])

# ---- Segments -----------------------------------------------------------
with tab_seg:
    mine = segments[(segments["ticker"] == ticker)
                    & (segments["frequency"] == frequency)]
    if mine.empty:
        st.info("No segment detail for {} at this frequency.".format(ticker))
    else:
        axis = st.radio("Breakdown", sorted(mine["axis"].unique()),
                        horizontal=True, key="detail_axis_{}".format(ticker))
        part = mine[mine["axis"] == axis]

        # Pick the fullest revenue series rather than the first one
        # alphabetically — "Intersegment Revenues" sorts before "Revenues" and
        # is usually a small elimination line, so the naive choice opened Deere
        # on a nearly empty chart.
        line_items = sorted(part["line_item"].unique())
        counts = part.groupby("line_item").size()
        revenue_items = [l for l in line_items if "revenue" in l.lower()]
        default = max(revenue_items or line_items, key=lambda l: counts.get(l, 0))
        line_item = st.selectbox(
            "Line item", line_items, index=line_items.index(default),
            # Keyed on the company and axis: a widget keyed only by name keeps
            # the previous company's selection, and Streamlit silently drops
            # options that no longer exist.
            key="detail_line_item_{}_{}".format(ticker, axis))

        series = part[part["line_item"] == line_item].copy()

        # Filers tag a rollup alongside its own components — Duke reports
        # "Total Reportable Segments" beside the five segments that make it
        # up. Plotted together the total double-counts everything below it and
        # flattens the scale for the smaller segments, so it starts
        # deselected. Still offered, because checking that the parts sum to
        # the whole is a fair thing to want.
        #
        # The extractor marks these now. It used to be guesswork here, looking
        # for "total", "consolidated" or "all other" in the member name, and
        # the guess was poor in both directions: of 70 flagged rollups it found
        # 21, missing NVIDIA's "Data Center" and Oracle's "Cloud Services And
        # License Support", while hiding ten real segments whose names merely
        # start with the word Total. The name test survives only for a snapshot
        # built before the flag existed.
        members = sorted(series["member"].unique())
        if "is_rollup" in series.columns:
            flagged = series.groupby("member")["is_rollup"].any()
            rollups = [m for m in members if bool(flagged.get(m, False))]
        else:
            rollups = [m for m in members
                       if any(w in m.lower()
                              for w in ("total", "consolidated", "all other"))]
        chosen_members = st.multiselect(
            "Segments", members,
            default=[m for m in members if m not in rollups] or members,
            # Same reason as the line item above: switching company must not
            # inherit the last one's segment names, which no longer exist as
            # options and leave the box empty and the chart blank.
            key="detail_members_{}_{}_{}".format(ticker, axis, line_item),
            help=("Reported totals start deselected — they are the sum of the "
                  "others, not another segment." if rollups else None),
        )
        series = series[series["member"].isin(chosen_members)]

        if series.empty:
            st.info("Nothing reported for that line item.")
        else:
            # Every segment value is money today, but the export carries a
            # unit now and a rate rendered as dollars is the bug that printed
            # Apple's 21% tax rate as "$0" on the footnotes tab. One line item
            # is one unit, so there is nothing to mix on the axis.
            unit = (series["unit"].dropna().iloc[0]
                    if "unit" in series.columns and series["unit"].notna().any()
                    else "money")
            as_percent = unit == "percent"
            figure = charts.time_series(
                series, series_col="member",
                kind="percent" if as_percent else "currency", height=400,
                title="{} · {}".format(axis, line_item),
                namespace="segment", label_last=series["member"].nunique() <= 4,
            )
            shown = series["value"].map(
                (lambda v: "{:.1%}".format(v)) if as_percent else currency)
            charts.render(
                figure,
                series.assign(Value=shown)[
                    ["member", "period_label", "Value"]
                ].rename(columns={"member": "Segment", "period_label": "Period"}),
                key="detail_segment_chart",
            )
            st.caption(
                "Segment history is parsed from a limited number of 10-K "
                "instance documents, so it is shallower than the statements — "
                "expect roughly a decade rather than the full filing history."
            )

# ---- Footnotes ----------------------------------------------------------
with tab_fn:
    mine = footnotes[(footnotes["ticker"] == ticker)
                     & (footnotes["frequency"] == frequency)]
    if mine.empty:
        st.info("No footnote detail for {} at this frequency.".format(ticker))
    else:
        schedule = st.radio("Schedule", sorted(mine["schedule"].unique()),
                            horizontal=True, key="detail_schedule_{}".format(ticker))
        part = mine[mine["schedule"] == schedule]

        # Schedules mix money and rates — the tax schedule carries both a
        # deferred tax balance in dollars and an effective tax rate as a
        # fraction. Formatting a 0.35 statutory rate as currency printed "$0".
        units = dict(zip(part["line_item"], part.get(
            "unit", pd.Series("money", index=part.index))))

        def show(line_item, value):
            if pd.isna(value):
                return "—"
            return ("{:.1f}%".format(value * 100)
                    if units.get(line_item) == "percent" else currency(value))

        table = part.pivot_table(
            index="line_item", columns="period_end", values="value",
            aggfunc="first")
        table = table[sorted(table.columns)[-10:]]
        table.columns = [pd.Timestamp(c).date() for c in table.columns]
        formatted = table.copy().astype(object)
        for line_item in formatted.index:
            formatted.loc[line_item] = [
                show(line_item, v) for v in table.loc[line_item]]
        st.dataframe(formatted, width="stretch")

        # A chart carries one y-scale, so money and rates cannot share it —
        # that is the dual-axis trap. Where a schedule has both, pick which.
        available_units = sorted(set(units.get(l, "money")
                                     for l in part["line_item"].unique()))
        unit = available_units[0]
        if len(available_units) > 1:
            unit = st.radio(
                "Units", available_units, horizontal=True,
                format_func=lambda u: {"money": "Amounts",
                                       "percent": "Rates"}.get(u, u.title()),
                key="detail_fn_unit_{}_{}".format(ticker, schedule),
                help="Amounts and rates cannot share one axis, so they chart "
                     "separately.",
            )

        line_items = sorted(l for l in part["line_item"].unique()
                            if units.get(l, "money") == unit)
        if not line_items:
            st.info("Nothing reported in those units for this schedule.")
            st.stop()
        chosen = st.multiselect(
            "Chart these", line_items, default=line_items[:1],
            key="detail_fn_items_{}_{}_{}".format(ticker, schedule, unit))
        if chosen:
            series = part[part["line_item"].isin(chosen)]
            kind = "percent" if unit == "percent" else "currency"
            # Name what is actually plotted, the way the Segments tab does —
            # "Debt" alone says nothing about which of its six line items is
            # on screen.
            title = schedule if len(chosen) > 1 else "{} · {}".format(
                schedule, chosen[0])
            charts.render(
                charts.time_series(
                    series, series_col="line_item", kind=kind,
                    height=360, title=title, namespace="footnote",
                    label_last=len(chosen) <= 4),
                series.assign(Value=[show(l, v) for l, v in
                                     zip(series["line_item"], series["value"])])[
                    ["line_item", "period_label", "Value"]
                ].rename(columns={"line_item": "Line item",
                                  "period_label": "Period"}),
                key="detail_fn_chart",
            )
