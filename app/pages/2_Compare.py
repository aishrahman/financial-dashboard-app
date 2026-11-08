"""Several companies against each other."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app import charts, data, filters, selection
from app.format import currency, metric_value, period_label

st.set_page_config(page_title="Compare", page_icon="⚖️", layout="wide")

if not filters.guard_dataset():
    st.stop()

registry = data.metric_registry()
universe = data.companies()

with st.sidebar:
    st.header("Filters")
    sectors = filters.sector_control("compare_sectors")
    templates = filters.template_control("compare_templates")
    frequency = filters.frequency_control("compare_frequency")
    start_year = filters.year_control("compare_start_year")

st.title("Compare")

pool = filters.apply_universe(sectors=sectors, templates=templates)
if pool.empty:
    st.warning("No companies match those filters.")
    st.stop()

names = dict(zip(universe["ticker"], universe["name"]))
key = "compare_tickers"
if key not in st.session_state:
    st.session_state[key] = [t for t in ["AAPL", "MSFT", "NVDA"] if t in set(pool["ticker"])] \
        or list(pool["ticker"])[:3]
# Keep the selection inside the filtered pool without repainting the survivors.
st.session_state[key] = [t for t in st.session_state[key] if t in set(pool["ticker"])]

tickers = st.multiselect(
    "Companies", list(pool["ticker"]), key=key, max_selections=6,
    format_func=lambda t: "{} · {}".format(t, names.get(t, t)),
    help=("Six lines is the practical limit for telling them apart. On a line "
          "chart every series is compared against every other, not just its "
          "neighbour in the legend, and distinguishability falls off quickly "
          "past that. For more companies at once use Small multiples below, "
          "or the Screener."),
)
if not tickers:
    st.info("Choose at least one company.")
    st.stop()

chosen_templates = sorted(pool[pool["ticker"].isin(tickers)]["template"].unique())
if len(chosen_templates) > 1:
    st.caption(
        "Mixed company types selected ({}). Only metrics valid for all of them "
        "are offered — a gross margin or a combined ratio is not comparable "
        "across these.".format(", ".join(t.title() for t in chosen_templates))
    )

# Metrics valid for EVERY selected template, so nothing on screen is
# meaningless for one of the companies shown.
available = data.available_metrics(chosen_templates, frequency)
for template in chosen_templates:
    valid = {m.id for m in registry.for_template(template)}
    available = available[available["metric"].isin(valid)]

if available.empty:
    st.warning(
        "No metric is valid for all of these company types at this frequency. "
        "Compare within one type, or switch frequency."
    )
    st.stop()

metric_id = filters.metric_control(
    chosen_templates, frequency, label="Metric",
    default="revenue_growth", key="compare_metric",
)
if metric_id is None or metric_id not in set(available["metric"]):
    metric_id = list(available["metric"])[0]

spec = registry.get(metric_id)
series = data.metric_series(tickers, [metric_id], frequency, start_year)

if series.empty:
    st.warning("No values for {} at this frequency in the selected period.".format(spec.label))
    st.stop()

st.subheader(spec.label)
if spec.description:
    st.caption(spec.description)

picked = selection.banner(["compare_scatter", "compare_latest"], tickers)

view = st.radio(
    "View", ["Over time", "Latest period", "Indexed to 100", "Small multiples"],
    horizontal=True, key="compare_view",
    help=("Indexing rebases every company to 100 at the first period, which "
          "compares shape rather than level — the honest alternative to a "
          "second y-axis. Small multiples give each company its own panel on "
          "a shared scale, which stays readable well past the point where "
          "overlapping lines stop being."),
)

table = series.assign(
    Period=series.apply(period_label, axis=1),
    Value=series["value"].map(lambda v: metric_value(v, spec.format)),
)[["ticker", "Period", "Value"] + (["band"] if "band" in series else [])]
table = table.rename(columns={"ticker": "Company", "band": "Band"})

if view == "Over time":
    # Aligned on CALENDAR period, not fiscal period end. Apple's September
    # year-end and Microsoft's June year-end are not the same date — plotted
    # against their own fiscal dates, the two lines visibly start and end at
    # different x positions even when the reader means to compare them period
    # for period. calendar_period_end buckets every filer's fiscal year onto
    # 31 December of the calendar year it belongs to (or the matching quarter
    # end for quarterly/TTM), so companies with different fiscal calendars
    # land on the same point.
    figure = charts.time_series(
        series, series_col="ticker", kind=spec.format, height=420,
        # Every line gets its name at its own end point. Colour then confirms
        # identity rather than carrying it alone, which is what makes six
        # series legible and is the condition the palette's own guidance puts
        # on using its later slots.
        label_last=True, highlight=picked,
        x_col="calendar_period_end", basis_col="basis",
    )
    for band in (spec.bands or []):
        if "max" in band:
            figure.add_hline(
                y=band["max"],
                line=dict(width=1, dash="dot",
                          color=charts.theme.STATUS.get(band.get("tone"), "#898781")),
                annotation_text=band["label"], annotation_position="right",
                annotation_font=dict(size=10),
            )
    charts.render(figure, table, key="compare_time")

elif view == "Latest period":
    latest = (series.sort_values("period_end")
              .groupby("ticker", as_index=False).tail(1))
    figure = charts.bars(
        latest.rename(columns={"ticker": "company"}), x="company", y="value",
        kind=spec.format, height=420, horizontal=len(tickers) > 5,
        title="{} · most recent period".format(spec.label),
        highlight=picked, selectable=True,
    )
    charts.render(figure, table.groupby("Company", as_index=False).tail(1),
                  key=selection.chart_key("compare_latest"), on_select=True)
    st.caption(
        "Each company at its own latest period end, which for non-December "
        "filers is not the same date. Use Over time to compare like for like."
    )

elif view == "Small multiples":
    charts.render(
        charts.small_multiples(
            series, series_col="ticker", x_col="calendar_period_end",
            kind=spec.format, title="{} · by company".format(spec.label)),
        table, key="compare_facets",
    )
    st.caption(
        "One panel per company, all on the same scale so the panels are "
        "comparable — without that a flat line and a steep one look alike. "
        "Trades precise crossings for seeing every company's shape at once."
    )

else:
    # Same calendar alignment as "Over time" — and more important here, since
    # an indexed chart is specifically a claim that every line starts from a
    # comparable point.
    charts.render(
        charts.indexed(series, series_col="ticker", height=420,
                       title="{} · indexed".format(spec.label),
                       x_col="calendar_period_end"),
        table, key="compare_indexed",
    )

st.divider()

# ---- Two metrics at once -------------------------------------------------
st.subheader("Two metrics against each other")
st.caption(
    "One point per company at its latest period. Two measures on two axes of "
    "a line chart would imply a relationship that is not in the data; a "
    "scatter shows the relationship if there is one. **Drag a box around "
    "points, or click one, to highlight those companies everywhere on this "
    "page.**"
)

left, right = st.columns(2)
with left:
    x_metric = filters.grouped_selectbox(
        "Horizontal", available, default="net_margin", key="scatter_x")
with right:
    y_metric = filters.grouped_selectbox(
        "Vertical", available, default="revenue_growth", key="scatter_y")

pair = data.metric_series(tickers, [x_metric, y_metric], frequency, start_year)
latest_pair = (pair.sort_values("period_end")
               .groupby(["ticker", "metric"], as_index=False).tail(1)
               .pivot_table(index="ticker", columns="metric",
                            values="value", aggfunc="first")
               .reset_index())

if x_metric in latest_pair and y_metric in latest_pair:
    frame = latest_pair.dropna(subset=[x_metric, y_metric])
    if frame.empty:
        st.info("No company has both metrics at its latest period.")
    else:
        x_spec, y_spec = registry.get(x_metric), registry.get(y_metric)
        figure = charts.scatter(
            frame.rename(columns={x_metric: x_spec.label, y_metric: y_spec.label}),
            x=x_spec.label, y=y_spec.label, label="ticker",
            x_kind=x_spec.format, y_kind=y_spec.format, height=460,
            highlight=picked, selectable=True,
        )
        readable = frame.copy()
        readable[x_metric] = readable[x_metric].map(lambda v: metric_value(v, x_spec.format))
        readable[y_metric] = readable[y_metric].map(lambda v: metric_value(v, y_spec.format))
        charts.render(
            figure,
            readable.rename(columns={"ticker": "Company", x_metric: x_spec.label,
                                     y_metric: y_spec.label}),
            key=selection.chart_key("compare_scatter"), on_select=True,
        )
else:
    st.info("Those two metrics do not overlap for the current selection.")

st.divider()

# ---- Side by side --------------------------------------------------------
st.subheader("Latest side by side")
family = st.selectbox(
    "Metric family", list(dict.fromkeys(available["family"])),
    format_func=lambda f: f.replace("_", " ").title(), key="compare_family")

subset = available[available["family"] == family]
shown = [t for t in tickers if not picked or t in picked]
rows = []
for ticker in shown:
    row = {"Company": ticker}
    for _, m in subset.iterrows():
        hit = data.metrics()
        hit = hit[(hit["ticker"] == ticker) & (hit["metric"] == m["metric"])
                  & (hit["frequency"] == frequency)]
        row[m["label"]] = (metric_value(hit.sort_values("period_end")["value"].iloc[-1],
                                        m["format"]) if len(hit) else "—")
    rows.append(row)
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
if picked and len(shown) < len(tickers):
    st.caption("Showing the {} highlighted compan{}. Clear the selection above "
               "for all {}.".format(len(shown), "y" if len(shown) == 1 else "ies",
                                    len(tickers)))
