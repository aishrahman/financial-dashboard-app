"""One company, over time."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app import charts, data, filters
from findash.metrics import peers as peer_lib
from app.format import currency, item_value, metric_value, period_label

st.set_page_config(page_title="Company", page_icon="📈", layout="wide")

if not filters.guard_dataset():
    st.stop()

universe = data.companies()
registry = data.metric_registry()

with st.sidebar:
    st.header("Filters")
    names = dict(zip(universe["ticker"], universe["name"]))
    ticker = st.selectbox(
        "Company", list(universe["ticker"]),
        format_func=lambda t: "{} · {}".format(t, names.get(t, t)),
        key="company_ticker",
    )
    frequency = filters.frequency_control("company_frequency")
    start_year = filters.year_control("company_start_year")
    peer_choice = st.selectbox(
        "Compare against", ["auto", "sector", "template", "all"],
        format_func=lambda k: {
            "auto": "Closest peers available", "sector": "Same sector",
            "template": "Same company type", "all": "All companies"}[k],
        key="company_peers",
        help=("Ratios are shown against the range their peers occupied in the "
              "same CALENDAR period, so filers with different year ends stay "
              "comparable. Only 7 of 15 sectors hold three or more companies, "
              "so 'closest available' falls back to company type where a "
              "sector is too thin to have a median."),
    )

row = universe[universe["ticker"] == ticker].iloc[0]
template = row["template"]

st.title("{} · {}".format(ticker, row["name"]))
meta = [str(row.get("sector") or "").strip(), template.title()]
if row.get("span"):
    meta.append("filings {}".format(row["span"]))
st.caption(" · ".join([m for m in meta if m]))

items = data.item_series([ticker], [], frequency)
items = data.items()
items = items[(items["ticker"] == ticker) & (items["frequency"] == frequency)]
metrics = data.metrics()
metrics = metrics[(metrics["ticker"] == ticker) & (metrics["frequency"] == frequency)]

if items.empty:
    st.warning("No {} data for {}.".format(data.FREQUENCY_LABELS[frequency].lower(), ticker))
    st.stop()

latest = items["period_end"].max()
prior_periods = sorted(items["period_end"].unique())
prior = prior_periods[-2] if len(prior_periods) > 1 else None


def item_at(code: str, period) -> float:
    hit = items[(items["item"] == code) & (items["period_end"] == period)]
    return float(hit["value"].iloc[0]) if len(hit) else float("nan")


def metric_at(mid: str, period) -> float:
    hit = metrics[(metrics["metric"] == mid) & (metrics["period_end"] == period)]
    return float(hit["value"].iloc[0]) if len(hit) else float("nan")


# ---- Headline ------------------------------------------------------------
label = period_label(items[items["period_end"] == latest].iloc[0])
st.subheader("Latest reported period · {}".format(label))

headline = [
    ("Revenue", "revenue", "currency"),
    ("Net income", "net_income", "currency"),
    ("Total assets", "total_assets", "currency"),
    ("Free cash flow", "free_cash_flow", "currency"),
]
cols = st.columns(4)
for col, (title, code, kind) in zip(cols, headline):
    now, before = item_at(code, latest), item_at(code, prior) if prior is not None else float("nan")
    change = None
    if pd.notna(now) and pd.notna(before) and before != 0:
        change = "{:+.1f}%".format((now / before - 1) * 100)
    col.metric(title, currency(now), change)

market = data.market()
market = market[(market["ticker"] == ticker) & (market["frequency"] == frequency)]
if not market.empty:
    at_latest = market[market["period_end"] == latest]
    if not at_latest.empty and pd.notna(at_latest["market_cap"].iloc[0]):
        cols = st.columns(4)
        cols[0].metric("Market cap", currency(at_latest["market_cap"].iloc[0]))
        cols[1].metric("P/E", metric_value(metric_at("pe_ratio", latest), "times"))
        cols[2].metric("EV / EBITDA", metric_value(metric_at("ev_to_ebitda", latest), "times"))
        cols[3].metric("FCF yield", metric_value(metric_at("fcf_yield", latest), "percent"))

st.divider()

# ---- Statements ----------------------------------------------------------
st.subheader("Financial statements")
statement_tabs = st.tabs(["Income statement", "Balance sheet", "Cash flow"])
registry_items = data.item_registry()
roles = {item.code: item.role for item in registry_items}

for tab, statement in zip(statement_tabs,
                          ["income_statement", "balance_sheet", "cash_flow"]):
    with tab:
        part = items[items["item"].isin(
            [i.code for i in registry_items.by_statement(statement)])]
        if part.empty:
            st.info("Nothing reported for this statement.")
            continue

        table = part.pivot_table(index="item", columns="period_end",
                                 values="value", aggfunc="first")
        periods = sorted(table.columns)[-12:]

        # Laid out as the filing presents it, from registry/statements.yaml,
        # not alphabetically — which had put Comprehensive Income above
        # Revenue. Section headings in capitals; a component is indented under
        # the subtotal it feeds, a subtotal or total sits flush.
        layout = data.statement_layout()[statement]
        placed = {code for _, codes in layout for code in codes}
        stray = [c for c in sorted(table.index) if c not in placed]
        labels, rows, ordered = [], [], []
        for section, codes in layout + ([("Other", stray)] if stray else []):
            present = [code for code in codes if code in table.index]
            if not present:
                continue
            labels.append(section.upper())
            rows.append([""] * len(periods))
            for code in present:
                role = roles.get(code, "")
                indent = "" if role in ("total", "subtotal") else "\u00a0" * 4
                labels.append(indent + registry_items.label(code))
                rows.append([item_value(table.at[code, period], role) for period in periods])
                ordered.append(code)
        st.dataframe(
            pd.DataFrame(rows, index=labels,
                         columns=[pd.Timestamp(period).date() for period in periods]),
            width="stretch", height=min(36 * (len(labels) + 1) + 4, 760),
        )

        codes = ordered
        default = "revenue" if "revenue" in codes else codes[0]
        chosen = st.multiselect(
            "Chart these lines", codes,
            default=[default],
            format_func=registry_items.label,
            key="lines_" + statement,
        )
        if chosen:
            series = part[part["item"].isin(chosen)].copy()
            series = series[series["calendar_year"] >= start_year]
            series["label"] = series["item"].map(registry_items.label)
            # Indexed when the lines differ in scale, so one axis stays honest.
            same_scale = len(chosen) == 1 or st.checkbox(
                "Index to 100 at the first period",
                value=len(chosen) > 1, key="idx_" + statement,
                help=("Two lines of different size on one axis is fine; two "
                      "AXES is not. Indexing compares shape without a second scale."),
            ) is False
            figure = (charts.time_series(series, series_col="label", kind="currency",
                                         height=340, namespace="item", basis_col="basis")
                      if same_scale else
                      charts.indexed(series, series_col="label", height=340,
                                     namespace="item"))
            charts.render(
                figure,
                series.assign(
                    Period=series.apply(period_label, axis=1),
                    Value=series["value"].map(currency),
                )[["label", "Period", "Value", "provenance"]].rename(
                    columns={"label": "Line item", "provenance": "Source"}),
                key="chart_" + statement,
            )

st.divider()

# ---- Metrics -------------------------------------------------------------
st.subheader("Ratios and scores")
available = data.available_metrics([template], frequency)
if available.empty:
    st.info("No metrics available at this frequency for a {} filer.".format(template))
    st.stop()

families = list(dict.fromkeys(available["family"]))
family_tabs = st.tabs([f.replace("_", " ").title() for f in families])

for tab, family in zip(family_tabs, families):
    with tab:
        subset = available[available["family"] == family]
        latest_rows = []
        for _, m in subset.iterrows():
            value = metric_at(m["metric"], latest)
            before = metric_at(m["metric"], prior) if prior is not None else float("nan")
            band = metrics[(metrics["metric"] == m["metric"])
                           & (metrics["period_end"] == latest)]
            found = data.peer_series(ticker, m["metric"], frequency, peer_choice)
            versus, peer_n = "", ""
            if found:
                _, _, rows = found
                here = rows[rows["calendar_period"].isin(
                    metrics.loc[metrics["period_end"] == latest, "calendar_period"])]
                if not here.empty and pd.notna(here["median"].iloc[0]) and pd.notna(value):
                    median = float(here["median"].iloc[0])
                    higher_better = m["direction"] != "lower_better"
                    ahead = (value > median) if higher_better else (value < median)
                    neutral = m["direction"] == "neutral"
                    mark = "–" if neutral else ("▲" if ahead else "▼")
                    versus = "{} {}".format(mark, metric_value(median, m["format"]))
                    peer_n = "{:.0f}".format(here["n"].iloc[0])
            latest_rows.append({
                "Metric": m["label"],
                "Latest": metric_value(value, m["format"]),
                "Prior": metric_value(before, m["format"]),
                "Peer median": versus,
                "Peers": peer_n,
                "Band": (band["band"].iloc[0] if len(band) and "band" in band
                         and pd.notna(band["band"].iloc[0]) else ""),
            })
        st.dataframe(pd.DataFrame(latest_rows), width="stretch", hide_index=True)
        st.caption(
            "**Peer median** is the group's value in the same calendar period. "
            "▲ means this company is on the better side of it for that metric, "
            "▼ the worse; – where the metric has no inherent direction."
        )

        chosen = st.selectbox(
            "Chart a metric", list(subset["metric"]),
            format_func=lambda m: subset.set_index("metric")["label"].get(m, m),
            key="metric_" + family,
        )
        if chosen:
            spec = registry.get(chosen)
            series = metrics[(metrics["metric"] == chosen)
                             & (metrics["calendar_year"] >= start_year)]
            if series.empty:
                st.info("No values for this metric in the selected period.")
            else:
                found = data.peer_series(ticker, chosen, frequency, peer_choice)
                peer_frame = None
                peer_note = None
                if found:
                    kind, group, rows = found
                    peer_frame = peer_lib.attach(
                        series[["period_end", "calendar_period"]].drop_duplicates(),
                        rows)
                    peer_note = peer_lib.describe(
                        kind, group,
                        peer_frame["n"].dropna().max() if "n" in peer_frame else None)
                figure = charts.time_series(
                    series, series_col="ticker", kind=spec.format, height=340,
                    title=spec.label, label_last=False, peer=peer_frame,
                    basis_col="basis")
                if spec.bands:
                    for band in spec.bands:
                        if "max" in band:
                            figure.add_hline(
                                y=band["max"], line=dict(width=1, dash="dot",
                                                         color=charts.theme.STATUS.get(
                                                             band.get("tone"), "#898781")),
                                annotation_text=band["label"],
                                annotation_position="right",
                                annotation_font=dict(size=10),
                            )
                charts.render(
                    figure,
                    series.assign(
                        Period=series.apply(period_label, axis=1),
                        Value=series["value"].map(
                            lambda v: metric_value(v, spec.format)),
                    )[["Period", "Value"] + (["band"] if "band" in series else [])],
                    key="metric_chart_" + family,
                )
                if peer_note:
                    st.caption(peer_note)
                elif peer_choice != "auto":
                    st.caption(
                        "No peer group at that level for this metric — the "
                        "group needs at least three companies reporting it.")
                if spec.description:
                    st.caption(spec.description)
