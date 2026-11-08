"""Loading the curated dataset, once per session.

Everything here reads parquet written by `python -m findash.build`. The app
never touches the extractor, the fact store or the network — which is what
keeps it responsive and what makes a yfinance outage a stale valuation column
rather than a broken dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from findash.config import Settings                      # noqa: E402
from findash.metrics import peers as peer_lib             # noqa: E402
from findash.metrics.engine import MetricRegistry        # noqa: E402
from findash.registry import ItemRegistry                # noqa: E402

FREQUENCY_LABELS = {
    "annual": "Annual",
    "quarterly": "Quarterly",
    "ttm": "Trailing 12M",
}


@st.cache_resource
def settings() -> Settings:
    return Settings.load()


@st.cache_resource
def item_registry() -> ItemRegistry:
    return ItemRegistry.load()


@st.cache_resource
def statement_layout():
    from findash.registry import load_statement_layout
    return load_statement_layout()


@st.cache_resource
def metric_registry() -> MetricRegistry:
    return MetricRegistry.load(items=item_registry())


# The columns a page may filter on, for artifacts that can legitimately be
# absent. The published app withholds market.parquet, and an absent artifact
# used to come back as a frame with NO columns — so `frame["ticker"]` raised
# KeyError and the public Company page crashed on load. It had done so since
# the first public deploy, because the bundle was checked on a sample of pages
# that did not include it.
_EMPTY_COLUMNS = {
    "market": ["ticker", "frequency", "period_end", "close", "adj_close",
               "shares_reported", "shares_adjusted", "shares_source", "market_cap"],
    "revisions": ["ticker", "frequency", "period_end", "item", "label",
                  "first_value", "value", "change", "pct_change",
                  "first_filed", "revised_filed", "versions"],
}


def _read(name: str) -> pd.DataFrame:
    path = settings().curated_dir / "{}.parquet".format(name)
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=_EMPTY_COLUMNS.get(name, []))


@st.cache_data(show_spinner=False)
def items() -> pd.DataFrame:
    return _read("items")


@st.cache_data(show_spinner=False)
def metrics() -> pd.DataFrame:
    return _read("metrics")


@st.cache_data(show_spinner=False)
def market() -> pd.DataFrame:
    return _read("market")


@st.cache_data(show_spinner=False)
def quality() -> pd.DataFrame:
    return _read("quality")


@st.cache_data(show_spinner=False)
def coverage() -> pd.DataFrame:
    return _read("coverage")


@st.cache_data(show_spinner=False)
def metric_report() -> pd.DataFrame:
    return _read("metric_report")


@st.cache_data(show_spinner=False)
def segments() -> pd.DataFrame:
    return _read("segments")


@st.cache_data(show_spinner=False)
def footnotes() -> pd.DataFrame:
    return _read("footnotes")


@st.cache_data(show_spinner=False)
def detail_summary() -> pd.DataFrame:
    return _read("detail_summary")


@st.cache_data(show_spinner=False)
def revisions() -> pd.DataFrame:
    return _read("revisions")


@st.cache_data(show_spinner=False)
def revision_report() -> pd.DataFrame:
    return _read("revision_report")


@st.cache_data(show_spinner=False)
def peers() -> pd.DataFrame:
    return _read("peers")


@st.cache_data(show_spinner=False)
def companies() -> pd.DataFrame:
    """One row per company: template, sector, name and quality context."""
    frame = items()[["ticker", "cik", "template"]].drop_duplicates("ticker")
    info = quality()
    if not info.empty:
        keep = [c for c in ("ticker", "name", "sector", "sic_description", "exchange",
                            "fill_rate", "identity_failures", "reconciliation_gaps",
                            "unmapped_concepts", "fiscal_year_end", "span")
                if c in info.columns]
        frame = frame.merge(info[keep], on="ticker", how="left")
    if "name" not in frame.columns:
        frame["name"] = frame["ticker"]
    frame["name"] = frame["name"].fillna(frame["ticker"])
    return frame.sort_values("ticker", ignore_index=True)


def dataset_is_built() -> bool:
    return not items().empty


def has_market_data() -> bool:
    return not market().empty


@st.cache_data(show_spinner=False)
def last_built() -> Optional[pd.Timestamp]:
    path = settings().curated_dir / "items.parquet"
    if not path.exists():
        return None
    return pd.Timestamp(path.stat().st_mtime, unit="s")


def metric_series(
    tickers: List[str],
    metric_ids: List[str],
    frequency: str,
    start_year: Optional[int] = None,
) -> pd.DataFrame:
    frame = metrics()
    frame = frame[
        frame["ticker"].isin(tickers)
        & frame["metric"].isin(metric_ids)
        & (frame["frequency"] == frequency)
    ]
    # Filtered on CALENDAR year, not the fiscal period's own year: NVIDIA's
    # year ending 2015-01-26 has period_end.dt.year == 2015 but belongs to
    # calendar_year 2014 (the five-month cutoff rule), so a period_end filter
    # let it through the "from 2015" slider a year early relative to Apple and
    # Microsoft — exactly the misalignment the chart's x-axis was fixed for.
    if start_year is not None:
        frame = frame[frame["calendar_year"] >= start_year]
    return frame.sort_values(["ticker", "metric", "period_end"])


def item_series(
    tickers: List[str],
    item_codes: List[str],
    frequency: str,
    start_year: Optional[int] = None,
) -> pd.DataFrame:
    frame = items()
    frame = frame[
        frame["ticker"].isin(tickers)
        & frame["item"].isin(item_codes)
        & (frame["frequency"] == frequency)
    ]
    if start_year is not None:  # see metric_series() for why calendar_year
        frame = frame[frame["calendar_year"] >= start_year]
    return frame.sort_values(["ticker", "item", "period_end"])


def latest_period(ticker: str, frequency: str = "annual") -> Optional[pd.Timestamp]:
    frame = items()
    rows = frame[(frame["ticker"] == ticker) & (frame["frequency"] == frequency)]
    return rows["period_end"].max() if len(rows) else None


def peer_series(
    ticker: str,
    metric: str,
    frequency: str,
    prefer: Optional[str] = None,
) -> Optional[tuple]:
    """Peer statistics for one company and metric.

    Returns (kind, group, rows) for the narrowest group that has data, or the
    group named by `prefer` when it does. None when no group qualifies — with
    fourteen of the forty-seven filers in a sector of one or two, that is a
    real outcome and the page says so rather than drawing an empty band.
    """
    frame = peers()
    if frame.empty:
        return None
    if prefer and prefer != "auto":
        group = peer_lib.groups_for(companies(), ticker).get(prefer)
        if group:
            rows = frame[
                (frame["group_kind"] == prefer) & (frame["group"] == group)
                & (frame["metric"] == metric) & (frame["frequency"] == frequency)
            ]
            if not rows.empty:
                return prefer, group, rows
        return None
    return peer_lib.best_group(frame, companies(), ticker, metric, frequency)


def available_metrics(templates: List[str], frequency: str) -> pd.DataFrame:
    """Metrics that are valid for these templates AND actually have values.

    Offering a metric the current selection can never produce is worse than
    omitting it: the reader picks it and sees an empty chart with no reason.
    """
    registry = metric_registry()
    frame = metrics()
    live = set(frame.loc[frame["frequency"] == frequency, "metric"].unique())

    rows = []
    for metric in registry:
        if metric.id not in live:
            continue
        if not any(t in metric.applies_to for t in templates):
            continue
        rows.append({
            "metric": metric.id, "label": metric.label, "family": metric.family,
            "format": metric.format, "direction": metric.direction,
            "description": metric.description,
        })
    return pd.DataFrame(rows)
