"""The shared filter row.

One set of controls, above everything they scope, held in session state so a
selection survives moving between pages. Per-chart filters are deliberately
absent: when each card filters itself, two charts side by side can silently be
showing different slices of the data.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st

from . import data

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA"]


def frequency_control(key: str = "frequency", allowed: Optional[List[str]] = None) -> str:
    options = allowed or ["annual", "ttm", "quarterly"]
    labels = [data.FREQUENCY_LABELS[o] for o in options]
    current = st.session_state.get(key, options[0])
    index = options.index(current) if current in options else 0
    chosen = st.radio(
        "Frequency", labels, index=index, horizontal=True, key=key + "_radio",
        help=(
            "Annual and Trailing 12M carry a full year in the numerator. "
            "Quarterly is offered only for metrics where a single quarter is "
            "meaningful — a margin, not a return on assets."
        ),
    )
    value = options[labels.index(chosen)]
    st.session_state[key] = value
    return value


def company_control(
    label: str = "Companies",
    default: Optional[List[str]] = None,
    max_selections: Optional[int] = None,
    key: str = "tickers",
) -> List[str]:
    frame = data.companies()
    options = list(frame["ticker"])
    names = dict(zip(frame["ticker"], frame["name"]))

    if key not in st.session_state:
        seed = [t for t in (default or DEFAULT_TICKERS) if t in options]
        st.session_state[key] = seed or options[:3]

    return st.multiselect(
        label, options, key=key,
        max_selections=max_selections,
        format_func=lambda t: "{} · {}".format(t, names.get(t, t)),
        help=("Eight is the limit for distinct colours. Beyond that, compare in "
              "the screener table instead." if max_selections else None),
    )


def sector_control(key: str = "sectors") -> List[str]:
    frame = data.companies()
    if "sector" not in frame.columns:
        return []
    options = sorted(frame["sector"].dropna().unique())
    return st.multiselect("Sector", options, key=key)


def template_control(key: str = "templates") -> List[str]:
    labels = {"industrial": "Industrial", "bank": "Bank",
              "insurer": "Insurer", "reit": "REIT"}
    options = list(labels)
    return st.multiselect(
        "Company type", options, key=key, format_func=lambda t: labels[t],
        help=("Metrics differ by type. A bank has no gross margin; a combined "
              "ratio means nothing outside insurance."),
    )


def year_control(key: str = "start_year", floor: int = 2006) -> int:
    current_max = pd.Timestamp.now().year
    return st.slider(
        "From year", min_value=floor, max_value=current_max - 1,
        value=st.session_state.get(key, 2015), key=key,
    )


def apply_universe(
    tickers: Optional[List[str]] = None,
    sectors: Optional[List[str]] = None,
    templates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """The company rows surviving the current filters."""
    frame = data.companies()
    if tickers:
        frame = frame[frame["ticker"].isin(tickers)]
    if sectors and "sector" in frame.columns:
        frame = frame[frame["sector"].isin(sectors)]
    if templates:
        frame = frame[frame["template"].isin(templates)]
    return frame


def metric_control(
    templates: List[str],
    frequency: str,
    label: str = "Metric",
    default: Optional[str] = None,
    key: str = "metric",
    family: Optional[str] = None,
) -> Optional[str]:
    """Pick from the metrics that are valid AND populated for this selection."""
    available = data.available_metrics(templates or ["industrial"], frequency)
    if family:
        available = available[available["family"] == family]
    if available.empty:
        st.info("No metrics available for this combination of company type and frequency.")
        return None

    return grouped_selectbox(label, available, default=default, key=key)


FAMILY_LABELS = {
    "growth": "Growth", "profitability": "Profitability", "returns": "Returns",
    "liquidity": "Liquidity", "leverage": "Leverage", "efficiency": "Efficiency",
    "cash_quality": "Cash Quality", "valuation": "Valuation",
    "forensic": "Forensic", "bank": "Bank", "insurer": "Insurer", "reit": "REIT",
}
# The order families appear in every picker. Roughly how an analyst reads a
# company — what it earns, what it returns, what it owes — rather than
# alphabetical, which would open on "Bank".
FAMILY_ORDER = ["growth", "profitability", "returns", "efficiency",
                "cash_quality", "liquidity", "leverage", "valuation",
                "forensic", "bank", "insurer", "reit"]


def group_metrics(available: pd.DataFrame) -> pd.DataFrame:
    """Order a metric table by family, then label.

    Streamlit's selectbox has no option groups, so the family travels in the
    option text instead — "Growth · Revenue Growth". Sorting by family then
    makes the list read as sections, which is the next best thing to real
    headings and the reason a long list stays navigable.
    """
    if available.empty:
        return available
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return available.assign(
        _family_rank=available["family"].map(lambda f: rank.get(f, 99))
    ).sort_values(["_family_rank", "label"]).drop(columns="_family_rank")


HEADER_PREFIX = "__family__"


def _short_label(label: str, family: str) -> str:
    """Drop a family word the header already carries.

    Under a "Growth" heading, "Revenue Growth" reads as "Growth / Revenue
    Growth". Trimming the redundant suffix gives Revenue, EBITDA, Net Income —
    which is how anyone would write the list by hand. Only fires where the
    label genuinely ends with the family word, so "Return on Assets" under
    Returns is left alone.
    """
    suffix = " " + FAMILY_LABELS.get(family, "")
    if len(suffix) > 1 and label.endswith(suffix) and len(label) > len(suffix) + 1:
        return label[: -len(suffix)]
    return label


def grouped_options(available: pd.DataFrame):
    """Option ids with non-selectable family headings woven in.

    Streamlit's selectbox has no option groups and cannot disable a single
    option, so a heading has to BE an option. It is rendered as a heading and
    `resolve_grouped` bounces any attempt to choose one, which makes it inert
    in practice — it still highlights on hover, which is the one part of a real
    option group that cannot be reproduced natively.
    """
    options, seen = [], set()
    for _, row in group_metrics(available).iterrows():
        family = row["family"]
        if family not in seen:
            seen.add(family)
            options.append(HEADER_PREFIX + family)
        options.append(row["metric"])
    return options


def grouped_label(available: pd.DataFrame, selected: Optional[str] = None):
    """format_func for `grouped_options`: headings plain, metrics indented.

    `selected` is rendered in full, unindented, because Streamlit uses one
    format_func for both the open list and the closed box. Shortening works
    only while the heading is visible above it: in the list, "Revenue" sits
    under "— Growth —" and plainly means revenue growth, but the closed box
    shows the option alone, where "Revenue" reads as the revenue level. The
    selected option therefore keeps its full label, which also marks it out
    in the open list.
    """
    labels = dict(zip(available["metric"], available["label"]))
    families = dict(zip(available["metric"], available["family"]))

    def render(option: str) -> str:
        if option.startswith(HEADER_PREFIX):
            family = option[len(HEADER_PREFIX):]
            return "— {} —".format(
                FAMILY_LABELS.get(family, family.replace("_", " ").title()))
        label = labels.get(option, option)
        if option == selected:
            return label
        # Non-breaking: HTML collapses a run of ordinary spaces, so plain
        # indentation never reaches the rendered list.
        return "\u00a0" * 4 + _short_label(label, families.get(option, ""))

    return render


def resolve_grouped(chosen: str, options, previous: Optional[str] = None) -> str:
    """Turn a possibly-heading selection back into a real metric."""
    if chosen and not chosen.startswith(HEADER_PREFIX):
        return chosen
    if previous and not previous.startswith(HEADER_PREFIX):
        return previous
    return next((o for o in options if not o.startswith(HEADER_PREFIX)), chosen)


def _bounce_heading(key: str, fallback: str) -> None:
    """Undo a heading selection, from inside the widget's own callback.

    This has to be a callback. Streamlit refuses to let a widget's key be
    written after that widget is instantiated, so correcting the value in the
    script body raises StreamlitAPIException — but a callback runs before the
    rerun that rebuilds the widget, where the write is allowed.
    """
    chosen = st.session_state.get(key)
    if not chosen or not str(chosen).startswith(HEADER_PREFIX):
        return
    previous = st.session_state.get(key + "__resolved")
    st.session_state[key] = (
        previous if previous and not str(previous).startswith(HEADER_PREFIX)
        else fallback
    )


def grouped_selectbox(
    label: str,
    available: pd.DataFrame,
    default: Optional[str] = None,
    key: str = "metric",
    help: Optional[str] = None,
) -> str:
    """A metric picker with family headings that behave as labels.

    Streamlit has no option groups and cannot disable a single option, so a
    heading has to be an option; picking one bounces straight back to the
    previous choice. It still highlights on hover — that is the one part of a
    real option group that cannot be reproduced natively.
    """
    options = grouped_options(available)
    real = [o for o in options if not o.startswith(HEADER_PREFIX)]
    if not real:
        return ""

    previous = st.session_state.get(key + "__resolved")
    start = default if default in real else (previous if previous in real else real[0])
    current = st.session_state.get(key)
    index = options.index(current) if current in options else options.index(start)

    showing = current if current in real else start
    chosen = st.selectbox(
        label, options, index=index, key=key,
        format_func=grouped_label(available, showing), help=help,
        on_change=_bounce_heading, args=(key, start),
    )
    resolved = resolve_grouped(chosen, options, previous)
    st.session_state[key + "__resolved"] = resolved
    return resolved


def metric_option_label(available: pd.DataFrame):
    """A format_func showing "Family · Metric" for a metric id."""
    labels = dict(zip(available["metric"], available["label"]))
    families = dict(zip(available["metric"], available["family"]))

    def render(metric_id: str) -> str:
        family = families.get(metric_id, "")
        return "{} · {}".format(
            FAMILY_LABELS.get(family, family.replace("_", " ").title()),
            labels.get(metric_id, metric_id))

    return render


def guard_dataset() -> bool:
    """Refuse to render pages against a dataset that has not been built."""
    if data.dataset_is_built():
        return True
    st.error("No curated dataset found.")
    st.markdown(
        "Build it first — this reads the extractor's output and writes parquet "
        "into `data/curated`:"
    )
    st.code("~/.venvs/findash/bin/python -m findash.build", language="bash")
    return False
