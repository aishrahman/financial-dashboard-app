"""Turning numbers into text a reader can scan.

Formatting is driven by each metric's declared `format`, so a percentage never
renders as a raw ratio and a currency figure never loses its unit.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def currency(value: Optional[float], places: int = 1) -> str:
    """Compact currency: 391.0bn rather than 391,035,000,000."""
    if value is None or pd.isna(value):
        return "—"
    sign = "-" if value < 0 else ""
    magnitude = abs(float(value))
    for cutoff, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if magnitude >= cutoff:
            return "{}${:,.{}f}{}".format(sign, magnitude / cutoff, places, suffix)
    return "{}${:,.0f}".format(sign, magnitude)


def metric_value(value: Optional[float], kind: str = "ratio") -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if kind == "percent":
        return "{:.1f}%".format(value * 100)
    if kind == "times":
        return "{:.2f}x".format(value)
    if kind == "days":
        return "{:.0f} d".format(value)
    if kind == "score":
        return "{:.2f}".format(value)
    if kind == "currency":
        return currency(value)
    if kind == "count":
        return "{:,.0f}".format(value)
    return "{:.3f}".format(value)


def axis_format(kind: str) -> Optional[str]:
    """A Plotly tick format matching the metric's units."""
    return {
        "percent": ".0%",
        "times": ".1f",
        "days": ".0f",
        "score": ".1f",
        # Currency deliberately returns None: d3's SI notation renders a
        # billion as "400G", which no one reads as money. Currency axes get
        # explicit ticks from `currency()` instead — see charts._currency_axis.
        "currency": None,
        "count": ",.0f",
        "ratio": ".2f",
    }.get(kind)


def hover_format(kind: str) -> str:
    return {
        "percent": ".2%",
        "times": ".2f",
        "days": ".0f",
        "score": ".2f",
        "currency": "$,.0f",
        "count": ",.0f",
        "ratio": ".3f",
    }.get(kind, ".3f")


def period_label(row) -> str:
    """Prefer the filer's own label; fall back to the date."""
    label = row.get("fiscal_label")
    if isinstance(label, str) and label.strip():
        return label
    end = row.get("period_end")
    return str(pd.to_datetime(end).date()) if end is not None else "—"


def delta(current: Optional[float], prior: Optional[float], kind: str = "ratio") -> str:
    if current is None or prior is None or pd.isna(current) or pd.isna(prior):
        return "—"
    if kind == "percent":
        return "{:+.1f} pt".format((current - prior) * 100)
    if prior == 0:
        return "—"
    return "{:+.1f}%".format((current / prior - 1) * 100)


def item_value(value: Optional[float], role: str = "") -> str:
    """A statement line formatted for what it is, not as if it were all money.

    Every line used to go through currency(), which rounds to whole dollars at
    this scale: Apple's diluted EPS of $7.46 printed as "$7", and a share count
    would have printed with a dollar sign.
    """
    if value is None or pd.isna(value):
        return "—"
    if role == "per_share":
        return "{}${:,.2f}".format("-" if value < 0 else "", abs(float(value)))
    if role == "count":
        return currency(value).replace("$", "")
    return currency(value)
