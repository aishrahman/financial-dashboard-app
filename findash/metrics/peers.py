"""Peer-group benchmarks.

A ratio on its own answers "what is it?". Set against comparable companies it
answers "is that good?", which is the question anyone actually has. This
computes, for every metric and period, the distribution across a peer group —
median and interquartile range — so a company's line can be read against the
band its peers occupy.

Three grouping levels, because one is not enough for this universe:

  sector    the most meaningful comparison, but thin. Only 7 of 15 SIC-derived
            sectors hold three or more filers, covering 33 of the 47.
  template  industrial, bank, insurer or REIT. Always at least five members,
            so always available, at the cost of comparing a software company
            against a railway.
  all       every filer. Coarse, and honest about being coarse.

The app picks the narrowest group that has enough members and says which it
used. A "median" of one company is not a benchmark, so groups below
MIN_PEERS produce nothing rather than a comforting line.

Alignment is on CALENDAR period, never fiscal period end. Apple's September
year and Microsoft's June year are not the same year, and averaging them on
filing date would silently compare different economic periods.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

# Below three members a median is not a distribution, it is a coin toss.
MIN_PEERS = 3
# The interquartile band needs a few more before its edges mean anything.
MIN_FOR_BAND = 5

ALL_GROUP = "All companies"
GROUP_KINDS = ("sector", "template", "all")

# An empty result still carries these, so callers can filter it without
# reaching for a column that is not there.
COLUMNS = ["group_kind", "group", "metric", "frequency", "calendar_period",
           "n", "p25", "median", "p75"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def build_peer_stats(
    metrics: pd.DataFrame,
    companies: pd.DataFrame,
    min_peers: int = MIN_PEERS,
) -> pd.DataFrame:
    """Distribution of each metric across each peer group, per calendar period.

    Returns one row per (group_kind, group, metric, frequency, calendar_period)
    with the member count and the quartiles. The count travels with the numbers
    because a median of three and a median of thirty are not equally
    trustworthy, and the reader should be able to see which they have.
    """
    if metrics.empty:
        return _empty()

    lookup = companies.set_index("ticker")
    frame = metrics.copy()
    frame["template"] = frame["ticker"].map(lookup["template"])
    if "sector" in lookup.columns:
        frame["sector"] = frame["ticker"].map(lookup["sector"])
    else:
        frame["sector"] = None
    frame["all"] = ALL_GROUP

    keys = ["metric", "frequency", "calendar_period"]
    out: List[pd.DataFrame] = []

    for kind in GROUP_KINDS:
        if kind not in frame.columns:
            continue
        part = frame[frame[kind].notna()]
        if part.empty:
            continue

        grouped = part.groupby([kind] + keys, observed=True)["value"]
        stats = grouped.agg(
            n="count",
            p25=lambda s: s.quantile(0.25),
            median="median",
            p75=lambda s: s.quantile(0.75),
        ).reset_index()
        stats = stats[stats["n"] >= min_peers]
        if stats.empty:
            continue
        stats = stats.rename(columns={kind: "group"})
        stats.insert(0, "group_kind", kind)
        out.append(stats)

    if not out:
        log.info("no peer group reached %d members", min_peers)
        return _empty()

    peers = pd.concat(out, ignore_index=True)
    log.info("peer stats: %d rows across %d group kinds",
             len(peers), peers["group_kind"].nunique())
    return peers.sort_values(
        ["group_kind", "group", "metric", "frequency", "calendar_period"],
        ignore_index=True)


def groups_for(companies: pd.DataFrame, ticker: str) -> Dict[str, str]:
    """The peer group this company belongs to at each level."""
    row = companies[companies["ticker"] == ticker]
    if row.empty:
        return {}
    row = row.iloc[0]
    found = {"all": ALL_GROUP}
    if row.get("template"):
        found["template"] = row["template"]
    if row.get("sector") and pd.notna(row.get("sector")):
        found["sector"] = row["sector"]
    return found


def best_group(
    peers: pd.DataFrame,
    companies: pd.DataFrame,
    ticker: str,
    metric: str,
    frequency: str,
) -> Optional[tuple]:
    """The narrowest peer group with data for this company and metric.

    Sector first, then company type, then everything — narrowest wins because
    a software company is better judged against other software companies than
    against a railway, but a comparison that exists beats one that does not.
    """
    available = groups_for(companies, ticker)
    for kind in GROUP_KINDS:
        group = available.get(kind)
        if not group:
            continue
        rows = peers[
            (peers["group_kind"] == kind)
            & (peers["group"] == group)
            & (peers["metric"] == metric)
            & (peers["frequency"] == frequency)
        ]
        if not rows.empty:
            return kind, group, rows
    return None


def attach(
    series: pd.DataFrame,
    peer_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Join peer statistics onto a company's own periods.

    Joined on calendar period, then drawn at the company's own period end, so
    the band sits under the line at the right place on the x-axis even when the
    filer's year does not end in December.
    """
    if series.empty or peer_rows.empty:
        return series
    columns = ["calendar_period", "n", "p25", "median", "p75"]
    return series.merge(peer_rows[columns], on="calendar_period", how="left")


LABELS = {
    "sector": "sector",
    "template": "company type",
    "all": "all companies",
}


def describe(kind: str, group: str, n: Optional[float]) -> str:
    """A caption saying exactly what the band represents."""
    label = LABELS.get(kind, kind)
    name = group.title() if kind == "template" else group
    count = " · {:.0f} companies".format(n) if n and pd.notna(n) else ""
    return "Peer group: {} ({}){}".format(name, label, count)
