"""Fiscal-to-calendar alignment.

11 of the 47 filers close their year somewhere other than December. Comparing
them on raw period_end puts Apple's September year against Microsoft's June
year against NVIDIA's January year and calls the result a peer group. Every
cross-company view needs a shared calendar axis instead.

Two different alignments, because the two frequencies need different rules:

  Annual    the Compustat convention — a fiscal year ending in January through
            May belongs to the PRIOR calendar year. NVIDIA's year ending
            2025-01-26 is mostly calendar 2024 and is compared as such.

  Quarterly snap the period end to the nearest calendar quarter end. A quarter
            ending 2024-09-28 is 2024Q3; one ending 2025-01-26 is 2024Q4,
            because it is 26 days from 31 December and 64 from 31 March.

Neither is a fact about the filing — both are analytical choices — so the
fiscal label is always carried alongside, and the dashboard can show either.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# A fiscal year ending on or before this month belongs to the prior calendar
# year. May is the conventional cut: a year ending in May has seven of its
# twelve months in the previous calendar year.
PRIOR_YEAR_CUTOFF_MONTH = 5



# Four consecutive quarter-ends span three quarters, ~273 days, or up to ~287
# for a 52/53-week filer with a 14-week quarter in the window. A skipped
# quarter pushes it past 360.
MAX_TTM_SPAN_DAYS = 300

def calendar_year(period_end: pd.Series) -> pd.Series:
    """Calendar year an annual period is compared under."""
    dates = pd.to_datetime(period_end)
    year = dates.dt.year
    return year.where(dates.dt.month > PRIOR_YEAR_CUTOFF_MONTH, year - 1).astype("Int64")


def calendar_quarter_end(period_end: pd.Series) -> pd.Series:
    """The calendar quarter end a period end snaps to.

    Distance is measured to both the quarter end at or before the date and the
    one after it, and the nearer wins. This handles 52/53-week calendars, where
    a "quarter end" can land a few days either side of the calendar boundary.
    """
    dates = pd.to_datetime(period_end)
    prior = dates + pd.offsets.QuarterEnd(0)          # this quarter end, or today if it is one
    prior = prior.where(prior >= dates, dates + pd.offsets.QuarterEnd(1))
    previous = dates - pd.offsets.QuarterEnd(1)
    take_prev = (dates - previous).abs() < (prior - dates).abs()
    return prior.where(~take_prev, previous)


def add_calendar_columns(frame: pd.DataFrame, period_col: str = "period_end") -> pd.DataFrame:
    """Attach the comparison axis: calendar year, quarter end, and a label.

    `calendar_period` is the single column cross-company charts group on —
    "2024" for annual rows, "2024Q3" for quarterly ones.
    """
    out = frame.copy()
    dates = pd.to_datetime(out[period_col])
    is_annual = out["frequency"].eq("annual")

    cal_year = calendar_year(dates)
    q_end = calendar_quarter_end(dates)
    q_year = q_end.dt.year
    q_num = q_end.dt.quarter

    out["calendar_year"] = cal_year.where(is_annual, q_year.astype("Int64"))
    out["calendar_quarter"] = pd.Series(q_num, index=out.index).where(~is_annual).astype("Int64")
    out["calendar_quarter_end"] = q_end.where(~is_annual)
    out["calendar_period"] = np.where(
        is_annual,
        cal_year.astype("string"),
        q_year.astype("string") + "Q" + q_num.astype("string"),
    )

    # One date column usable as an x-axis regardless of frequency, so that
    # companies with different fiscal year ends land on the SAME point when
    # plotted together. Annual rows get 31 December of their calendar year
    # (a display date, not a claim about when the fiscal year actually ended);
    # quarterly and TTM rows already have a real calendar quarter end.
    #
    # Without this, a multi-company "indexed to 100" chart rebases each line
    # at its own fiscal date — Apple in September, Microsoft in June, NVIDIA
    # in January — so the lines visibly start and end at different x
    # positions even though they are meant to be compared period for period.
    annual_end = pd.to_datetime(
        cal_year.astype("Int64").astype("string") + "-12-31", errors="coerce"
    )
    out["calendar_period_end"] = annual_end.where(is_annual, q_end)
    return out


def ttm_frame(
    wide: pd.DataFrame,
    flow_items: set,
    mean_items: set = frozenset(),
    min_quarters: int = 4,
) -> pd.DataFrame:
    """Trailing-twelve-month view, built from the discrete quarterly frame.

    The extractor emits DISCRETE quarters, not year-to-date cumulations (verified
    against the panel), so a trailing year is a plain 4-quarter rolling sum —
    no back-out needed here.

    Flows are summed. Instants are NOT: a rolling sum of a balance sheet is
    meaningless, so the balance at the end of the window carries through
    unchanged.

    Weighted-average share counts are a third case. They are flows — each
    belongs to its quarter — but they are averages, and averages do not add:
    summing four quarters of diluted shares reports a company with four times
    its actual equity. They take a rolling MEAN instead. Per-share figures are
    left to sum, because trailing EPS conventionally is the sum of four
    quarterly EPS.

    Requires a full four quarters; a partial window would understate every flow
    and is left null instead.
    """
    if wide.empty:
        return wide

    quarterly = wide[wide.index.get_level_values("frequency") == "quarterly"]
    if quarterly.empty:
        return quarterly

    out = []
    for ticker, group in quarterly.groupby(level="ticker", sort=False):
        group = group.sort_index(level="period_end")
        result = group.copy()

        # The window counts rows, and a missing quarter means four rows reach
        # back fifteen months. ConocoPhillips' trailing year to March 2011 and
        # Digital Realty's to December 2010 were both sums of four quarters
        # that were not the trailing four. Consecutive quarter-ends sit about
        # 273 days apart end to end; anything wider has skipped one.
        ends = pd.Series(pd.to_datetime(group.index.get_level_values("period_end")),
                         index=group.index)
        consecutive = (ends - ends.shift(3)).dt.days <= MAX_TTM_SPAN_DAYS

        flows = [c for c in group.columns if c in flow_items and c not in mean_items]
        if flows:
            result[flows] = (
                group[flows].rolling(window=4, min_periods=min_quarters).sum()
            ).where(consecutive, axis=0)
        averages = [c for c in group.columns if c in mean_items]
        if averages:
            result[averages] = (
                group[averages].rolling(window=4, min_periods=min_quarters).mean()
            ).where(consecutive, axis=0)
        out.append(result)

    ttm = pd.concat(out)
    # Rebuild the index rather than relabelling the level in place: the sliced
    # frame still carries "annual" among its level categories, and set_levels
    # would leave the codes pointing past the end of the new level.
    ttm.index = pd.MultiIndex.from_arrays(
        [
            ttm.index.get_level_values("ticker"),
            pd.Index(["ttm"] * len(ttm), name="frequency"),
            ttm.index.get_level_values("period_end"),
        ]
    )
    return ttm
