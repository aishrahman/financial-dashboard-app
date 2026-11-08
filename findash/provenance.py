"""How a value came to exist, and how to say so.

Kept apart from the derivation code on purpose, with nothing but pandas behind
it. The pages need this vocabulary to describe a value, and the published app
carries only the modules the pages import. Living inside derive.py, it took
the fact store and the whole build layer along with it — and when the bundle
left derive.py out, the public Home page failed to import for a month.

Provenance reads "source[:detail][|flag]":

    reported                      tagged by the filer for this period
    reported|sign-normalized      ...with its sign corrected
    ttm                           four quarters summed here
    differenced                   a quarter backed out of year-to-date figures
    summed:1/2                    one of two expected parts present — the total
                                  may be incomplete
    derived:total_assets - ...    an identity applied here
    derived:extractor:...         an identity applied upstream, its formula
    store:Assets                  read from the fact store
"""
from __future__ import annotations

import pandas as pd

REPORTED = "reported"

# The extractor labels each panel row with how it got the number, and those
# labels are not interchangeable with "reported". Mapped straight through
# rather than collapsed, because collapsing them is exactly the error this
# exists to prevent: 43,194 values that the extractor had already told us were
# computed were being presented as filed.
EXTRACTOR_SOURCE = {
    "reported": REPORTED,
    "differenced": "differenced",   # a quarter backed out of year-to-date
    "summed": "summed",             # components added up
    "derived": "derived:extractor",  # an identity applied upstream
}

# Every source this pipeline can attach, and how to describe it to a reader.
# One definition, so the pages cannot drift from the data or from each other.
SOURCE_LABELS = {
    REPORTED: "Straight from the filing as tagged",
    "differenced": "A quarter the filer never reported, backed out of "
                   "year-to-date figures",
    "summed": "Components added up by the extractor",
    "derived": "Recovered from an accounting identity",
    "store": "An XBRL concept the panel does not map",
    "ttm": "A trailing-twelve-month sum of four quarters",
}
SOURCE_ORDER = (REPORTED, "ttm", "derived", "differenced", "summed", "store")

# What a reader would not call "as filed".
INFERRED_SOURCES = ("derived", "differenced", "summed", "store")

_PROVENANCE_SOURCE = r"^([a-z_]+)"
_PARTS = r"^summed:(\d+)/(\d+)"

# What a chart needs to know about a point: was it filed, worked out, or worked
# out from fewer parts than the total needs. Ordered, so a value built from
# several inputs takes the weakest of them.
BASIS_ORDER = ("reported", "inferred", "partial")
BASIS_LABELS = {
    "reported": "As filed",
    "inferred": "Computed, not as filed",
    "partial": "Summed from some of its parts — may be incomplete",
}


def provenance_source(provenance: pd.Series) -> pd.Series:
    """The bare source of a value: reported, store, derived or ttm."""
    return provenance.astype("string").str.extract(_PROVENANCE_SOURCE, expand=False)


def was_sign_normalized(provenance: pd.Series) -> pd.Series:
    return provenance.astype("string").str.contains("sign-normalized", na=False)


def is_partial(provenance: pd.Series) -> pd.Series:
    """Summed from fewer parts than the total is made of."""
    parts = provenance.astype("string").str.extract(_PARTS)
    present = pd.to_numeric(parts[0], errors="coerce")
    expected = pd.to_numeric(parts[1], errors="coerce")
    return (present < expected).fillna(False).astype(bool)


def provenance_basis(provenance: pd.Series) -> pd.Series:
    """reported, inferred or partial, for one value.

    A trailing-twelve-month figure counts as reported here: it is a frequency
    the reader chose, labelled as such, and marking every TTM point would say
    nothing. What it was summed FROM still shows on the quarterly view.
    """
    source = provenance_source(provenance)
    basis = pd.Series("reported", index=provenance.index, dtype=object)
    basis[source.isin(INFERRED_SOURCES).fillna(False).to_numpy()] = "inferred"
    basis[is_partial(provenance).to_numpy()] = "partial"
    basis[provenance.isna().to_numpy()] = None
    return basis
