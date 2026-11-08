"""Distress and earnings-quality models, decomposed.

A score on its own is not analysis. Every model here is shown with the terms
that produced it, because the finding is almost never "the number is 1.9" — it
is "the number is 1.9 because retained earnings are negative".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from app import charts, data, filters
from app.format import metric_value, period_label

st.set_page_config(page_title="Forensics", page_icon="🔬", layout="wide")

if not filters.guard_dataset():
    st.stop()

registry = data.metric_registry()
universe = data.companies()

# Each model: the composite, its terms, and the coefficient each term carries.
MODELS = {
    "altman_z": {
        "title": "Altman Z-Score",
        "terms": [("z_working_capital", 1.2), ("z_retained_earnings", 1.4),
                  ("z_ebit", 3.3), ("z_equity_to_liabilities", 0.6),
                  ("z_asset_turnover", 1.0)],
        "note": ("The 1968 model for listed manufacturers. The market-equity "
                 "term dominates for a company the market likes, which is why "
                 "the book-based variants below often read very differently."),
    },
    "altman_z_prime": {
        "title": "Altman Z'-Score (book equity)",
        "terms": [("z_working_capital", 0.717), ("z_retained_earnings", 0.847),
                  ("z_ebit", 3.107), ("z_book_equity_to_liabilities", 0.420),
                  ("z_asset_turnover", 0.998)],
        "note": "Altman's private-firm refitting. Needs no market price.",
    },
    "altman_z_double_prime": {
        "title": "Altman Z''-Score (non-manufacturer)",
        "terms": [("z_working_capital", 6.56), ("z_retained_earnings", 3.26),
                  ("z_ebit", 6.72), ("z_book_equity_to_liabilities", 1.05)],
        "note": ("Drops the sales/assets term, which penalises asset-heavy and "
                 "service businesses for their industry rather than their solvency."),
    },
    "beneish_m": {
        "title": "Beneish M-Score",
        "terms": [("beneish_dsri", 0.920), ("beneish_gmi", 0.528),
                  ("beneish_aqi", 0.404), ("beneish_sgi", 0.892),
                  ("beneish_depi", 0.115), ("beneish_sgai", -0.172),
                  ("beneish_tata", 4.679), ("beneish_lvgi", -0.327)],
        "intercept": -4.84,
        "note": ("An earnings-manipulation screen, annual only. A reading above "
                 "-1.78 says this company resembles the manipulators in "
                 "Beneish's sample — it is a prompt to read the filing, not a "
                 "conclusion about anyone's conduct."),
    },
    "piotroski_f": {
        "title": "Piotroski F-Score",
        "terms": [("piotroski_roa_positive", 1), ("piotroski_cfo_positive", 1),
                  ("piotroski_roa_improving", 1), ("piotroski_accruals", 1),
                  ("piotroski_leverage", 1), ("piotroski_liquidity", 1),
                  ("piotroski_no_dilution", 1), ("piotroski_margin", 1),
                  ("piotroski_turnover", 1)],
        "note": "Nine tests of fundamental improvement, one point each.",
    },
}

# Only offer a model the build actually produced. The published snapshot
# withholds the price layer, which takes the 1968 Altman Z with it — its
# market-equity term is Yahoo-derived — and the page used to default to
# exactly that, opening on "No values for this model". The book-based
# refittings, which are the ones worth reading anyway, are unaffected.
_computed = set(data.metrics()["metric"].unique())
AVAILABLE = {k: v for k, v in MODELS.items() if k in _computed} or MODELS

with st.sidebar:
    st.header("Filters")
    model_id = st.selectbox(
        "Model", list(AVAILABLE), format_func=lambda m: MODELS[m]["title"],
        key="forensic_model")
    sectors = filters.sector_control("forensic_sectors")

st.title("Forensics")
model = MODELS[model_id]
spec = registry.get(model_id)

st.caption(
    "These models are defined for industrial and, where noted, REIT filers. "
    "They are not computed for banks or insurers: neither was fitted on a "
    "balance sheet of loans or policy reserves, and applying the coefficients "
    "there produces a number with no meaning."
)

metrics = data.metrics()
scores = metrics[(metrics["metric"] == model_id) & (metrics["frequency"].isin(spec.frequency))]
if sectors:
    keep = set(universe.loc[universe["sector"].isin(sectors), "ticker"])
    scores = scores[scores["ticker"].isin(keep)]

if scores.empty:
    st.warning("No values for this model with the current filters.")
    st.stop()

frequency = "annual" if "annual" in spec.frequency else spec.frequency[0]
scores = scores[scores["frequency"] == frequency]

st.subheader("{} · latest period".format(model["title"]))
if spec.description:
    st.caption(spec.description)

latest = scores.sort_values("period_end").groupby("ticker", as_index=False).tail(1)
latest = latest.merge(universe[["ticker", "name", "sector"]], on="ticker", how="left")
ascending = spec.direction == "lower_better"
latest = latest.sort_values("value", ascending=ascending)

# ---- Band summary --------------------------------------------------------
if "band" in latest and latest["band"].notna().any():
    # Ordered by SEVERITY, following the registry's own band order, rather than
    # by how many companies happen to fall in each. Reading "Safe, Distress,
    # Grey zone" makes the reader re-sort three tiles in their head.
    order = [b["label"] for b in (spec.bands or [])]
    raw = latest["band"].value_counts()
    counts = pd.Series(
        {label: int(raw.get(label, 0)) for label in order if label in raw.index})
    cols = st.columns(len(counts))
    tones = latest.drop_duplicates("band").set_index("band")["tone"].to_dict()
    icons = {"positive": "✓", "warning": "!", "negative": "✕"}
    for col, (band, n) in zip(cols, counts.items()):
        tone = tones.get(band)
        col.metric(
            "{} {}".format(icons.get(tone, "·"), band), n,
            help="Companies currently in this band. The ranking below uses the "
                 "same colours.",
        )

# The band is the point of the chart, so it carries the colour. Status hues
# are legitimate here — they mean distress/watch/safe rather than identity —
# and every bar keeps its numeric label and its band in the table beside it, so
# nothing is conveyed by colour alone.
# Lead with the companies that need attention: worst at the top, whichever
# direction "worst" runs in for this model. Horizontal bars draw row 0 at the
# BOTTOM, so for a lower-is-better score (Beneish) ascending puts the highest —
# the flagged ones — on top, and for a higher-is-better score (Altman)
# descending puts the lowest there.
#
# This must be the SAME sort `bars()` applies internally, because the colour
# list below is positional: sort the two differently and every band gets
# painted onto the wrong company.
ordered = latest.sort_values("value", ascending=ascending)
band_colour = [
    charts.theme.STATUS.get(tone, charts.theme.CHROME[charts.mode()]["muted"])
    for tone in ordered.get("tone", pd.Series([None] * len(ordered)))
]
figure = charts.bars(
    ordered.rename(columns={"ticker": "company"}), x="company", y="value",
    kind=spec.format, height=max(340, 22 * len(latest)), horizontal=True,
    title="{} by company".format(model["title"]),
    colours=band_colour, ascending=ascending,
)
# Threshold lines without labels: at these scales the band boundaries sit a
# few points apart on an axis running to sixty, and stacked annotations
# collide into an unreadable smear. The bands are named in the tiles above.
for band in (spec.bands or []):
    if "max" in band:
        figure.add_vline(
            x=band["max"],
            line=dict(width=1, dash="dot",
                      color=charts.theme.STATUS.get(band.get("tone"), "#898781")),
        )
st.caption(
    "Ordered worst first — {} readings are the ones to look at.".format(
        "lower" if not ascending else "higher"))
charts.render(
    figure,
    ordered[["ticker", "name", "value"] + (["band"] if "band" in ordered else [])]
    .assign(value=ordered["value"].map(lambda v: metric_value(v, spec.format)))
    .rename(columns={"ticker": "Ticker", "name": "Company",
                     "value": model["title"], "band": "Band"}),
    key="forensic_ranking",
)

st.divider()

# ---- Decomposition -------------------------------------------------------
st.subheader("What is driving the score")
ticker = st.selectbox(
    "Company", list(latest["ticker"]),
    format_func=lambda t: "{} · {}".format(
        t, universe.set_index("ticker")["name"].get(t, t)),
    key="forensic_ticker",
)

company_scores = scores[scores["ticker"] == ticker].sort_values("period_end")
periods = list(company_scores["period_end"])
period = st.select_slider(
    "Period", periods, value=periods[-1],
    format_func=lambda p: pd.Timestamp(p).date().isoformat(),
    key="forensic_period",
) if len(periods) > 1 else periods[-1]

terms = metrics[(metrics["ticker"] == ticker)
                & (metrics["frequency"] == frequency)
                & (metrics["period_end"] == period)
                & (metrics["metric"].isin([t for t, _ in model["terms"]]))]
term_values = dict(zip(terms["metric"], terms["value"]))

labels, contributions, rows = [], [], []
for metric_id, weight in model["terms"]:
    raw = term_values.get(metric_id)
    if raw is None or pd.isna(raw):
        continue
    term_spec = registry.get(metric_id)
    contribution = float(raw) * weight
    labels.append(term_spec.label.replace("Z: ", "").replace("M: ", "").replace("F: ", ""))
    contributions.append(contribution)
    rows.append({
        "Term": term_spec.label,
        "Value": metric_value(raw, term_spec.format),
        "Weight": "{:+.3f}".format(weight) if weight != 1 else "1",
        "Contribution": "{:+.3f}".format(contribution),
    })

total = company_scores[company_scores["period_end"] == period]["value"]
total = float(total.iloc[0]) if len(total) else None

if not labels:
    st.info("No component values for this period.")
else:
    if "intercept" in model:
        labels.insert(0, "Intercept")
        contributions.insert(0, model["intercept"])
        rows.insert(0, {"Term": "Intercept", "Value": "—", "Weight": "—",
                        "Contribution": "{:+.3f}".format(model["intercept"])})
    charts.render(
        charts.waterfall(
            labels, contributions, total_label=model["title"], total=total,
            title="{} · {}".format(ticker, pd.Timestamp(period).date()),
            height=420,
            higher_is_better=spec.direction != "lower_better",
        ),
        pd.DataFrame(rows), key="forensic_waterfall",
    )
    st.caption(model["note"])

st.divider()

# ---- History -------------------------------------------------------------
st.subheader("History")
history = scores[scores["ticker"] == ticker]
figure = charts.time_series(
    history, series_col="ticker", kind=spec.format, height=340, label_last=False,
    title="{} · {}".format(ticker, model["title"]),
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
charts.render(
    figure,
    history.assign(
        Period=history.apply(period_label, axis=1),
        Value=history["value"].map(lambda v: metric_value(v, spec.format)),
    )[["Period", "Value"] + (["band"] if "band" in history else [])],
    key="forensic_history",
)
