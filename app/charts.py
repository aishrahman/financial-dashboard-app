"""Chart factory.

One place where the rules live, so every chart in the app obeys them:

  One y-axis, always. Two measures of different scale become two charts or an
  indexed view — never two scales on one plot, which invents a correlation the
  data does not contain.

  Colour follows the company, carried in session state, so filtering one out
  never repaints the others.

  A legend whenever there is more than one series; none for a single series,
  where the title already names it. Selective direct labels on the final point
  rather than a number on every point.

  Every chart has a table twin, so no value is reachable only by hover.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from . import theme
from .format import axis_format, currency, hover_format, metric_value

LINE_WIDTH = 2
MARKER_SIZE = 8
MAX_DIRECT_LABELS = 6
STATUS_GOOD = theme.STATUS["positive"]
STATUS_BAD = theme.STATUS["negative"]



# What the hover adds after a value that was not as filed.
BASIS_NOTE = {"inferred": "  · computed, not as filed",
              "partial": "  · summed from some of its parts — may be incomplete"}
# The legend key is short on purpose: the full sentence wrapped a four-entry
# legend into four rows on a half-width chart, and it ran into the title. The
# hover carries the full explanation at the point it applies to.
BASIS_KEY = {"inferred": "Computed", "partial": "May be incomplete"}

def _nice_step(rough: float) -> float:
    """Round a tick interval up to 1, 2, 2.5 or 5 times a power of ten."""
    import math
    if rough <= 0:
        return 1.0
    exponent = math.floor(math.log10(rough))
    base = rough / (10 ** exponent)
    for candidate in (1, 2, 2.5, 5, 10):
        if base <= candidate:
            return candidate * (10 ** exponent)
    return 10 ** (exponent + 1)


def _currency_axis(figure: go.Figure, values, axis: str = "y") -> None:
    """Label a money axis in bn/m rather than d3's SI G and M.

    Plotly's SI tick format renders four hundred billion dollars as "$400G".
    It is technically correct and nobody reads it, so currency axes get their
    ticks written out by the same formatter the tables use.
    """
    finite = [float(v) for v in values if v is not None and pd.notna(v)]
    if not finite:
        return
    low, high = min(finite), max(finite)
    if low == high:
        low, high = min(low, 0.0), max(high, 0.0) or 1.0
    low = min(low, 0.0) if low > 0 and low / (high or 1) > 0.6 else low

    step = _nice_step((high - low) / 4 or 1)
    start = math.floor(low / step) * step
    ticks, current = [], start
    while current <= high + step * 0.5 and len(ticks) < 12:
        ticks.append(current)
        current += step

    setter = figure.update_yaxes if axis == "y" else figure.update_xaxes
    setter(tickmode="array", tickvals=ticks,
           ticktext=[currency(t, 0 if abs(t) >= 1e11 else 1) for t in ticks])


def mode() -> str:
    """The viewer's theme, so the palette is selected for the right surface."""
    try:
        current = st.context.theme.type
        if current in ("light", "dark"):
            return current
    except Exception:
        pass
    try:
        return "dark" if st.get_option("theme.base") == "dark" else "light"
    except Exception:
        return "light"


def selected_entities(event) -> List[str]:
    """Companies picked out of a chart selection, in the order clicked.

    Every trace carries its company in `customdata`, which is what survives the
    round trip to the browser and back — a trace index would not, because the
    figure is rebuilt on each rerun and the traces can reorder.
    """
    if not event:
        return []
    points = (event.get("selection") or {}).get("points") or []
    found: List[str] = []
    for point in points:
        data = point.get("customdata")
        name = data[0] if isinstance(data, (list, tuple)) and data else data
        if name and name not in found:
            found.append(str(name))
    return found


def muted() -> str:
    return theme.CHROME[mode()]["muted"]


def _emphasis(entity: str, colours: Dict[str, str], highlight: Optional[Sequence[str]]):
    """Colour and weight for one series given the current selection.

    Selection DE-EMPHASISES the rest; it never repaints them. A reader who
    learned Apple is blue must still find Apple blue when someone else is
    selected — so the unselected go grey and thin, and their own hue returns
    intact the moment the selection clears.
    """
    if not highlight or entity in highlight:
        return colours[entity], LINE_WIDTH, 1.0
    return muted(), 1, 0.35


def colour_map(entities: Sequence[str], namespace: str = "ticker") -> Dict[str, str]:
    """Stable colours for this session, assigned in first-seen order.

    Namespaced, so that different KINDS of series draw from separate pools.
    Without it the two are shared first-come-first-served: charting Revenue and
    Gross Profit on the company page took the first slots, and Apple — blue
    everywhere else — came out orange on that page alone. A company's colour
    has to mean the same thing on every screen.
    """
    assignment = st.session_state.setdefault("colour_slots", {}).setdefault(namespace, {})
    current = mode()
    active = set(map(str, entities))
    return {e: theme.series_colour(e, assignment, current, active) for e in entities}


def _peer_band(
    figure: go.Figure, peer: pd.DataFrame, fmt: str, label: str,
    x_col: str = "period_end",
) -> None:
    """The interquartile range of the peer group, with its median.

    Drawn in the muted ink rather than a palette colour: the band is the
    backdrop a company is read against, not a competitor on the chart. Giving
    it a hue would make a reader hunt for it in the legend.
    """
    band = peer.dropna(subset=["p25", "p75"]).sort_values(x_col)
    middle = peer.dropna(subset=["median"]).sort_values(x_col)
    ink = muted()

    if len(band) > 1:
        figure.add_trace(go.Scatter(
            x=list(band[x_col]) + list(band[x_col][::-1]),
            y=list(band["p75"]) + list(band["p25"][::-1]),
            fill="toself",
            fillcolor="rgba(137,135,129,0.13)",
            line=dict(width=0),
            name="{} (25th–75th)".format(label),
            hoverinfo="skip",
            showlegend=True,
        ))
    if len(middle) > 1:
        figure.add_trace(go.Scatter(
            x=middle[x_col], y=middle["median"],
            mode="lines", name="Peer median",
            line=dict(color=ink, width=1.5, dash="dash"),
            hovertemplate="Peer median  %{{y:{}}}<extra></extra>".format(fmt),
        ))


# A horizontal legend wraps onto extra rows, and Plotly reserves no space for
# the ones after the first — they grow upward into the title. Roughly how many
# characters fit on a row at 12px in a typical chart width.
# Tuned against Deere's five segment names, which wrap to three rows in
# practice. Deliberately conservative: over-reserving costs a little
# whitespace, under-reserving puts the legend back on top of the title.
LEGEND_CHARS_PER_ROW = 60
LEGEND_ROW_HEIGHT = 20
TITLE_BAND = 40


def _legend_rows(figure: go.Figure) -> int:
    """How many rows the horizontal legend will wrap onto.

    Estimated from the label text, because Plotly lays the legend out in the
    browser and the server cannot measure it. Deere's five segment names —
    "Production and Precision Agriculture" among them — run to three rows,
    and the third landed on top of the chart title.
    """
    labels = [t.name for t in figure.data
              if getattr(t, "showlegend", True) is not False and t.name]
    if not labels:
        return 0
    # Each entry carries its swatch and padding beyond the text itself.
    width = sum(len(str(name)) + 6 for name in labels)
    return max(1, min(4, -(-width // LEGEND_CHARS_PER_ROW)))


def _finish(figure: go.Figure, height: int, legend: bool) -> go.Figure:
    template = theme.register_template(mode())
    figure.update_layout(
        template=template,
        height=height,
        showlegend=legend,
        hovermode="x unified",
    )
    # Reserve room for the title AND however many rows the legend needs, so a
    # chart with many long series names does not crowd its own heading.
    if legend:
        rows = _legend_rows(figure)
        figure.update_layout(
            margin=dict(l=8, r=8, b=8,
                        t=TITLE_BAND + rows * LEGEND_ROW_HEIGHT + 12))
    return figure


def time_series(
    frame: pd.DataFrame,
    value_col: str = "value",
    series_col: str = "ticker",
    label_col: Optional[str] = None,
    kind: str = "ratio",
    title: str = "",
    height: int = 380,
    label_last: bool = True,
    highlight: Optional[Sequence[str]] = None,
    peer: Optional[pd.DataFrame] = None,
    peer_label: str = "Peer range",
    namespace: str = "ticker",
    x_col: str = "period_end",
    basis_col: Optional[str] = None,
) -> go.Figure:
    """Lines over time, one per company.

    Hover is unified across series at the cursor's period, so the reader
    compares companies at a point in time — which is the question a comparison
    chart is asked — rather than reading one series at a time.
    """
    figure = go.Figure()
    entities = list(dict.fromkeys(frame[series_col]))
    colours = colour_map(entities, namespace)
    fmt = hover_format(kind)
    labelled: List[str] = []
    flagged: set = set()

    # Peers first, so they sit BEHIND the company lines. They are context, not
    # a series: no categorical hue, because that would read as another company.
    if peer is not None and not peer.empty:
        _peer_band(figure, peer, fmt, peer_label, x_col)

    for entity in entities:
        part = frame[frame[series_col] == entity].sort_values(x_col)
        if part.empty:
            continue
        colour, width, opacity = _emphasis(entity, colours, highlight)
        notes = ([BASIS_NOTE.get(b, "") for b in part[basis_col]]
                 if basis_col and basis_col in part else [""] * len(part))
        figure.add_trace(go.Scatter(
            x=part[x_col], y=part[value_col],
            name=str(entity), mode="lines",
            line=dict(color=colour, width=width),
            opacity=opacity,
            customdata=[[str(entity), note] for note in notes],
            hovertemplate=(
                "<b>%{{fullData.name}}</b>  %{{y:{}}}%{{customdata[1]}}<extra></extra>".format(fmt)
            ),
        ))
        # Where a point is not as filed, say so on the line itself. Shape, not
        # colour: colour already says which company, and a second meaning on
        # the same channel would be read as a second company.
        if basis_col and basis_col in part:
            for basis, symbol, size in (("inferred", "circle-open", 8),
                                        ("partial", "diamond-open", 11)):
                marked = part[part[basis_col] == basis].dropna(subset=[value_col])
                if marked.empty:
                    continue
                flagged.add(basis)
                figure.add_trace(go.Scatter(
                    x=marked[x_col], y=marked[value_col], mode="markers",
                    marker=dict(symbol=symbol, size=size, color=colour,
                                line=dict(width=2, color=colour)),
                    opacity=opacity, showlegend=False, hoverinfo="skip"))
        # One label, at the end of the line: enough to identify the series
        # without a number on every point.
        # Label every line up to the point where the labels themselves would
        # collide; beyond that only what is highlighted. Direct labels are the
        # secondary encoding that lets six series share one chart without
        # identity resting on hue alone.
        label_this = label_last and (
            len(entities) <= MAX_DIRECT_LABELS or (highlight and entity in highlight))
        if label_this:
            final = part.dropna(subset=[value_col]).tail(1)
            if not final.empty:
                figure.add_trace(go.Scatter(
                    x=final[x_col], y=final[value_col],
                    mode="markers+text", text=[str(entity)],
                    textposition="middle right",
                    textfont=dict(color=colour, size=11),
                    marker=dict(color=colour, size=MARKER_SIZE,
                                line=dict(width=2, color=theme.CHROME[mode()]["surface"])),
                    opacity=opacity,
                    showlegend=False, hoverinfo="skip",
                ))
                labelled.append(str(entity))

    figure.update_yaxes(tickformat=axis_format(kind), title=None)
    figure.update_xaxes(title=None)
    if kind == "currency":
        _currency_axis(figure, frame[value_col])
    if title:
        figure.update_layout(title=title)
    has_peer = peer is not None and not peer.empty
    # One key per mark actually drawn, in neutral grey: it explains a shape,
    # and is not another series.
    for basis, symbol, size in (("inferred", "circle-open", 8), ("partial", "diamond-open", 11)):
        if basis in flagged:
            figure.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=BASIS_KEY[basis],
                marker=dict(symbol=symbol, size=size, color=muted(),
                            line=dict(width=2, color=muted())),
                hoverinfo="skip"))
    figure = _finish(figure, height,
                     legend=len(entities) > 1 or has_peer or bool(flagged))
    if labelled:
        # An end-of-line label sits OUTSIDE the plot area, and Plotly reserves
        # nothing for it — with short tickers it fits in the default margin, but
        # a footnote line item like "Interest Paid, Net" runs off the right edge
        # and gets clipped. Reserve room for the longest one actually drawn.
        widest = max(len(name) for name in labelled)
        figure.update_layout(margin=dict(
            l=figure.layout.margin.l, t=figure.layout.margin.t,
            b=figure.layout.margin.b,
            r=min(160, 14 + widest * 6)))
    return figure


def bars(
    frame: pd.DataFrame,
    x: str,
    y: str,
    kind: str = "ratio",
    title: str = "",
    height: int = 380,
    colour_by_entity: bool = True,
    horizontal: bool = False,
    colours: Optional[Sequence[str]] = None,
    highlight: Optional[Sequence[str]] = None,
    selectable: bool = False,
    namespace: str = "ticker",
    ascending: Optional[bool] = None,
) -> go.Figure:
    """A bar per company for one period.

    Bars are one colour when the categories have no order — colouring each bar
    darker-where-bigger double-encodes the length and burns the only free
    channel that could carry something new. A ranking of twenty companies is
    ONE series and gets one hue.

    Entity colours are for the handful of companies being tracked across a
    page, so the same firm keeps its colour between charts. `colours` overrides
    both, for the case where a bar's category genuinely carries state — a
    distress band — and the colour is doing real work.
    """
    # Horizontal bars draw the first row at the BOTTOM, so ascending puts the
    # largest at the top. Callers override when "largest" is not what should
    # lead — a distress ranking should open with the distressed.
    #
    # `colours` is aligned with the rows AS PASSED, and travels with them
    # through the sort. It used to be applied positionally after sorting, so
    # a caller whose frame was in any other order got every colour on the
    # wrong bar; Forensics only escaped by pre-sorting to match, under a
    # comment warning that it must. Stable, so tied values keep their order.
    part = frame if colours is None else frame.assign(_fill=list(colours))
    part = part.sort_values(
        y, ascending=horizontal if ascending is None else ascending, kind="stable")
    entities = list(part[x])
    if colours is not None:
        fill = list(part["_fill"])
    elif colour_by_entity and len(entities) <= len(theme.palette(mode())):
        assigned = colour_map(entities, namespace)
        fill = [assigned[e] for e in entities]
    else:
        fill = theme.palette(mode())[0]
    fmt = hover_format(kind)

    if highlight:
        base = fill if isinstance(fill, list) else [fill] * len(entities)
        fill = [c if e in highlight else muted() for e, c in zip(entities, base)]

    figure = go.Figure(go.Bar(
        x=part[y] if horizontal else part[x],
        y=part[x] if horizontal else part[y],
        orientation="h" if horizontal else "v",
        marker=dict(color=fill, line=dict(width=0)),
        customdata=[[str(e)] for e in entities],
        text=[metric_value(v, kind) for v in part[y]],
        textposition="outside",
        cliponaxis=False,          # never clip a label against the plot edge
        hovertemplate=("<b>%{y}</b>  %{x:" + fmt + "}<extra></extra>" if horizontal
                       else "<b>%{x}</b>  %{y:" + fmt + "}<extra></extra>"),
        showlegend=False,
    ))
    if horizontal:
        figure.update_xaxes(tickformat=axis_format(kind))
        if kind == "currency":
            _currency_axis(figure, part[y], axis="x")
    else:
        figure.update_yaxes(tickformat=axis_format(kind))
        if kind == "currency":
            _currency_axis(figure, part[y])
    if title:
        figure.update_layout(title=title)
    figure.update_layout(bargap=0.35)
    figure = _finish(figure, height, legend=False)
    if selectable:
        figure.update_layout(dragmode="select", clickmode="event+select")
    return figure


def dumbbell(
    frame: pd.DataFrame,
    label: str,
    start: str,
    end: str,
    kind: str = "currency",
    title: str = "",
    height: int = 380,
    colours: Optional[Sequence[str]] = None,
    start_name: str = "Before",
    end_name: str = "After",
) -> go.Figure:
    """Two values per row joined by a line, rows top to bottom as passed.

    The form for a before-and-after. A bar of the percentage change explodes
    wherever the starting figure is small — one company's 1,961% flattened
    every other bar on the chart into a sliver — and hides both values behind
    a ratio of them. Here the gap between the dots IS the change, in the units
    it happened in, and a sign flip reads as a line crossing zero.

    The start is a hollow ring and the end a filled dot, so direction survives
    without colour; `colours` (aligned with the rows) can still carry it.
    """
    rows = frame.reset_index(drop=True)
    fill = list(colours) if colours is not None else [theme.palette(mode())[0]] * len(rows)
    fmt = hover_format(kind)
    segment_x, segment_y = [], []
    for _, row in rows.iterrows():
        segment_x += [row[start], row[end], None]
        segment_y += [row[label], row[label], None]

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=segment_x, y=segment_y, mode="lines", hoverinfo="skip", showlegend=False,
        line=dict(color=muted(), width=2)))
    figure.add_trace(go.Scatter(
        x=rows[start], y=rows[label], mode="markers", name=start_name,
        marker=dict(size=9, color="rgba(0,0,0,0)", line=dict(color=muted(), width=2)),
        hovertemplate="<b>%{y}</b>  " + start_name + " %{x:" + fmt + "}<extra></extra>"))
    figure.add_trace(go.Scatter(
        x=rows[end], y=rows[label], mode="markers", name=end_name,
        marker=dict(size=10, color=fill, line=dict(width=0)),
        hovertemplate="<b>%{y}</b>  " + end_name + " %{x:" + fmt + "}<extra></extra>"))

    figure.update_yaxes(categoryorder="array", categoryarray=list(rows[label])[::-1],
                        showgrid=False)
    figure.update_xaxes(tickformat=axis_format(kind), zeroline=True,
                        zerolinecolor=muted(), zerolinewidth=1)
    if kind == "currency":
        _currency_axis(figure, pd.concat([rows[start], rows[end]]), axis="x")
    if title:
        figure.update_layout(title=title)
    figure = _finish(figure, height, legend=True)
    figure.update_layout(hovermode="closest")
    return figure


def indexed(
    frame: pd.DataFrame,
    value_col: str = "value",
    series_col: str = "ticker",
    title: str = "",
    height: int = 380,
    namespace: str = "ticker",
    x_col: str = "period_end",
) -> go.Figure:
    """Every series rebased to 100 at its first period.

    The honest answer to "these two measures have different scales": put them on
    one axis in common units rather than giving each its own y-scale.
    """
    parts = []
    for entity, part in frame.groupby(series_col, sort=False):
        part = part.sort_values(x_col).copy()
        base = part[value_col].dropna()
        if base.empty or base.iloc[0] == 0:
            continue
        part[value_col] = part[value_col] / base.iloc[0] * 100
        parts.append(part)
    if not parts:
        return go.Figure()

    figure = time_series(
        pd.concat(parts), value_col=value_col, series_col=series_col,
        kind="score", title=title, height=height, namespace=namespace,
        x_col=x_col,
    )
    figure.add_hline(y=100, line=dict(color=theme.CHROME[mode()]["axis"], width=1))
    figure.update_yaxes(tickformat=".0f", title="Indexed to 100")
    return figure


def small_multiples(
    frame: pd.DataFrame,
    value_col: str = "value",
    series_col: str = "ticker",
    x_col: str = "period_end",
    kind: str = "ratio",
    title: str = "",
    columns: int = 3,
    row_height: int = 150,
    shared_scale: bool = True,
) -> go.Figure:
    """One small chart per company, on a shared scale.

    The honest answer to comparing more series than a single chart can hold.
    Past six overlapping lines the reader is decoding a legend rather than
    reading a trend; faceting trades the ability to compare exact crossings for
    the ability to see every company's shape at once, which is usually the
    better trade.

    The shared y-axis is what makes it a comparison rather than a set of
    unrelated charts — without it each panel silently rescales and a flat line
    can look like a steep one.
    """
    entities = list(dict.fromkeys(frame[series_col]))
    if not entities:
        return go.Figure()

    rows = -(-len(entities) // columns)
    colours = colour_map(entities)
    current = mode()
    figure = make_subplots(
        rows=rows, cols=columns,
        subplot_titles=[str(e) for e in entities],
        shared_yaxes=False, shared_xaxes=True,
        vertical_spacing=max(0.06, 0.30 / max(rows, 1)),
        horizontal_spacing=0.06,
    )

    finite = frame[value_col].replace([np.inf, -np.inf], np.nan).dropna()
    span = (float(finite.min()), float(finite.max())) if len(finite) else None
    fmt = hover_format(kind)

    for i, entity in enumerate(entities):
        row, col = i // columns + 1, i % columns + 1
        part = frame[frame[series_col] == entity].sort_values(x_col)
        figure.add_trace(
            go.Scatter(
                x=part[x_col], y=part[value_col], mode="lines",
                name=str(entity),
                line=dict(color=colours[entity], width=LINE_WIDTH),
                showlegend=False,
                hovertemplate="%{{x|%Y}}  %{{y:{}}}<extra></extra>".format(fmt),
            ),
            row=row, col=col,
        )
        if span and shared_scale:
            pad = (span[1] - span[0]) * 0.08 or 1
            figure.update_yaxes(range=[span[0] - pad, span[1] + pad],
                                row=row, col=col)
        figure.update_yaxes(tickformat=axis_format(kind), row=row, col=col)
        # Only the leftmost panel carries tick labels; repeating them on every
        # panel is noise once the scale is shared.
        if col != 1:
            figure.update_yaxes(showticklabels=False, row=row, col=col)

    for annotation in figure.layout.annotations:
        annotation.font.size = 12
        annotation.font.color = theme.CHROME[current]["primary"]

    figure.update_layout(
        template=theme.register_template(current),
        height=max(260, rows * row_height + 90),
        showlegend=False,
        hovermode="closest",
        margin=dict(l=8, r=8, t=64 if title else 40, b=8),
    )
    if title:
        figure.update_layout(title=title)
    return figure


def waterfall(
    labels: Sequence[str],
    values: Sequence[float],
    total_label: str = "Total",
    total: Optional[float] = None,
    title: str = "",
    height: int = 380,
    higher_is_better: bool = True,
) -> go.Figure:
    """Contribution of each term to a composite score.

    Used to decompose the forensic models: a Z-Score of 1.9 is a number, but
    seeing that the retained-earnings term contributed -0.6 of it is the
    finding.

    Colour follows the metric's DIRECTION, not the arithmetic sign. On the
    Beneish M-Score a higher reading is worse, so a term that pushes the score
    up is pushing the company toward the manipulation flag — painting it green
    because the number went up tells the reader the opposite of the truth.
    """
    current = mode()
    good, bad = STATUS_GOOD, STATUS_BAD
    if not higher_is_better:
        good, bad = bad, good
    figure = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(values) + (["total"] if total is not None else []),
        x=list(labels) + ([total_label] if total is not None else []),
        y=list(values) + ([total] if total is not None else []),
        text=["{:+.2f}".format(v) for v in values]
             + (["{:.2f}".format(total)] if total is not None else []),
        textposition="outside",
        cliponaxis=False,
        connector=dict(line=dict(color=theme.CHROME[current]["axis"], width=1)),
        increasing=dict(marker=dict(color=good)),
        decreasing=dict(marker=dict(color=bad)),
        totals=dict(marker=dict(color=theme.palette(current)[0])),
    ))
    if title:
        figure.update_layout(title=title)
    return _finish(figure, height, legend=False)


def heatmap(
    frame: pd.DataFrame,
    title: str = "",
    height: int = 420,
    kind: str = "ratio",
    centre: Optional[float] = None,
    higher_is_better: bool = True,
) -> go.Figure:
    """A company-by-period grid.

    Diverging scale only where the measure has a real midpoint — a margin above
    or below zero, a score above or below its threshold. Otherwise one hue,
    light to dark, because magnitude has no poles.

    The red end is always the BAD end. Red reads as alarming whichever measure
    is on screen, so on a metric where higher is better the ramp is reversed:
    contraction red, growth blue. Leaving it unreversed painted the fastest
    growers in the most alarming colour on the page.
    """
    current = mode()
    values = pd.Series(frame.to_numpy(dtype=float).ravel()).replace(
        [np.inf, -np.inf], np.nan).dropna()

    if centre is not None:
        # Scaled to the 5th–95th percentile rather than the extremes. One
        # outlier otherwise sets the whole range and every other cell collapses
        # into the middle of the ramp — ConocoPhillips' post-2020 revenue
        # rebound flattened seventy other companies into a single shade.
        # Values beyond the range still render, at the end colour.
        span = max(
            abs(values.quantile(0.95) - centre),
            abs(values.quantile(0.05) - centre),
            1e-9,
        ) if len(values) else 1e-9
        scale = theme.diverging(current)
        if higher_is_better:
            scale = list(reversed(scale))
        zmin, zmax = centre - span, centre + span
    else:
        scale, zmin, zmax = theme.SEQUENTIAL, None, None
        if len(values):
            zmin, zmax = float(values.quantile(0.05)), float(values.quantile(0.95))

    figure = go.Figure(go.Heatmap(
        z=frame.to_numpy(), x=[str(c) for c in frame.columns],
        y=[str(i) for i in frame.index],
        colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)],
        zmin=zmin, zmax=zmax,
        xgap=2, ygap=2,           # a surface gap, never a border around cells
        hovertemplate="<b>%{y}</b> · %{x}<br>%{z:" + hover_format(kind) + "}<extra></extra>",
        colorbar=dict(
            thickness=10, outlinewidth=0, len=0.8,
            tickformat=axis_format(kind),
            tickfont=dict(color=theme.CHROME[current]["muted"], size=10),
        ),
    ))
    if title:
        figure.update_layout(title=title)
    figure.update_layout(hovermode="closest")
    return _finish(figure, height, legend=False)


def scatter(
    frame: pd.DataFrame,
    x: str, y: str, label: str,
    x_kind: str = "ratio", y_kind: str = "ratio",
    title: str = "", height: int = 440,
    highlight: Optional[Sequence[str]] = None,
    selectable: bool = False,
    namespace: str = "ticker",
) -> go.Figure:
    """Two measures against each other, one point per company."""
    entities = list(frame[label])
    colours = colour_map(entities, namespace)
    fills = [colours[e] if (not highlight or e in highlight) else muted()
             for e in entities]
    sizes = [12 if (not highlight or e in highlight) else 8 for e in entities]
    figure = go.Figure(go.Scatter(
        x=frame[x], y=frame[y], mode="markers+text",
        text=entities, textposition="top center",
        textfont=dict(size=10, color=theme.CHROME[mode()]["secondary"]),
        customdata=[[str(e)] for e in entities],
        marker=dict(
            size=sizes, color=fills,
            line=dict(width=2, color=theme.CHROME[mode()]["surface"]),
        ),
        hovertemplate=("<b>%{text}</b><br>" + x + ": %{x:" + hover_format(x_kind) + "}"
                       + "<br>" + y + ": %{y:" + hover_format(y_kind) + "}<extra></extra>"),
        showlegend=False,
    ))
    figure.update_xaxes(tickformat=axis_format(x_kind), title=x)
    figure.update_yaxes(tickformat=axis_format(y_kind), title=y)
    if title:
        figure.update_layout(title=title)
    figure.update_layout(hovermode="closest")
    figure = _finish(figure, height, legend=False)
    if selectable:
        # Drag draws a selection box rather than zooming — this chart exists to
        # pick companies out of, which is the commoner intent here.
        figure.update_layout(dragmode="select", clickmode="event+select")
    return figure


def render(
    figure: go.Figure,
    table: Optional[pd.DataFrame] = None,
    key: str = "",
    on_select: bool = False,
):
    """Draw a chart with its table twin behind a disclosure.

    The table is not optional decoration: a tooltip must never be the only
    route to a value.

    With `on_select`, returns the companies the reader picked, so other charts
    on the page can respond to them.
    """
    remove = ["lasso2d", "select2d"] if not on_select else []
    event = st.plotly_chart(
        figure, width="stretch", key=key or None,
        on_select="rerun" if on_select else "ignore",
        selection_mode=("points", "box", "lasso"),
        config={"displaylogo": False, "modeBarButtonsToRemove": remove},
    )
    if table is not None and not table.empty:
        with st.expander("Table view"):
            st.dataframe(table, width="stretch", hide_index=True)
    return selected_entities(event) if on_select else None
