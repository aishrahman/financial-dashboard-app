"""Palette and Plotly template.

The categorical order below is a validated set: eight hues checked for
colour-vision separation, chroma and lightness against both surfaces. The
ORDER is the safety mechanism, not decoration — adjacent slots are the pairs a
reader compares, so they are the pairs that must stay distinguishable. Do not
reorder, and do not add a ninth: past eight, fold the tail into "Other" or use
small multiples.

Colour follows the COMPANY, never its rank in the current view. A reader who
learns Apple is blue must not find it orange after filtering someone out, so
assignment is sticky for a session (see `series_colour`).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

import plotly.graph_objects as go
import plotly.io as pio

CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]

# Reserved. A status colour never stands in for "series 5", and always ships
# with a label beside it — hue alone never carries the meaning.
STATUS = {
    "positive": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "negative": "#d03b3b",
}

SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
              "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
              "#184f95", "#104281", "#0d366b"]

# Two hues that read as opposite, with a NEUTRAL midpoint: a hue in the middle
# would imply the centre means something.
DIVERGING_LIGHT = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
                   "#f0a3a3", "#d03b3b", "#7d1f1f"]
DIVERGING_DARK = ["#0d366b", "#3987e5", "#9ec5f4", "#383835",
                  "#f0a3a3", "#d03b3b", "#7d1f1f"]

CHROME = {
    "light": {
        "surface": "#fcfcfb", "plane": "#f9f9f7", "primary": "#0b0b0b",
        "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
        "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
    },
    "dark": {
        "surface": "#1a1a19", "plane": "#0d0d0d", "primary": "#ffffff",
        "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "axis": "#383835", "border": "rgba(255,255,255,0.10)",
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
OTHER = "Other"


def palette(mode: str = "light") -> List[str]:
    return CATEGORICAL_DARK if mode == "dark" else CATEGORICAL_LIGHT


def diverging(mode: str = "light") -> List[str]:
    return DIVERGING_DARK if mode == "dark" else DIVERGING_LIGHT


def series_colour(
    entity: str,
    assignment: Dict[str, int],
    mode: str = "light",
    active: Optional[Set[str]] = None,
) -> str:
    """The colour for one company, stable while it is on screen.

    A company keeps the slot it was first given, so removing one from a
    comparison never repaints the rest — that is the rule this exists to serve.

    But the slot is only held while the company is still being shown. An
    earlier version held them for the life of the session, which meant every
    company looked at on any page consumed one permanently: after browsing
    seven or eight of them, returning to Compare produced a chart where
    everything was grey except whichever company happened to have been
    assigned first. Slots are now reclaimed from companies that have left the
    screen, so the palette cannot be exhausted by browsing.

    `active` is the set currently being drawn. Without it the old behaviour
    applies, which is right for a single chart that knows nothing of the rest.
    """
    colours = palette(mode)
    if entity not in assignment:
        used = set(assignment.values())
        free = next((i for i in range(len(colours)) if i not in used), None)
        if free is None and active:
            for held in [k for k in assignment if k not in active]:
                del assignment[held]
            used = set(assignment.values())
            free = next((i for i in range(len(colours)) if i not in used), None)
        if free is None:
            return CHROME[mode]["muted"]
        assignment[entity] = free
    slot = assignment[entity]
    return colours[slot] if slot < len(colours) else CHROME[mode]["muted"]


def register_template(mode: str = "light") -> str:
    """Install the Plotly template and return its name.

    Chrome is deliberately recessive: hairline solid gridlines one shade off
    the surface, no vertical grid at all on time series, no plot border. The
    data should be the only thing with weight.
    """
    name = "findash_{}".format(mode)
    chrome = CHROME[mode]

    axis = dict(
        showgrid=True,
        gridcolor=chrome["grid"],
        gridwidth=1,
        griddash="solid",          # never dashed: dashing reads as a threshold
        zeroline=False,
        linecolor=chrome["axis"],
        linewidth=1,
        ticks="outside",
        ticklen=4,
        tickcolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=11),
        title=dict(font=dict(color=chrome["secondary"], size=12)),
        automargin=True,
    )

    template = go.layout.Template(
        layout=go.Layout(
            colorway=palette(mode),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT, color=chrome["primary"], size=13),
            # Title and legend both sit in the top margin band, but they use
            # DIFFERENT coordinate systems by default — title.yref is
            # "container" (the whole figure) while legend.y is "paper"
            # (the plot area). Left alone they compete for the same strip
            # and overlap whenever a chart carries both, which every
            # multi-series chart does. Pinning both to explicit, separate
            # bands — title above, legend below it, plot below that — and
            # sizing the margin to fit both is what keeps them apart.
            # yanchor="top" is the part that matters: at yref="container" with
            # y=1 (the very top of the figure), yanchor="bottom" anchors the
            # text's BOTTOM edge there, so the whole title extends upward and
            # off the canvas — invisible, not overlapping. "top" extends it
            # downward into the margin instead, where it is actually drawn.
            title=dict(
                font=dict(size=15, color=chrome["primary"]),
                x=0, xanchor="left",
                y=1, yanchor="top", yref="container",
                pad=dict(t=4, b=8),
            ),
            xaxis={**axis, "showgrid": False},
            yaxis=axis,
            hoverlabel=dict(
                bgcolor=chrome["surface"],
                bordercolor=chrome["border"],
                font=dict(family=FONT, size=12, color=chrome["primary"]),
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                font=dict(size=12, color=chrome["secondary"]),
                bgcolor="rgba(0,0,0,0)",
                itemsizing="constant",
            ),
            margin=dict(l=8, r=8, t=76, b=8),
            colorscale=dict(
                sequential=[[i / (len(SEQUENTIAL) - 1), c]
                            for i, c in enumerate(SEQUENTIAL)],
                diverging=[[i / (len(diverging(mode)) - 1), c]
                           for i, c in enumerate(diverging(mode))],
            ),
        )
    )
    pio.templates[name] = template
    return name


def tone_colour(tone: Optional[str]) -> Optional[str]:
    return STATUS.get(tone) if tone else None
