"""Cross-chart selection.

Streamlit persists a chart's selection under its widget key, and that value is
already readable at the TOP of the next run — before any chart has rendered.
That single fact is what makes cross-filtering work here: the page reads what
was selected, draws every chart against it, and the selectable chart comes last
without needing to have run first.

An earlier version kept its own copy in session state and called st.rerun() on
every change. It fought Streamlit's widget state and settled on "cleared" —
the selection registered and then immediately erased itself. There is no second
source of truth here now.

Selection EMPHASISES rather than filters. Removing the unselected would destroy
the context that makes a selection worth making: you pick two companies out of
thirty to see them against the other twenty-eight, not instead of them.
"""
from __future__ import annotations

from typing import List, Sequence

import streamlit as st

from . import charts

GENERATION = "selection_generation"


def chart_key(name: str) -> str:
    """A widget key that changes when the reader clears the selection.

    Streamlit will not let a widget's own state be reassigned from script, so
    clearing is done by asking for a NEW widget — same chart, fresh key, empty
    selection.
    """
    return "{}__{}".format(name, st.session_state.get(GENERATION, 0))


def picked(*names: str, limit_to: Sequence[str] = ()) -> List[str]:
    """Companies selected in any of the named charts, in the order found."""
    found: List[str] = []
    for name in names:
        event = st.session_state.get(chart_key(name))
        for entity in charts.selected_entities(event):
            if entity not in found and (not limit_to or entity in limit_to):
                found.append(entity)
    return found


def clear() -> None:
    st.session_state[GENERATION] = st.session_state.get(GENERATION, 0) + 1


def banner(names: Sequence[str], limit_to: Sequence[str] = ()) -> List[str]:
    """Show what is highlighted, with a way out. Returns the selection.

    A selection the reader cannot see or undo is a trap: the page looks broken
    rather than filtered.
    """
    chosen = picked(*names, limit_to=limit_to)
    if not chosen:
        return []
    left, right = st.columns([5, 1])
    with left:
        st.caption(
            "Highlighting **{}** — the rest stay on the chart, greyed.".format(
                ", ".join(chosen)))
    with right:
        if st.button("Clear", key="clear_selection", width="stretch"):
            clear()
            st.rerun()
    return chosen
