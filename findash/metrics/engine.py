"""The metric engine.

Metrics are declarations, not code. Each one names an expression over item
codes, market columns and other metrics; the engine sorts them into dependency
order and evaluates each as a single vectorized pass over the whole universe.

Three things it refuses to do, because each is a way a ratio dashboard misleads:

  It will not compute a metric for a template it is not valid for. A bank has
  no gross margin and no inventory; a combined ratio outside insurance is
  noise. `applies_to` is enforced, not advisory.

  It will not annualise by accident. A ratio of a flow to a balance — ROA,
  asset turnover — is an annual quantity. Computed against a single quarter's
  numerator it reads about a quarter of the truth, so those metrics are
  restricted to annual and trailing-twelve-month rows.

  It will not emit a number whose guard fails. A P/E on negative earnings, a
  return on negative equity, growth from a loss-making base: each is
  arithmetically computable and analytically meaningless.

Every exclusion is counted and reported, so a gap in the dashboard can always
be explained.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
import yaml

from ..model.calendar import add_calendar_columns
from ..registry import Expr, ExprError, ItemRegistry, eval_expr, parse_expr

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GROUP_LEVELS = ["ticker", "frequency"]

# Columns the price layer contributes. Named here so an expression referencing
# one is recognised rather than reported as an unknown item.
MARKET_COLUMNS = ("market_cap", "close", "adj_close", "shares_adjusted")


@dataclass
class Metric:
    id: str
    label: str
    family: str
    expr: Expr
    applies_to: List[str]
    frequency: List[str]
    format: str = "ratio"
    direction: str = "neutral"
    bands: List[Dict[str, Any]] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    guard: Optional[Expr] = None
    description: str = ""

    @property
    def depends_on(self) -> Set[str]:
        names = set(self.expr.required) | set(self.expr.optional)
        if self.guard is not None:
            names |= set(self.guard.required) | set(self.guard.optional)
        return names

    def band_for(self, value: float) -> Optional[Dict[str, Any]]:
        """The named band a value falls in, or None when unbanded."""
        if value is None or pd.isna(value):
            return None
        for band in self.bands:
            if "max" not in band or value <= band["max"]:
                return band
        return None


@dataclass
class MetricRegistry:
    metrics: Dict[str, Metric]

    def __iter__(self):
        return iter(self.metrics.values())

    def __len__(self) -> int:
        return len(self.metrics)

    def __contains__(self, key: str) -> bool:
        return key in self.metrics

    def get(self, key: str) -> Optional[Metric]:
        return self.metrics.get(key)

    def families(self) -> List[str]:
        seen: List[str] = []
        for metric in self.metrics.values():
            if metric.family not in seen:
                seen.append(metric.family)
        return seen

    def by_family(self, family: str) -> List[Metric]:
        return [m for m in self.metrics.values() if m.family == family]

    def for_template(self, template: str) -> List[Metric]:
        return [m for m in self.metrics.values() if template in m.applies_to]

    def evaluation_order(self) -> List[Metric]:
        """Metrics sorted so a metric's dependencies are computed before it.

        The composite scores are all built this way — Beneish's M-Score from
        its eight indices, Piotroski's F from its nine tests — which is what
        lets the dashboard decompose a score into the parts that moved it.
        """
        pending = dict(self.metrics)
        deps = {
            key: {n for n in metric.depends_on if n in pending and n != key}
            for key, metric in pending.items()
        }
        ordered: List[Metric] = []
        done: Set[str] = set()
        while pending:
            ready = sorted(k for k, d in deps.items() if k in pending and d <= done)
            if not ready:
                raise ExprError(
                    "circular metric dependency among: " + ", ".join(sorted(pending))
                )
            for key in ready:
                ordered.append(pending.pop(key))
                done.add(key)
        return ordered

    @classmethod
    def load(
        cls,
        path: Optional[Path] = None,
        items: Optional[ItemRegistry] = None,
    ) -> "MetricRegistry":
        path = Path(path) if path else PROJECT_ROOT / "registry" / "metrics.yaml"
        raw = yaml.safe_load(path.read_text()) or {}
        defaults = raw.get("defaults", {})
        metrics: Dict[str, Metric] = {}

        for key, spec in (raw.get("metrics") or {}).items():
            spec = spec or {}
            metrics[key] = Metric(
                id=key,
                label=spec.get("label", key.replace("_", " ").title()),
                family=spec.get("family", "other"),
                expr=parse_expr(spec["expr"]),
                applies_to=list(spec.get("applies_to", defaults.get("applies_to", []))),
                frequency=list(spec.get("frequency", defaults.get("frequency", []))),
                format=spec.get("format", defaults.get("format", "ratio")),
                direction=spec.get("direction", defaults.get("direction", "neutral")),
                bands=list(spec.get("bands") or []),
                requires=list(spec.get("requires") or []),
                guard=parse_expr(spec["guard"]) if spec.get("guard") else None,
                description=(spec.get("description") or "").strip(),
            )

        registry = cls(metrics=metrics)
        if items is not None:
            known = set(items.items) | set(metrics) | set(MARKET_COLUMNS)
            unknown = {
                name
                for metric in metrics.values()
                for name in metric.depends_on
                if name not in known
            }
            if unknown:
                raise ExprError(
                    "metrics reference unknown name(s): " + ", ".join(sorted(unknown))
                )
        registry.evaluation_order()  # surface cycles at load, not mid-build
        return registry


def build_inputs(
    curated: pd.DataFrame,
    market: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """The wide frame metrics are evaluated over: items plus market columns."""
    wide = curated.pivot_table(
        index=GROUP_LEVELS + ["period_end"],
        columns="item", values="value", aggfunc="first",
    )
    wide.columns.name = None

    if market is not None and not market.empty:
        columns = [c for c in MARKET_COLUMNS if c in market.columns]
        joined = market.set_index(GROUP_LEVELS + ["period_end"])[columns]
        wide = wide.join(joined, how="left")

    return wide.sort_index()


def evaluate(
    inputs: pd.DataFrame,
    metrics: MetricRegistry,
    templates: pd.Series,
    have_market: bool = True,
) -> tuple:
    """Compute every metric. Returns (wide results, per-metric diagnostics).

    `templates` maps ticker to template, and drives the applies_to mask.
    """
    index = inputs.index
    row_template = pd.Series(index.get_level_values("ticker"), index=index).map(templates)
    row_frequency = pd.Series(index.get_level_values("frequency"), index=index)

    columns: Dict[str, pd.Series] = {c: inputs[c] for c in inputs.columns}
    results: Dict[str, pd.Series] = {}
    report: List[Dict[str, Any]] = []

    for metric in metrics.evaluation_order():
        row = {"metric": metric.id, "family": metric.family, "label": metric.label}

        if "market" in metric.requires and not have_market:
            report.append({**row, "computed": 0, "skipped": "no market data"})
            results[metric.id] = pd.Series(np.nan, index=index)
            columns[metric.id] = results[metric.id]
            continue

        missing = sorted(metric.expr.required - set(columns))
        if missing:
            report.append({
                **row, "computed": 0,
                "skipped": "missing input: " + ", ".join(missing),
            })
            results[metric.id] = pd.Series(np.nan, index=index)
            columns[metric.id] = results[metric.id]
            continue

        value = eval_expr(metric.expr, columns, index, GROUP_LEVELS)
        if value is None:
            value = pd.Series(np.nan, index=index)

        eligible = row_template.isin(metric.applies_to) & row_frequency.isin(metric.frequency)
        blocked_template = int((value.notna() & ~row_template.isin(metric.applies_to)).sum())
        blocked_frequency = int((
            value.notna() & row_template.isin(metric.applies_to)
            & ~row_frequency.isin(metric.frequency)
        ).sum())

        blocked_guard = 0
        if metric.guard is not None:
            passed = eval_expr(metric.guard, columns, index, GROUP_LEVELS)
            if passed is None:
                passed = pd.Series(np.nan, index=index)
            guard_ok = passed.fillna(0) > 0
            blocked_guard = int((value.notna() & eligible & ~guard_ok).sum())
            eligible = eligible & guard_ok

        value = value.where(eligible)
        results[metric.id] = value
        columns[metric.id] = value

        report.append({
            **row,
            "computed": int(value.notna().sum()),
            "blocked_template": blocked_template,
            "blocked_frequency": blocked_frequency,
            "blocked_guard": blocked_guard,
            "skipped": None,
        })

    frame = pd.DataFrame(results, index=index)
    return frame, pd.DataFrame(report)


def to_long(wide: pd.DataFrame, metrics: MetricRegistry) -> pd.DataFrame:
    """Melt to one row per company/period/metric, banded and labelled.

    Carries the calendar alignment as well as the fiscal period end. Peer
    benchmarks are grouped on calendar period — Apple's September year and
    Microsoft's June year are not the same year, and pooling them by filing
    date would average different economic periods together.
    """
    stacked = wide.stack(future_stack=True).rename("value")
    frame = stacked[stacked.notna()].reset_index()
    frame = frame.rename(columns={"level_3": "metric"})
    if frame.empty:
        return frame

    frame["family"] = frame["metric"].map(
        lambda m: metrics.metrics[m].family if m in metrics else "other")

    frame = add_calendar_columns(frame)

    # Built as FULL columns rather than a partial .loc assignment. Assigning to
    # a subset leaves NaN in the untouched rows, and pandas 3.0's str dtype
    # stringifies that on the parquet round trip — the missing band came back
    # as the literal text "nan" and rendered as a band called nan in the UI.
    banded = {m.id: m for m in metrics if m.bands}
    if banded:
        labels, tones = [], []
        for metric_id, value in zip(frame["metric"], frame["value"]):
            spec = banded.get(metric_id)
            band = spec.band_for(value) if spec else None
            labels.append((band or {}).get("label"))
            tones.append((band or {}).get("tone"))
        frame["band"] = pd.Series(labels, index=frame.index, dtype="string")
        frame["tone"] = pd.Series(tones, index=frame.index, dtype="string")
    return frame.sort_values(
        ["ticker", "frequency", "period_end", "metric"], ignore_index=True)


# ---- How sure a metric value is -------------------------------------------

def item_dependencies(metrics: MetricRegistry) -> Dict[str, Set[str]]:
    """Every line item a metric reads, following metrics that read metrics.

    Altman's Z names five terms and no line items at all; what it rests on is
    whatever those terms rest on.
    """
    resolved: Dict[str, Set[str]] = {}

    def walk(metric_id: str, path: frozenset) -> Set[str]:
        if metric_id in resolved:
            return resolved[metric_id]
        out: Set[str] = set()
        for name in metrics.metrics[metric_id].depends_on:
            if name in metrics.metrics and name not in path:
                out |= walk(name, path | {name})
            elif name not in metrics.metrics:
                out.add(name)
        resolved[metric_id] = out
        return out

    for metric in metrics:
        walk(metric.id, frozenset({metric.id}))
    return resolved


def item_basis(curated: pd.DataFrame, items) -> pd.DataFrame:
    """Basis codes per item and period (0 reported, 1 inferred, 2 partial).

    Two kinds of derived value are not the same thing. Recovering a line the
    filer reports but did not tag — total liabilities from assets less equity
    — is inferred: the figure exists in the filing and was not taken from it.
    An analytical aggregate this layer builds by definition — EBITDA, net
    debt, total debt — is not a line in any filing, so calling it "not as
    filed" says nothing; it is as filed as what it was built from.

    Treating aggregates as inferred made every leverage ratio inferred (786
    of 786 annual net debt / EBITDA), and it broke the chain that matters
    most: short-term debt summed from one part of two feeds total debt, then
    net debt, then leverage, and the partial-ness stopped at the first
    aggregate. Aggregates now take the weakest basis among the inputs of the
    formula actually used for that period, which provenance records.
    """
    from ..provenance import BASIS_ORDER, provenance_basis
    from ..registry import parse_expr

    rank = {b: i for i, b in enumerate(BASIS_ORDER)}
    prov = curated.pivot_table(index=GROUP_LEVELS + ["period_end"], columns="item",
                               values="provenance", aggfunc="first")
    codes = prov.apply(lambda col: provenance_basis(col).map(rank)).astype(float)

    aggregates = [i.code for i in items if i.statement == "derived" and i.code in prov]
    parsed: Dict[str, Set[str]] = {}
    for _ in range(len(aggregates) + 1):          # aggregates of aggregates
        changed = False
        for code in aggregates:
            column = prov[code]
            for text in column.dropna().unique():
                if not str(text).startswith("derived:") or str(text).startswith("derived:extractor"):
                    continue
                expr = str(text)[len("derived:"):].split("|")[0]
                if expr not in parsed:
                    try:
                        node = parse_expr(expr)
                        parsed[expr] = set(node.required) | set(node.optional)
                    except Exception:
                        parsed[expr] = set()
                inputs = [c for c in parsed[expr] if c in codes.columns]
                if not inputs:
                    continue
                rows = column == text
                worst = codes.loc[rows, inputs].max(axis=1)
                before = codes.loc[rows, code]
                if not worst.equals(before):
                    codes.loc[rows, code] = worst
                    changed = True
        if not changed:
            break
    return codes


def metric_basis(
    codes: pd.DataFrame,
    metric_wide: pd.DataFrame,
    metrics: MetricRegistry,
) -> pd.DataFrame:
    """reported / inferred / partial for every metric value, from its inputs.

    A ratio is only as filed as the least-filed thing in it. The weakest input
    wins: partial over inferred over reported.

    Read at the period itself. A growth rate also leans on the prior period,
    and that period's provenance is not carried here — a limitation, recorded
    rather than hidden.
    """
    from ..provenance import BASIS_ORDER

    if codes.empty or metric_wide.empty:
        return pd.DataFrame(columns=GROUP_LEVELS + ["period_end", "metric", "basis"])
    codes = codes.reindex(metric_wide.index)
    deps = item_dependencies(metrics)
    out = {}
    for metric_id in metric_wide.columns:
        inputs = [c for c in deps.get(metric_id, ()) if c in codes.columns]
        worst = codes[inputs].max(axis=1) if inputs else pd.Series(0.0, index=codes.index)
        out[metric_id] = worst.where(metric_wide[metric_id].notna())
    frame = (pd.DataFrame(out).stack(future_stack=True).dropna()
             .rename("code").reset_index().rename(columns={"level_3": "metric"}))
    frame["basis"] = frame["code"].astype(int).map(dict(enumerate(BASIS_ORDER)))
    return frame.drop(columns="code")
