"""The item registry, and the small expression language its derivations use.

Derivations are evaluated over pandas Series rather than scalars, so filling a
column for every ticker-period is one vectorized pass instead of 200,000
interpreted ones.

The expression language is deliberately tiny — arithmetic over item codes, plus
opt() and coalesce(). It is parsed with `ast` and walked by hand; nothing is
ever passed to eval(), so a typo in the YAML is a clear error rather than
arbitrary code execution.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

# Comparisons return 1.0 or 0.0 rather than booleans, so a scoring model can
# add them up. Piotroski's F-Score is nine such tests summed, and writing it
# that way keeps it in the registry instead of in Python.
_COMPARISONS = {
    ast.Gt: operator.gt,
    ast.Lt: operator.lt,
    ast.GtE: operator.ge,
    ast.LtE: operator.le,
}


class ExprError(ValueError):
    """A derivation in the registry could not be parsed."""


@dataclass(frozen=True)
class Expr:
    """A parsed derivation.

    `required` names must be present as columns for the expression to be
    evaluable at all. `optional` names appear inside opt() and default to zero.
    """

    source: str
    tree: ast.expr
    required: FrozenSet[str]
    optional: FrozenSet[str]



# Nominal spacing between consecutive periods, in days, by frequency.
_PERIOD_DAYS = {"annual": 365.25, "quarterly": 91.31, "ttm": 91.31}
# A 52/53-week filer's quarter runs 13 or 14 weeks and its year 52 or 53, so a
# true prior period can sit a week off the nominal distance. Three weeks is
# generous for that and still far short of a skipped quarter.
_LAG_TOLERANCE_DAYS = 21


def _lag_is_calendar_true(index, group_levels, periods: int):
    """Whether `periods` rows back is also `periods` periods back in time.

    A shift counts rows, and rows are not time. Where a quarter is missing from
    the data, four rows back is fifteen months back — and a year-over-year
    growth rate silently becomes a fifteen-month one. Digital Realty's first
    quarter of 2011 was being compared with a quarter 639 days earlier. Where
    the distance is wrong the prior value is withheld, so the metric is absent
    rather than mislabelled.
    """
    names = list(index.names)
    if "period_end" not in names or "frequency" not in names:
        return pd.Series(True, index=index)
    dates = pd.Series(pd.to_datetime(index.get_level_values("period_end")), index=index)
    prior = dates.groupby(level=group_levels, sort=False).shift(periods)
    span = (dates - prior).dt.days
    nominal = pd.Series(index.get_level_values("frequency"), index=index).map(_PERIOD_DAYS) * periods
    ok = (span - nominal).abs() <= _LAG_TOLERANCE_DAYS
    return ok | nominal.isna()       # an unknown frequency keeps the old behaviour


# Rows per year at each frequency. lag() counts YEARS, not rows.
_ROWS_PER_YEAR = {"annual": 1, "quarterly": 4, "ttm": 4}


def _lag_years(series: pd.Series, group_levels, years: int) -> pd.Series:
    """The value `years` years earlier, within each company and frequency.

    lag(x, 1) used to mean one ROW back, and every formula in metrics.yaml was
    written meaning one YEAR back — a prior year's revenue, a year-start
    balance. On annual data those coincide. On TTM and quarterly data a row is
    a quarter, so growth compared a trailing year with the trailing year one
    quarter earlier; ROA averaged this quarter-end with last quarter-end; and
    Beneish and Piotroski compared two windows overlapping by three quarters,
    which flattens every year-on-year signal toward "no change". Caterpillar's
    fourth quarter of 2025 showed revenue growth of 8.5% — that was against
    the previous quarter. Against the same quarter a year earlier it was 18%.

    Shifted WITHIN each company and frequency, so one company's earliest
    period never takes the previous company's latest as its prior year, and
    checked against the calendar, so a missing period yields no value rather
    than a comparison with the wrong one.
    """
    if "frequency" not in series.index.names:
        return series.groupby(level=group_levels, sort=False).shift(years)
    freq = np.asarray(series.index.get_level_values("frequency"))
    out = np.full(len(series), np.nan)
    for frequency in np.unique(freq):
        mask = freq == frequency
        rows = _ROWS_PER_YEAR.get(str(frequency), 1) * years
        part = series[mask]
        shifted = part.groupby(level=group_levels, sort=False).shift(rows)
        shifted = shifted.where(_lag_is_calendar_true(part.index, group_levels, rows))
        out[mask] = shifted.to_numpy(dtype=float)
    return pd.Series(out, index=series.index)

def parse_expr(source: str) -> Expr:
    try:
        tree = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        raise ExprError("cannot parse derivation {!r}: {}".format(source, exc)) from exc

    required: set = set()
    optional: set = set()

    def walk(node: ast.expr, soft: bool) -> None:
        if isinstance(node, ast.Name):
            (optional if soft else required).add(node.id)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _OPS:
                raise ExprError("unsupported operator in {!r}".format(source))
            walk(node.left, soft)
            walk(node.right, soft)
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise ExprError("unsupported unary operator in {!r}".format(source))
            walk(node.operand, soft)
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARISONS:
                raise ExprError("unsupported comparison in {!r}".format(source))
            walk(node.left, soft)
            walk(node.comparators[0], soft)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", None)
            if name == "lag":
                if len(node.args) != 2 or not isinstance(node.args[1], ast.Constant):
                    raise ExprError(
                        "lag() takes a name and a constant number of YEARS "
                        "in {!r}".format(source)
                    )
                walk(node.args[0], soft)
            elif name == "opt":
                if len(node.args) != 1:
                    raise ExprError("opt() takes exactly one argument in {!r}".format(source))
                walk(node.args[0], True)
            elif name == "coalesce":
                if len(node.args) < 2:
                    raise ExprError("coalesce() needs at least two arguments in {!r}".format(source))
                # No single argument is required — any one of them will do — so
                # they all register as optional and evaluation checks the set.
                for arg in node.args:
                    walk(arg, True)
            else:
                raise ExprError("unknown function {!r} in {!r}".format(name, source))
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ExprError("only numeric constants are allowed in {!r}".format(source))
        else:
            raise ExprError("unsupported syntax in derivation {!r}".format(source))

    walk(tree, False)
    return Expr(source, tree, frozenset(required), frozenset(optional))


def eval_expr(
    expr: Expr,
    columns: Mapping[str, pd.Series],
    index: pd.Index,
    group_levels: Optional[List[str]] = None,
) -> Optional[pd.Series]:
    """Evaluate a derivation. Returns None when a required input is absent.

    A required input that is *present but null* for some rows yields null for
    those rows — that is correct, and different from the input being absent
    entirely, which makes the whole derivation impossible.
    """
    if not expr.required.issubset(columns.keys()):
        return None

    zero = pd.Series(0.0, index=index)

    def run(node: ast.expr) -> pd.Series:
        if isinstance(node, ast.Name):
            col = columns.get(node.id)
            return zero if col is None else col
        if isinstance(node, ast.Constant):
            return pd.Series(float(node.value), index=index)
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](run(node.left), run(node.right))
        if isinstance(node, ast.Compare):
            op = _COMPARISONS[type(node.ops[0])]
            left, right = run(node.left), run(node.comparators[0])
            result = op(left, right).astype("float64")
            # A test on a missing input has no answer, and scoring it zero
            # would silently mark the company as failing it.
            return result.where(left.notna() & right.notna())
        if isinstance(node, ast.UnaryOp):
            return -run(node.operand) if isinstance(node.op, ast.USub) else run(node.operand)
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None)
            if name == "lag":
                years = int(node.args[1].value)
                series = run(node.args[0])
                if group_levels is None:
                    return series.shift(years)
                return _lag_years(series, group_levels, years)
            if name == "opt":
                inner = node.args[0]
                if isinstance(inner, ast.Name) and inner.id not in columns:
                    return zero
                return run(inner).fillna(0.0)
            if name == "coalesce":
                out: Optional[pd.Series] = None
                for arg in node.args:
                    if isinstance(arg, ast.Name) and arg.id not in columns:
                        continue
                    series = run(arg)
                    out = series if out is None else out.combine_first(series)
                return zero if out is None else out
        raise ExprError("unsupported node during evaluation: {!r}".format(expr.source))

    result = run(expr.tree)
    # Division by a zero denominator yields inf, which is not a number anyone
    # wants downstream; treat it as missing.
    return result.replace([np.inf, -np.inf], np.nan)


@dataclass
class Derivation:
    expr: Expr
    confidence: str = "medium"


@dataclass
class Item:
    code: str
    label: str
    statement: str
    role: str
    period_type: str
    sign: str
    templates: List[str]
    derive: List[Derivation] = field(default_factory=list)
    store: List[str] = field(default_factory=list)
    zero_is_missing: bool = False
    at_least: List[str] = field(default_factory=list)

    @property
    def is_flow(self) -> bool:
        return self.period_type == "flow"

    @property
    def is_instant(self) -> bool:
        return self.period_type == "instant"

    def applies_to(self, template: str) -> bool:
        return template in self.templates


@dataclass
class ItemRegistry:
    items: Dict[str, Item]

    def __getitem__(self, code: str) -> Item:
        return self.items[code]

    def __contains__(self, code: str) -> bool:
        return code in self.items

    def __iter__(self):
        return iter(self.items.values())

    def __len__(self) -> int:
        return len(self.items)

    def get(self, code: str) -> Optional[Item]:
        return self.items.get(code)

    def label(self, code: str) -> str:
        item = self.items.get(code)
        return item.label if item else code

    def for_template(self, template: str) -> List[Item]:
        return [i for i in self.items.values() if i.applies_to(template)]

    def by_statement(self, statement: str) -> List[Item]:
        return [i for i in self.items.values() if i.statement == statement]

    def fill_items(self) -> List[Item]:
        """Tier 1 — statement items recoverable from their reported siblings.

        These are evaluated against the REPORTED baseline, never against each
        other's output, which is what makes mutually-defined identities safe:
        total_liabilities = total_assets - total_equity and its mirror
        total_equity = total_assets - total_liabilities are both correct, and
        each fills exactly the rows the other cannot. Same for the
        gross_profit / cost_of_revenue pair.

        The build runs this tier twice. The second pass sees the first pass's
        fills, which recovers short chains (a missing total_liabilities whose
        total_equity was itself derived) without ever letting a value depend on
        itself — each pass only fills nulls and never overwrites.
        """
        return [i for i in self.items.values() if i.derive and i.statement != "derived"]

    def derived_order(self) -> List[Item]:
        """Tier 2 — aggregates that exist only as identities, in dependency order.

        ebitda depends on ebit, net_debt on total_debt. Registry order would
        compute them backwards and silently yield nulls, so sort topologically
        and fail loudly on a cycle. Dependencies on tier-1 items are free:
        tier 1 has already run by the time this tier is evaluated.
        """
        pending = {
            i.code: i
            for i in self.items.values()
            if i.derive and i.statement == "derived"
        }
        deps = {
            code: {
                name
                for d in item.derive
                for name in (d.expr.required | d.expr.optional)
                if name in pending and name != code
            }
            for code, item in pending.items()
        }
        ordered: List[Item] = []
        done: set = set()
        while pending:
            ready = sorted(c for c, d in deps.items() if c in pending and d <= done)
            if not ready:
                raise ExprError(
                    "circular derivation among aggregates: " + ", ".join(sorted(pending))
                )
            for code in ready:
                ordered.append(pending.pop(code))
                done.add(code)
        return ordered

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ItemRegistry":
        path = Path(path) if path else PROJECT_ROOT / "registry" / "items.yaml"
        raw: Dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        defaults: Dict[str, Any] = raw.get("defaults", {})
        items: Dict[str, Item] = {}

        for code, spec in (raw.get("items") or {}).items():
            spec = spec or {}
            derivations = [
                Derivation(parse_expr(d["expr"]), d.get("confidence", "medium"))
                for d in (spec.get("derive") or [])
            ]
            items[code] = Item(
                code=code,
                label=spec.get("label", code.replace("_", " ").title()),
                statement=spec.get("statement", "derived"),
                role=spec.get("role", defaults.get("role", "component")),
                period_type=spec.get("period_type", defaults.get("period_type", "instant")),
                sign=spec.get("sign", defaults.get("sign", "positive")),
                templates=list(spec.get("templates", defaults.get("templates", []))),
                derive=derivations,
                store=list(spec.get("store") or []),
                zero_is_missing=bool(spec.get("zero_is_missing", False)),
                at_least=list(spec.get("at_least") or []),
            )

        registry = cls(items=items)
        unknown = {
            name
            for item in items.values()
            for d in item.derive
            for name in (d.expr.required | d.expr.optional)
            if name not in items
        }
        if unknown:
            raise ExprError(
                "derivations reference unknown item(s): " + ", ".join(sorted(unknown))
            )
        registry.derived_order()  # surfaces cycles at load time, not mid-build
        return registry


STATEMENTS = ("income_statement", "balance_sheet", "cash_flow")


def load_statement_layout(path: Optional[Path] = None) -> Dict[str, List[Tuple[str, List[str]]]]:
    """Sections and line order for each statement, from registry/statements.yaml.

    Returned as {statement: [(section, [item codes in order]), ...]}.
    """
    path = Path(path) if path else Path(__file__).resolve().parent.parent / "registry" / "statements.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    return {statement: [(block["section"], list(block["items"])) for block in raw.get(statement, [])]
            for statement in STATEMENTS}
