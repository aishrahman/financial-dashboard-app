"""Settings loading.

Follows the same shape as the extractor's sec_extractor/config.py: a dataclass
with defaults, loaded from one YAML file, with unknown keys rejected loudly
rather than silently ignored.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_VAR = re.compile(r"\$\{(\w+)\}")


def _expand(value: str, base: Path = PROJECT_ROOT) -> Path:
    """Resolve ~, and make relative paths relative to the project root.

    Relative-to-project (rather than relative-to-cwd) matters because Streamlit
    is often launched from somewhere else entirely.
    """
    text = os.path.expanduser(str(value))
    path = Path(text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _interpolate(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ${key} references between settings, so paths stay DRY."""
    out = dict(raw)
    for _ in range(5):  # bounded: references are shallow by design
        changed = False
        for key, value in out.items():
            if not isinstance(value, str):
                continue
            new = _VAR.sub(lambda m: str(out.get(m.group(1), m.group(0))), value)
            if new != value:
                out[key], changed = new, True
        if not changed:
            break
    return out


@dataclass
class Settings:
    extractor_dir: Path = PROJECT_ROOT.parent / "3. 10K & 10Q Extractor"
    panel: Path = PROJECT_ROOT.parent / "3. 10K & 10Q Extractor/output/_panel.parquet"
    segments: Optional[Path] = None
    footnotes: Optional[Path] = None
    run_summary: Optional[Path] = None
    resolve_report: Optional[Path] = None
    restatements: Optional[Path] = None
    facts_db: Optional[Path] = None
    exclude_status: List[str] = field(default_factory=lambda: ["foreign", "error"])
    curated_dir: Path = PROJECT_ROOT / "data/curated"
    prices_dir: Path = PROJECT_ROOT / "data/prices"
    price_source: str = "yfinance"
    price_start: str = "2005-01-01"
    price_max_age_days: int = 7

    PATH_KEYS = (
        "extractor_dir", "panel", "segments", "footnotes",
        "run_summary", "resolve_report", "restatements",
        "facts_db", "curated_dir", "prices_dir",
    )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Settings":
        path = Path(path) if path else PROJECT_ROOT / "config" / "settings.yaml"
        raw: Dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        raw = _interpolate(raw)

        for key in cls.PATH_KEYS:
            if raw.get(key):
                raw[key] = _expand(raw[key])

        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                "Unknown key(s) in settings.yaml: " + ", ".join(sorted(unknown))
            )
        return cls(**raw)

    def resolved_panel(self) -> Path:
        """The panel file, or a clear error saying the extractor has not run.

        Deliberately not checked in `load`. The dashboard reads only the
        curated parquet — it never opens the panel, the fact store or the
        network — so a deployment that ships `data/curated` needs no extractor
        output at all. Checking at load time made `settings()` raise on any
        machine without it, which took the app down on boot rather than at the
        one point that actually needs the file.
        """
        if self.panel.exists():
            return self.panel
        # The extractor writes both formats; accept whichever survives.
        fallback = self.panel.with_suffix(
            ".csv" if self.panel.suffix == ".parquet" else ".parquet")
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            "Panel not found at {} (nor {}).\nRun the extractor first, "
            "or fix the inbound paths in config/settings.yaml.".format(
                self.panel, fallback.name)
        )

    def ensure_dirs(self) -> None:
        for d in (self.curated_dir, self.prices_dir):
            d.mkdir(parents=True, exist_ok=True)
