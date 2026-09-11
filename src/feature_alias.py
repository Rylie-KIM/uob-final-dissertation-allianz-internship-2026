"""Rename real Allianz feature names to their anonymised alias before a figure is drawn.

Reads features/registry/feature_alias_map.json (built by features/build_feature_alias.py, one
combined file covering all three versions, never committed — see config.alias_map_path()).
Every SHAP/DiD figure that could land in the thesis renames its axis/legend labels through
``to_alias`` instead of plotting a version's raw column names directly.

Deliberately strict: an unmapped name raises rather than passing the real name through unchanged,
because a silent passthrough is exactly the failure mode this module exists to prevent.
"""

from __future__ import annotations

import json
import pathlib
from functools import lru_cache
from typing import Iterable

import config


@lru_cache(maxsize=1)
def _load_map() -> dict[str, dict]:
    p = config.alias_map_path()
    if not p.is_file():
        raise SystemExit(
            f"{p} does not exist. Build it first (analysis .venv):\n"
            f"    uv run python features/build_feature_alias.py\n"
        )
    return json.loads(p.read_text(encoding="utf-8"))


def load_version(version: str) -> dict:
    """That version's {order, real_to_alias, alias_to_real}."""
    payload = _load_map()
    if version not in payload:
        raise SystemExit(
            f"{config.alias_map_path()} has no entry for {version!r} (has: "
            f"{', '.join(payload) or 'nothing'}). Rebuild features/build_feature_alias.py on the "
            f"company laptop once that version's registry exists."
        )
    return payload[version]


def to_alias(version: str, names: Iterable[str]) -> list[str]:
    """Real column names -> anonymised aliases (v1_feat_01, ...), for axis/legend labels."""
    real_to_alias = load_version(version)["real_to_alias"]
    missing = [n for n in names if n not in real_to_alias]
    if missing:
        raise KeyError(
            f"{version}: {len(missing)} name(s) not in the alias map (first 5: {missing[:5]}). "
            f"Rebuild features/build_feature_alias.py — the registry and the alias map have "
            f"drifted apart."
        )
    return [real_to_alias[n] for n in names]


def to_real(version: str, aliases: Iterable[str]) -> list[str]:
    """Aliases -> real names. Debugging only — never feed the result into a figure."""
    alias_to_real = load_version(version)["alias_to_real"]
    return [alias_to_real[a] for a in aliases]
