"""Canonical column names, and the one function that translates into them.

Every version's production log calls the same thing something different (v1 `lossdate`,
v2 `ReportedDate`, v3 `ReportedDate_CLAIM`). Application code must not know that. So the real
names are declared in `config.VERSIONS[v]["columns"]`, translated ONCE here at ingest, and after
that everything downstream says only the canonical names below.

The names are deliberately few. They are the structural columns an SFP analysis needs — key, time,
treatment, outcome, segmenter — and nothing else. In causal terms:

    CLAIM_ID   the unit
    DATE       time
    SCORE      the model's output
    DECISION   the TREATMENT (1 = fast-tracked to scrap, 0 = sent to garage)
    OBSERVED   the OUTCOME as recorded — contaminated by construction, never a ground truth
    MOBILITY   the segmenting covariate v1's decision rule keys on

FEATURE columns are NOT canonicalised and must never be added here. Their encoded feature names are the
measurement itself — collapsing v1's 55 `make_*` against v2's 41 would change the dissertation's
central concentration statistic. Cross-version feature correspondence is governed separately, by
the hand-confirmed mapping in features/feature_overlap.json (built by features/check_overlap.py
from the Excel sheet — never by string matching).
"""

from __future__ import annotations

import pandas as pd

import config

CLAIM_ID = "claim_id"
DATE = "date"
SCORE = "score" # only used for v2 logs for now (2026-08-19)
DECISION = "decision"
OBSERVED = "observed"
MOBILITY = "mobility"

#: What any per-version log must provide once translated. MOBILITY is excluded: only v1 has it.
REQUIRED = (CLAIM_ID, DATE, SCORE, DECISION, OBSERVED)


def to_canonical(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """Rename one version's real column names to the canonical ones.

    Columns not mentioned in that version's mapping are left untouched — feature columns pass
    through with their raw names, which is what we want.
    """
    return df.rename(columns=config.rename_map(version))


def require(df: pd.DataFrame, version: str, columns=REQUIRED) -> None:
    """Fail loudly, and specifically, if a canonical column is missing after translation.

    The error names the version and points at the config entry to fix, because that is always the
    cause: either the real column name was declared wrongly, or it was never declared at all.
    """
    missing = [c for c in columns if c not in df.columns]
    if not missing:
        return
    detail = []
    for c in missing:
        declared = config.VERSIONS[version].get("columns", {}).get(c)
        if declared is None:
            detail.append(f"    {c!r}: declared as None — this version claims not to have it")
        elif declared == config.PLACEHOLDER:
            detail.append(f"    {c!r}: not declared yet (config.VERSIONS['{version}']['columns'])")
        else:
            detail.append(f"    {c!r}: declared as {declared!r}, but no such column in the file")
    raise KeyError(
        f"[{version}] missing canonical columns after translation:\n" + "\n".join(detail)
    )
