"""Analysis-side score loader. Reads precomputed per-version score files and merges on claim_id.

No model dependency, no environment awareness — runs in the analysis .venv and never loads a model.
Each score file was produced offline by predict.py inside that version's own env.
"""
from __future__ import annotations

from functools import reduce

import pandas as pd


def load_scores(score_paths: dict[str, str], id_col: str = "claim_id") -> pd.DataFrame:
    """Merge precomputed per-version score files on claim_id.

    score_paths = {"v1": ".../v1_scores.parquet", "v2": ..., "v3": ...}
    """
    frames = [pd.read_parquet(p) for p in score_paths.values()]
    return reduce(lambda l, r: l.merge(r, on=id_col, how="outer"), frames)
