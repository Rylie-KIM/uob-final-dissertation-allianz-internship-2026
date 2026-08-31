"""How concentrated is a model's attribution mass — the thesis's central descriptive statistic.

A model that spreads its reasoning over many features and one that funnels it into a handful can
score identically. The SFP claim is about the *second* thing: as forced labels accumulate, later
versions are expected to lean harder on the features that drive the fast-track decision, so the
mean|phi| profile should concentrate. These are the measures that make that claim numeric.

All of them are functions of one vector — mean|phi| per feature, turned into shares p_i summing to
1 — and they are the standard diversity family, imported from ecology (Hill 1973), where the same
question ("how many effective types are there?") is asked of species abundances:

    richness      D0 = number of features with any mass at all
    shannon       H  = -sum p log p                       (entropy, nats)
    exp_shannon   D1 = exp(H)                             effective number of features
    simpson       S  = sum p^2                            concentration; = HHI
    inv_simpson   D2 = 1/S                                effective number, dominance-weighted
    hill(q)       Dq = (sum p^q)^(1/(1-q))                the one-parameter family; q>1 weights
                                                          dominant features more heavily
    gini          inequality of the share distribution, 0 = uniform, ->1 = all on one feature
    top_k_share   the blunt one: how much mass the k biggest features hold

D0 >= D1 >= D2 always. Reporting the three together is more honest than one number: D0 moving while
D2 does not means the tail changed, not the drivers.

TWO THINGS THAT WILL INVALIDATE A COMPARISON, so they are checked, not assumed:

* **Encoded feature names only** — each model's own post-preprocessing columns, exactly as the
  booster sees them (`features/registry/<v>.json` -> `model_features`), never collapsed back to
  the concept they encode. Merging v1's 55 `make_*` columns against v2's 41 into one `make`
  changes every number here — that is the preprocessing-divergence confound, not a finding.
  Cross-version correspondence comes only from the hand-confirmed mapping
  (features/check_overlap.py -> features/feature_overlap.json).
* **Same backend.** Interventional and tree-path-dependent TreeSHAP use different references, so
  mixing them puts a distributional artefact into the comparison. `require_comparable()` reads the
  sidecar meta JSONs and refuses.

Note on the feature COUNT. D0 differs across versions simply because the versions have different
numbers of columns, so UNADJUSTED D1/D2 are not comparable on their own either. `profile()`
therefore also returns Hill's ratio **evenness** (D1/D0, D2/D0) alongside the unadjusted numbers —
report both, and say which one the claim rests on.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

BASE_COL = "_base_value"     # written by scoring/attribute.py; never a feature
SPLIT_COL = "split"          # added by loaders when split="all"; a marker, never a feature


# ======================================================================================
# from attributions to shares
# ======================================================================================


def mean_abs(attributions: pd.DataFrame, id_col: str = "claim_id") -> pd.Series:
    """mean|phi| per feature, descending. The input to every measure below.

    Drops the id, base-value and split-marker columns; everything else is a feature and keeps its
    own encoded feature name. `split` is dropped rather than tolerated: it is a string, so leaving
    it in would raise deep inside .abs() rather than here, where the reason is obvious.
    """
    drop = [c for c in (id_col, BASE_COL, SPLIT_COL) if c in attributions.columns]
    phi = attributions.drop(columns=drop)
    return phi.abs().mean().sort_values(ascending=False)


def shares(mabs: pd.Series | np.ndarray) -> np.ndarray:
    """Normalise mean|phi| to p_i summing to 1, dropping zero-mass features."""
    p = np.asarray(mabs, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    total = p.sum()
    if total <= 0:
        raise ValueError("all attribution mass is zero — nothing to measure")
    return p / total


# ======================================================================================
# the measures
# ======================================================================================


def richness(mabs) -> int:
    """D0 — features carrying any mass. Bounded above by the version's feature count."""
    return int(len(shares(mabs)))


def shannon(mabs) -> float:
    """H = -sum p log p, in nats."""
    p = shares(mabs)
    return float(-(p * np.log(p)).sum())


def simpson(mabs) -> float:
    """sum p^2 — the concentration index (identical to the HHI used in the earlier notebooks)."""
    p = shares(mabs)
    return float((p ** 2).sum())


def hill(mabs, q: float) -> float:
    """Hill number of order q — the effective number of features at that weighting.

    q=0 richness · q->1 exp(shannon) · q=2 inverse Simpson. Continuous in q; the q=1 limit is
    taken analytically because the formula is 0/0 there.
    """
    p = shares(mabs)
    if np.isclose(q, 1.0):
        return float(np.exp(-(p * np.log(p)).sum()))
    return float((p ** q).sum() ** (1.0 / (1.0 - q)))


def inv_simpson(mabs) -> float:
    """D2 = 1 / sum p^2."""
    return 1.0 / simpson(mabs)


def gini(mabs) -> float:
    """Inequality of the shares: 0 = every feature equal, ->1 = one feature holds everything."""
    p = np.sort(shares(mabs))
    n = len(p)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * p).sum()) / (n * p.sum()) - (n + 1) / n)


def top_k_share(mabs, k: int = 5) -> float:
    """Mass held by the k largest features. Blunt, but the one a reader understands unaided."""
    p = np.sort(shares(mabs))[::-1]
    return float(p[:k].sum())


def profile(mabs, k: int = 5) -> dict:
    """Every measure at once, plus the unit-free evenness ratios. One row per version."""
    d0 = richness(mabs)
    d1 = hill(mabs, 1)
    d2 = inv_simpson(mabs)
    return {
        "richness_D0": d0,
        "shannon_H": shannon(mabs),
        "exp_shannon_D1": d1,
        "inv_simpson_D2": d2,
        "simpson_S": simpson(mabs),
        "gini": gini(mabs),
        f"top{k}_share": top_k_share(mabs, k),
        "evenness_D1_D0": d1 / d0,          # unit-free: comparable across different feature counts
        "evenness_D2_D0": d2 / d0,
    }


def profile_table(mabs_by_version: dict[str, pd.Series], k: int = 5) -> pd.DataFrame:
    """profile() for each version, as a table — the shape that goes into the write-up."""
    return pd.DataFrame({v: profile(m, k) for v, m in mabs_by_version.items()}).T


def hill_curve(mabs, q_values=None) -> pd.Series:
    """Dq over a range of q — the diversity profile. Curves that cross tell a subtler story than
    any single q, and are the honest thing to plot when D1 and D2 disagree."""
    q_values = np.linspace(0, 4, 41) if q_values is None else np.asarray(q_values, dtype=float)
    return pd.Series([hill(mabs, q) for q in q_values], index=q_values, name="D_q")


# ======================================================================================
# guards
# ======================================================================================


def require_comparable(metas: dict[str, dict]) -> None:
    """Refuse to compare versions attributed under different backends or references.

    `metas` is {version: the sidecar meta JSON}. Raises with what differs; a warning would be too
    weak, because the resulting numbers look perfectly reasonable.
    """
    keys = ("backend", "perturbation", "model_output")
    seen = {v: tuple(m.get(k) for k in keys) for v, m in metas.items()}
    if len(set(seen.values())) > 1:
        detail = "\n".join(f"    {v}: " + ", ".join(f"{k}={x}" for k, x in zip(keys, s))
                           for v, s in seen.items())
        raise ValueError(
            "these versions were attributed under different SHAP settings, so their concentration "
            f"numbers are not comparable:\n{detail}\n"
            "Re-run src/scoring/attribute_all.py with one --backend for all of them."
        )

    # NOT an error, and usually not fixable: the versions' data windows leave no claim common to
    # all three (v2 2018-01..2020-09, v3 2023-06..2026-05), so per-version sampling is the norm.
    # The note exists so the caveat travels with the number instead of being remembered.
    ids = {v: m.get("explain_ids_file") for v, m in metas.items()}
    if len(set(ids.values())) > 1 or None in ids.values():
        print("NOTE: the versions were explained on their own rows, not one shared claim set "
              f"({ids}). Every difference below is confounded with case-mix — state that wherever "
              "it is reported. A shared set is only reachable for a pair whose windows overlap "
              "(v1/v2): attribute_all.py --shared-claims.")

    # Split names are per-version and deliberately not unified (v3's holdout is "oot", v2's is
    # "test"), so differing names are NOT an error — but one version measured on train against
    # another on a holdout is a real confound, and only the reader can tell which case this is.
    splits = {v: m.get("split") for v, m in metas.items()}
    if None in splits.values():
        print(f"WARNING: some attributions predate the --split flag ({splits}) — which rows they "
              f"describe is unrecorded. Re-run attribute_all.py to stamp it.")
    elif len(set(splits.values())) > 1:
        print(f"NOTE: versions were attributed on differently-named splits ({splits}). Fine when "
              f"they play the same role (v2 'test' and v3 'oot' are both holdouts); a confound "
              f"when one is train and another is not. State which where it is reported.")


def load_meta(path) -> dict:
    """Read the sidecar meta JSON that sits next to an attributions parquet."""
    p = pathlib.Path(path)
    if p.suffix == ".parquet":
        p = p.with_name(p.stem + "_meta.json")
    return json.loads(p.read_text(encoding="utf-8"))
