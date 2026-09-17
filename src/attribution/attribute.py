"""Per-row SHAP attributions — predict.py's Version-Layer counterpart; runs INSIDE a version's own env.

WHY THIS IS A SEPARATE PROCESS FROM THE NOTEBOOK
------------------------------------------------
SHAP needs the model *function*, not merely its scores, so it must open the pickle — and a pickle
only unpickles inside the env it was serialised in (`ModuleNotFoundError` otherwise; see
src/docs/DESIGN.md § "Attribution ingestion"). So attribution is Version Layer work: this file runs
under `src/envs/v<k>/.venv/bin/python`, writes `detection/shap/<v>/<v>_attributions.parquet`, and the
notebook — running in the shared analysis `.venv` — reads that parquet and never touches a model.
Nothing about the numbers is version-specific once they are on disk.

WHY THIS LIVES IN `attribution/`, NOT `scoring/` (moved 2026-09-17)
--------------------------------------------------------------------
`scoring/predict.py` and this file share the same MECHANICAL reason for being Version-Layer code —
both must open a pkl, so both must run inside that version's own env — but they answer different
questions. `predict.py` produces the model's actual decision (a score). This file explains an
already-scored decision (why); its output feeds `estimator/`, never `detector/`'s scoring path. That
split-by-concern mirrors why `scoring/` and `training/` are already separate directories despite both
being Version-Layer / pkl-opening code. `backfill_feature_order.py` / `_v1.py` moved here alongside
it for the same reason: they patch attribution metas, not scores.

THE ACTUAL TREESHAP DISPATCH LIVES IN `backend.py` (split out 2026-09-17)
---------------------------------------------------------------------------
`_via_shap_backend` / `_via_native_booster_backend` / `_compute_attribute` come from
`attribution/backend.py` — the same three functions `shap_kit.py`'s `compute()` calls for the
notebook side. They used to be reimplemented here (`_shap_backend`/`_native_backend`/`_attribute`)
with a stricter fallback policy (ImportError-only, xgboost `iteration_range` respected) than
shap_kit.py's more lenient one (any exception falls back, no iteration_range) — real behavioural
drift between two copies of the same computation. shap_kit.py's policy is the one both now use, on
the reasoning that it is the version 00_SHAP.ipynb already runs in practice; see backend.py's own
docstring for the full comparison.

--features is the POST-preprocessing matrix, exactly as predict.py consumes it (confirmed
2026-07-31: the real repos pickle the preprocessor separately, and predict_proba takes the
already-transformed columns). Feature columns therefore keep their **encoded feature names**, which is what
SHAP and the concentration measures must be indexed by. Cross-version correspondence comes only
from the hand-confirmed mapping (features/check_overlap.py -> features/feature_overlap.json).

BACKENDS — and why the choice is recorded, not hidden
-----------------------------------------------------
  shap    shap.TreeExplainer(model, data=background, feature_perturbation="interventional",
          model_output="raw").  Fixing ONE shared background means a difference between versions
          is a difference in the model function, not in the reference distribution. Preferred.
  native  the booster's own TreeSHAP (`pred_contribs=True` / `pred_contrib=True`). Needs no `shap`
          package at all — useful when a frozen version env cannot take a new dependency — but it
          is TREE-PATH-DEPENDENT: it uses each tree's own cover statistics as the reference, so a
          cross-version difference partly reflects the training distributions, not only the
          functions.

The backend and perturbation land in the sidecar meta JSON. `estimator/concentration.py` and the
notebook refuse to compare versions that were attributed under different backends — silently mixing
them would put a distributional artefact into the thesis's central concentration statistic.

USAGE — one call per version, each with THAT version's interpreter:

    src/envs/v2/.venv/bin/python src/attribution/attribute.py \
        --model src/models/real/baseline/v2.pkl \
        --features src/data/real/inputs/features_v2.parquet \
        --version v2 --split test \
        --out src/data/real/detection/shap/v2/v2_attributions_test.parquet \
        --explain-ids src/data/real/inputs/shap_explain_ids.parquet \
        --background-ids src/data/real/inputs/shap_background_ids.parquet

Run once per version, in that version's own env — there is no all-versions driver, since every
version's model only ever unpickles inside its own env anyway. --explain-ids / --background-ids
are optional: pass them to fix a specific claim set (e.g. a shared v1/v2 sample built by hand),
or omit them and --rows / --background draw a per-version sample instead.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

# The two repo imports this worker makes. It runs inside a version env via subprocess and takes
# every path on the command line, so it deliberately knows nothing about config. `select_features`
# is the same selection predict.py/retrain.py fit and score by — see `_select_columns` below for
# why attribute.py needs it too (fixed 2026-09-17; it did not before, and could not have been run
# against a real --features file until now). `attribution.backend` holds the actual TreeSHAP
# dispatch, shared with shap_kit.py — see the module docstring above.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trained_order import select_features  # noqa: E402
from attribution.backend import _compute_attribute  # noqa: E402

BASE_COL = "_base_value"   # the additive constant; phi columns + this = the raw margin


# ======================================================================================
# inputs
# ======================================================================================


def _read_ids(path: str | None, id_col: str) -> list | None:
    """Read a one-column claim_id table (parquet or csv). None means 'no restriction'."""
    if not path:
        return None
    p = pathlib.Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if id_col not in df.columns:
        raise SystemExit(f"{p} has no {id_col!r} column (columns: {list(df.columns)})")
    return df[id_col].tolist()


def _select(df: pd.DataFrame, ids: list | None, id_col: str, n: int | None, seed: int,
            label: str) -> pd.DataFrame:
    """Rows for `ids` (order preserved, missing ones reported), else a seeded sample of n."""
    if ids is not None:
        sub = df[df[id_col].isin(set(ids))]
        missing = len(set(ids)) - sub[id_col].nunique()
        if missing:
            print(f"  {label}: {missing} of {len(set(ids))} requested ids are absent from "
                  f"--features; continuing with the {sub[id_col].nunique()} present")
        if sub.empty:
            raise SystemExit(f"  {label}: none of the requested ids are in --features — stopping "
                             f"rather than silently attributing a different set of claims")
        return sub
    if n is not None and n < len(df):
        return df.sample(n=n, random_state=seed)
    return df


def _estimator(obj):
    """The thing that owns the trees.

    --features is already preprocessed, so a bare estimator is expected. If a full Pipeline was
    pickled anyway, the preprocessing head would have to run first — which would mean --features
    was the WRONG artefact, so we stop instead of guessing.
    """
    if hasattr(obj, "steps"):
        raise SystemExit(
            "the pickle is a Pipeline, not a bare estimator. --features is the POST-preprocessing "
            "matrix, so feeding it through the pipeline's own preprocessing head would transform "
            "it twice. Point --model at the estimator pickle "
            "(config.VERSIONS[v]['paths']['model']), or --features at the raw input."
        )
    return obj


def _select_columns(df: pd.DataFrame, est, id_col: str, features_json: str | None) -> tuple[list[str], str]:
    """The model's own columns, in trained order — same rule predict.py/retrain.py select by.

    FIXED 2026-09-17. Until this, attribute.py treated every non-id column of `--features` as a
    feature (`[c for c in df.columns if c != id_col]`) and then REFUSED (SystemExit) the instant
    that didn't exactly match the trained set — which is every real run, because `processed_inputs`
    is not feature-only: every version's export carries the target beside the model inputs, and
    v3's also carries its own saved predictions (`src/docs/STRUCTURE.md` § "Which columns of the
    matrix are model inputs"). That "refuse rather than repair" stance was the right instinct for a
    script producing a trusted artefact, but it was refusing on the CORRECT file every time, not
    only a wrong one — predict.py and retrain.py hit the identical bug and were fixed 2026-09-02;
    this file was built 2026-08-01, before that invariant was even confirmed, and never got the
    same fix. `select_features` still refuses hard when a TRAINED column is genuinely missing from
    `--features` (SystemExit, from trained_order.py) — that failure mode is real and unchanged.
    """
    return select_features(df, est, id_col, features_json)


# ======================================================================================
# backends
# ======================================================================================


def _params(est) -> dict:
    """The estimator's hyperparameters, recorded beside the attributions on purpose.

    problem.md §1.4c: no adjacent version pair shares a configuration (v2 regularises via penalties,
    v3 via tree structure), so a concentration difference has a competing mechanical explanation.
    The requirement there is that **every version's configuration is printed beside its
    concentration figures** — which is only possible if it is captured here, in the one process that
    can open the pickle. Non-JSON values are stringified rather than dropped.
    """
    try:
        raw = est.get_params()
    except Exception:
        raw = {k: v for k, v in vars(est).items() if not k.startswith("_")}
    out = {}
    for k, v in sorted(raw.items()):
        out[k] = v if isinstance(v, (int, float, bool, str)) or v is None else str(v)
    return out


# ======================================================================================


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)   # that version's processed_inputs matrix
                                                    # (claim_id + model inputs; extras such as the
                                                    # target are set aside by _select_columns)
    p.add_argument("--version", required=True)    # label only, e.g. "v2"
    p.add_argument("--split", required=True,      # WHICH split these phi describe. Recorded in
                   help="split these attributions describe, e.g. train/test/oot")
    p.add_argument("--out", required=True)
    p.add_argument("--id-col", default="claim_id")
    p.add_argument("--features-json", default=None,
                   help="features/registry/<v>.json, written by features/extract_features.py. "
                        "Only consulted when the estimator exposes no feature names.")
    p.add_argument("--explain-ids", default=None,
                   help="parquet/csv of claim_ids to explain — pass the SAME file to every version")
    p.add_argument("--background-ids", default=None,
                   help="parquet/csv of claim_ids for the interventional background")
    p.add_argument("--rows", type=int, default=None,
                   help="sample size when --explain-ids is not given (default: all rows)")
    p.add_argument("--background", type=int, default=500,
                   help="background size when --background-ids is not given (0 = no background, "
                        "which makes the shap backend tree-path-dependent)")
    p.add_argument("--backend", default="auto", choices=("auto", "shap", "native"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    df = pd.read_parquet(a.features)
    if a.id_col not in df.columns:
        raise SystemExit(f"--features has no {a.id_col!r} column (columns: {list(df.columns)[:10]})")
    if BASE_COL in df.columns:
        raise SystemExit(f"a feature is literally named {BASE_COL!r}, which collides with the "
                         f"base-value column written to --out. Rename it upstream.")

    est = _estimator(joblib.load(a.model))       # needs that version's repo importable in THIS env
    feature_cols, order_source = _select_columns(df, est, a.id_col, a.features_json)
    order = "exact" if order_source == "booster" else f"exact (via registry: {order_source})"

    X_exp_rows = _select(df, _read_ids(a.explain_ids, a.id_col), a.id_col, a.rows, a.seed, "explain")
    bg_ids = _read_ids(a.background_ids, a.id_col)
    if bg_ids is None and not a.background:
        X_bg = None
    else:
        X_bg = _select(df, bg_ids, a.id_col, a.background, a.seed + 1, "background")[feature_cols]

    X = X_exp_rows[feature_cols]
    try:
        phi, base, meta = _compute_attribute(est, X, background=X_bg, backend=a.backend)
    except Exception as exc:
        # _compute_attribute (attribution/backend.py, shared with shap_kit.py) re-raises
        # whatever the shap attempt threw when --backend shap was explicit -- shap_kit.py is
        # fine with the bare traceback (a notebook cell), but this is a CLI a person or
        # pipeline.py reads the exit message from, so give it something actionable.
        if a.backend != "shap":
            raise
        raise SystemExit(
            f"--backend shap was requested but failed in this env "
            f"({type(exc).__name__}: {exc}). If shap is simply missing, `uv pip install shap` "
            f"into it; otherwise use --backend native (tree-path-dependent — note it in the "
            f"write-up, and use the SAME backend for every version)."
        ) from exc

    if phi.shape != X.shape:
        raise SystemExit(f"attribution shape {phi.shape} != feature matrix {X.shape}")

    out = pd.DataFrame(phi, columns=feature_cols, index=X.index)
    out.insert(0, a.id_col, X_exp_rows[a.id_col].values)
    out[BASE_COL] = base

    out_path = pathlib.Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    meta.update({
        "version": a.version,
        "split": a.split,                   # concentration on train != concentration on a
                                            #   holdout; the number is unreadable without this
        "model_path": str(a.model),
        "features_path": str(a.features),
        "estimator": type(est).__name__,
        "estimator_params": _params(est),   # printed beside the concentration figures — §1.4c
        "feature_order": order,
        "n_rows": int(len(out)),
        "n_features": len(feature_cols),
        "feature_names": feature_cols,          # the model's own names — the level SHAP must be compared at
        "background_n": 0 if X_bg is None else int(len(X_bg)),
        "explain_ids_file": a.explain_ids,
        "background_ids_file": a.background_ids,
        "seed": a.seed,
        "base_value": float(np.mean(base)),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    })
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mabs = np.abs(phi).mean(axis=0)
    top = sorted(zip(feature_cols, mabs), key=lambda kv: -kv[1])[:5]
    print(f"[{a.version}/{a.split}] {len(out)} rows x {len(feature_cols)} features "
          f"({meta['backend']}/{meta['perturbation']}) -> {out_path}")
    print(f"[{a.version}] top mean|phi|: " + ", ".join(f"{n}={v:.4f}" for n, v in top))


if __name__ == "__main__":
    main()
