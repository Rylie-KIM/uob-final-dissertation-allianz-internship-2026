"""Per-row SHAP attributions — the sibling of predict.py; runs INSIDE a version's own env.

WHY THIS IS A SEPARATE PROCESS FROM THE NOTEBOOK
------------------------------------------------
SHAP needs the model *function*, not merely its scores, so it must open the pickle — and a pickle
only unpickles inside the env it was serialised in (`ModuleNotFoundError` otherwise; see
src/docs/DESIGN.md § "Attribution ingestion"). So attribution is Version Layer work: this file runs
under `src/envs/v<k>/.venv/bin/python`, writes `detection/shap/<v>_attributions.parquet`, and the
notebook — running in the shared analysis `.venv` — reads that parquet and never touches a model.
Nothing about the numbers is version-specific once they are on disk.

--features is the POST-preprocessing matrix, exactly as predict.py consumes it (confirmed
2026-07-31: the real repos pickle the preprocessor separately, and predict_proba takes the
already-transformed columns). Feature columns therefore keep their **L0 raw names**, which is what
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

    src/envs/v2/.venv/bin/python src/scoring/attribute.py \
        --model src/models/real/baseline/v2.pkl \
        --features src/data/real/inputs/features_v2.parquet \
        --version v2 --split test \
        --out src/data/real/detection/shap/v2_attributions_test.parquet \
        --explain-ids src/data/real/inputs/shap_explain_ids.parquet \
        --background-ids src/data/real/inputs/shap_background_ids.parquet

(In practice run src/scoring/attribute_all.py instead — it resolves every path from config and
builds the two id files so all versions are explained on the SAME claims.)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

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


def _check_feature_order(est, columns: list[str]) -> str:
    """Column order must match what the booster was trained on — SHAP is positional underneath."""
    trained = None
    if hasattr(est, "get_booster"):
        try:
            trained = est.get_booster().feature_names
        except Exception:
            trained = None
    if trained is None:
        trained = getattr(est, "feature_name_", None)          # lightgbm
    if trained is None:
        names = getattr(est, "feature_names_in_", None)
        trained = list(names) if names is not None else None
    if not trained:
        return "unverified (the estimator exposes no trained feature names)"
    trained = [str(c) for c in trained]
    if trained == columns:
        return "exact"
    if sorted(trained) == sorted(columns):
        raise SystemExit(
            "--features has the right columns in the WRONG ORDER. SHAP indexes positionally, so "
            "this would attribute every value to the wrong feature. Reorder --features to the "
            f"trained order, e.g. df[{trained[:4]} ...]."
        )
    only_model = [c for c in trained if c not in set(columns)][:8]
    only_file = [c for c in columns if c not in set(trained)][:8]
    raise SystemExit(
        f"--features does not match the model's features.\n"
        f"  in model, not in file: {only_model}\n"
        f"  in file, not in model: {only_file}"
    )


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


def _iteration_range(est):
    """Respect early stopping, so phi sums to the margin predict_proba actually used."""
    best = getattr(est, "best_iteration", None)
    return (0, int(best) + 1) if best is not None else None


def _shap_backend(est, X: pd.DataFrame, X_bg: pd.DataFrame | None):
    """Interventional TreeSHAP against a fixed background. Returns (phi, base, meta)."""
    import shap

    kwargs = {"model_output": "raw"}
    if X_bg is not None:
        kwargs["data"] = X_bg
        kwargs["feature_perturbation"] = "interventional"
    try:
        expl = shap.TreeExplainer(est, **kwargs)
    except TypeError:                       # older/newer shap dropped one of the kwargs
        expl = shap.TreeExplainer(est, data=X_bg) if X_bg is not None else shap.TreeExplainer(est)

    values = expl.shap_values(X)
    if isinstance(values, list):            # multiclass / some binary shap versions
        values = values[-1]                 # the positive class
    values = np.asarray(values)
    if values.ndim == 3:                    # (n, features, classes)
        values = values[:, :, -1]

    base = expl.expected_value
    base = np.asarray(base).ravel()[-1] if np.ndim(base) else float(base)
    meta = {
        "backend": "shap",
        "shap_version": shap.__version__,
        "perturbation": "interventional" if X_bg is not None else "tree_path_dependent",
        "model_output": "raw",
    }
    return values, np.full(len(X), float(base)), meta


def _native_backend(est, X: pd.DataFrame):
    """The booster's own TreeSHAP — no `shap` package needed. Tree-path-dependent."""
    if hasattr(est, "get_booster"):                                  # xgboost
        import xgboost as xgb

        booster = est.get_booster()
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        kw = {"pred_contribs": True}
        rng = _iteration_range(est)
        if rng is not None:
            kw["iteration_range"] = rng
        try:
            contribs = booster.predict(dm, **kw)
        except TypeError:                                            # xgboost < 1.4
            kw.pop("iteration_range", None)
            contribs = booster.predict(dm, **kw)
        contribs = np.asarray(contribs)
        if contribs.ndim == 3:
            contribs = contribs[:, -1, :]
        meta = {"backend": "native", "library": "xgboost", "library_version": xgb.__version__,
                "perturbation": "tree_path_dependent", "model_output": "raw"}
        return contribs[:, :-1], contribs[:, -1], meta

    if hasattr(est, "booster_"):                                     # lightgbm
        import lightgbm as lgb

        contribs = np.asarray(est.predict(X, pred_contrib=True))
        meta = {"backend": "native", "library": "lightgbm", "library_version": lgb.__version__,
                "perturbation": "tree_path_dependent", "model_output": "raw"}
        return contribs[:, :-1], contribs[:, -1], meta

    raise SystemExit(
        f"no native TreeSHAP for {type(est).__name__}. Install `shap` in this version's env and "
        f"rerun with --backend shap, or add a branch here for this estimator family."
    )


def _attribute(est, X: pd.DataFrame, X_bg: pd.DataFrame | None, backend: str):
    """Dispatch, with `auto` preferring the interventional (comparable) route."""
    if backend in ("auto", "shap"):
        try:
            return _shap_backend(est, X, X_bg)
        except ImportError:
            if backend == "shap":
                raise SystemExit(
                    "--backend shap was requested but `shap` is not installed in this env. Either "
                    "`uv pip install shap` into it, or use --backend native (tree-path-dependent — "
                    "note it in the write-up, and use the SAME backend for every version)."
                )
            print("  shap not installed in this env -> falling back to the booster's own TreeSHAP")
    return _native_backend(est, X)


# ======================================================================================


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)   # claim_id + preprocessed features ONLY
    p.add_argument("--version", required=True)    # label only, e.g. "v2"
    p.add_argument("--split", required=True,      # WHICH split these phi describe. Recorded in
                   help="split these attributions describe, e.g. train/test/oot")
    p.add_argument("--out", required=True)
    p.add_argument("--id-col", default="claim_id")
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

    feature_cols = [c for c in df.columns if c != a.id_col]
    if BASE_COL in feature_cols:
        raise SystemExit(f"a feature is literally named {BASE_COL!r}, which collides with the "
                         f"base-value column written to --out. Rename it upstream.")

    est = _estimator(joblib.load(a.model))       # needs that version's repo importable in THIS env
    order = _check_feature_order(est, feature_cols)

    X_exp_rows = _select(df, _read_ids(a.explain_ids, a.id_col), a.id_col, a.rows, a.seed, "explain")
    bg_ids = _read_ids(a.background_ids, a.id_col)
    if bg_ids is None and not a.background:
        X_bg = None
    else:
        X_bg = _select(df, bg_ids, a.id_col, a.background, a.seed + 1, "background")[feature_cols]

    X = X_exp_rows[feature_cols]
    phi, base, meta = _attribute(est, X, X_bg, a.backend)

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
        "feature_names": feature_cols,          # L0 — the level SHAP must be compared at
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
