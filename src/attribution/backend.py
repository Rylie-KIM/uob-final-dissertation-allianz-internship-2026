"""TreeSHAP interventional/native dispatch -- the ONE implementation shared by shap_kit.py
(notebook toolkit, v2/v3) and attribution/attribute.py (the headless CLI worker pipeline.py shells
out to for its per-version "estimate" step).

Until 2026-09-17 this logic was implemented twice, with real behavioural drift between the two
copies: attribute.py's native branch respected xgboost early stopping (`iteration_range`) and its
dispatcher only fell back from shap to native on `ImportError`; shap_kit.py's did neither -- any
shap failure fell back silently, and early stopping was never consulted. shap_kit.py's behaviour is
the one kept here (it is the version 00_SHAP.ipynb already runs against in practice), so this file
is a rename-only extraction of shap_kit.py's compute()/_via_shap()/_via_booster() into attribute.py's
naming convention, returning the plain (phi, base, meta) tuple attribute.py's on-disk artefact
needs rather than a shap_kit.Attribution (which stays defined in shap_kit.py -- it is
plotting-facing, and this module knows nothing about plotting).

Deliberately imports nothing notebook- or config-facing: attribute.py runs headless, once per
version env, and must not depend on either. `align` comes from trained_order.py (stdlib-only,
py3.5-clean) for the same reason shap_kit.py re-exports it rather than importing shap_kit here.

    from attribution.backend import _compute_attribute
    phi, base, meta = _compute_attribute(est, X, background=X_bg, backend="auto")
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from trained_order import align  # noqa: E402


def _via_shap_backend(est, X: pd.DataFrame, background: pd.DataFrame | None):
    """Interventional TreeSHAP against a fixed background. Returns (phi, base, meta)."""
    import shap

    kwargs = {"model_output": "raw"}
    if background is not None:
        kwargs["data"] = background
        kwargs["feature_perturbation"] = "interventional"
    try:
        expl = shap.TreeExplainer(est, **kwargs)
    except TypeError:                                  # older/newer shap dropped a kwarg
        expl = shap.TreeExplainer(est, background) if background is not None else shap.TreeExplainer(est)

    values = expl.shap_values(X)
    if isinstance(values, list):                       # some versions return one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    base = expl.expected_value
    base = float(np.asarray(base).ravel()[-1]) if np.ndim(base) else float(base)
    meta = {
        "backend": "shap",
        "shap_version": shap.__version__,
        "perturbation": "interventional" if background is not None else "tree_path_dependent",
        "model_output": "raw",
    }
    return values, np.full(len(X), base), meta


def _via_native_booster_backend(est, X: pd.DataFrame):
    """The booster's own TreeSHAP -- no `shap` package needed. Tree-path-dependent."""
    if hasattr(est, "get_booster"):                     # xgboost
        import xgboost as xgb

        booster = est.get_booster()
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        contribs = np.asarray(booster.predict(dmatrix, pred_contribs=True))
        if contribs.ndim == 3:
            contribs = contribs[:, -1, :]
        meta = {"backend": "native", "library": "xgboost", "library_version": xgb.__version__,
                "perturbation": "tree_path_dependent", "model_output": "raw"}
        return contribs[:, :-1], contribs[:, -1], meta

    if hasattr(est, "booster_"):                        # lightgbm
        import lightgbm as lgb

        contribs = np.asarray(est.predict(X, pred_contrib=True))
        meta = {"backend": "native", "library": "lightgbm", "library_version": lgb.__version__,
                "perturbation": "tree_path_dependent", "model_output": "raw"}
        return contribs[:, :-1], contribs[:, -1], meta

    raise RuntimeError(f"no TreeSHAP route for {type(est).__name__}; install shap in this env")


def _compute_attribute(est, X: pd.DataFrame, background: pd.DataFrame | None = None,
                        backend: str = "auto"):
    """Dispatch: shap+background preferred, falls back to the booster's own TreeSHAP.

    Returns (phi, base, meta). `backend="shap"` disables the fallback (re-raises instead of
    falling back) -- any other exception during the shap attempt (missing package, an internal
    shap error, ...) falls back to native the same way, matching shap_kit.py's existing policy.
    """
    X = align(X, est)
    if backend in ("auto", "shap"):
        try:
            return _via_shap_backend(est, X, background)
        except Exception as exc:
            if backend == "shap":
                raise
            print(f"  shap unavailable or unhappy here ({type(exc).__name__}: {exc}) "
                  f"-> using the booster's own TreeSHAP")
    return _via_native_booster_backend(est, X)
