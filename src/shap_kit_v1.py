# -*- coding: utf-8 -*-
"""Python 3.5 twin of shap_kit.py, for env-v1 only.

env-v1 is Python 3.5.6, so it cannot import shap_kit.py, config.py or figstyle.py (all are
3.7+ syntax). This module carries the v1 constants those files would have provided, plus the
native-TreeSHAP subset that notebook/real/00_SHAP_v1.ipynb needs: xgboost's own
`pred_contribs=True` (exact TreeSHAP, tree_path_dependent, no shap package required).
Same rule as features/extract_features_v1.py: never retrofit v1 support into the shared files.
"""
import json
import os
import re
import sys
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, RED, GREY = "#2a78d6", "#e34948", "#898781"
INK, AXIS = "#0b0b0b", "#c3c2b7"
SERIES = [BLUE, "#1baf7a", "#eda100", "#008300", "#4a3aa7", RED, "#e87ba4", "#eb6834"]
CMAP = LinearSegmentedColormap.from_list("sfp_div", [BLUE, "#dcdcd6", RED])
CMAP_SEQ = LinearSegmentedColormap.from_list("sfp_seq", ["#dcdcd6", BLUE])
FIG_W = 6.3

VERSION = "v1"
XGBOOST_PIN = "0.72"
SPLITS = ("train", "test", "val1", "val2")
OOT_SPLIT = "val2"
ID_COLUMN = "claimnumber"
TARGET_COLUMN = "veh_total_loss"
TRAINING_CONFIG = OrderedDict([("eval_metric", "mlogloss"), ("n_jobs", 20), ("silent", False)])

# Which attribution lands on the CANONICAL (no-suffix) filename, and what the others are called.
# Identical to 00_SHAP.ipynb's ACTIVE_BACKEND/OUT_SUFFIX for v2/v3, so all three versions spell
# the same backend the same way: 00_shap_attribution.ipynb's interventional runs read the
# no-suffix file and its "path_dependent" run reads "_native" (attribute_all.py --out-suffix's
# spelling, which is why it is not "_tree_path_dependent").
CANONICAL_PERTURBATION = "interventional"
BACKEND_SUFFIX = OrderedDict([("interventional", "_shap"), ("tree_path_dependent", "_native")])
DECISION_RULE = OrderedDict([
    ("shape", "segmented"),
    ("segment_by", "mobility"),
    ("thresholds", OrderedDict([("immobile", 0.75), ("mobile", 0.85)])),
    ("mobile_values", ("Mobile", "Mobile Not Roadworthy", "Mobile Not Secure")),
    ("overlap_band", (0.75, 0.85)),
])

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
FIG_DIR = os.path.join(ROOT, "figures")
REGISTRY_PATH = os.path.join(ROOT, "features", "registry", "v1.json")


def features_csv_path(split):
    if split not in SPLITS:
        raise KeyError("unknown split {!r}; expected one of {}".format(split, SPLITS))
    return os.path.join(ROOT, "src", "data", "real", "inputs",
                        "features_v1_{}.csv".format(split))


def out_suffix(perturbation):
    """Filename suffix for one backend: "" for the canonical slot, else e.g. "_native".

    v1 is the version that can produce only tree_path_dependent when `shap` is not installed
    here (the booster's pred_contribs). That output must NOT take the canonical name — that
    slot means "interventional" to 00_shap_attribution.ipynb, and mixing a native file into an
    interventional run is exactly what concentration.require_comparable refuses.
    """
    if perturbation == CANONICAL_PERTURBATION:
        return ""
    if perturbation not in BACKEND_SUFFIX:
        raise KeyError("unknown perturbation {!r}; expected one of {}".format(
            perturbation, tuple(BACKEND_SUFFIX)))
    return BACKEND_SUFFIX[perturbation]


def attributions_csv_path(split, suffix=""):
    """CSV path for one (split, backend). Convert to parquet in the analysis .venv before
    00_shap_attribution.ipynb reads it — env-v1 has no parquet engine.

    `suffix` is appended AFTER the split, matching config.path("attributions", ...) +
    attribute_all.py's --out-suffix: v1_attributions_val2_native.csv. Pass out_suffix(
    att.perturbation) rather than a literal, so the two spellings cannot drift.

    config.py's "attributions" template gives every version its OWN directory under shap/
    ("src/data/{source}/detection/shap/{v}/{v}_attributions.parquet"), and that is where
    00_shap_attribution.ipynb looks. Dropping the "v1" component here would leave the
    converted parquet one level up, unreadable by the cross-version notebook.
    """
    if split not in SPLITS:
        raise KeyError("unknown split {!r}; expected one of {}".format(split, SPLITS))
    return os.path.join(ROOT, "src", "data", "real", "detection", "shap", "v1",
                        "v1_attributions_{}{}.csv".format(split, suffix))


def registry_features():
    if not os.path.exists(REGISTRY_PATH):
        raise RuntimeError(
            "the estimator exposes no trained feature names and {} does not exist. Build it in "
            "THIS env first:\n    <env-v1 python> features/extract_features_v1.py "
            "--model <path to fasttracker_xgb.pkl>".format(REGISTRY_PATH))
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        registry = json.load(fh)
    trained = [str(c) for c in (registry.get("model_features") or [])]
    if not trained:
        raise RuntimeError("{} has no `model_features`".format(REGISTRY_PATH))
    return trained


def registry_features_source():
    """WHERE the registry's column ORDER came from, or "unrecorded" for a pre-2026-08-31 file.

    extract_features_v1.py writes this as `model_features_source`. It matters because the rungs
    are not equally strong: "booster.feature_names" IS the fit order, while a preprocessing
    head's "get_feature_names_out" is only the order that head emitted -- a good inference, not
    a record. A verdict of "exact (via registry)" is worth different amounts in the two cases.
    """
    if not os.path.exists(REGISTRY_PATH):
        return "unrecorded"
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        registry = json.load(fh)
    return str(registry.get("model_features_source") or "unrecorded")


def style():
    import matplotlib as mpl
    wanted = OrderedDict([
        ("font.family", "sans-serif"),
        ("font.sans-serif", ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]),
        ("font.size", 10.0),
        ("axes.titlesize", 12.0),
        ("axes.labelsize", 10.5),
        ("axes.grid", True),
        ("axes.axisbelow", True),
        ("grid.color", "#e1e0d9"),
        ("axes.edgecolor", AXIS),
        ("axes.spines.top", False),
        ("axes.spines.right", False),
        ("figure.facecolor", "white"),
        ("axes.facecolor", "white"),
        ("savefig.facecolor", "white"),
        ("savefig.dpi", 150),
    ])
    try:
        from cycler import cycler
        wanted["axes.prop_cycle"] = cycler(color=SERIES)
    except Exception:
        pass
    skipped = []
    for key, value in wanted.items():
        try:
            mpl.rcParams[key] = value
        except Exception:
            skipped.append(key)
    if skipped:
        print("  (matplotlib {} does not know {} -- skipped)".format(mpl.__version__, skipped))


def save(fig, name):
    if not os.path.isdir(FIG_DIR):
        os.makedirs(FIG_DIR)
    png = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    return png


def env_report(strict=True):
    info = OrderedDict([("version", VERSION),
                        ("python", sys.version.split()[0]),
                        ("executable", sys.executable)])
    try:
        import xgboost
        info["xgboost"] = xgboost.__version__
    except Exception as exc:
        info["xgboost"] = "NOT IMPORTABLE ({})".format(exc)
    for name in ("numpy", "pandas", "matplotlib", "joblib"):
        try:
            info[name] = __import__(name).__version__
        except Exception:
            info[name] = "-"
    info["xgboost_expected"] = XGBOOST_PIN

    width = max(len(k) for k in info)
    print("environment")
    for k, v in info.items():
        print("  {}  {}".format(k.ljust(width), v))

    if sys.version_info[:2] != (3, 5):
        warnings.warn("this module targets env-v1 (Python 3.5.6); running on {}".format(
            info["python"]))
    if info["xgboost"] != XGBOOST_PIN:
        message = ("this kernel has xgboost {}, but v1 was serialised with {}. Either the "
                   "kernel or the pickle is the wrong one -- and both would still produce "
                   "figures.".format(info["xgboost"], XGBOOST_PIN))
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)
    return info


def _load_any(path):
    """Unpickle v1's artefact with whichever loader the 2018 stack actually used.

    Plain `joblib.load` is NOT enough here. v1 was serialised on sklearn < 0.23, whose
    `sklearn.externals.joblib` is a DIFFERENT vendored copy — standalone joblib's
    NumpyUnpickler desyncs on that stream and dies inside pickle.py with `KeyError: <n>`
    (n = whatever byte it landed on, commonly 0), after passing through numpy_pickle.py.
    Same ladder, same order as features/extract_features_v1.py, which is the version proven
    against the real pickle on the company laptop.

    Returns (object, loader_name).
    """
    attempts = []

    def _sklearn_joblib():
        from sklearn.externals import joblib as sk_joblib   # sklearn < 0.23 only
        return sk_joblib.load(path)

    def _standalone_joblib():
        import joblib
        return joblib.load(path)

    def _joblib_compat():
        # joblib <= 0.9 wrote the arrays as separate .npy sidecars beside the pkl.
        from joblib.numpy_pickle_compat import load_compatibility
        return load_compatibility(path)

    def _plain_pickle():
        import pickle
        fh = open(path, "rb")
        try:
            return pickle.load(fh)
        finally:
            fh.close()

    for name, fn in (("sklearn.externals.joblib", _sklearn_joblib),
                     ("joblib", _standalone_joblib),
                     ("joblib.load_compatibility", _joblib_compat),
                     ("pickle", _plain_pickle)):
        try:
            obj = fn()
        except ImportError as exc:
            attempts.append((name, "not available: {0}".format(exc)))
        except Exception as exc:
            attempts.append((name, "{0}: {1}".format(type(exc).__name__, exc)))
        else:
            return obj, name

    lines = ["could not unpickle {0} with any loader:".format(path)]
    for name, why in attempts:
        lines.append("    {0:<28} {1}".format(name, str(why)[:120]))
    lines.append("")
    lines.append("A `KeyError: <n>` from EVERY loader means the pickle stream desynced -- the file")
    lines.append("was written by a stack this env cannot reproduce. Check `dir` beside the pkl for")
    lines.append(".npy sidecars (very old joblib), and compare this env's scikit-learn/numpy")
    lines.append("against the v1 repo's conda_dependencies_local.yml.")
    raise RuntimeError("\n".join(lines))


def load_estimator(path):
    obj, loader = _load_any(path)
    print("  loaded with {}".format(loader))
    if hasattr(obj, "steps"):
        print("  note: {} is a Pipeline ({}). Explaining the final step; X must be the "
              "POST-preprocessing matrix.".format(
                  os.path.basename(str(path)), [n for n, _ in obj.steps]))
        return obj.steps[-1][1]
    return obj


def model_feature_names(est):
    getters = (lambda: list(est.get_booster().feature_names),
               lambda: list(est.feature_name_),
               lambda: [str(c) for c in est.feature_names_in_])
    for getter in getters:
        try:
            names = getter()
        except Exception:
            continue
        if names and not all(re.match(r"^f\d+$", str(n)) for n in names):
            return [str(n) for n in names]
    return []


UNVERIFIED_ORDER = "unverified (the estimator exposes no trained feature names)"


def feature_order(est, columns, trained=None, trained_source=None):
    """Status of `columns` against the booster's trained order -- the meta's `feature_order`.

    Same four words as shap_kit.feature_order in the analysis env, so the field means one thing
    across v1, v2 and v3 metas: "exact", "reordered", "set_mismatch", "unverified ...".

    v1 reaches "unverified" more easily than the other two, and that is not a defect here: this
    module's model_feature_names() rejects xgboost 0.72's generic f0/f1/... names, because a
    positional placeholder is not evidence that the order is right. An honest "unverified" is
    the correct record in that case.
    """
    source = ""
    if trained is None:
        trained = model_feature_names(est)
    else:
        # Pass the list the caller actually ordered X by. 00_SHAP_v1 resolves it from
        # registry_features() when the booster exposes nothing, and without this the meta would
        # read "unverified" for a run whose order WAS checked -- against the registry.
        trained = [str(c) for c in trained]
        source = " (via registry: {})".format(trained_source or "unrecorded")
    cols = [str(c) for c in columns]
    if not trained:
        return UNVERIFIED_ORDER
    if trained == cols:
        return "exact" + source
    if sorted(trained) == sorted(cols):
        return "reordered" + source
    return "set_mismatch" + source


def align(X, est):
    status = feature_order(est, X.columns)
    if status == UNVERIFIED_ORDER:
        print("  the estimator exposes no feature names -- column order is UNVERIFIED. "
              "Check it by hand before trusting any per-feature claim.")
        return X
    trained = model_feature_names(est)
    if status == "set_mismatch":
        have, want = set(X.columns), set(trained)
        raise ValueError(
            "X does not match the model's features.\n"
            "  in model, not in X : {}\n"
            "  in X, not in model : {}".format(sorted(want - have)[:12], sorted(have - want)[:12]))
    if status == "reordered":
        print("  reordered X into the booster's trained column order")
    return X[trained]


def describe_features(X):
    rows = OrderedDict()
    for col in X.columns:
        s = X[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        values = pd.to_numeric(s, errors="coerce") if numeric else s
        uniq = int(s.nunique(dropna=True))
        rows[col] = OrderedDict([
            ("dtype", str(s.dtype)),
            ("n_unique", uniq),
            ("kind", ("constant" if uniq <= 1 else
                      "categorical" if not numeric else
                      "binary" if uniq == 2 else "numeric")),
            ("pct_missing", float(s.isnull().mean() * 100)),
            ("pct_zero", float((values == 0).mean() * 100) if numeric else np.nan),
            ("mean", float(values.mean()) if numeric else np.nan),
            ("std", float(values.std()) if numeric else np.nan),
            ("min", float(values.min()) if numeric else np.nan),
            ("max", float(values.max()) if numeric else np.nan),
            ("family", col.rsplit("_", 1)[0] if "_" in col else col),
        ])
    return pd.DataFrame(rows).T


def feature_summary(X):
    d = describe_features(X)
    fams = d[d["kind"] == "binary"]["family"].value_counts()
    return pd.Series(OrderedDict([
        ("n_features", len(X.columns)),
        ("n_rows", len(X)),
        ("n_numeric", int((d["kind"] == "numeric").sum())),
        ("n_binary", int((d["kind"] == "binary").sum())),
        ("n_categorical", int((d["kind"] == "categorical").sum())),
        ("n_constant", int((d["kind"] == "constant").sum())),
        ("n_onehot_families", int((fams > 1).sum())),
        ("largest_onehot_family",
         "{} ({})".format(fams.index[0], fams.iloc[0]) if len(fams) else "-"),
        ("pct_cells_missing", round(float(X.isnull().mean().mean() * 100), 3)),
    ]))


class Attribution(object):

    def __init__(self, phi, base, X, backend, perturbation, note=""):
        self.phi = np.asarray(phi, dtype=float)
        self.base = np.asarray(base, dtype=float).ravel()
        self.X = X
        self.backend = backend
        self.perturbation = perturbation
        self.note = note
        self.features = list(X.columns)

    def __repr__(self):
        return "<Attribution {} rows x {} features, {}/{}>".format(
            self.phi.shape[0], self.phi.shape[1], self.backend, self.perturbation)

    @property
    def mean_abs(self):
        return pd.Series(np.abs(self.phi).mean(axis=0),
                         index=self.features).sort_values(ascending=False)

    @property
    def margin(self):
        return self.base + self.phi.sum(axis=1)

    def top(self, n=15):
        return list(self.mean_abs.head(n).index)

    def frame(self, id_values=None, id_col="claim_id"):
        out = pd.DataFrame(self.phi, columns=self.features, index=self.X.index)
        if id_values is not None:
            out.insert(0, id_col, np.asarray(id_values))
        out["_base_value"] = self.base
        return out

    def subset(self, mask):
        mask = np.asarray(mask)
        return Attribution(self.phi[mask], self.base[mask], self.X.loc[mask],
                           self.backend, self.perturbation, self.note)


def has_shap():
    """The installed `shap` version, or None. env-v1 can run without it, at a cost."""
    try:
        import shap
    except Exception:
        return None
    return getattr(shap, "__version__", "?")


def _via_shap(est, X, background):
    """TreeSHAP through the `shap` package. background=None => tree_path_dependent.

    This is the route v2/v3 take in 00_SHAP.ipynb, so using it here too makes v1's sidecar meta
    say backend="shap" like theirs -- concentration.require_comparable compares that field and
    refuses a run whose versions disagree. The booster route below is the fallback, not the
    equal alternative.
    """
    import shap
    if background is None:
        expl = shap.TreeExplainer(est)
    else:
        try:
            expl = shap.TreeExplainer(est, background, feature_perturbation="interventional")
        except TypeError:                      # shap <= 0.29 spelled it differently
            expl = shap.TreeExplainer(est, background, feature_dependence="independent")

    values = expl.shap_values(X)
    if isinstance(values, list):               # one array per class on some versions
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    base = expl.expected_value
    base = float(np.asarray(base).ravel()[-1]) if np.ndim(base) else float(base)
    return Attribution(values, np.full(len(X), base), X, "shap",
                       "tree_path_dependent" if background is None else "interventional",
                       note="shap {}".format(shap.__version__))


def _via_booster(est, X):
    """xgboost's own pred_contribs. Exact TreeSHAP, but tree_path_dependent ONLY -- there is no
    background to intervene with, which is why an interventional request cannot come here."""
    import xgboost as xgb
    booster = est.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = np.asarray(booster.predict(dmatrix, pred_contribs=True))
    if contribs.ndim == 3:
        contribs = contribs[:, -1, :]
    return Attribution(contribs[:, :-1], contribs[:, -1], X, "native", "tree_path_dependent",
                       note="xgboost {}".format(xgb.__version__))


def compute(est, X, background=None, backend="auto"):
    """Per-row TreeSHAP for `X`.

    background : a reference sample => INTERVENTIONAL. None => tree_path_dependent, each tree's
                 own cover as the reference.
    backend    : "auto" prefers `shap` and falls back to the booster; "shap" and "native" force
                 one. An interventional request can never be served by "native" -- it raises
                 instead of quietly returning path-dependent numbers under an interventional
                 label, which is the one failure that survives every downstream check.
    """
    X = align(X, est)
    if backend not in ("auto", "shap", "native"):
        raise ValueError("backend must be auto|shap|native, not {!r}".format(backend))

    if backend == "native" or (backend == "auto" and has_shap() is None):
        if background is not None:
            raise RuntimeError(
                "interventional attribution needs the `shap` package, which the booster route "
                "cannot substitute for. Install it in THIS env (see src/envs/v1/requirements.txt "
                "-- shap==0.35.0 is the last cp35 wheel) and verify with "
                "src/envs/v1/check_shap.py, or ask for background=None.")
        return _via_booster(est, X)
    return _via_shap(est, X, background)


def model_margin(est, X):
    """The model's RAW margin for X -- the quantity `sum(phi) + base` must reproduce.

    NOT `est.predict(X, output_margin=True)`. xgboost's sklearn wrapper does not honour
    output_margin before 0.81 (fixed upstream there; shap refuses model_output != "raw" on
    < 0.81 for exactly this reason), and env-v1 is pinned at 0.72 -- it returns PROBABILITIES
    instead, so every additivity check fails by the logit-vs-probability difference, a gap of
    order 5-10 that is identical for every backend because the reference, not the attribution,
    is wrong.

    Ask the booster directly; fall back to logit(predict_proba), which is exact for
    binary:logistic. `ntree_limit` follows the estimator so early stopping cannot make the
    predicted margin cover a different tree set than the attribution summed.
    """
    try:
        import xgboost as xgb
        booster = est.get_booster()
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        limit = getattr(est, "best_ntree_limit", 0) or 0
        margin = booster.predict(dmatrix, output_margin=True, ntree_limit=limit)
        return np.asarray(margin).ravel()
    except Exception:
        proba = est.predict_proba(X)
        proba = np.asarray(proba)
        proba = proba[:, -1] if proba.ndim > 1 else proba
        return np.log(np.clip(proba, 1e-12, 1 - 1e-12) / np.clip(1 - proba, 1e-12, 1))


def check_additivity(att, est, tol=1e-3):
    margin = model_margin(est, att.X)
    gap = float(np.max(np.abs(att.margin - np.asarray(margin).ravel())))
    verdict = "OK" if gap < tol else "*** MISMATCH ***"
    print("  additivity: max |sum(phi) + base - margin| = {:.2e}  {}".format(gap, verdict))
    return gap


def _colour(values):
    v = pd.to_numeric(pd.Series(values), errors="coerce").values.astype(float)
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return v, 0.0, 1.0, finite
    lo, hi = np.nanpercentile(v[finite], [5, 95])
    if hi <= lo:
        lo, hi = float(np.nanmin(v[finite])), float(np.nanmax(v[finite]) + 1e-12)
    return v, float(lo), float(hi), finite


def _swarm_offset(values, max_offset=0.38, nbins=80):
    v = np.asarray(values, dtype=float)
    off = np.zeros(len(v))
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return off
    lo, hi = np.min(v[finite]), np.max(v[finite])
    if hi <= lo:
        return off
    bins = np.clip(((v - lo) / (hi - lo) * nbins).astype(int), 0, nbins - 1)
    for b in np.unique(bins[finite]):
        idx = np.where((bins == b) & finite)[0]
        k = len(idx)
        step = min(2.0 * max_offset / max(k, 1), 0.055)
        off[idx] = (np.arange(k) - (k - 1) / 2.0) * step
    return np.clip(off, -max_offset, max_offset)


def _finish(fig, title=None):
    if title:
        fig.suptitle(title)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


def plot_bar(att, top_n=20, title=None, ax=None):
    m = att.mean_abs.head(top_n)[::-1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(FIG_W, 0.28 * len(m) + 1.3))
    else:
        fig = ax.figure
    ax.barh(np.arange(len(m)), m.values, color=BLUE, height=0.72)
    ax.set_yticks(np.arange(len(m)))
    ax.set_yticklabels(m.index, fontsize=8)
    ax.set_xlabel("mean |SHAP|  (log-odds)")
    ax.set_ylim(-0.7, len(m) - 0.3)
    return _finish(fig, title or "Global importance -- top {} of {}".format(
        len(m), len(att.features)))


def plot_beeswarm(att, top_n=20, title=None, max_points=4000):
    feats = att.top(top_n)[::-1]
    n = att.phi.shape[0]
    take = np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).astype(int)

    fig, ax = plt.subplots(figsize=(FIG_W + 1.2, 0.32 * len(feats) + 1.4))
    scatter = None
    for i, feat in enumerate(feats):
        j = att.features.index(feat)
        phi = att.phi[take, j]
        colour, lo, hi, finite = _colour(att.X[feat].values[take])
        y = i + _swarm_offset(phi)
        if (~finite).any():
            ax.scatter(phi[~finite], y[~finite], s=7, color=GREY, alpha=0.6, linewidths=0)
        scatter = ax.scatter(phi[finite], y[finite], c=colour[finite], cmap=CMAP,
                             vmin=lo, vmax=hi, s=7, alpha=0.85, linewidths=0)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("SHAP value  (log-odds contribution)")
    ax.set_ylim(-0.7, len(feats) - 0.3)
    ax.grid(False, axis="y")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.01, aspect=30)
        cbar.set_label("feature value  (low to high)", fontsize=9)
        cbar.set_ticks([])
    return _finish(fig, title or "SHAP beeswarm -- top {} features".format(len(feats)))


def plot_beeswarm_abs(att, top_n=20, title=None, max_points=4000):
    m = att.mean_abs.head(top_n)
    feats = list(m.index)[::-1]
    n = att.phi.shape[0]
    take = np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).astype(int)
    lo, hi = float(m.min()), float(m.max() + 1e-12)

    fig, ax = plt.subplots(figsize=(FIG_W + 1.2, 0.32 * len(feats) + 1.4))
    scatter = None
    for i, feat in enumerate(feats):
        j = att.features.index(feat)
        a = np.abs(att.phi[take, j])
        ax.barh(i, m[feat], color="#eeeeea", height=0.66, zorder=0)
        scatter = ax.scatter(a, i + _swarm_offset(a), c=np.full(len(a), m[feat]), cmap=CMAP_SEQ,
                             vmin=lo, vmax=hi, s=7, alpha=0.85, linewidths=0, zorder=2)
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("|SHAP|   (bar = mean |SHAP|)")
    ax.set_ylim(-0.7, len(feats) - 0.3)
    ax.grid(False, axis="y")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.01, aspect=30)
        cbar.set_label("mean |SHAP| of the feature", fontsize=9)
    return _finish(fig, title or "Magnitude beeswarm -- importance spread per feature")


def plot_dependence(att, feature, interaction=None, threshold_x=None,
                    title=None, ax=None, max_points=6000):
    j = att.features.index(feature)
    phi = att.phi[:, j]
    x = pd.to_numeric(att.X[feature], errors="coerce").values.astype(float)
    n = len(x)
    take = np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).astype(int)

    if interaction == "auto":
        interaction = _pick_interaction(att, j)
    if ax is None:
        fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
    else:
        fig = ax.figure

    if interaction:
        colour, lo, hi, finite = _colour(att.X[interaction].values)
        sc = ax.scatter(x[take], phi[take], c=colour[take], cmap=CMAP, vmin=lo, vmax=hi,
                        s=9, alpha=0.8, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, pad=0.01)
        cbar.set_label(interaction, fontsize=8)
    else:
        ax.scatter(x[take], phi[take], color=BLUE, s=9, alpha=0.6, linewidths=0)

    ax.axhline(0, color=AXIS, lw=1)
    if threshold_x is not None:
        ax.axvline(threshold_x, color=RED, lw=1.2, ls="--")
        ax.text(threshold_x, ax.get_ylim()[1], " tau", color=RED, va="top", fontsize=9)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value")
    return _finish(fig, title or "Dependence -- {}".format(feature))


def _pick_interaction(att, j):
    phi = att.phi[:, j]
    best, best_r = None, 0.15
    numeric = att.X.select_dtypes(include=[np.number])
    for col in numeric.columns:
        if col == att.features[j]:
            continue
        v = numeric[col].values.astype(float)
        ok = np.isfinite(v) & np.isfinite(phi)
        if ok.sum() < 50 or np.nanstd(v[ok]) == 0 or np.nanstd(phi[ok]) == 0:
            continue
        r = abs(float(np.corrcoef(v[ok], phi[ok])[0, 1]))
        if np.isfinite(r) and r > best_r:
            best, best_r = col, r
    return best


def feature_association(X, method="spearman", features=None, max_features=80,
                        sample=20000, seed=0):
    cols = list(features) if features is not None else list(X.columns)
    if len(cols) > max_features:
        print("  {} features -> keeping the {} with the highest variance".format(
            len(cols), max_features))
        var = X[cols].apply(lambda s: pd.to_numeric(s, errors="coerce").var())
        cols = list(var.sort_values(ascending=False).head(max_features).index)

    sub = X[cols]
    if len(sub) > sample:
        sub = sub.sample(n=sample, random_state=seed)
    sub = sub.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    if method in ("pearson", "spearman"):
        return sub.corr(method=method)
    raise ValueError("unknown method {!r}; expected pearson or spearman".format(method))


def top_associated_pairs(matrix, k=15, absolute=True):
    values = np.abs(matrix.values) if absolute else matrix.values
    iu = np.triu_indices_from(values, k=1)
    order = np.argsort(-values[iu])[:k]
    return pd.DataFrame([OrderedDict([("feature_a", matrix.index[iu[0][p]]),
                                      ("feature_b", matrix.columns[iu[1][p]]),
                                      ("association",
                                       float(matrix.values[iu[0][p], iu[1][p]]))])
                         for p in order])


def plot_waterfall(att, row, top_n=12, label=None, ax=None):
    phi = att.phi[row]
    base = float(att.base[row])
    order = np.argsort(-np.abs(phi))
    keep, rest = order[:top_n], order[top_n:]

    names = ["{} = {}".format(att.features[j], _fmt(att.X.iloc[row, j])) for j in keep]
    values = [phi[j] for j in keep]
    if len(rest):
        names.append("{} other features".format(len(rest)))
        values.append(float(phi[rest].sum()))

    names, values = names[::-1], values[::-1]
    starts = base + np.concatenate([[0.0], np.cumsum(values)[:-1]])

    if ax is None:
        fig, ax = plt.subplots(figsize=(FIG_W + 0.8, 0.34 * len(values) + 1.6))
    else:
        fig = ax.figure
    colours = [RED if v > 0 else BLUE for v in values]
    ax.barh(np.arange(len(values)), values, left=starts, color=colours, height=0.66)

    ends = starts + np.asarray(values)
    lo = float(min(starts.min(), ends.min(), base))
    hi = float(max(starts.max(), ends.max(), base))
    pad = max((hi - lo) * 0.18, 1e-6)
    ax.set_xlim(lo - pad, hi + pad)
    for i, (s0, v) in enumerate(zip(starts, values)):
        if abs(v) < (hi - lo) * 0.02:
            continue
        ax.text(s0 + v + (0.012 if v >= 0 else -0.012) * (hi - lo), i,
                "{:+.3f}".format(v), va="center", fontsize=7,
                ha="left" if v >= 0 else "right", color=INK)
    ax.axvline(base, color=GREY, lw=1, ls=":")
    ax.axvline(base + float(np.sum(phi)), color=INK, lw=1.2)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("log-odds       base E[f(x)] = {:.3f}   to   f(x) = {:.3f}".format(
        base, base + phi.sum()))
    ax.grid(False, axis="y")
    return _finish(fig, label or "Waterfall -- row {}".format(row))


def plot_force(att, row, top_n=10, label=None, ax=None):
    phi = att.phi[row]
    base, out = float(att.base[row]), float(att.base[row] + phi.sum())
    order = np.argsort(-np.abs(phi))[:top_n]
    pos = [(att.features[j], phi[j]) for j in order if phi[j] > 0]
    neg = [(att.features[j], phi[j]) for j in order if phi[j] < 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(FIG_W + 1.0, 1.9))
    else:
        fig = ax.figure

    span = max(abs(out - base), 1e-9)
    for group, colour, sign in ((sorted(neg, key=lambda kv: kv[1]), BLUE, 1),
                                (sorted(pos, key=lambda kv: -kv[1]), RED, -1)):
        cursor, depth = base, 0
        for name, v in group:
            ax.barh(0, v, left=cursor, color=colour, height=0.45,
                    edgecolor="white", linewidth=0.7)
            if abs(v) > 0.06 * span:
                depth += 1
                ax.text(cursor + v / 2.0, sign * (0.34 + 0.26 * (depth % 2)), name,
                        ha="center", va="bottom" if sign > 0 else "top",
                        fontsize=6.5, color=INK)
            cursor += v

    ax.axvline(base, color=GREY, lw=1, ls=":")
    ax.axvline(out, color=INK, lw=1.4)
    ax.text(out, 1.02, " f(x)={:.3f}".format(out), fontsize=8, color=INK, va="top")
    ax.text(base, 1.02, "base={:.3f} ".format(base), fontsize=8, color=GREY,
            ha="right", va="top")
    ax.set_yticks([])
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("log-odds")
    ax.grid(False)
    return _finish(fig, label or "Force -- row {}".format(row))


def _fmt(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "NaN"
    if float(f).is_integer() and abs(f) < 1e6:
        return "{:.0f}".format(f)
    return "{:.3g}".format(f)


def plot_distributions(X, features, scores=None, tau=None, above=None,
                       bins=40, ncols=3, title=None):
    features = list(features)
    if above is not None:
        above = np.asarray(above, dtype=bool)
    else:
        above = None if (scores is None or tau is None) else np.asarray(scores) > tau
    nrows = int(np.ceil(len(features) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, features):
        if not pd.api.types.is_numeric_dtype(X[feat]):
            top = X[feat].astype(str).value_counts().head(6)
            pos = np.arange(len(top))
            if above is None:
                ax.bar(pos, top.values, color=BLUE, width=0.6)
            else:
                sub = X[feat].astype(str)
                ax.bar(pos - 0.18, [int((sub[~above] == c).sum()) for c in top.index],
                       color=GREY, width=0.34, label="below tau")
                ax.bar(pos + 0.18, [int((sub[above] == c).sum()) for c in top.index],
                       color=RED, width=0.34, label="above tau")
            ax.set_xticks(pos)
            ax.set_xticklabels(top.index, fontsize=6, rotation=30, ha="right")
            ax.set_ylabel("claims", fontsize=8)
            ax.set_title("{}  (categorical, top {})".format(feat, len(top)), fontsize=9)
            continue

        s = pd.to_numeric(X[feat], errors="coerce")
        if s.nunique(dropna=True) <= 2:
            if above is None:
                ax.bar([0], [float(s.mean())], color=BLUE, width=0.5)
                ax.set_xticks([0])
                ax.set_xticklabels(["all"])
            else:
                ax.bar([0, 1], [float(s[~above].mean()), float(s[above].mean())],
                       color=[GREY, RED], width=0.55)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["below tau", "above tau"], fontsize=8)
            ax.set_ylabel("rate", fontsize=8)
        else:
            v = s.values.astype(float)
            finite = np.isfinite(v)
            edges = np.histogram(v[finite], bins=bins)[1] if finite.any() else bins
            if above is None:
                ax.hist(v[finite], bins=edges, color=BLUE, alpha=0.9)
            else:
                ax.hist(v[finite & ~above], bins=edges, color=GREY, alpha=0.75,
                        label="below tau")
                ax.hist(v[finite & above], bins=edges, color=RED, alpha=0.75,
                        label="above tau")
            ax.set_ylabel("claims", fontsize=8)
        miss = float(X[feat].isnull().mean() * 100)
        ax.set_title(feat + ("   ({:.0f}% missing)".format(miss) if miss > 0.5 else ""),
                     fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes[len(features):]:
        ax.axis("off")
    if above is not None and len(features):
        for ax in axes[:len(features)]:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(fontsize=7, loc="upper right")
                break
    return _finish(fig, title or "Model input distributions" +
                   ("" if above is None else "  -  below vs above the fast-track cutoff"))


def score_bands(scores, tau, quantiles=(0.5, 0.9, 0.99), above=None):
    s = pd.Series(np.asarray(scores, dtype=float))
    labels = pd.Series(["below tau"] * len(s), index=s.index)
    if above is not None:
        above = pd.Series(np.asarray(above, dtype=bool), index=s.index)
        lower = float(s[above].min()) if above.any() else float(tau)
    else:
        above = s > tau
        lower = float(tau)
    if above.sum() == 0:
        return pd.Series(pd.Categorical(labels, categories=["below tau"], ordered=True))

    edges = [lower] + [float(s[above].quantile(q)) for q in quantiles]
    points = ["tau"] + ["p{}".format(int(round(q * 100))) for q in quantiles] + ["max"]
    names = ["{}-{}".format(points[i], points[i + 1]) for i in range(len(points) - 1)]
    for i, name in enumerate(names):
        upper = edges[i + 1] if i + 1 < len(edges) else None
        sel = above & (s >= edges[i]) & ((s < upper) if upper is not None else True)
        labels[sel] = name
    order = ["below tau"] + names
    return pd.Series(pd.Categorical(labels, categories=order, ordered=True), index=s.index)


def plot_band_bars(att, bands, top_n=12, title=None):
    if hasattr(bands, "cat"):
        order = list(bands.cat.categories)
    else:
        order = list(pd.unique(bands))
    bands = pd.Series(np.asarray(bands, dtype=object), index=range(len(bands)))
    feats = att.top(top_n)[::-1]
    order = [b for b in order if (bands == b).sum() > 0]

    table = OrderedDict()
    for band in order:
        mask = (bands == band).values
        table["{}  (n={})".format(band, int(mask.sum()))] = pd.Series(
            np.abs(att.phi[mask]).mean(axis=0), index=att.features).reindex(feats)
    table = pd.DataFrame(table)

    fig, ax = plt.subplots(figsize=(FIG_W + 1.4, 0.36 * len(feats) + 1.5))
    y = np.arange(len(feats))
    h = 0.8 / max(len(table.columns), 1)
    for i, col in enumerate(table.columns):
        ax.barh(y + (i - (len(table.columns) - 1) / 2.0) * h, table[col].values, height=h,
                color=SERIES[i % len(SERIES)], label=col)
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("mean |SHAP| within the band")
    ax.set_ylim(-0.7, len(feats) - 0.3)
    ax.grid(False, axis="y")
    ax.legend(fontsize=7, loc="lower right")
    return _finish(fig, title or "Attribution mass by score band"), table
