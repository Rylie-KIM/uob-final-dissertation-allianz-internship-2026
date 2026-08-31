"""SHAP toolkit for ONE model version — the same notebook runs inside env-v1, env-v2 and env-v3.

WHAT THIS IS FOR
----------------
`notebook/real/00_SHAP.ipynb` opens a version's pickle directly, so it must run in that version's
own environment. Those environments are far apart — **xgboost 0.72 (v1) / 1.4.2 (v2) / 3.2.0
(v3)** — and their matplotlib, numpy and shap releases differ with them. Anything the notebook
relies on therefore has to work across all three, which rules out two obvious shortcuts:

* **shap's own plotting layer.** `shap.plots.beeswarm` / `shap.plots.waterfall` and the
  `Explanation` object only exist in newer shap; the 2018-era stack that pairs with xgboost 0.72
  has `shap.summary_plot` and little else. Every plot here is therefore drawn from the raw phi
  matrix with plain matplotlib — same figure, same house style, in all three envs.
* **shap being installed at all.** `compute()` prefers `shap.TreeExplainer`, but falls back to the
  booster's own `pred_contribs=True`, which has existed since xgboost 0.6 and needs no extra
  package. The fallback is exact TreeSHAP; what it changes is the *reference* — see `Attribution`.

Only numpy, pandas and matplotlib are required. `figstyle` is used if it imports, and its rcParams
are applied one key at a time so an old matplotlib that has never heard of `axes.titlecolor` skips
that key instead of failing the whole notebook.

    import shap_kit as sk

    env = sk.env_report()                    # which version am I? does xgboost match config?
    est = sk.load_estimator(model_path)
    att = sk.compute(est, X, background=X.sample(500))

    att.mean_abs.head(15)                    # global ranking
    sk.plot_bar(att)                          sk.plot_beeswarm(att)
    sk.plot_dependence(att, "repair_ratio")   sk.plot_waterfall(att, row=0)

Every plot function returns a matplotlib Figure — save it with `figstyle.save(fig, name)`.
"""
from __future__ import annotations

import contextlib
import os
import sys
import warnings

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:                  # a headless caller (scoring/attribute.py) imports this
    plt = None                       # module for feature_order() alone and never plots. Only
    LinearSegmentedColormap = None   # the plotting half needs matplotlib; style() says so.

# ── house style, applied defensively ──────────────────────────────────────────────────
try:
    import figstyle                                  # noqa: F401  (same dir as this file)
    _HAS_FIGSTYLE = True
except Exception:                                    # pragma: no cover - a frozen env may lack it
    figstyle = None
    _HAS_FIGSTYLE = False

BLUE, RED, GREY = "#2a78d6", "#e34948", "#898781"
INK, AXIS = "#0b0b0b", "#c3c2b7"
if _HAS_FIGSTYLE:
    BLUE, RED, GREY = figstyle.BLUE, figstyle.RED, figstyle.NEUTRAL
    INK, AXIS = figstyle.INK, figstyle.AXIS

#: low value -> blue, high value -> red. The convention every SHAP reader already knows.
CMAP = LinearSegmentedColormap.from_list("sfp_div", [BLUE, "#dcdcd6", RED])
CMAP_SEQ = LinearSegmentedColormap.from_list("sfp_seq", ["#dcdcd6", BLUE])

FIG_W = 6.3          # dissertation text column


def style() -> None:
    """Install the house style, skipping rcParams this matplotlib does not know.

    `mpl.rcParams.update(dict)` is all-or-nothing: one unknown key (``axes.titlecolor`` arrived in
    matplotlib 3.4) raises and leaves the session unstyled. Old envs are exactly where that bites.
    """
    if not _HAS_FIGSTYLE:
        return
    import matplotlib as mpl

    before = dict(mpl.rcParams)
    try:
        figstyle.apply()
        return
    except Exception:
        mpl.rcParams.update(before)

    from cycler import cycler
    wanted = {"font.size": 10.0, "axes.titlesize": 12.0, "axes.labelsize": 10.5,
              "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e1e0d9",
              "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
              "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 150,
              "savefig.bbox": "tight", "axes.prop_cycle": cycler(color=figstyle.SERIES)}
    skipped = []
    for key, value in wanted.items():
        try:
            mpl.rcParams[key] = value
        except Exception:
            skipped.append(key)
    if skipped:
        print(f"  (matplotlib {mpl.__version__} does not know {skipped} — skipped)")


# ======================================================================================
# 1. Which version am I running as?
# ======================================================================================


def detect_version(default: str | None = None) -> str | None:
    """Infer v1/v2/v3 from the running interpreter's path (``src/envs/v2/.venv/bin/python``).

    Path-based rather than a variable the notebook sets, because the failure it prevents is
    running the v2 notebook on the v3 kernel — which produces a complete set of plausible figures
    for the wrong model. Falls back to $SFP_VERSION, then `default`.
    """
    exe = os.path.normpath(sys.executable).replace("\\", "/")
    for label in ("v1", "v2", "v3"):
        if f"/envs/{label}/" in exe:
            return label
    return os.environ.get("SFP_VERSION", default)


def env_report(version: str | None = None, strict: bool = True) -> dict:
    """Print, and return, what is actually running — and check it against config.

    The xgboost release is not cosmetic: a pickle is bound to the release it was serialised with,
    and the three versions are 0.72 / 1.4.2 / 3.2.0. A mismatch means either the wrong kernel or
    the wrong pickle, and both produce figures rather than errors — so with ``strict`` this raises.
    """
    version = version or detect_version()
    info = {"version": version, "python": sys.version.split()[0], "executable": sys.executable}

    try:
        import xgboost
        info["xgboost"] = xgboost.__version__
    except Exception as exc:
        info["xgboost"] = f"NOT IMPORTABLE ({exc})"
    for name in ("numpy", "pandas", "matplotlib", "shap", "sklearn", "joblib"):
        try:
            info[name] = __import__(name).__version__
        except Exception:
            info[name] = "—"

    expected = None
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import config
        expected = config.xgboost_pin(version) if version else None
    except Exception:
        pass
    info["xgboost_expected"] = expected or "(not declared in config)"

    width = max(len(k) for k in info)
    print("environment")
    for k, v in info.items():
        print(f"  {k:<{width}}  {v}")

    if version is None:
        raise RuntimeError(
            "cannot tell which version this kernel is. Run the notebook with a kernel built on "
            "src/envs/v<k>/.venv, or set VERSION explicitly in the first cell."
        )
    if expected and info["xgboost"] != expected:
        message = (f"this kernel has xgboost {info['xgboost']}, but config declares {version} was "
                   f"serialised with {expected}. Either the kernel or the pickle is the wrong one — "
                   f"and both would still produce figures.")
        if strict:
            raise RuntimeError(message)
        warnings.warn(message)
    return info


def training_config(version: str) -> pd.Series:
    """That version's training call from config — print it beside every concentration figure."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    return pd.Series(config.training_config(version), name=version)


# ======================================================================================
# 2. Model + features
# ======================================================================================


def load_estimator(path):
    """Unpickle, and hand back the thing that owns the trees.

    Accepts either a bare estimator or a Pipeline. For a Pipeline the last step is returned along
    with a warning, because SHAP then explains the estimator's view of an ALREADY-preprocessed
    matrix — feed it the transformed X, not raw claims.
    """
    import joblib

    obj = joblib.load(path)
    if hasattr(obj, "steps"):
        print(f"  note: {os.path.basename(str(path))} is a Pipeline "
              f"({[n for n, _ in obj.steps]}). Explaining the final step; X must be the "
              f"POST-preprocessing matrix.")
        return obj.steps[-1][1]
    return obj


def model_feature_names(est) -> list:
    """The column names the booster was trained on, in trained order — or [] if unrecoverable."""
    for getter in (lambda: list(est.get_booster().feature_names),
                   lambda: list(est.feature_name_),
                   lambda: [str(c) for c in est.feature_names_in_]):
        try:
            names = getter()
            if names:
                return [str(n) for n in names]
        except Exception:
            continue
    return []


UNVERIFIED_ORDER = "unverified (the estimator exposes no trained feature names)"


def feature_order(est, columns, trained=None, trained_source=None) -> str:
    """Status of `columns` against the booster's trained order — the meta's `feature_order` field.

    ONE VOCABULARY, shared by every producer of an attributions file (this module, shap_kit_v1.py,
    and scoring/attribute.py), so the field means the same thing in every `_meta.json`:

        "exact"         the trained names, in the trained order
        "reordered"     same set, different order
        "set_mismatch"  different columns altogether
        "unverified …"  the estimator exposes no trained names, so nothing can be checked

    Producers differ in what they DO about it — align() below reorders, attribute.py refuses —
    but they all record the same word. Recording it is the point: without the field a reader
    cannot tell a checked file from an unchecked one, and the two look identical. SHAP is
    positional underneath, so a wrong order attributes every value to the wrong feature.
    """
    source = ""
    if trained is None:
        trained = model_feature_names(est)
    else:
        # The caller resolved the order from somewhere other than the pickle — v1 falls back to
        # features/registry/v1.json when xgboost 0.72 exposes no names. Recording WHICH source
        # answered matters: "unverified" would be wrong (the order WAS checked) and a bare
        # "exact" would hide that the pickle itself never confirmed it.
        trained = [str(c) for c in trained]
        source = f" (via registry: {trained_source or 'unrecorded'})"
    cols = [str(c) for c in columns]
    if not trained:
        return UNVERIFIED_ORDER
    if trained == cols:
        return "exact" + source
    return ("reordered" if sorted(trained) == sorted(cols) else "set_mismatch") + source


def align(X: pd.DataFrame, est) -> pd.DataFrame:
    """Put X into the booster's trained column order, or say exactly what does not match."""
    status = feature_order(est, X.columns)
    if status == UNVERIFIED_ORDER:
        print("  the estimator exposes no feature names — column order is UNVERIFIED. "
              "Check it by hand before trusting any per-feature claim.")
        return X
    trained = model_feature_names(est)
    if status == "set_mismatch":
        have, want = set(X.columns), set(trained)
        raise ValueError(
            "X does not match the model's features.\n"
            f"  in model, not in X : {sorted(want - have)[:12]}\n"
            f"  in X, not in model : {sorted(have - want)[:12]}"
        )
    if status == "reordered":
        print("  reordered X into the booster's trained column order")
    return X[trained]


def describe_features(X: pd.DataFrame) -> pd.DataFrame:
    """Per-column profile: type, cardinality, missingness, spread, and its one-hot family.

    The family is the prefix before the last underscore — a heuristic for grouping `make_FORD`,
    `make_BMW`, … It is for READING the table only. Never aggregate SHAP by it: the encoded feature names
    are the measurement; cross-version correspondence lives ONLY in the hand-confirmed mapping
    (features/check_overlap.py -> features/feature_overlap.json).
    """
    rows = {}
    for col in X.columns:
        s = X[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        values = pd.to_numeric(s, errors="coerce") if numeric else s
        uniq = int(s.nunique(dropna=True))
        rows[col] = {
            "dtype": str(s.dtype),
            "n_unique": uniq,
            "kind": ("constant" if uniq <= 1 else
                     "categorical" if not numeric else
                     "binary" if uniq == 2 else "numeric"),
            "pct_missing": float(s.isna().mean() * 100),
            "pct_zero": float((values == 0).mean() * 100) if numeric else np.nan,
            "mean": float(values.mean()) if numeric else np.nan,
            "std": float(values.std()) if numeric else np.nan,
            "min": float(values.min()) if numeric else np.nan,
            "max": float(values.max()) if numeric else np.nan,
            "family": col.rsplit("_", 1)[0] if "_" in col else col,
        }
    return pd.DataFrame(rows).T


def feature_summary(X: pd.DataFrame) -> pd.Series:
    """The headline counts for "what does this repo's feature space look like?"."""
    d = describe_features(X)
    fams = d[d["kind"] == "binary"]["family"].value_counts()
    return pd.Series({
        "n_features": len(X.columns),
        "n_rows": len(X),
        "n_numeric": int((d["kind"] == "numeric").sum()),
        "n_binary": int((d["kind"] == "binary").sum()),
        "n_categorical": int((d["kind"] == "categorical").sum()),
        "n_constant": int((d["kind"] == "constant").sum()),
        "n_onehot_families": int((fams > 1).sum()),
        "largest_onehot_family": f"{fams.index[0]} ({fams.iloc[0]})" if len(fams) else "—",
        "pct_cells_missing": round(float(X.isna().mean().mean() * 100), 3),
    })


# ======================================================================================
# 3. Attributions
# ======================================================================================


class Attribution(object):
    """phi for one model on one matrix, plus the record of HOW it was produced.

    `backend`/`perturbation` are carried around rather than forgotten because they change what the
    numbers mean: ``interventional`` measures against a supplied background sample, so it answers
    "versus these reference claims"; ``tree_path_dependent`` measures against each tree's own cover
    statistics, i.e. versus that model's training distribution. Both are exact TreeSHAP. They are
    not the same quantity, and neither is comparable to the other across models.
    """

    def __init__(self, phi, base, X, backend, perturbation, note=""):
        self.phi = np.asarray(phi, dtype=float)
        self.base = np.asarray(base, dtype=float).ravel()
        self.X = X
        self.backend = backend
        self.perturbation = perturbation
        self.note = note
        self.features = list(X.columns)

    def __repr__(self):
        return (f"<Attribution {self.phi.shape[0]} rows x {self.phi.shape[1]} features, "
                f"{self.backend}/{self.perturbation}>")

    @property
    def mean_abs(self) -> pd.Series:
        """mean|phi| per feature, descending — the global ranking."""
        return pd.Series(np.abs(self.phi).mean(axis=0),
                         index=self.features).sort_values(ascending=False)

    @property
    def margin(self) -> np.ndarray:
        """The reconstructed raw (log-odds) output: base + sum of phi. Additivity is the check."""
        return self.base + self.phi.sum(axis=1)

    def top(self, n: int = 15) -> list:
        return list(self.mean_abs.head(n).index)

    def frame(self, id_values=None, id_col: str = "claim_id") -> pd.DataFrame:
        """phi as a tidy DataFrame — what gets written to detection/shap/<v>/<v>_attributions.parquet."""
        out = pd.DataFrame(self.phi, columns=self.features, index=self.X.index)
        if id_values is not None:
            out.insert(0, id_col, np.asarray(id_values))
        out["_base_value"] = self.base
        return out

    def subset(self, mask) -> "Attribution":
        """The same attribution restricted to a row mask — e.g. only fast-tracked claims."""
        mask = np.asarray(mask)
        return Attribution(self.phi[mask], self.base[mask], self.X.loc[mask],
                           self.backend, self.perturbation, self.note)


def compute(est, X: pd.DataFrame, background=None, backend: str = "auto") -> Attribution:
    """TreeSHAP for `est` on `X`. Prefers shap+background, falls back to the booster itself.

    `background` (a sample of rows) switches shap to the INTERVENTIONAL reference. Pass one when
    the question is "versus this population"; leave it None to stay tree-path-dependent, which is
    also what the native fallback always is.
    """
    X = align(X, est)

    # auto refers to trying shap >> then, navtive fall back 
    if backend in ("auto", "shap"):
        try:
            return _via_shap(est, X, background)
        except Exception as exc:
            if backend == "shap":
                raise
            print(f"  shap unavailable or unhappy here ({type(exc).__name__}: {exc}) "
                  f"-> using the booster's own TreeSHAP")
    return _via_booster(est, X)


def _via_shap(est, X, background):
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
    return Attribution(values, np.full(len(X), base), X, "shap",
                       "interventional" if background is not None else "tree_path_dependent",
                       note="shap %s" % shap.__version__)


def _via_booster(est, X):
    if hasattr(est, "get_booster"):
        import xgboost as xgb

        booster = est.get_booster()
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        contribs = np.asarray(booster.predict(dmatrix, pred_contribs=True))
        if contribs.ndim == 3:
            contribs = contribs[:, -1, :]
        return Attribution(contribs[:, :-1], contribs[:, -1], X, "native",
                           "tree_path_dependent", note="xgboost %s" % xgb.__version__)
    if hasattr(est, "booster_"):                       # lightgbm
        contribs = np.asarray(est.predict(X, pred_contrib=True))
        return Attribution(contribs[:, :-1], contribs[:, -1], X, "native",
                           "tree_path_dependent", note="lightgbm")
    raise RuntimeError(f"no TreeSHAP route for {type(est).__name__}; install shap in this env")


def check_additivity(att: Attribution, est, tol: float = 1e-3) -> float:
    """phi must sum to the model's own raw margin. The one check that catches a wrong X.

    Returns the max absolute gap. A gap of the order of the margin itself (not 1e-6) means the
    matrix, the column order or the estimator is not what the attribution assumed.
    """
    try:
        margin = est.predict(att.X, output_margin=True)
    except TypeError:                                  # old sklearn wrapper without the kwarg
        proba = est.predict_proba(att.X)[:, 1]
        margin = np.log(np.clip(proba, 1e-12, 1 - 1e-12) / np.clip(1 - proba, 1e-12, 1))
    gap = float(np.max(np.abs(att.margin - np.asarray(margin).ravel())))
    verdict = "OK" if gap < tol else "*** MISMATCH ***"
    print(f"  additivity: max |sum(phi) + base - margin| = {gap:.2e}  {verdict}")
    return gap


# ======================================================================================
# 4. Plots — all drawn from phi with plain matplotlib, so every env renders the same figure
# ======================================================================================


def _colour(values):
    """Feature values -> colour scale, robust to outliers (5th–95th percentile) and NaN."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").values.astype(float)
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return v, 0.0, 1.0, finite
    lo, hi = np.nanpercentile(v[finite], [5, 95])
    if hi <= lo:
        lo, hi = float(np.nanmin(v[finite])), float(np.nanmax(v[finite]) + 1e-12)
    return v, float(lo), float(hi), finite


def _swarm_offset(values, max_offset=0.38, nbins=80):
    """Vertical spread that makes density visible — the thing that makes a beeswarm readable."""
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


# ---- global -------------------------------------------------------------------------


def plot_bar(att: Attribution, top_n: int = 20, title=None, ax=None):
    """mean|phi| per feature — the global importance ranking, over all explained rows."""
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
    return _finish(fig, title or f"Global importance — top {len(m)} of {len(att.features)}")


def plot_beeswarm(att: Attribution, top_n: int = 20, title=None, max_points: int = 4000):
    """One point per claim per feature: x = SHAP value, colour = that claim's feature value.

    This is the plot that separates "the feature matters" from "HOW it matters": a red cloud on
    the right means high values push towards total loss; red on both sides means the effect is
    conditional on something else, which is what the dependence plots then chase.
    """
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
    ax.grid(axis="y", visible=False)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.01, aspect=30)
        cbar.set_label("feature value  (low to high)", fontsize=9)
        cbar.set_ticks([])
    return _finish(fig, title or f"SHAP beeswarm — top {len(feats)} features")


def plot_beeswarm_abs(att: Attribution, top_n: int = 20, title=None, max_points: int = 4000):
    """|SHAP| beeswarm, each feature's cloud coloured by its OWN mean|SHAP|.

    Same rows as the signed beeswarm, but folded to magnitude: it shows how a feature's importance
    is *distributed* — a feature that is huge for 2% of claims and irrelevant for the rest has the
    same mean as one that matters mildly everywhere, and only this plot tells them apart. The bar
    behind each row marks that mean, so the colour and the bar are the same quantity.
    """
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
    ax.grid(axis="y", visible=False)
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.01, aspect=30)
        cbar.set_label("mean |SHAP| of the feature", fontsize=9)
    return _finish(fig, title or "Magnitude beeswarm — importance spread per feature")


# ======================================================================================
# shap's OWN plotting layer — available only where shap is modern enough
# ======================================================================================
#
# Everything above draws from the raw phi matrix with plain matplotlib, because this module has to
# run under env-v1 as well (Python 3.5, shap 0.28 at best, xgboost 0.72) and that stack predates
# shap's modern API entirely. The helpers below are the escape hatch for the envs where that
# constraint does NOT bind: env-v2 carries shap 0.40 and env-v3 shap 0.49, both well past the 0.36
# release that introduced Explanation.
#
# They add pictures; they do not replace any. A cross-version figure must be drawn by the SAME code
# for all three versions, or part of the difference between two panels is a difference between two
# renderers (different ordering, clipping, colour normalisation). Use these to read ONE version
# closely; keep plot_bar/plot_beeswarm for anything that appears beside another version.

MODERN_SHAP_MIN = (0, 36)   # shap 0.36 (Sep 2020) introduced Explanation + shap.plots.*


def shap_capability() -> dict:
    """What shap can do in THIS env: importable at all, and modern enough for Explanation.

    Returned rather than asserted, because the answer differs per env by design and the notebook
    has to say which branch it took instead of silently taking one.
    """
    out = {"installed": False, "version": None, "explanation": False, "plots": False}
    try:
        import shap
    except Exception:
        return out
    out["installed"] = True
    out["version"] = getattr(shap, "__version__", "?")
    out["explanation"] = hasattr(shap, "Explanation")
    out["plots"] = hasattr(getattr(shap, "plots", None), "beeswarm")
    return out


def explanation(att: Attribution, feature_names=None):
    """Wrap an ALREADY COMPUTED Attribution as a `shap.Explanation`. Nothing is recomputed.

    phi, the base value and the feature matrix are exactly what an Explanation holds, and we have
    all three. So shap's plotting layer draws our numbers without a second TreeSHAP pass and
    without a second reference distribution: these plots and the hand-drawn ones above are
    guaranteed to be the same values.

    Raises where shap is too old, naming the cause, because that is a fact about the environment
    rather than something the caller can work around.
    """
    cap = shap_capability()
    if not cap["explanation"]:
        raise RuntimeError(
            "this env's shap has no Explanation object (installed=%s, version=%s). That is env-v1: "
            "Python 3.5 caps shap at the 2018 releases. Use plot_bar / plot_beeswarm instead — "
            "they draw the same phi with plain matplotlib." % (cap["installed"], cap["version"])
        )
    import shap

    return shap.Explanation(
        values=np.asarray(att.phi),
        base_values=np.asarray(att.base),
        data=np.asarray(att.X.values),
        feature_names=list(feature_names or att.features),
    )


@contextlib.contextmanager
def _shap_plot_compat():
    """Let an old shap draw on a modern matplotlib. Two independent defects in shap 0.40.

    **1. The beeswarm colour legend.** shap draws it as ``pl.colorbar(m, ticks=[0, 1],
    aspect=1000)`` (``shap/plots/_beeswarm.py:364``) with a bare ``ScalarMappable`` — no ``.axes``,
    no ``ax=``. Up to matplotlib 3.5 that silently stole space from ``gca()``; 3.6 turned it into
    ``ValueError: Unable to determine Axes to steal space for Colorbar``. Injecting
    ``ax=plt.gca()`` restores exactly the pre-3.6 behaviour, so the figure is the figure shap
    always drew — unlike ``color_bar=False``, which would compile but drop the legend that makes
    the colour axis readable. Only that one line needs it: the rest of shap's colorbar block
    (``set_ticklabels`` / ``set_label`` / ``tick_params`` / ``set_alpha`` / ``outline``) still
    exists in current matplotlib, and ``draw_all`` is already commented out upstream. A mappable
    that *does* carry an Axes is left alone, so this is a no-op for every other caller.

    **2. ``plt`` is undefined in the waterfall.** ``shap/plots/_waterfall.py`` imports pyplot as
    ``pl`` (line 4) but then calls ``plt.ioff()`` (line 43) and ``return plt.gcf()`` (line 297) —
    an upstream ``NameError``, on the ``show=False`` branch specifically, which is the branch
    ``native_plot`` needs to get the figure back. It fires on every matplotlib, so it is not a
    version-compatibility problem at all; matplotlib just happened to fail first. Binding the name
    on the module is what the author meant, since ``plt`` and ``pl`` are the same module (later
    shap releases renamed the import). Checked across all of ``shap/plots/`` in 0.40:
    ``_waterfall.py`` is the only module with the bug, but the loop is written generically because
    giving a module an unused ``plt`` attribute costs nothing.
    """
    for name, mod in list(sys.modules.items()):
        if name.startswith("shap.plots") and mod is not None and not hasattr(mod, "plt"):
            try:
                mod.plt = plt
            except Exception:                     # C extension or frozen module — nothing to fix
                pass

    original = plt.colorbar

    def with_ax(mappable=None, cax=None, ax=None, **kw):
        if cax is None and ax is None and getattr(mappable, "axes", None) is None:
            ax = plt.gca()
        return original(mappable, cax=cax, ax=ax, **kw)

    plt.colorbar = with_ax
    try:
        yield
    finally:
        plt.colorbar = original


def native_plot(kind: str, expl, show: bool = False, **kwargs):
    """Call one of shap.plots.* and hand BACK the figure, so figstyle.save can take it.

    shap draws onto the current figure and calls plt.show() itself unless told not to; every
    plotting function here takes `show=False`, and we then grab plt.gcf(). Without that the figure
    is closed before it can be saved, which is the usual reason a shap plot "does not save".

    The `_shap_plot_compat` wrapper is what lets an old shap draw on a modern stack — see its
    docstring. It belongs here rather than in the notebook because this is the seam every version's
    kernel already goes through; §6b calls this function and nothing else.
    """
    import matplotlib.pyplot as plt
    import shap

    fn = getattr(getattr(shap, "plots", None), kind, None)
    if fn is None:
        raise AttributeError(
            "shap.plots has no %r in shap %s" % (kind, shap_capability()["version"]))
    with _shap_plot_compat():
        fn(expl, show=show, **kwargs)
    return plt.gcf()


def plot_dependence(att: Attribution, feature: str, interaction=None, threshold_x=None,
                    title=None, ax=None, max_points: int = 6000, strength=None):
    """x = feature value, y = its SHAP contribution. The learned response curve, per claim.

    **Vertical dispersion at a fixed x is the interaction.** If this feature acted alone, every
    claim with the same value would receive the same contribution and the plot would be a curve.
    The cloud's thickness is what the model does with this feature *conditional on the others* —
    so colouring by the right second feature turns that thickness into a readable pattern.

    `interaction` picks that second feature:
      * a name          — colour by that feature
      * ``"auto"``      — with `strength` (a SHAP interaction matrix from `interaction_strength`)
                          the true strongest interactor; without it, a correlation proxy
      * ``None``        — no colouring

    Vertical steps in x are the tree's split points.
    """
    j = att.features.index(feature)
    phi = att.phi[:, j]
    x = pd.to_numeric(att.X[feature], errors="coerce").values.astype(float)
    n = len(x)
    take = np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).astype(int)

    if interaction == "auto":
        interaction = (pick_interaction(strength, feature) if strength is not None
                       else _pick_interaction(att, j))
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
        ax.text(threshold_x, ax.get_ylim()[1], " τ", color=RED, va="top", fontsize=9)
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value")
    return _finish(fig, title or f"Dependence — {feature}")


def _pick_interaction(att: Attribution, j: int):
    """Cheap proxy: the other feature most correlated with phi_j.

    Used only when no interaction matrix is available. It is a PROXY — a feature can be strongly
    correlated with phi_j without interacting with it at all (it may simply share information with
    feature j). `interaction_strength()` answers the question properly.
    """
    phi = att.phi[:, j]
    best, best_r = None, 0.15                    # ignore anything weaker than this
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


# ======================================================================================
# 5. Interactions — three different questions, often confused for one
# ======================================================================================
#
#   (a) SHAP interaction values   — what the MODEL does: how much of a claim's prediction comes
#                                   from feature i AND j jointly, beyond each acting alone. This
#                                   is what explains vertical dispersion in a dependence plot.
#   (b) association in X          — what the DATA does: are the two columns statistically related
#                                   at all (Pearson / Spearman / mutual information). Cheap, and
#                                   answers a question about the population, not the model.
#   (c) Friedman's H-statistic    — a third route: how much of the joint partial-dependence
#                                   surface is non-additive. Not implemented here; it needs many
#                                   model evaluations per pair and TreeSHAP gives the exact answer
#                                   for tree models anyway.
#
# (a) and (b) genuinely disagree in normal cases. Two features can be highly correlated and never
# interact (the model reads one and ignores the other); two independent features can interact
# strongly (a split on one changes what the other's split is worth). Reporting one as if it were
# the other is a real error, so they are kept as separate functions with separate names.


def interaction_values(est, X: pd.DataFrame, max_rows: int = 300, backend: str = "auto",
                       memory_limit_gb: float = 2.0, seed: int = 0):
    """Full per-row SHAP interaction matrix: phi_ij for every pair, shape (rows, p, p).

    ⚠️ **This is the expensive one.** Cost is O(rows × p²) in memory and far worse than plain
    TreeSHAP in time — with 300 one-hot columns, 300 rows already costs ~216 MB. So it is computed
    on a ROW subsample by default, and it refuses rather than swapping if the request exceeds
    `memory_limit_gb`. The ranking of interacting pairs is stable long before the values are, so a
    few hundred rows is normally enough to CHOOSE what to plot; use the full φ matrix (not this)
    for anything that gets reported as a magnitude.

    ⚠️ X must carry **every** model feature — rows are the only lever. Passing a column subset
    (`X[att.top(40)]`) cannot work: the model needs all its inputs to predict, and `align` will
    refuse it. Restrict FEATURES only downstream, when slicing the returned strength matrix.

    Note the diagonal is the **main effect** of each feature (its contribution net of interactions)
    and the off-diagonals split each pair's interaction in half: phi_ij = phi_ji.
    """
    n_all, p = X.shape
    n = min(max_rows, n_all)
    need_gb = n * p * p * 8 / 1e9
    if need_gb > memory_limit_gb:
        raise MemoryError(
            f"interaction values for {n} rows x {p} features need ~{need_gb * 1000:.0f} MB "
            f"(limit {memory_limit_gb} GB). Lower max_rows to "
            f"~{int(memory_limit_gb * 1e9 / (p * p * 8))} or raise memory_limit_gb deliberately. "
            f"(Dropping feature COLUMNS is not an option — the model needs all its inputs.)"
        )
    if n < n_all:
        idx = np.random.RandomState(seed).choice(n_all, size=n, replace=False)
        X = X.iloc[np.sort(idx)]
    X = align(X, est)
    print(f"  interaction values on {n} rows x {p} features (~{need_gb * 1000:.1f} MB)")

    if backend in ("auto", "shap"):
        try:
            import shap
            # No background here on purpose: interaction values are tree_path_dependent only.
            inter = np.asarray(shap.TreeExplainer(est).shap_interaction_values(X))
            if inter.ndim == 4:                       # (rows, p, p, classes)
                inter = inter[..., -1]
            return inter, X, "shap"
        except Exception as exc:
            if backend == "shap":
                raise
            print(f"  shap interaction route unavailable ({type(exc).__name__}: {exc}) "
                  f"-> trying the booster")

    if hasattr(est, "get_booster"):
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
        try:
            inter = np.asarray(est.get_booster().predict(dmatrix, pred_interactions=True))
        except Exception as exc:
            raise RuntimeError(
                "neither shap nor this xgboost build can produce interaction values "
                f"({exc}). `pred_interactions` arrived in xgboost 0.81, so on the v1 env (0.72) "
                "install `shap` there, or fall back to feature_association(), which measures a "
                "different thing — see this module's section 5 header."
            )
        if inter.ndim == 4:
            inter = inter[:, -1, :, :]
        return inter[:, :-1, :-1], X, "native"        # drop the bias row/column

    raise RuntimeError(f"no interaction route for {type(est).__name__}")


def interaction_strength(inter, features) -> pd.DataFrame:
    """mean|phi_ij| over rows, as a p x p table. Off-diagonals doubled = the pair's full strength.

    The diagonal is the main effect, kept so it can be compared against the interactions: a feature
    whose off-diagonal row sums rival its diagonal is doing most of its work jointly with others.
    """
    inter = np.asarray(inter)
    m = np.abs(inter).mean(axis=0)
    off = ~np.eye(m.shape[0], dtype=bool)
    m = m.copy()
    m[off] *= 2.0                                    # phi_ij and phi_ji each carry half
    return pd.DataFrame(m, index=list(features), columns=list(features))


def top_interactions(strength: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """The k strongest PAIRS (each once), with each feature's main effect alongside for scale."""
    values = strength.values
    iu = np.triu_indices_from(values, k=1)
    order = np.argsort(-values[iu])[:k]
    rows = []
    for pos in order:
        i, j = iu[0][pos], iu[1][pos]
        a, b = strength.index[i], strength.columns[j]
        rows.append({"feature_a": a, "feature_b": b,
                     "interaction": float(values[i, j]),
                     "main_effect_a": float(values[i, i]),
                     "main_effect_b": float(values[j, j])})
    return pd.DataFrame(rows)


def pick_interaction(strength: pd.DataFrame, feature: str):
    """The feature that interacts most strongly with `feature` — the right colour for a dependence
    plot. Returns None if the feature is absent or interacts with nothing."""
    if strength is None or feature not in strength.index:
        return None
    row = strength.loc[feature].drop(index=[feature], errors="ignore")
    if len(row) == 0 or float(row.max()) <= 0:
        return None
    return str(row.idxmax())


def plot_interaction_heatmap(strength: pd.DataFrame, top_n: int = 15, title=None):
    """Interaction strength among the top features, main effects blanked out.

    The diagonal is removed on purpose: it is the main effect and is typically an order of
    magnitude larger, so leaving it in would flatten every interaction to the same colour.
    """
    order = strength.values.diagonal().argsort()[::-1][:top_n]
    names = [strength.index[i] for i in order]
    block = np.array(strength.loc[names, names].values, dtype=float, copy=True)
    np.fill_diagonal(block, np.nan)

    fig, ax = plt.subplots(figsize=(0.42 * len(names) + 3.0, 0.42 * len(names) + 2.4))
    image = ax.imshow(block, cmap=CMAP_SEQ, aspect="auto")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.grid(visible=False)
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("mean |SHAP interaction|", fontsize=8)
    return _finish(fig, title or f"Pairwise interaction strength — top {len(names)} features")


def feature_association(X: pd.DataFrame, method: str = "spearman", features=None,
                        max_features: int = 80, sample: int = 20000, seed: int = 0):
    """Association between COLUMNS of X — a property of the data, not of the model.

    Use it to answer "are these two inputs related at all", never "does the model use them
    jointly". Two correlated features often produce NO interaction (the trees read one and ignore
    the other), and independent features can interact strongly.

      pearson       linear; on 0/1 one-hot columns this is the phi coefficient
      spearman      monotone; the default, since claim features are skewed and heavy-tailed
      mutual_info   any dependence, including non-monotone; needs scikit-learn and is much slower
    """
    cols = list(features) if features is not None else list(X.columns)
    if len(cols) > max_features:
        print(f"  {len(cols)} features -> keeping the {max_features} with the highest variance "
              f"(pass features=... to choose explicitly)")
        var = X[cols].apply(lambda s: pd.to_numeric(s, errors="coerce").var())
        cols = list(var.sort_values(ascending=False).head(max_features).index)

    sub = X[cols]
    if len(sub) > sample:
        sub = sub.sample(n=sample, random_state=seed)
    sub = sub.apply(lambda s: pd.to_numeric(s, errors="coerce"))

    if method in ("pearson", "spearman"):
        return sub.corr(method=method)

    if method == "mutual_info":
        from sklearn.feature_selection import mutual_info_regression

        filled = sub.fillna(sub.median())
        out = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
        for col in cols:
            mi = mutual_info_regression(filled, filled[col], random_state=seed)
            out[col] = mi
        return (out + out.T) / 2.0                   # symmetrise: MI is symmetric, the estimator is not

    raise ValueError(f"unknown method {method!r}; expected pearson, spearman or mutual_info")


def top_associated_pairs(matrix: pd.DataFrame, k: int = 15, absolute: bool = True) -> pd.DataFrame:
    """The k most strongly associated column pairs of `feature_association`'s output."""
    values = np.abs(matrix.values) if absolute else matrix.values
    iu = np.triu_indices_from(values, k=1)
    order = np.argsort(-values[iu])[:k]
    return pd.DataFrame([{"feature_a": matrix.index[iu[0][p]],
                          "feature_b": matrix.columns[iu[1][p]],
                          "association": float(matrix.values[iu[0][p], iu[1][p]])}
                         for p in order])


# ---- local --------------------------------------------------------------------------


def plot_waterfall(att: Attribution, row: int, top_n: int = 12, label=None, ax=None):
    """One claim, decomposed: from the base value to this claim's own log-odds output.

    Local interpretability in its most defensible form — the bars are exact and they sum. For a
    fast-tracked claim this is the answer to "why was THIS car written off without an inspection".
    """
    phi = att.phi[row]
    base = float(att.base[row])
    order = np.argsort(-np.abs(phi))
    keep, rest = order[:top_n], order[top_n:]

    names = [f"{att.features[j]} = {_fmt(att.X.iloc[row, j])}" for j in keep]
    values = [phi[j] for j in keep]
    if len(rest):
        names.append(f"{len(rest)} other features")
        values.append(float(phi[rest].sum()))

    names, values = names[::-1], values[::-1]          # least important at the bottom
    starts = base + np.concatenate([[0.0], np.cumsum(values)[:-1]])

    if ax is None:
        fig, ax = plt.subplots(figsize=(FIG_W + 0.8, 0.34 * len(values) + 1.6))
    else:
        fig = ax.figure
    colours = [RED if v > 0 else BLUE for v in values]
    ax.barh(np.arange(len(values)), values, left=starts, color=colours, height=0.66)

    # Label placement is fiddly on purpose: the bars span three orders of magnitude, so a label
    # hung off a tiny bar lands on the y tick text. Anything under 2% of the drawn range is left
    # unlabelled (the bar is visibly nil anyway), and the axis is padded so the rest fit.
    ends = starts + np.asarray(values)
    lo, hi = float(min(starts.min(), ends.min(), base)), float(max(starts.max(), ends.max(), base))
    pad = max((hi - lo) * 0.18, 1e-6)
    ax.set_xlim(lo - pad, hi + pad)
    for i, (s0, v) in enumerate(zip(starts, values)):
        if abs(v) < (hi - lo) * 0.02:
            continue
        ax.text(s0 + v + (0.012 if v >= 0 else -0.012) * (hi - lo), i, f"{v:+.3f}", va="center",
                fontsize=7, ha="left" if v >= 0 else "right", color=INK)
    ax.axvline(base, color=GREY, lw=1, ls=":")
    ax.axvline(base + float(np.sum(phi)), color=INK, lw=1.2)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(f"log-odds       base E[f(x)] = {base:.3f}   to   f(x) = {base + phi.sum():.3f}")
    ax.grid(axis="y", visible=False)
    return _finish(fig, label or f"Waterfall — row {row}")


def plot_force(att: Attribution, row: int, top_n: int = 10, label=None, ax=None):
    """The same decomposition as one pushed bar: forces towards (red) and away from (blue) scrap.

    Compact enough to put several claims side by side — which is how it earns its place next to
    the waterfall: comparing three claims at once, not explaining one in full.
    """
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
    for group, colour, sign in ((sorted(neg, key=lambda kv: kv[1]), BLUE, +1),
                                (sorted(pos, key=lambda kv: -kv[1]), RED, -1)):
        cursor, depth = base, 0
        for name, v in group:
            ax.barh(0, v, left=cursor, color=colour, height=0.45,
                    edgecolor="white", linewidth=0.7)     # the segment boundaries must be visible
            # Only wide-enough segments are labelled, and consecutive labels alternate depth —
            # otherwise the thin tail segments print on top of each other, as they always do near
            # f(x) where the last few contributions bunch together.
            if abs(v) > 0.06 * span:
                depth += 1
                ax.text(cursor + v / 2.0, sign * (0.34 + 0.26 * (depth % 2)), name, ha="center",
                        va="bottom" if sign > 0 else "top", fontsize=6.5, color=INK)
            cursor += v

    ax.axvline(base, color=GREY, lw=1, ls=":")
    ax.axvline(out, color=INK, lw=1.4)
    ax.text(out, 1.02, f" f(x)={out:.3f}", fontsize=8, color=INK, va="top")
    ax.text(base, 1.02, f"base={base:.3f} ", fontsize=8, color=GREY, ha="right", va="top")
    ax.set_yticks([])
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("log-odds")
    ax.grid(visible=False)
    return _finish(fig, label or f"Force — row {row}")


def _fmt(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "NaN"
    return f"{f:.0f}" if float(f).is_integer() and abs(f) < 1e6 else f"{f:.3g}"


# ---- distributions and the threshold region ------------------------------------------


def plot_distributions(X: pd.DataFrame, features, scores=None, tau=None, above=None,
                       bins: int = 40, ncols: int = 3, title=None):
    """Input distributions, split by the fast-track cutoff when scores and tau are given.

    The split is the point: a tree model with a hard threshold acts on a REGION of input space, so
    the question is not just what each feature looks like overall but how the above-τ population
    differs. Binary features get rate bars; continuous features get overlaid histograms.

    `above`, if given, is an exact per-row treatment mask (e.g. `src/threshold.py:apply`) and wins
    over the `scores > tau` approximation — pass it whenever the exact rule is known, since a
    piecewise (v2) or segmented (v1) rule's real boundary is not any single scalar tau.
    """
    features = list(features)
    if above is not None:
        above = np.asarray(above, dtype=bool)
    else:
        # STRICT, matching src/threshold.py — scoring exactly tau is garaged, not scrapped.
        above = None if (scores is None or tau is None) else np.asarray(scores) > tau
    nrows = int(np.ceil(len(features) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, features):
        if not pd.api.types.is_numeric_dtype(X[feat]):        # a passthrough string column
            top = X[feat].astype(str).value_counts().head(6)
            pos = np.arange(len(top))
            if above is None:
                ax.bar(pos, top.values, color=BLUE, width=0.6)
            else:
                sub = X[feat].astype(str)
                ax.bar(pos - 0.18, [int((sub[~above] == c).sum()) for c in top.index],
                       color=GREY, width=0.34, label="below τ")
                ax.bar(pos + 0.18, [int((sub[above] == c).sum()) for c in top.index],
                       color=RED, width=0.34, label="above τ")
            ax.set_xticks(pos)
            ax.set_xticklabels(top.index, fontsize=6, rotation=30, ha="right")
            ax.set_ylabel("claims", fontsize=8)
            ax.set_title(f"{feat}  (categorical, top {len(top)})", fontsize=9)
            continue

        s = pd.to_numeric(X[feat], errors="coerce")
        if s.nunique(dropna=True) <= 2:                       # binary / one-hot: rates, not bins
            if above is None:
                ax.bar([0], [float(s.mean())], color=BLUE, width=0.5)
                ax.set_xticks([0]); ax.set_xticklabels(["all"])
            else:
                ax.bar([0, 1], [float(s[~above].mean()), float(s[above].mean())],
                       color=[GREY, RED], width=0.55)
                ax.set_xticks([0, 1]); ax.set_xticklabels(["below τ", "above τ"], fontsize=8)
            ax.set_ylabel("rate", fontsize=8)
        else:
            v = s.values.astype(float)
            finite = np.isfinite(v)
            edges = np.histogram_bin_edges(v[finite], bins=bins) if finite.any() else bins
            if above is None:
                ax.hist(v[finite], bins=edges, color=BLUE, alpha=0.9)
            else:
                ax.hist(v[finite & ~above], bins=edges, color=GREY, alpha=0.75, label="below τ")
                ax.hist(v[finite & above], bins=edges, color=RED, alpha=0.75, label="above τ")
            ax.set_ylabel("claims", fontsize=8)
        miss = float(X[feat].isna().mean() * 100)
        ax.set_title(f"{feat}" + (f"   ({miss:.0f}% missing)" if miss > 0.5 else ""), fontsize=9)
        ax.tick_params(labelsize=7)

    for ax in axes[len(features):]:
        ax.axis("off")
    if above is not None and len(features):
        # the first panel may be a binary/rate panel, which carries no labelled artists
        for ax in axes[:len(features)]:
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(fontsize=7, loc="upper right")
                break
    return _finish(fig, title or "Model input distributions" +
                   ("" if above is None else "  ·  below vs above the fast-track cutoff"))


def score_bands(scores, tau, quantiles=(0.5, 0.9, 0.99), above=None) -> "pd.Series":
    """Label each claim: 'below τ', then quantile bands of the score WITHIN the above-τ region.

    Above the cutoff every claim gets the same action, so the interesting variation is how deep
    into the region it sits — a claim at τ+0.001 and one at 0.999 are treated identically by the
    policy but are not the same claim, and their attributions should not be pooled.

    `above`, if given, is an exact per-row treatment mask (e.g. `src/threshold.py:apply`) and wins
    over the `scores > tau` approximation — the lower band edge then comes from the mask's own
    minimum score rather than the scalar tau, so a claim it marks "above" is never left mislabelled
    "below τ" for scoring under tau's own value.
    """
    s = pd.Series(np.asarray(scores, dtype=float))
    labels = pd.Series(["below τ"] * len(s), index=s.index)
    if above is not None:
        above = pd.Series(np.asarray(above, dtype=bool), index=s.index)
        lower = float(s[above].min()) if above.any() else float(tau)
    else:
        above = s > tau       # STRICT, matching src/threshold.py — scoring exactly tau is garaged
        lower = float(tau)
    if above.sum() == 0:
        return pd.Series(pd.Categorical(labels, categories=["below τ"], ordered=True))

    # Named by PERCENTILE, not by the score value: above the cutoff the scores bunch against 1.0,
    # so value-based labels come out as "0.999..1.000" twice and stop distinguishing anything.
    edges = [lower] + [float(s[above].quantile(q)) for q in quantiles]
    points = ["τ"] + [f"p{int(round(q * 100))}" for q in quantiles] + ["max"]
    names = [f"{points[i]}–{points[i + 1]}" for i in range(len(points) - 1)]
    for i, name in enumerate(names):
        upper = edges[i + 1] if i + 1 < len(edges) else None
        sel = above & (s >= edges[i]) & ((s < upper) if upper is not None else True)
        labels[sel] = name
    order = ["below τ"] + names
    return pd.Series(pd.Categorical(labels, categories=order, ordered=True), index=s.index)


def plot_band_bars(att: Attribution, bands: pd.Series, top_n: int = 12, title=None):
    """mean|SHAP| per feature, one bar group per score band — what drives the deep-scrap region.

    Read it as: does the same feature carry the decision everywhere, or does the model switch to a
    different set of reasons for the claims it is most certain about? A feature whose mass grows
    monotonically into the top band is the one the fast-track rule is effectively keyed on.
    """
    # Band order must come from the bands themselves — sorting the labels alphabetically puts
    # "below τ" between two above-τ bands and the reader silently mis-reads the trend.
    order = list(getattr(bands, "cat").categories) if hasattr(bands, "cat") else list(pd.unique(bands))
    bands = pd.Series(np.asarray(bands, dtype=object), index=range(len(bands)))
    feats = att.top(top_n)[::-1]
    order = [b for b in order if (bands == b).sum() > 0]

    table = {}
    for band in order:
        mask = (bands == band).values
        table[f"{band}  (n={int(mask.sum())})"] = pd.Series(
            np.abs(att.phi[mask]).mean(axis=0), index=att.features).reindex(feats)
    table = pd.DataFrame(table)

    fig, ax = plt.subplots(figsize=(FIG_W + 1.4, 0.36 * len(feats) + 1.5))
    y = np.arange(len(feats))
    h = 0.8 / max(len(table.columns), 1)
    palette = (figstyle.SERIES if _HAS_FIGSTYLE else [GREY, BLUE, RED, "#eda100", "#4a3aa7"])
    for i, col in enumerate(table.columns):
        ax.barh(y + (i - (len(table.columns) - 1) / 2.0) * h, table[col].values, height=h,
                color=palette[i % len(palette)], label=col)
    ax.set_yticks(y)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("mean |SHAP| within the band")
    ax.set_ylim(-0.7, len(feats) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower right")
    return _finish(fig, title or "Attribution mass by score band"), table
