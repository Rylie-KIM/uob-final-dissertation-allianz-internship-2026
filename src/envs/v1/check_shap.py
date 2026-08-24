"""Probe whether a `shap` release actually works inside env-v1 (Python 3.5.6, xgboost 0.72).

env-v1 is the one version env with NO shap: `shap_kit_v1.compute()` falls back to the booster's
`pred_contribs`, which is tree_path_dependent ONLY. That leaves v1 unable to produce the
interventional attribution the cross-version runs in 00_shap_attribution.ipynb compare, so v1's
file lands in the canonical (interventional) slot carrying `backend="native"` and
`concentration.require_comparable` refuses the comparison.

shap 0.35.0 is the last release with a cp35 win_amd64 wheel (2020-02-27), it accepts
`feature_perturbation="interventional"`, and its XGBTreeModelLoader parses the pre-1.0 raw model
buffer that xgboost 0.72 writes. Whether the wheel's compiled `_cext` matches THIS env's numpy is
the open question — that is what this script answers, before any notebook is changed.

Run it INSIDE env-v1, from the repo root (PowerShell):

    src\\envs\\v1\\.venv\\python.exe src\\envs\\v1\\check_shap.py ^
        --model model_repos\\real\\<v1 repo>\\outputs\\fasttracker_xgb.pkl ^
        --split train --rows 300 --background 100

Exit code 0 = interventional works and its numbers differ from the native path-dependent ones.
Any other exit code, or a "*** MISMATCH ***" line, means do NOT switch v1 over to shap yet.

Python 3.5: no f-strings, no annotations, `.format()` throughout.
"""

import argparse
import os
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))       # src/envs/v1 -> walk up to the repo root
while not os.path.exists(os.path.join(ROOT, "src", "config.py")) and ROOT != os.path.dirname(ROOT):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import shap_kit_v1 as sk                                 # noqa: E402


def report_versions():
    print("== environment ==")
    print("  python     {}".format(sys.version.split()[0]))
    versions = {}
    for name in ("numpy", "scipy", "pandas", "sklearn", "xgboost"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "?")
            print("  {:<10} {}".format(name, versions[name]))
        except Exception as exc:
            print("  {:<10} NOT IMPORTABLE ({})".format(name, exc))
    # The probe answers a question about ONE env. Run it with any other interpreter and the
    # answer is about that interpreter, which is worse than no answer.
    if not sys.version.startswith("3.5") or versions.get("xgboost") != sk.XGBOOST_PIN:
        print("  !! this is NOT env-v1 (expected python 3.5.x + xgboost {}). Re-run with"
              .format(sk.XGBOOST_PIN))
        print("     src\\envs\\v1\\.venv\\python.exe — results below say nothing about env-v1.")


def import_shap():
    """Return (module_or_None, TreeExplainer_or_None, how).

    There is no partial route in: `shap/__init__.py` imports KernelExplainer on its FIRST line,
    and importing any submodule (`shap.explainers.tree`) runs that same __init__ first. So
    whatever __init__ needs, this env needs — tqdm included.
    """
    try:
        import shap
        return shap, shap.TreeExplainer, "import shap"
    except ImportError as exc:
        msg = str(exc)
        print("  `import shap` failed: {}: {}".format(type(exc).__name__, msg))
        if "tqdm" in msg:
            print("  -> shap/__init__.py line 8 imports KernelExplainer, and kernel.py does")
            print("     `from tqdm.auto import tqdm` at module level. tqdm gates importing shap")
            print("     AT ALL here, not just progress bars. --no-deps skipped it, so add it:")
            print('     ...\\python.exe -m pip install --no-deps "tqdm==4.64.1"')
        return None, None, None
    except Exception as exc:
        print("  `import shap` failed: {}: {}".format(type(exc).__name__, exc))
        traceback.print_exc()
        print("  -> if this comes from shap/__init__.py's plot imports, matplotlib 2.2.5 is the")
        print("     likely cause; there is no way to import TreeExplainer around it (the package")
        print("     __init__ always runs). Step down the shap ladder instead.")
        return None, None, None


def build_matrix(est, features_path, rows, seed):
    frame = pd.read_csv(features_path)
    id_col = "claim_id"
    if sk.ID_COLUMN in frame.columns and id_col not in frame.columns:
        frame = frame.rename(columns={sk.ID_COLUMN: id_col})
    X = frame.drop([id_col], axis=1) if id_col in frame.columns else frame

    trained = sk.model_feature_names(est) or sk.registry_features()
    missing = [c for c in trained if c not in X.columns]
    if missing:
        raise RuntimeError("matrix is missing {} model column(s), e.g. {}".format(
            len(missing), missing[:8]))
    X = sk.align(X[trained], est)

    if rows and rows < len(X):
        X = X.sample(n=rows, random_state=seed)
    return X


def explain_interventional(TreeExplainer, est, X, background):
    """Try the modern kwarg first, then the pre-0.30 spelling.

    `feature_perturbation="interventional"` was named `feature_dependence="independent"` up to
    shap 0.29 — same estimand, so the ladder below (0.35 -> 0.34 -> 0.33 -> 0.32.1 -> 0.29.3)
    stays usable if a newer wheel will not load against this env's numpy.
    """
    try:
        expl = TreeExplainer(est, background, feature_perturbation="interventional")
        spelling = 'feature_perturbation="interventional"'
    except TypeError:
        expl = TreeExplainer(est, background, feature_dependence="independent")
        spelling = 'feature_dependence="independent"  (pre-0.30 spelling)'
    print("  kwarg accepted: {}".format(spelling))

    values = expl.shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]

    base = expl.expected_value
    base = float(np.asarray(base).ravel()[-1]) if np.ndim(base) else float(base)
    return sk.Attribution(values, np.full(len(X), base), X, "shap", "interventional",
                          note="shap probe")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="path to v1's fasttracker_xgb.pkl")
    p.add_argument("--features", default=None,
                   help="features CSV; default = shap_kit_v1.features_csv_path(--split)")
    p.add_argument("--split", default="train", choices=list(sk.SPLITS))
    p.add_argument("--rows", type=int, default=300, help="rows to explain (keep it small)")
    p.add_argument("--background", type=int, default=100,
                   help="interventional reference sample size")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    report_versions()

    print("\n== shap ==")
    shap_mod, TreeExplainer, how = import_shap()
    if TreeExplainer is None:
        print("  shap is not usable in this env -> stay on the native pred_contribs route.")
        return 2
    print("  shap {}  ({})".format(getattr(shap_mod, "__version__", "?"), how))
    print("  {}".format(getattr(shap_mod, "__file__", "?")))

    features_path = a.features or sk.features_csv_path(a.split)
    print("\n== data ==")
    print("  model    {}".format(a.model))
    print("  features {}".format(features_path))
    est = sk.load_estimator(a.model)
    X = build_matrix(est, features_path, a.rows, a.seed)
    background = X.sample(n=min(a.background, len(X)), random_state=a.seed + 1)
    print("  explaining {} rows x {} features, background {} rows".format(
        X.shape[0], X.shape[1], len(background)))

    print("\n== interventional (shap) ==")
    try:
        inter = explain_interventional(TreeExplainer, est, X, background)
    except Exception as exc:
        print("  FAILED: {}: {}".format(type(exc).__name__, exc))
        traceback.print_exc()
        print("\n  If this is a numpy binary-compatibility error, step down the ladder:")
        print("    shap==0.34.0 -> 0.33.0 -> 0.32.1 -> 0.29.3 (older wheels, older numpy ABI)")
        return 3
    print("  {}".format(inter))
    gap_inter = sk.check_additivity(inter, est)

    print("\n== native pred_contribs (tree_path_dependent) ==")
    native = sk.compute(est, X)
    print("  {}".format(native))
    gap_native = sk.check_additivity(native, est)

    print("\n== do the two backends actually differ? ==")
    delta = float(np.max(np.abs(inter.phi - native.phi)))
    k = min(10, inter.phi.shape[1])
    a_top = list(inter.mean_abs.head(k).index)
    b_top = list(native.mean_abs.head(k).index)
    print("  max |phi_interventional - phi_path_dependent| = {:.3e}".format(delta))
    print("  top-{} features identical order: {}".format(k, a_top == b_top))
    print("  top-{} overlap: {}/{}".format(k, len(set(a_top) & set(b_top)), k))
    if delta < 1e-9:
        print("  *** the two backends collapsed to the same numbers — the interventional")
        print("      background was ignored. Do NOT record this as interventional. ***")
        return 4

    ok = gap_inter < 1e-3 and gap_native < 1e-3
    print("\n== verdict ==")
    print("  {}".format("USABLE — v1 can produce interventional attributions."
                        if ok else "NOT usable — additivity failed; investigate before switching."))
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
