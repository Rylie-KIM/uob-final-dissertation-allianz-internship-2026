"""v1-only feature-name extractor -- the Python 3.5 twin of extract_features.py.

WHY A SEPARATE FILE. `extract_features.py` already produced `features/registry/v2.json` and
`features/registry/v3.json` and must not be touched. It cannot produce v1's, for two reasons that
are both properties of env-v1 and not of the code:

  * env-v1 is **Python 3.5.6**, and extract_features.py uses `from __future__ import annotations`
    (3.7), `list[str]` annotations (3.9) and f-strings (3.6). It is a SyntaxError there, before a
    single line runs.
  * it falls back to `import config` for the pickle path, and `src/config.py` is itself 3.7+, so
    that import is a SyntaxError too. This script therefore takes the paths as arguments and never
    imports config.

WHAT IT WRITES -- byte-for-byte the same schema as extract_features.py, so `check_overlap.py` and
every registry reader treat all three versions identically:

    {
      "version":        "v1",
      "model_path":     "<the pickle this was read from>",
      "pipeline_repr":  "<class name of the loaded object>",
      "raw_features":   [...],   raw claim columns the PREPROCESSOR expects
      "model_features": [...]    columns the BOOSTER consumes, post-preprocessing
    }

No extra keys: a v1 file that carried fields the others lack would make the three registries
non-comparable, which is the one thing this registry exists to prevent.

--preprocessor IS OPTIONAL, AND USUALLY NOT WORTH PASSING. v1 keeps the fitted preprocessing and
the estimator in SEPARATE files (confirmed 2026-07-31); the estimator gives `model_features`, the
preprocessor gives `raw_features`. But **nothing in this repo reads `raw_features`** -- the
registry's only consumer, `check_overlap.py`, validates against `model_features` alone. Meanwhile
the preprocessor is the pickle that holds custom transformer classes, so it is the one that raises
`ModuleNotFoundError` unless the version repo is on sys.path (see ENV_MANAGEMENT.md, the fttl.pth
step). Passing --model on its own avoids that entirely and loses nothing that is used.

Names read off v1's own training script:

    outputs/fasttracker_xgb.pkl     the estimator   -> model_features
    outputs/fttl_pipeline.pkl      the preprocessor -> raw_features

  (verify the spelling against `dir outputs` -- "fasttracker" vs "fasttracker" is transcribed, not
   confirmed. A typo just fails with FileNotFoundError, so it is safe to try as-is.)

USAGE -- run with env-v1's own interpreter, from the repo root. ONE LINE, so no continuation
character is involved and it is the same in PowerShell and cmd:

    src\\envs\\v1\\.venv\\python.exe features\\extract_features_v1.py --model model_repos\\real\\<v1-repo>\\outputs\\fasttracker_xgb.pkl

If you do split it over lines, the continuation character is SHELL-SPECIFIC -- ` in PowerShell,
^ in cmd. This repo's SETUP.md is written for PowerShell, where a stray ^ is passed to the script
as a literal argument and the remaining lines run as separate commands. Backtick, and nothing
after it on the line:

    src\\envs\\v1\\.venv\\python.exe features\\extract_features_v1.py `
        --model        model_repos\\real\\<v1-repo>\\outputs\\fasttracker_xgb.pkl `
       

Writes features/registry/v1.json. Add `--dry-run` to print what it found without writing.

=============================================================================
!! PYTHON 3.5 COMPATIBLE ON PURPOSE -- DO NOT MODERNISE !!
No f-strings, no future annotations, no builtin generics, no pathlib-only APIs
that arrived after 3.5. Old sklearn (pre-1.0) has no `feature_names_in_`, so
raw-column recovery walks the pipeline for the attributes bespoke transformers
actually use -- see `_raw_from_pipeline`.
=============================================================================
"""
# PYTHON 3.5 COMPATIBLE ON PURPOSE
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "registry")

VERSION = "v1"

COLUMN_ATTRS = ("feature_names_in_", "columns", "columns_", "feature_names", "feature_names_",
                "input_columns", "input_columns_", "cols", "cols_", "variables", "variables_")


def _load_any(path):
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

    lines = ["\ncould not unpickle {0} with any loader:".format(path)]
    for name, why in attempts:
        lines.append("    {0:<28} {1}".format(name, str(why)[:120]))
    lines.append("")
    lines.append("A `KeyError: <n>` from every loader means the pickle stream desynced -- the file")
    lines.append("was written by a stack this env cannot reproduce. Check `dir` beside the pkl for")
    lines.append(".npy sidecars (very old joblib), and compare this env's scikit-learn/numpy")
    lines.append("against the v1 repo's conda_dependencies_local.yml.")
    raise SystemExit("\n".join(lines))


def _as_names(value):
    """A list-like of column names -> list of str, or None if `value` is not one."""
    if value is None:
        return None
    if isinstance(value, str):
        return None
    try:
        names = [str(v) for v in value]
    except TypeError:
        return None
    return names or None


def _raw_from_object(obj):
    """The first attribute on `obj` that looks like a raw column list."""
    for attr in COLUMN_ATTRS:
        names = _as_names(getattr(obj, attr, None))
        if names:
            return names, attr
    return None, None


def _raw_from_pipeline(prep):
    """Raw claim columns the preprocessor expects."""
    if prep is None:
        return [], "no preprocessor given"

    names, attr = _raw_from_object(prep)
    if names:
        return names, "pipeline." + attr

    steps = getattr(prep, "steps", None)
    if steps:
        for name, step in steps:
            found, attr = _raw_from_object(step)
            if found:
                return found, "step '" + str(name) + "'." + attr

    return [], "not recoverable from this pipeline"


def _model_from_estimator(est):
    """Post-preprocessing column names -- what the booster literally consumed."""
    if hasattr(est, "steps"):     # a Pipeline was pickled after all
        est = est.steps[-1][1]

    if hasattr(est, "get_booster"):
        try:
            names = _as_names(est.get_booster().feature_names)
            if names:
                return names, "booster.feature_names"
        except Exception as exc:
            print("  booster.feature_names unavailable: {0}".format(exc))

    names, attr = _raw_from_object(est)
    if names:
        return names, "estimator." + attr
    return [], "not recoverable from this estimator"


def main():
    p = argparse.ArgumentParser(description="Write features/registry/v1.json")
    p.add_argument("--model", required=True,
                   help="v1's ESTIMATOR pickle")
    p.add_argument("--preprocessor", default=None,
                   help="OPTIONAL, and usually not worth passing. It only fills "
                        "raw_features, which nothing in this repo reads, and it is the pickle "
                        "that needs the version repo importable -- leaving it off avoids "
                        "ModuleNotFoundError entirely.")
    p.add_argument("--out", default=None, help="default: features/registry/v1.json")
    p.add_argument("--dry-run", action="store_true", help="print what was found, write nothing")
    a = p.parse_args()

    print("python {0}".format(sys.version.split()[0]))
    try:
        import xgboost
        print("xgboost {0}   (config declares 0.72 for v1)".format(xgboost.__version__))
    except Exception as exc:
        print("xgboost not importable: {0}".format(exc))

    if not os.path.isfile(a.model):
        raise SystemExit("\n--model {0} does not exist.\n".format(a.model))

    est, est_loader = _load_any(a.model)
    print("estimator: {0}  (via {1})".format(type(est).__name__, est_loader))

    prep = None
    if a.preprocessor:
        if not os.path.isfile(a.preprocessor):
            raise SystemExit("\n--preprocessor {0} does not exist.\n".format(a.preprocessor))
        prep, prep_loader = _load_any(a.preprocessor)
        print("preprocessor: {0}  (via {1})".format(type(prep).__name__, prep_loader))
        steps = getattr(prep, "steps", None)
        if steps:
            print("  steps: {0}".format([str(n) for n, _ in steps]))
    else:
        print("preprocessor: not given -> raw_features will be empty")

    model_features, model_src = _model_from_estimator(est)
    raw_features, raw_src = _raw_from_pipeline(prep)

    print("\nraw_features   {0:>4}   (from {1})".format(len(raw_features), raw_src))
    print("model_features {0:>4}   (from {1})".format(len(model_features), model_src))
    if model_features:
        print("  first 8: {0}".format(model_features[:8]))

    # `pipeline_repr` is the loaded object's class name -- same field extract_features.py writes.
    payload = {
        "version": VERSION,
        "model_path": os.path.abspath(a.model),
        "pipeline_repr": type(prep).__name__ if prep is not None else type(est).__name__,
        "raw_features": raw_features,
        "model_features": model_features,
    }

    if not model_features:
        print("\n  WARNING: no model-ready feature names recovered. SHAP output will be "
              "positional only -- resolve this before comparing across versions.")
    if not raw_features and prep is not None:
        print("\n  NOTE: the preprocessor exposes no raw column list under any of the names tried")
        print("  ({0}).".format(", ".join(COLUMN_ATTRS)))

    if a.dry_run:
        print("\ndry run -- nothing written")
        return

    if not os.path.isdir(REGISTRY):
        os.makedirs(REGISTRY)
    out = a.out if a.out else os.path.join(REGISTRY, VERSION + ".json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\n[{0}] raw={1} model={2} -> {3}".format(
        VERSION, len(raw_features), len(model_features), out))


if __name__ == "__main__":
    main()
