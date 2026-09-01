"""Unpickling a fitted model -- ONE definition, for every env.

WHY A LADDER AND NOT `joblib.load`. v1's artefact was serialised in 2018 against
scikit-learn < 0.23, whose `sklearn.externals.joblib` is a DIFFERENT vendored copy of joblib.
Standalone joblib's NumpyUnpickler desyncs on that stream and dies inside pickle.py with
`KeyError: <n>` (n = whatever byte it landed on, commonly 0). So the loader is tried in order,
strongest-for-this-repo first, and the one that answered is REPORTED -- which loader opened a
pickle is part of reading any result that came out of it.

v2 and v3 fall through to plain joblib on the first or second rung; the cost of the extra
attempt is one ImportError.

WHY IT IS ITS OWN FILE. This ladder was copy-pasted in shap_kit_v1.py and
features/extract_features_v1.py, with a note in each saying it had to stay identical to the
other -- a rule no code enforced. Same reasoning as trained_order.py: one definition, imported.

Python 3.5 syntax and ASCII-only strings, because env-v1 imports it -- return
annotations included, which 3.5 has had since 3.0. Every loader is imported
lazily, inside its own rung, so an env that lacks one is simply a rung that does not apply.
"""
import os
from typing import Tuple


def load_any(path) -> Tuple[object, str]:
    """Unpickle `path` with whichever loader that stack actually used.

    Returns (object, loader_name). Raises RuntimeError, listing what each rung said, when none
    of them can open the file -- the caller decides whether that is a traceback or a clean exit.
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
    lines.append("against that version repo's dependency spec.")
    raise RuntimeError("\n".join(lines))


def load_estimator(path, quiet=False) -> Tuple[object, str]:
    """load_any(), then hand back the thing that owns the trees.

    A Pipeline is unwrapped to its final step, with a note: everything downstream (SHAP, scoring)
    works on the POST-preprocessing matrix, so the caller must feed the transformed X, not raw
    claims. Returns (estimator, loader_name).
    """
    obj, loader = load_any(path)
    if not quiet:
        print("  loaded with {0}".format(loader))
    if hasattr(obj, "steps"):
        if not quiet:
            print("  note: {0} is a Pipeline ({1}). Using the final step; X must be the "
                  "POST-preprocessing matrix.".format(
                      os.path.basename(str(path)), [n for n, _ in obj.steps]))
        return obj.steps[-1][1], loader
    return obj, loader
