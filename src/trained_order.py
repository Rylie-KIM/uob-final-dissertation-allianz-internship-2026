"""The trained column order -- ONE definition, shared by every env.

WHAT IT ANSWERS. "Which columns did this model consume, and in what order?", and "does the frame
I am holding match that?". SHAP and xgboost are both POSITIONAL underneath: a frame carrying the
right columns in the wrong order attributes every phi to the wrong feature and scores every row
against the wrong splits. Where the names are present that raises; where they are absent it does
not, which is the case this module exists for.

WHY IT IS ITS OWN FILE. This used to be copy-pasted into shap_kit.py (v2/v3 SHAP), shap_kit_v1.py
(v1 SHAP) and training/retrain.py (retrain, and predict.py through it) -- and the copies had
already drifted: only the v1 one rejected xgboost's fabricated f0/f1/... names, and only the
retrain one returned None instead of [] when nothing was recoverable. The same pickle could be
called "exact" by one caller and "unverified" by another, which is precisely what a single
vocabulary is meant to make impossible. One definition now; the three modules re-export these
names unchanged, so no call site moved.

TWO CONSTRAINTS, both inherited from env-v1 (Python 3.5.6 on Windows):
  * Python 3.5 syntax only. Function annotations are FINE (PEP 3107 predates 3.0, and
    `typing` shipped with 3.5) -- what 3.5 lacks is variable annotations (PEP 526), builtin
    generics `list[str]` (PEP 585), `X | None` unions (PEP 604), f-strings and
    `from __future__ import annotations`. shap_kit_v1.py imports this and runs there.
    `align` returns whatever frame it was given, annotated as a forward-ref string so the
    stdlib-only rule below survives.
  * ASCII only in anything printed or raised: that console is not UTF-8, and an em dash raises
    UnicodeEncodeError halfway through a run.
Stdlib only, for the same reason -- it has to import cleanly in all three version envs AND the
analysis env, so it never touches pandas, numpy or config.

NOT the producer side. features/extract_features.py builds features/registry/<v>.json off a
DIFFERENT ladder (the preprocessing head first, recording which rung answered). That ladder is
deliberate and verified; it is not unified here.
"""
import json
import os
import re
from typing import List, Optional, Tuple

# f0, f1, f2, ... -- what old xgboost hands back for a matrix that carried no column names.
# See model_feature_names for why they are refused rather than believed.
_PLACEHOLDER = re.compile(r"^f\d+$")


def model_feature_names(est) -> List[str]:
    """The columns the estimator was TRAINED on, in trained order -- or [] if unrecoverable.

    Three rungs, strongest first: xgboost's booster, lightgbm, sklearn >= 1.0.

    Fabricated names are rejected. Fit the 0.x-era xgboost on a bare numpy array and the booster
    reports f0, f1, f2, ... -- that release generates them from the column COUNT on first access
    (modern xgboost returns None instead, which lands on the same [] by a different route). They
    are positions wearing a name's clothes: checking a frame against them proves nothing, so the
    caller gets [] and the verdict "unverified". Being told the order was never checked is worth
    more than a green light that means nothing.

    This is aimed at v1, whose env is xgboost 0.72 -- the only one of the three that can produce
    them. All three current pickles expose their real names (the 2026-08-31 backfill reported a
    plain "exact" for each), so the rule costs nothing today.

    The test is `all`, not `any`: a real matrix is allowed to contain one column called "f1".
    Only a list that is ENTIRELY f<number> is the generated one, because that release names every
    column or none.
    """
    getters = (lambda: list(est.get_booster().feature_names),    # xgboost, all three releases
               lambda: list(est.feature_name_),                  # lightgbm
               lambda: [str(c) for c in est.feature_names_in_])  # sklearn >= 1.0
    for getter in getters:
        try:
            names = getter()
        except Exception:
            continue
        if not names:
            continue
        if all(_PLACEHOLDER.match(str(n)) for n in names):
            continue
        return [str(n) for n in names]
    return []


UNVERIFIED_ORDER = "unverified (the estimator exposes no trained feature names)"


def feature_order(est, columns, trained=None, trained_source=None) -> str:
    """Status of `columns` against the trained order -- the meta's `feature_order` field.

    ONE VOCABULARY, shared by every producer of an attributions file, so the field means the same
    thing in every `_meta.json`:

        "exact"           the trained names, in the trained order
        "reordered"       same set, different order
        "set_mismatch"    different columns altogether
        "unverified ..."  no trained names are exposed, so nothing can be checked

    Producers differ in what they DO about it -- align() below reorders, attribute.py refuses --
    but they all record the same word. Recording it is the point: without the field a reader
    cannot tell a checked file from an unchecked one, and the two look identical.

    `trained` lets a caller supply an order resolved from somewhere other than the pickle: v1
    falls back to features/registry/v1.json when xgboost 0.72 exposes nothing. The verdict then
    carries "(via registry: ...)", because "unverified" would be wrong -- the order WAS checked --
    and a bare "exact" would hide that the pickle itself never confirmed it.
    """
    source = ""
    if trained is None:
        trained = model_feature_names(est)
    else:
        trained = [str(c) for c in trained]
        source = " (via registry: {0})".format(trained_source or "unrecorded")
    cols = [str(c) for c in columns]
    if not trained:
        return UNVERIFIED_ORDER
    if trained == cols:
        return "exact" + source
    if sorted(trained) == sorted(cols):
        return "reordered" + source
    return "set_mismatch" + source


def align(X, est) -> "pandas.DataFrame":
    """Put X into the trained column order, or say exactly what does not match.

    Selection is not this function's job: a frame carrying columns the model never saw is a
    set_mismatch and raises. Callers holding an export matrix (which carries the target beside
    the inputs) select first -- X[model_feature_names(est)] -- and align nothing afterwards.
    """
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
            "  in model, not in X : {0}\n"
            "  in X, not in model : {1}".format(sorted(want - have)[:12], sorted(have - want)[:12]))
    if status == "reordered":
        print("  reordered X into the booster's trained column order")
    return X[trained]


def read_registry(path) -> Optional[List[str]]:
    """features/registry/<v>.json -> its `model_features` list, or None.

    The registry is written by features/extract_features.py INSIDE each version env, off that
    version's own pickles, so it is the same authority as the booster -- just recorded on disk
    where a process that cannot open the pickle can still read it.
    """
    if not path:
        return None
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise SystemExit(
            "--features-json {0} does not exist.\n"
            "Build it inside this version's env:\n"
            "    <this python> features/extract_features.py --version <v>\n".format(p))
    fh = open(p)
    try:
        payload = json.load(fh)
    finally:
        fh.close()
    names = payload.get("model_features") or None
    if names is None:
        raise SystemExit(
            "{0} has no `model_features` -- extract_features.py could not recover them from the "
            "pickle (it prints a WARNING when that happens). Resolve that first: without the "
            "column list a feature cannot be told from the target.".format(p))
    return [str(n) for n in names]


def select_features(X, est, id_col, registry_path=None) -> Tuple[List[str], str]:
    """The feature columns of X, in the order the model was trained on.

    SHARED by training/retrain.py and scoring/predict.py: fitting and scoring have to pick and
    order the columns by the same rule, or the mitigated model is scored on a matrix it was never
    fitted on. Order matters as much as membership -- xgboost matches positionally once names are
    absent, so a reordered matrix trains a different model while raising nothing.

    TWO ROUTES, one authority. Both answer "which columns did this model consume", and both come
    from the fitted objects, never from a hand-written list:

      1. the booster the caller already has open (--baseline in retrain, --model in predict);
      2. features/registry/<v>.json, written by features/extract_features.py inside this env.

    (1) is preferred because it needs nothing else on disk. (2) is the fallback for a booster
    fitted from a bare numpy array, which carries no names of its own.

    Selection, not just ordering: `processed_inputs` is NOT feature-only. Every version's exported
    matrix carries the target beside the model inputs, and v3's also carries its own saved
    predictions, so "everything except claim_id" would hand the model its own answer as an input.
    Returns (columns, source) -- the caller does X[columns].
    """
    trained = model_feature_names(est)
    source = "booster"
    if not trained:
        trained = read_registry(registry_path)
        source = "registry"
    if not trained:
        raise SystemExit(
            "the estimator exposes no trained feature names, and no --features-json was given,\n"
            "so the feature columns cannot be determined.\n"
            "Run the driver instead (training/retrain_all.py, scoring/score_all.py) -- it passes\n"
            "the registry path from config.\n"
            "To build the registry:  <this python> features/extract_features.py --version <v>\n")

    missing = [c for c in trained if c not in X.columns]
    if missing:
        raise SystemExit(
            "the feature file is missing {0} column(s) the model was trained on, e.g. {1}.\n"
            "Point --features at that version's processed_inputs matrix for this "
            "split.".format(len(missing), missing[:8]))
    extra = [c for c in X.columns if c not in trained and c != id_col]
    if extra:
        print("  set aside {0} non-model column(s): {1}{2}".format(
            len(extra), extra[:8], " ..." if len(extra) > 8 else ""))
    return trained, source
