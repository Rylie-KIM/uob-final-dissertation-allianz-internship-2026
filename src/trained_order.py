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
  * Python 3.5 syntax only -- no f-strings, no future annotations, no variable annotations.
    shap_kit_v1.py imports this and runs there.
  * ASCII only in anything printed or raised: that console is not UTF-8, and an em dash raises
    UnicodeEncodeError halfway through a run.
Stdlib only, for the same reason -- it has to import cleanly in all three version envs AND the
analysis env, so it never touches pandas, numpy or config.

NOT the producer side. features/extract_features.py builds features/registry/<v>.json off a
DIFFERENT ladder (the preprocessing head first, recording which rung answered). That ladder is
deliberate and verified; it is not unified here.
"""
import re

# f0, f1, f2, ... -- what old xgboost hands back for a matrix that carried no column names.
# See model_feature_names for why they are refused rather than believed.
_PLACEHOLDER = re.compile(r"^f\d+$")


def model_feature_names(est):
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


def feature_order(est, columns, trained=None, trained_source=None):
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


def align(X, est):
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
