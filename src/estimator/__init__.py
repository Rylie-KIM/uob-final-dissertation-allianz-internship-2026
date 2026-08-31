"""Estimator layer — "how bad, and by what mechanism?". Reads artefacts; never opens a pkl.

Only `concentration` exists so far. The `EffectEstimator` ABC and its RDD / SHAP-DiD / logit
subclasses are still [plan] (see src/docs/STRUCTURE.md § "Five layers") — they are promoted out of
notebooks 04_01–04_03 when the real data supports them, and this package is where they will land.

    import sys; sys.path.insert(0, "../src")
    from estimator import concentration as conc
    conc.profile_table({v: conc.mean_abs(d.attributions) for v, d in versions.items()})
"""
from estimator import concentration

__all__ = ["concentration"]
