"""Estimator layer — "how bad, and by what mechanism?". Reads artefacts; never opens a pkl.

`concentration` (functions, shared measurement primitives — used by, but not itself part of, the
strategy axis below) + the pluggable `EffectEstimator` strategies under `estimator.effect`
(`EffectEstimator` ABC → `ShapDiDEstimator`, built 2026-09-16 — see src/docs/STRUCTURE.md
§ "Five layers"). `effect/` mirrors `detector/algorithm/` and `mitigator/corrector/`'s split
between "the ABC-conforming strategy" and loose per-layer utilities (added 2026-09-17, once
`estimator/` needed the same separation). `ShapDiDEstimator` is the only causal estimator this
thesis runs on real data; RDD and logit-adjustment were explored on the synthetic DGP only and
have no real-data counterpart, so no `RDDEstimator`/`LogitAdjustEstimator` exists here.

    import sys; sys.path.insert(0, "../src")
    from estimator import concentration as conc
    conc.profile_table({v: conc.mean_abs(d.attributions) for v, d in versions.items()})

    from estimator import ShapDiDEstimator
    est = ShapDiDEstimator()
    report = est.estimate("v2", "test", "v3", "oot")
"""
from estimator import concentration
from estimator.effect import Assumption, EffectEstimator, EffectReport, FalsificationReport, ShapDiDEstimator

__all__ = [
    "concentration",
    "EffectEstimator",
    "Assumption",
    "FalsificationReport",
    "EffectReport",
    "ShapDiDEstimator",
]
