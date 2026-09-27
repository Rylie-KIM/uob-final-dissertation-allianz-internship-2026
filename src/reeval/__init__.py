"""Re-evaluation (Analysis Layer — loads no model). "What changed?" — the only layer that reads
TWO artefact sets (a baseline and a Layer-3-corrected one) for the same comparison.

Public API: the pluggable `ReEvalMetric` strategies under `reeval.metrics`, called directly
(`ShapDiDDelta().compute(v_before, split_before, v_after, split_after, tag=...)`). No
context/wrapper class holds them — see `reeval/metrics/base.py`'s module docstring for why no
`ReEvaluator` is built here, the same reasoning that left `EffectEstimator` without one and got
`SFPDetector`/`SFPMitigator` deleted 2026-09-16.
"""
from reeval.metrics import ReEvalMetric, ReEvalReport, ShapDiDDelta

__all__ = ["ReEvalMetric", "ReEvalReport", "ShapDiDDelta"]
