"""SFP mitigation (Analysis Layer — loads no model).

Public API: the pluggable `TrainingDataCorrector` strategies under `mitigator.corrector`, called
directly (`ReweightCorrector(scheme=...).correct(features, labels, feature_cols)`). No
context/wrapper class holds them — `SFPMitigator` was deleted 2026-09-16: it was imported by no
real-data notebook (they call `ReweightCorrector` directly), and its only caller was
`pipeline/pipeline.py`, which now calls the corrector directly too.
"""
from mitigator.corrector import ReweightCorrector, TrainingDataCorrector

__all__ = [
    "TrainingDataCorrector",
    "ReweightCorrector",
]
