"""Pluggable causal-effect estimation strategies (Analysis Layer — loads no model)."""
from estimator.effect.base import Assumption, EffectEstimator, EffectReport, FalsificationReport
from estimator.effect.shap_did import ShapDiDEstimator

__all__ = ["EffectEstimator", "Assumption", "FalsificationReport", "EffectReport", "ShapDiDEstimator"]
