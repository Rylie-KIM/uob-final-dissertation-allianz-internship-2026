"""Pluggable training-data correctors (Analysis Layer — loads no model)."""
from mitigator.corrector.base import TrainingDataCorrector
from mitigator.corrector.ips import IPSCorrector

__all__ = ["TrainingDataCorrector", "IPSCorrector"]
