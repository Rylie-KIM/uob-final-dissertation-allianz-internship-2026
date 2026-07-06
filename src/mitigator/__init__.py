"""SFP mitigation (Analysis Layer — loads no model).

Public API: `SFPMitigator` (holds a corrector + optional policy) and the pluggable strategies under
`mitigator.corrector` and `mitigator.policy`.
"""
from mitigator.corrector import IPSCorrector, TrainingDataCorrector
from mitigator.policy import InvestigationPolicy
from mitigator.sfp_mitigator import SFPMitigator

__all__ = [
    "SFPMitigator",
    "TrainingDataCorrector",
    "IPSCorrector",
    "InvestigationPolicy",
]
