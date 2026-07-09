"""SFP mitigation (Analysis Layer — loads no model).

Public API: `SFPMitigator` (holds a corrector + optional policy) and the pluggable strategies under
`mitigator.corrector` and `mitigator.policy`.

Runtime type-checking: `beartype_this_package()` installs an import hook so every function and
method in this package (incl. `mitigator.corrector` / `mitigator.policy`) has its type hints
enforced at call time.
"""
from beartype.claw import beartype_this_package

beartype_this_package()

from mitigator.corrector import IPSCorrector, TrainingDataCorrector  # noqa: E402
from mitigator.policy import InvestigationPolicy  # noqa: E402
from mitigator.sfp_mitigator import SFPMitigator  # noqa: E402

__all__ = [
    "SFPMitigator",
    "TrainingDataCorrector",
    "IPSCorrector",
    "InvestigationPolicy",
]
