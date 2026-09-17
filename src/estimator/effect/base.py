"""EffectEstimator — the causal-quantity axis of the Strategy Pattern (`src/docs/DESIGN.md`
§ "Strategy Pattern"), the `estimator/` sibling of `detector/algorithm/base.py`'s
`DetectionAlgorithm` / `mitigator/corrector/base.py`'s `TrainingDataCorrector`. Renamed from
`estimator/effect_estimator.py` 2026-09-17, then moved into its own `estimator/effect/` package
the same day — `algorithm/` and `corrector/` are named after their ABC's own tail word, but
`EffectEstimator`'s tail word is "Estimator" (the package's own name), so this subpackage is
named after the ABC's head word instead. Distinct from `detector/`: a `DetectionAlgorithm` returns
yes/no, an `EffectEstimator` returns a CAUSAL QUANTITY, and is required to say what makes that
quantity identified before it reports one.

RDD and DiD are identified only under assumptions that are, in general, untestable (continuity at
the cutoff; parallel trends). The discipline this ABC encodes — state the assumption, test its
observable implications, carry the diagnostic alongside the number rather than silently deciding
for the reader (Imbens & Wooldridge, P7) — is made structural here: `estimate()` must attach
`falsify()`'s diagnostics to whatever it returns, so a number can never travel without the check
that qualifies it.

Only `ShapDiDEstimator` exists so far (`shap_did.py`) — the only causal estimator this thesis
actually runs on real data. RDD and logit-adjustment were explored on the SYNTHETIC DGP only
(`notebook/04_01_p7_RDD.ipynb`, `notebook/04_03_p7_logit_adjustment.ipynb`) and have no real-data
counterpart today; do not add `RDDEstimator`/`LogitAdjustEstimator` here unless a real notebook
actually needs one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Assumption:
    """One named identifying assumption — stated, not buried in a docstring."""
    name: str
    description: str


@dataclass
class FalsificationReport:
    """The observable diagnostics behind one or more of an estimator's assumptions.

    `passed` is the estimator's OWN verdict on whether its diagnostics are even interpretable —
    not a verdict on whether the assumption itself holds. `ShapDiDEstimator`, for instance,
    computes and returns its DiD regardless of what this says (an earlier design tried to
    hard-gate on a dose threshold and broke — see `shap_did.py`'s module docstring), so
    `passed=False` here means "read this number against the diagnostics below by hand before
    trusting it", never "no number was computed".
    """
    passed: bool
    diagnostics: dict[str, object] = field(default_factory=dict)
    note: str = ""


@dataclass
class EffectReport:
    """One estimator's headline number, with its FalsificationReport attached — never separable,
    so the number can never be read without the diagnostic that qualifies it."""
    estimate: float
    falsification: FalsificationReport
    diagnostics: dict[str, object] = field(default_factory=dict)


class EffectEstimator(ABC):
    """`assumptions()` + `falsify()` + `estimate()` — see the module docstring for why all three
    are mandatory rather than `estimate()` alone. Concrete subclasses give these full, typed
    signatures for their own inputs (see `ShapDiDEstimator`); the ABC only marks that they exist.
    """

    @abstractmethod
    def assumptions(self) -> list[Assumption]:
        """The identifying assumption(s) this estimator relies on, named up front."""
        ...

    @abstractmethod
    def falsify(self, *args: object, **kwargs: object) -> FalsificationReport:
        """Test the observable implications of `assumptions()` — a gate that CAN fail."""
        ...

    @abstractmethod
    def estimate(self, *args: object, **kwargs: object) -> EffectReport:
        """Compute the causal quantity. Must run `falsify()` and attach its report."""
        ...
