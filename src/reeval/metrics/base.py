"""ReEvalMetric — the re-evaluation axis of the Strategy Pattern (`src/docs/DESIGN.md`
§ "Strategy Pattern"), the `reeval/` sibling of `detector/algorithm/base.py`'s
`DetectionAlgorithm` / `mitigator/corrector/base.py`'s `TrainingDataCorrector` /
`estimator/effect/base.py`'s `EffectEstimator`. Built 2026-09-23 — `reeval/` had sat as `[plan]`
in `src/docs/STRUCTURE.md`/`DESIGN.md` since 2026-07-14 and in the appendix figure
(`report/figures/appx_codebase_architecture.png`, `appx:codebase`) ever since, while its one real
member, `ShapDiDDelta`, already ran as a plain function inside
`notebook/real/shap_did/04_03_shap_did_mitigation_delta.ipynb`.

Distinct from the other three axes by WHAT it reads: a `DetectionAlgorithm` looks at ONE
population, an `EffectEstimator` looks at ONE version pair, a `TrainingDataCorrector` looks at
ONE (features, labels) set — a `ReEvalMetric` re-runs the SAME upstream quantity twice, once on
the contaminated ("before"/baseline) world and once on a Layer-3-corrected ("after"/mitigated)
world, and reports the difference. It exists to answer "did the fix actually reach the
mechanism?", never "is there a problem?" (that is detection) or "how big is the effect?" (that is
estimation) — `DESIGN.md`'s own framing: composition over an existing detector/estimator, not a
fresh measurement.

**No context/wrapper class.** `src/docs/STRUCTURE.md`'s `[plan]` sketch once paired this ABC with
a `reevaluator.py` (`ReEvaluator`, "composes detector/estimator over (before, after)"), mirroring
the `SFPDetector`/`SFPMitigator` context classes `detector/`/`mitigator/` used to have. Both of
those were deleted 2026-09-16 — unused by any real caller, since `pipeline.py` (the one place
that could have used them) calls the concrete strategy directly — and `EffectEstimator` was built
2026-09-16 with no context class from the start for the same reason
(`estimator/effect/base.py`'s docstring). The appendix figure drawn since then shows the same
thing: `ReEvalMetric` floats with no arrow into `SFPPipeline`, unlike `DetectionAlgorithm`/
`TrainingDataCorrector`. So `reevaluator.py` is not built here either — add it only once a real
notebook needs to hold more than one `ReEvalMetric` at a time and calling each directly
(`ShapDiDDelta().compute(...)`) has become the thing actually getting duplicated.

Only `ShapDiDDelta` exists so far (`shap_did_delta.py`), the one member with a real-data notebook
behind it (`subsec:shapdiddelta`, `eq:shapdiddelta`). `DecisionFlipCount` and `DetectionDelta` are
named in `DESIGN.md`'s "Four Strategy Axes" table and the appendix figure as further members but
have no notebook computing them yet on real data — do not add stub classes for them until one
does (the same rule `estimator/effect/base.py` states for `RDDEstimator`/`LogitAdjustEstimator`).
`OracleValidation` needs `true_garage_outcome`, which does not exist in real Allianz data by
construction — it is synthetic-only and out of scope under the REAL DATA ONLY working mode
(`.claude/CLAUDE.md`); never add it here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ReEvalReport:
    """One re-evaluation metric's before/after pair and their difference — never separable, so a
    delta can never be read without the two numbers it was computed from."""

    before: float
    after: float
    delta: float
    diagnostics: dict[str, object] = field(default_factory=dict)


class ReEvalMetric(ABC):
    """`compute()` re-runs the same upstream quantity on a contaminated and a corrected world and
    returns their difference. Concrete subclasses give this a full, typed signature for their own
    inputs (see `ShapDiDDelta`); the ABC only marks that it exists.
    """

    @abstractmethod
    def compute(self, *args: object, **kwargs: object) -> ReEvalReport:
        """Return (before, after, delta) for one before/after comparison."""
        ...
