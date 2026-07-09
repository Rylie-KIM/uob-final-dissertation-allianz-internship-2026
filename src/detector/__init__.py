"""SFP detection (Analysis Layer — loads no model).

Public API: `SFPDetector` (holds a strategy, returns `DetectionReport`) + the pluggable
`DetectionAlgorithm` strategies under `detector.algorithm`.

Runtime type-checking: `beartype_this_package()` installs an import hook so every function and
method in this package (incl. `detector.algorithm`) has its type hints enforced at call time.
"""
from beartype.claw import beartype_this_package

beartype_this_package()

from detector.algorithm import DetectionAlgorithm, ResidualPeakAlgorithm  # noqa: E402
from detector.sfp_detector import DetectionReport, SFPDetector  # noqa: E402

__all__ = ["SFPDetector", "DetectionReport", "DetectionAlgorithm", "ResidualPeakAlgorithm"]
