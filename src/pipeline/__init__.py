"""SFP pipeline orchestration (Analysis Layer — loads no model).

Runtime type-checking: `beartype_this_package()` installs an import hook so every function and
method in this package has its type hints enforced at call time.
"""
from beartype.claw import beartype_this_package

beartype_this_package()

from pipeline.pipeline import SFPPipeline, CycleResult  # noqa: E402

__all__ = ["SFPPipeline", "CycleResult"]
