"""Synthetic-data generation package (runs in the Analysis env — xgboost et al.).

Runtime type-checking: `beartype_this_package()` installs an import hook so every function and
method in this package has its type hints enforced at call time.
"""
from beartype.claw import beartype_this_package

beartype_this_package()
