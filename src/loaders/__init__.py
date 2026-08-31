"""Data loading for notebooks and analysis code — config-driven, never path-driven.

    import sys; sys.path.insert(0, "../src")
    from loaders import load, load_all, scores_wide

    d = load("v2", split="test")   # features/targets/scores exist ONLY per split
    d.frame        # targets + scores on claim_id
    d.tau          # read off the log, with the determinism check
    d.decisions    # that version's own rule applied

Split names are each version's own (config.SPLITS) — v1 and v2 invert what "test" means, and
only v3 has "oot". Pass split="all" (config.ALL_SPLITS) to pool every split; the frame then
carries a `split` column so the pooling is visible wherever the numbers are read.
"""
from loaders.version_data import VersionData, load, load_all, scores_wide

__all__ = ["VersionData", "load", "load_all", "scores_wide"]
