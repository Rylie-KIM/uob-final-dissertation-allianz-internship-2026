"""Extract a version's feature names from its pickled Pipeline — run INSIDE env-vX.

The names are not readable from any source file: preprocessing is frozen inside the pickle
(README § "Model artefacts and reproduction"), so the only authoritative source is the fitted
pipeline itself. This script is version-agnostic; the active environment decides which version
it is extracting, exactly like src/scoring/predict.py.

Two name sets are written, and they are NOT interchangeable:

  input_features  — raw claim columns the pipeline expects  (pipe.feature_names_in_)
                    -> use for concept-level overlap across versions
  model_features  — columns the booster actually sees, post-preprocessing
                    -> use for SHAP; this is what shap values are indexed by

Usage — one call per version, each run with THAT version's interpreter. The pickle path comes
from src/config.py, so nothing here hard-codes a repo name:

    <v2 python> features/extract_features.py --version v2

Override the path if needed:

    <v2 python> features/extract_features.py --version v2 --model /some/other/v2.pkl

Writes features/registry/<version>.json, which check_overlap.py reads to validate the
hand-confirmed Excel mapping's names against the pickles (the typo check).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import joblib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = pathlib.Path(__file__).parent / "registry"
sys.path.insert(0, str(ROOT / "src"))


def _input_features(pipe) -> list[str]:
    names = getattr(pipe, "feature_names_in_", None)
    if names is None and hasattr(pipe, "steps"):
        names = getattr(pipe.steps[0][1], "feature_names_in_", None)
    return [str(n) for n in names] if names is not None else []


def _model_features(pipe) -> list[str]:
    """Post-preprocessing column names, tried in order of reliability."""
    # 1. the preprocessing head can name its own outputs
    if hasattr(pipe, "__getitem__") and hasattr(pipe, "steps") and len(pipe.steps) > 1:
        head = pipe[:-1]
        if hasattr(head, "get_feature_names_out"):
            try:
                return [str(n) for n in head.get_feature_names_out()]
            except Exception as exc:  # bespoke transformers often do not implement it
                print(f"  get_feature_names_out() unavailable on preprocess head: {exc}")

    # 2. ask the estimator itself
    est = pipe.steps[-1][1] if hasattr(pipe, "steps") else pipe
    if hasattr(est, "get_booster"):
        try:
            names = est.get_booster().feature_names
            if names:
                return [str(n) for n in names]
        except Exception as exc:
            print(f"  booster.feature_names unavailable: {exc}")
    names = getattr(est, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]
    return []


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", required=True, help="our label: v1, v2 or v3")
    p.add_argument("--model", default=None, help="override; defaults to config.model_path(version)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    if a.model:
        model = pathlib.Path(a.model)
    else:
        import config  # resolved from src/
        try:
            model = config.model_path(a.version)
        except ValueError as exc:
            raise SystemExit(f"\n{exc}\n")

    pipe = joblib.load(model)  # needs that version's repo importable in THIS env
    payload = {
        "version": a.version,
        "model_path": str(model),
        "pipeline_repr": type(pipe).__name__,
        "input_features": _input_features(pipe),
        "model_features": _model_features(pipe),
    }

    if not payload["model_features"]:
        print(
            "  WARNING: no model-ready feature names recovered. SHAP output will be "
            "positional only — resolve this before comparing across versions."
        )

    REGISTRY.mkdir(exist_ok=True)
    out = pathlib.Path(a.out) if a.out else REGISTRY / f"{a.version}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[{a.version}] input={len(payload['input_features'])} "
        f"model={len(payload['model_features'])} -> {out}"
    )


if __name__ == "__main__":
    main()

# src\envs\v1\.venv\python.exe          features\extract_features.py --version v1
# src\envs\v2\.venv\Scripts\python.exe  features\extract_features.py --version v2
# src\envs\v3\.venv\Scripts\python.exe  features\extract_features.py --version v3