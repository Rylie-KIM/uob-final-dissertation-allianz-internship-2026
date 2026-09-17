"""Batch-run notebook/real/00_SHAP.ipynb over every (version, split, axis) combination via papermill.

Company laptop only -- needs the real `mitigated` pickles on disk and the `fttl-v2`/`fttl-v3`
Jupyter kernels registered (the same kernels 00_SHAP.ipynb is already run under by hand). Run this
from any environment that has `papermill` installed (e.g. the analysis .venv: `uv add papermill`
once, then `uv run python scripts/run_00_shap_batch.py`) -- papermill dispatches execution through
each run's own `--kernel`, so the orchestrating environment does not need xgboost/pandas itself.

What one run does: sets 00_SHAP.ipynb's `SPLIT`/`MODEL_PATH` parameters cell, executes the
notebook under that version's kernel, and writes the executed copy to
    notebook/real/00_shap_runs/00_SHAP_<version>_<split>_<tag>.ipynb
(`<tag>` is "baseline" or the mitigation axis) -- open that copy to see exactly what ran, including
any error traceback, without touching the source notebook.

Mitigated axes are discovered by globbing `<version>_train_*.pkl` under the `mitigated` kind's own
directory (config.split_path) rather than a hardcoded list, so a newly retrained axis is picked up
automatically and a removed one is not silently still requested.

Usage:
    python scripts/run_00_shap_batch.py                    # every version, every split, every axis
    python scripts/run_00_shap_batch.py --versions v2       # v2 only
    python scripts/run_00_shap_batch.py --dry-run           # print the papermill commands, run nothing
    python scripts/run_00_shap_batch.py --only-baseline     # skip mitigated axes
    python scripts/run_00_shap_batch.py --only-mitigated    # skip baseline
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "src" / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import config  

KERNELS = {"v2": "fttl-v2", "v3": "fttl-v3"}
NOTEBOOK = ROOT / "notebook" / "real" / "00_SHAP.ipynb"
OUT_DIR = ROOT / "notebook" / "real" / "00_shap_runs"


def mitigated_axes(version: str) -> list[tuple[str, Path]]:
    """[(axis, pkl_path), ...] for every `<version>_train_<axis>.pkl` actually on disk."""
    train_split = config.SPLITS[version][0]
    mitigated_dir = config.split_path("mitigated", version, train_split).parent
    prefix = f"{version}_{train_split}_"
    out = []
    for p in sorted(mitigated_dir.glob(f"{prefix}*.pkl")):
        out.append((p.stem[len(prefix):], p))
    return out


def runs_for(version: str, include_baseline: bool, include_mitigated: bool) -> list[dict]:
    splits = [config.SPLITS[version][0], config.OOT_SPLIT[version]]  # train + OOT, deduped below
    splits = list(dict.fromkeys(splits))
    tags: list[tuple[str, Path | None]] = []
    if include_baseline:
        tags.append(("baseline", None))
    if include_mitigated:
        tags.extend(mitigated_axes(version))
    return [
        {"version": version, "split": split, "tag": tag, "model_path": model_path}
        for split in splits
        for tag, model_path in tags
    ]


def run_one(spec: dict, dry_run: bool) -> None:
    out_path = OUT_DIR / f"00_SHAP_{spec['version']}_{spec['split']}_{spec['tag']}.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "papermill", str(NOTEBOOK), str(out_path),
        "--kernel", KERNELS[spec["version"]],
        "-p", "SPLIT", spec["split"],
    ]
    if spec["model_path"] is not None:
        cmd += ["-p", "MODEL_PATH", str(spec["model_path"])]
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="batch 00_SHAP")
    ap.add_argument("--versions", nargs="+", default=["v2", "v3"], choices=["v2", "v3"])
    ap.add_argument("--only-baseline", action="store_true")
    ap.add_argument("--only-mitigated", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.only_baseline and args.only_mitigated:
        ap.error("--only-baseline and --only-mitigated are mutually exclusive")
    include_baseline = not args.only_mitigated
    include_mitigated = not args.only_baseline

    specs = [
        s
        for v in args.versions
        for s in runs_for(v, include_baseline, include_mitigated)
    ]
    if not specs:
        print("nothing to run")
        return
    print(f"{len(specs)} run(s) queued\n")

    failed = []
    for i, spec in enumerate(specs, 1):
        print(f"[{i}/{len(specs)}] {spec['version']} / {spec['split']} / {spec['tag']}")
        try:
            run_one(spec, args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"  FAILED (exit {exc.returncode}) -- see the executed copy under {OUT_DIR} "
                  f"for the traceback")
            failed.append(spec)
        print()

    if failed:
        print(f"{len(failed)}/{len(specs)} run(s) failed:")
        for s in failed:
            print(f"  - {s['version']} / {s['split']} / {s['tag']}")
        sys.exit(1)
    print("all runs complete")


if __name__ == "__main__":
    main()
