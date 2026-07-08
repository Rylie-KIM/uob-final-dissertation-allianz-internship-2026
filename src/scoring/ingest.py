"""Ingest each version's emitted production log into the canonical logs/ landing zone.

The model application for a version emits its log under whatever name IT chose
(model_repos/.../log_v2.parquet here; on real data e.g. v2_prod_scored_2025Q1.parquet). This
pipeline, though, expects one canonical per-version name: logs/<version>.parquet. ingest.py is the
SINGLE place that translates 'their name' -> 'our name': it reads the source path from
logs/manifest.json, validates the schema, and archives a copy as logs/<version>.parquet.

Everything downstream (build_inputs.py) reads only the canonical logs/, never the source name. So
when a file name changes, you edit manifest.json alone — no pipeline code changes.

  python src/scoring/ingest.py
"""
from __future__ import annotations

import json
import pathlib
import shutil

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT / "src/data/synthetic/logs"
MANIFEST = LOGS_DIR / "manifest.json"


def _required_cols(v: str) -> list[str]:
    # the columns build_inputs.py needs; per-version score/decision are named model_<v>_*
    return ["claim_id", "observed_outcome", "true_garage_outcome", f"model_{v}_decision"]


def main() -> None:
    sources = json.loads(MANIFEST.read_text())["sources"]

    for v, rel in sources.items():
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(f"[{v}] manifest points at a missing log: {src}")

        cols = pd.read_parquet(src).columns
        missing = [c for c in _required_cols(v) if c not in cols]
        if missing:
            raise ValueError(f"[{v}] {src.name} is missing required columns: {missing}")

        dst = LOGS_DIR / f"{v}.parquet"
        shutil.copyfile(src, dst)                 # archive verbatim under the canonical name
        print(f"[{v}] {src.name}  ->  logs/{v}.parquet")

    print(f"Ingested {len(sources)} logs -> {LOGS_DIR}")


if __name__ == "__main__":
    main()
