"""Split each version's canonical ingested log into the per-version scoring contract.

ingest.py has already archived every version's emitted log under the canonical name
logs/<version>.parquet (translating whatever name the model app chose). This step reads only that
canonical log — never a source name — and splits it into TWO files per version, because predict.py
must never see the labels:

  inputs/features_v{k}.parquet : claim_id + raw features   (the model-scoring INPUT)
  inputs/labels_v{k}.parquet   : claim_id + observed_outcome + decision + true_garage_outcome

On real data these come from different places (features from the version's preprocessing, labels
from production + the garage). Here we split one log into the two; downstream code is identical.

  python src/scoring/ingest.py        # first: source log -> logs/<version>.parquet
  python src/scoring/build_inputs.py  # then:  logs/<version>.parquet -> inputs/
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW_FEATURES = ["make", "repair_ratio"]            # the stable raw schema (per-version preprocessing diverges)
LOGS_DIR = ROOT / "src/data/synthetic/logs"        # canonical ingested logs (populated by ingest.py)
# features + labels are the fixed per-version "inputs" (shared material for scoring & retrain)
INPUTS_DIR = ROOT / "src/data/synthetic/inputs"


def main() -> None:
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for v in ("v1", "v2", "v3"):
        log_path = LOGS_DIR / f"{v}.parquet"
        log = pd.read_parquet(log_path)

        features = log[["claim_id", *RAW_FEATURES]]
        features.to_parquet(INPUTS_DIR / f"features_{v}.parquet", index=False)

        labels = log[["claim_id", "observed_outcome", "true_garage_outcome"]].copy()
        labels["decision"] = log[f"model_{v}_decision"]
        labels.to_parquet(INPUTS_DIR / f"labels_{v}.parquet", index=False)

        print(f"[{v}] features={len(features)} rows, labels: "
              f"scrap_rate={labels['decision'].mean():.3f} -> features_{v}.parquet + labels_{v}.parquet")


if __name__ == "__main__":
    main()
