"""Version-agnostic batch scorer — runs INSIDE a version's own env (env-v1/v2/v3).

The active environment + CLI args decide which version is scored; this script has no knowledge of
which version it is running. It loads that version's pickled Pipeline(preprocess + model) and emits
P(total_loss) per claim. Loading the pkl requires that version's repo importable in the active env —
which is exactly why it must run in env-vX (see src/DESIGN.md).

  src/envs/v2/.venv/bin/python src/scoring/predict.py \
      --model src/models/synthetic/baseline/v2.pkl \
      --features src/data/synthetic/inputs/features_v2.parquet \
      --version v2 --out src/data/synthetic/detection/v2_scores.parquet
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)   # claim_id + raw features ONLY (no labels)
    p.add_argument("--version", required=True)    # label only, e.g. "v2"
    p.add_argument("--out", required=True)
    p.add_argument("--id-col", default="claim_id")
    a = p.parse_args()

    df = pd.read_parquet(a.features)
    model = joblib.load(a.model)                  # needs fttl_vX importable in THIS env

    # @TODO: do we need this...? This shoud be in the "pipeline of the .pkl file from v# model pkl file"
    X = df.drop(columns=[a.id_col])               # everything except claim_id → the pipeline
    scores = model.predict_proba(X)[:, 1]         # P(total_loss); pipeline runs preprocess + model

    out = pd.DataFrame({a.id_col: df[a.id_col], f"model_{a.version}_score": scores})
    out.to_parquet(a.out, index=False)
    print(f"[{a.version}] wrote {len(out)} scores -> {a.out}")


if __name__ == "__main__":
    main()
