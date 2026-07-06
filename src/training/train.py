"""Version-agnostic BASELINE trainer — runs INSIDE env-vX like predict.py / retrain.py.

Produces the baseline pkl (src/models/synthetic/baseline/vX.pkl) that reproduces the version's
PRODUCTION scorer. The pkls are not in git, so this rebuilds them locally from the fixed per-version
inputs. Distinct from retrain.py:

  train.py   (this)  : fits prep AND model FRESH from raw features + the production observed label.
                       Nothing is inherited. Output = the baseline production reproduction.
  retrain.py         : REUSES the baseline's already-fitted `prep` verbatim and fits only a new
                       weighted model on corrected labels. Output = the mitigated model.

Preprocessing is the version's own `V{N}FeatureBuilder` (fttl_vX.preprocess), imported dynamically —
which is exactly why this must run in env-vX (its repo must be importable). The output pkl is a full
Pipeline(prep + model) saved together, so predict.py can score it directly from raw features.

  src/envs/v2/.venv/bin/python src/training/train.py \
      --features src/data/synthetic/inputs/features_v2.parquet \
      --labels   src/data/synthetic/inputs/labels_v2.parquet \
      --version v2 --out-model src/models/synthetic/baseline/v2.pkl
"""
from __future__ import annotations

import argparse
import importlib

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def _feature_builder(version: str):
    """Import that version's own preprocessing transformer, e.g. v2 -> fttl_v2.preprocess.V2FeatureBuilder."""
    mod = importlib.import_module(f"fttl_{version}.preprocess")
    return getattr(mod, f"V{version[1:]}FeatureBuilder")()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)   # claim_id + raw features (make, repair_ratio)
    p.add_argument("--labels", required=True)     # claim_id + observed_outcome (the production label)
    p.add_argument("--version", required=True)
    p.add_argument("--out-model", required=True)
    p.add_argument("--id-col", default="claim_id")
    p.add_argument("--label-col", default="observed_outcome")   # train on the PRODUCTION label
    a = p.parse_args()

    X = pd.read_parquet(a.features)
    lab = pd.read_parquet(a.labels)
    df = X.merge(lab[[a.id_col, a.label_col]], on=a.id_col)      # inner join on claim_id
    feats = [c for c in X.columns if c != a.id_col]

    # fresh prep + model, both fitted here (baseline). Same estimator config as the version repos.
    pipe = Pipeline([
        ("prep", _feature_builder(a.version)),
        ("model", XGBClassifier(n_estimators=80, max_depth=3, eval_metric="logloss", random_state=42)),
    ])
    pipe.fit(df[feats], df[a.label_col])

    joblib.dump(pipe, a.out_model)                # full pipeline (preprocess + model) in one pkl
    rate = float(df[a.label_col].mean())
    print(f"[{a.version}] trained baseline on {len(df)} rows "
          f"(pos_rate={rate:.3f}, label={a.label_col}) -> {a.out_model}")


if __name__ == "__main__":
    main()
