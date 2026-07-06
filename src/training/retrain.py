"""Version-agnostic RETRAINER for the re-evaluation step — runs INSIDE env-vX like predict.py.

Fits a fresh model on that version's FIXED feature file plus a supplied corrected-label file
(claim_id + label + optional weight). The label/weight is the ONLY thing that differs between the
baseline and the mitigated run; preprocessing is held fixed, so the before->after score change is
attributable to the mitigation, not to a preprocessing change (the re-evaluation invariant).

How preprocessing is held fixed: we load the baseline pipeline and REUSE its already-fitted
`prep` transformer verbatim (never re-fit it), then fit only a new weighted model on top. The
output pkl is a full Pipeline(prep + model) — preprocessing and model saved TOGETHER — so
predict.py can score it directly from raw features.

  src/envs/v2/.venv/bin/python src/training/retrain.py \
      --baseline src/models/synthetic/baseline/v2.pkl \
      --features src/data/synthetic/inputs/features_v2.parquet \
      --labels   src/data/synthetic/mitigation/v2_corrected.parquet \
      --version v2 --out-model src/models/synthetic/mitigated/v2.pkl
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)   # baseline pkl → reuse its FITTED preprocessing
    p.add_argument("--features", required=True)   # that version's preprocessed-input X (held fixed)
    p.add_argument("--labels", required=True)     # corrected target: claim_id + label (+ weight)
    p.add_argument("--version", required=True)
    p.add_argument("--out-model", required=True)
    p.add_argument("--id-col", default="claim_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--weight-col", default="weight")
    a = p.parse_args()

    base = joblib.load(a.baseline)                # needs fttl_vX importable in THIS env
    prep = base.named_steps["prep"]               # ALREADY fitted on baseline data — reuse verbatim

    X = pd.read_parquet(a.features)
    lab = pd.read_parquet(a.labels)
    df = X.merge(lab, on=a.id_col)                # inner join → only the corrected (labelled) rows
    feats = [c for c in X.columns if c != a.id_col]

    Xt = prep.transform(df[feats])                # transform with the FIXED preprocessing
    w = df[a.weight_col] if a.weight_col in df.columns else None

    model = XGBClassifier(n_estimators=80, max_depth=3, eval_metric="logloss", random_state=42)
    model.fit(Xt, df[a.label_col], sample_weight=w)

    mitigated = Pipeline([("prep", prep), ("model", model)])   # fixed prep + new weighted model
    joblib.dump(mitigated, a.out_model)           # full pipeline (preprocess + model) in one pkl
    wtxt = "unweighted" if w is None else f"weighted (mean w={float(w.mean()):.2f})"
    print(f"[{a.version}] retrained on {len(df)} rows ({wtxt}) -> {a.out_model}")


if __name__ == "__main__":
    main()
