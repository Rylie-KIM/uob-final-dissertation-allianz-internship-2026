"""MITIGATED retrainer — the one trainer left. Runs INSIDE env-v2 / env-v3.

Fits a fresh estimator on that version's FIXED feature matrix plus a supplied corrected-label
file (claim_id + label + optional weight). The label/weight is the ONLY thing that differs
between the baseline and the mitigated run; preprocessing is held fixed, so the before->after
score change is attributable to the mitigation and not to a preprocessing change (the
re-evaluation invariant).

How preprocessing is held fixed: it is never touched. The real repos pickle the preprocessing
pipeline SEPARATELY from the model (confirmed 2026-07-31) and `features` already holds the
POST-preprocessing matrix, so the mitigated model is fitted on the very same matrix the baseline
was. The invariant holds by construction rather than by careful code.

Hyperparameters are CLONED off the baseline estimator, never written here. Real v2 uses
reg_alpha=20 / scale_pos_weight=4.5 where v1 uses neither, so re-specifying them by hand would
silently change the model between baseline and mitigated — breaking the invariant this script
exists to hold. Note clone() throws the fitted trees away and keeps only the configuration: this
is a fresh fit on corrected labels, NOT a warm start from the baseline's boosters.

MODERN PYTHON (>=3.10) SINCE 2026-09-02. This file used to carry the py3.5 discipline, but v1 is
never retrained (its training data was destroyed; 03_03 refuses the env-v1 kernel), so that
discipline protected an interpreter with no way to reach this code. It now assumes env-v2/env-v3
and is IMPORTABLE: `notebook/real/mitigation/03_03_retrain.ipynb` runs on the version kernel and
calls `retrain()` directly, and the CLI main wraps the same function for the config-aware driver
`training/retrain_all.py` and for `pipeline/pipeline.py`. Paths still arrive as ARGUMENTS, never
from config — the caller resolves them (the driver in the analysis env, the notebook through its
own `import config`), which is the repo's standard driver/worker split.

  src/envs/v2/.venv/Scripts/python.exe src/training/retrain.py \
      --baseline src/models/real/baseline/v2.pkl \
      --features src/data/real/inputs/features_v2_train.parquet \
      --labels   src/data/real/mitigation/v2_corrected_train.parquet \
      --version v2 --out-model src/models/real/mitigated/v2_train.pkl

(In practice run src/training/retrain_all.py instead — it resolves all of those from config.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

# The trained-column-order vocabulary, shared with predict.py and shap_kit.py: scoring and
# fitting must answer "which columns did this model consume" with the SAME function, or
# predict.py can score a matrix retrain.py would not have fitted. Resolved off this file's own
# location, never the working directory, because this worker is launched by a driver from the
# repo root, imported by the 03_03 kernel, and run by hand from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trained_order import select_features  # noqa: E402


def _read_table(path: str | Path) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".csv"):
        return pd.read_csv(p, low_memory=False)
    if p.endswith(".pkl"):
        return pd.read_pickle(p)
    return pd.read_parquet(p)


def retrain(
    baseline: str | Path,
    features: str | Path,
    labels: str | Path,
    version: str,
    out_model: str | Path,
    id_col: str = "claim_id",
    label_col: str = "label",
    weight_col: str = "weight",
    features_json: str | Path | None = None,
) -> dict:
    """Clone the baseline's hyperparameters, fit fresh on the corrected labels, write pkl + meta.

    Returns the meta dict that is also written to `<out_model minus .pkl>_meta.json`. Must run
    inside the version's own env: loading the baseline pkl needs that version's repo importable.
    `features_json` (features/registry/<v>.json) is only consulted when the baseline booster
    exposes no feature names.
    """
    base = joblib.load(baseline)                  # needs that version's repo importable HERE
    if not hasattr(base, "get_params"):
        raise TypeError(
            f"baseline is a {type(base).__name__}, which sklearn.base.clone cannot copy "
            f"(no get_params). This needs the SKLEARN-API estimator (XGBClassifier), not a "
            f"bare xgboost.Booster."
        )

    X = _read_table(features)
    lab = _read_table(labels)

    if id_col not in X.columns:
        raise ValueError(f"features file has no {id_col!r} column")
    if id_col not in lab.columns:
        raise ValueError(f"labels file has no {id_col!r} column")

    feats, how = select_features(X, base, id_col,
                                 None if features_json is None else str(features_json))

    df = X.merge(lab, on=id_col)                  # inner join -> only the corrected (labelled) rows
    if len(df) == 0:
        raise ValueError(
            f"features and labels share no {id_col}. They must describe the SAME split — the "
            f"corrector runs per split, and pairing one split's matrix with another's labels "
            f"is what produces an empty join."
        )
    if len(df) < len(lab):
        print(f"  note: {len(lab) - len(df)} of {len(lab)} corrected rows are absent from features")

    w = df[weight_col] if weight_col in df.columns else None

    # clone() copies the baseline estimator's hyperparameters and nothing else (unfitted), so the
    # mitigated model differs from the baseline in its TARGET alone — never in its configuration.
    model = clone(base)
    model.fit(df[feats], df[label_col], sample_weight=w)

    out_model = Path(out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)                 # estimator only; the preprocessor is untouched

    # The sidecar records what this fit actually consumed. Without it, "the mitigated model" is an
    # unlabelled pickle: which split, which corrected labels, how the feature columns were chosen,
    # and whether weights were applied are all unrecoverable from the file itself.
    meta = {
        "version": version,
        "baseline": str(baseline),
        "features": str(features),
        "labels": str(labels),
        "n_rows": int(len(df)),
        "n_features": len(feats),
        "feature_selection": how,
        "weighted": w is not None,
        "weight_mean": None if w is None else round(float(w.mean()), 4),
        "weight_max": None if w is None else round(float(w.max()), 4),
        "label_pos_rate": round(float(df[label_col].mean()), 6),
        "estimator": type(base).__name__,
    }
    meta_path = str(out_model.with_suffix("")) + "_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    wtxt = "unweighted" if w is None else f"weighted (mean w={float(w.mean()):.2f})"
    print(f"[{version}] retrained on {len(df)} rows x {len(feats)} features "
          f"({wtxt}, feature cols via {how}) -> {out_model}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)   # baseline pkl -> clone its HYPERPARAMETERS
    p.add_argument("--features", required=True)   # that version's processed_inputs (held fixed)
    p.add_argument("--labels", required=True)     # corrected target: claim_id + label (+ weight)
    p.add_argument("--version", required=True)
    p.add_argument("--out-model", required=True)
    p.add_argument("--id-col", default="claim_id")
    p.add_argument("--label-col", default="label")
    p.add_argument("--weight-col", default="weight")
    p.add_argument("--features-json", default=None,
                   help="features/registry/<v>.json, written by features/extract_features.py. "
                        "Only consulted when the baseline booster exposes no feature names.")
    a = p.parse_args()
    try:
        retrain(a.baseline, a.features, a.labels, a.version, a.out_model,
                id_col=a.id_col, label_col=a.label_col, weight_col=a.weight_col,
                features_json=a.features_json)
    except (TypeError, ValueError) as exc:        # clean CLI message, full traceback when imported
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
