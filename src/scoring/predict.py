"""Version-agnostic batch scorer — the MODERN shared worker, runs INSIDE env-v2 / env-v3.

The active environment plus the arguments decide which version is scored; this file has no
knowledge of which version it is running. It loads that version's pickled ESTIMATOR and emits
P(total_loss) per claim. Loading the pkl requires that version's repo importable in the active
env — which is exactly why it must run in env-vX (see src/docs/DESIGN.md).

v2/v3 ONLY. v1 has no twin of this file (`predict_v1.py` deleted 2026-09-17): v1's `scores` kind
is not recomputed at all — `01_export_v1.ipynb` writes `detection/v1_scores_<split>.csv` directly
from `predictions.pkl`, the original 2018 pipeline's own saved predictions, for every split. v1 is
also never retrained (`03_03_retrain.ipynb` refuses the env-v1 kernel), so there is no mitigated
v1 model to score either — nothing left for a v1 scorer to do. The pipeline this project designs
going forward diagnoses/mitigates the ML systems *after* v2/v3, not v1 itself; v1 only ever enters
as the placebo/validity-check side of the v1->v2 SHAP-DiD pair (`estimator/shap_did.py`), which
needs v1's scores as data, never as something this file computes.

PATTERN B SINCE 2026-09-02 (was Pattern A, when a since-deleted py3.5 v1 twin existed). This file
is also IMPORTABLE: 03_03_retrain.ipynb runs on the version kernel and calls `predict()`
directly, and the CLI main wraps the same function for pipeline/pipeline.py.

`features` holds the POST-preprocessing matrix (confirmed 2026-07-31: the real repos pickle the
preprocessor separately from the model, and predict_proba takes the already-transformed columns).
No preprocessing happens here.

WHICH COLUMNS GO IN, AND IN WHAT ORDER. Not "everything except id_col": every version's exported
matrix carries the TARGET beside the model inputs, and v3's also carries its own saved
predictions, so that rule feeds the model its own answer. The columns come from the fitted
estimator (trained_order.select_features — the same function retrain.py fits by, so scoring and
fitting cannot disagree), and they are selected IN TRAINED ORDER: xgboost matches positionally
once names are absent, and raises `feature_names mismatch` when they are present, so a reordered
frame is either silently wrong or a hard stop, never harmless.

  src/envs/v2/.venv/bin/python src/scoring/predict.py \
      --model src/models/real/baseline/v2.pkl \
      --features src/data/real/inputs/features_v2_test.parquet \
      --version v2 --out src/data/real/detection/v2_scores_test.parquet

Run once per version (v2, v3), in that version's own env — there is no all-versions driver.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_io                              # noqa: E402  loader ladder
from trained_order import select_features    # noqa: E402  same rule retrain.py fits by


def read_table(path: str | Path) -> pd.DataFrame:
    """Branch on the extension — the caller's path encodes which artefact twin is wanted."""
    p = str(path)
    if p.endswith(".csv"):
        return pd.read_csv(p, low_memory=False)
    if p.endswith(".pkl"):
        return pd.read_pickle(p)
    return pd.read_parquet(p)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """Match the OUT extension the caller asked for."""
    p = str(path)
    if p.endswith(".csv"):
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)


def predict(
    model: str | Path,
    features: str | Path,
    version: str,
    out: str | Path,
    id_col: str = "claim_id",
    features_json: str | Path | None = None,
) -> pd.DataFrame:
    """Score `features` with `model`, write `<id_col>, model_<version>_score` to `out`, return it.

    Must run inside the version's own env: unpickling the model needs that version's repo
    importable. `features_json` (features/registry/<v>.json) is only consulted when the
    estimator exposes no feature names.
    """
    df = read_table(features)
    est, _loader = model_io.load_estimator(str(model))

    # the model's own columns, in the order it was trained on — selection and order in one step
    feature_cols, source = select_features(df, est, id_col,
                                           None if features_json is None else str(features_json))
    scores = est.predict_proba(df[feature_cols])[:, 1]   # P(total_loss); preprocessed input

    score_col = f"model_{version}_score"
    out_df = pd.DataFrame({id_col: df[id_col].values, score_col: scores})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_table(out_df[[id_col, score_col]], out)
    print(f"[{version}] {len(feature_cols)} features (from the {source}) "
          f"-> wrote {len(out_df)} scores -> {out}")
    return out_df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)   # that version's processed_inputs matrix
    p.add_argument("--version", required=True)    # label only, e.g. "v2"
    p.add_argument("--out", required=True)
    p.add_argument("--id-col", default="claim_id")
    p.add_argument("--features-json", default=None,
                   help="features/registry/<v>.json, written by features/extract_features.py. "
                        "Only consulted when the estimator exposes no feature names.")
    a = p.parse_args()
    predict(a.model, a.features, a.version, a.out,
            id_col=a.id_col, features_json=a.features_json)


if __name__ == "__main__":
    main()
