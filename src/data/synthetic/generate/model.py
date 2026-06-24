import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from generate.config import (
    SCRAP_THRESHOLD,
    SEED,
    MATURATION_BUFFER_MONTHS,
    OOT_MONTHS,
    MIN_TRAIN_ROWS,
    ModelSpec,
    FEATURE_COLS,
    CATEGORICAL_COLS,
)
from generate.imputer import EnrichmentImputer


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)


def _get_feature_matrix(df_encoded: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    dummy_cols = [c for c in df_encoded.columns if c not in base_cols]
    all_feature_cols = FEATURE_COLS + dummy_cols
    available = [c for c in all_feature_cols if c in df_encoded.columns]
    return df_encoded[available].astype(float)


def _apply_threshold(scores: pd.Series) -> pd.Series:
    return (scores >= SCRAP_THRESHOLD).astype(int)


def _train_cutoff(end_year: int) -> pd.Timestamp:
    era_end = pd.Timestamp(year=end_year, month=12, day=31)
    return era_end - pd.DateOffset(months=MATURATION_BUFFER_MONTHS + OOT_MONTHS)


def _build_train_mask(
    df: pd.DataFrame,
    dates: pd.Series,
    spec: ModelSpec,
    train_cutoff: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns (train_mask, y_train) built from all segments in spec.
    Segments with end_year_cap use a hard year ceiling; others use train_cutoff.
    """
    masks, ys = [], []

    for seg in spec.segments:
        mask = dates.dt.year >= seg.start_year

        if seg.end_year_cap is not None:
            mask &= dates.dt.year <= seg.end_year_cap
        else:
            mask &= dates <= train_cutoff

        mask &= df[seg.target_col].notna()
        masks.append(mask)
        ys.append(df.loc[mask, seg.target_col].astype(int))

    combined_mask = masks[0]
    for m in masks[1:]:
        combined_mask |= m

    y_train = pd.concat(ys).sort_index()
    return combined_mask, y_train


def train_and_apply(
    df: pd.DataFrame,
    spec: ModelSpec,
    garage_outcome: pd.Series,
    X_all: pd.DataFrame | None = None,
    enrichment_table: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, EnrichmentImputer | None]:
    """
    Train an XGBoost triage model according to spec and score the full dataset.

    For the first model (X_all=None), pass enrichment_table so the imputer
    can be fitted on the training window and X_all computed once.
    Subsequent models reuse the same X_all.

    Returns (df, X_all, imputer).  imputer is None for v2/v3 calls.
    """
    dates        = pd.to_datetime(df["claim_date"])
    train_cutoff = _train_cutoff(spec.cutoff_end_year)

    train_mask, y_train = _build_train_mask(df, dates, spec, train_cutoff)

    n_train = train_mask.sum()
    if n_train < MIN_TRAIN_ROWS:
        raise ValueError(
            f"{spec.tag} training set has {n_train:,} rows — minimum is {MIN_TRAIN_ROWS:,}."
        )

    imputer = None
    if X_all is None:
        if enrichment_table is None:
            raise ValueError("enrichment_table is required when X_all has not been computed yet.")
        imputer = EnrichmentImputer()
        imputer.fit(enrichment_table, df.loc[train_mask])
        df = imputer.transform(df)
        df_enc = _encode_categoricals(df)
        X_all  = _get_feature_matrix(df_enc, list(df.columns))

    model = XGBClassifier(n_estimators=100, random_state=SEED, eval_metric="logloss")
    model.fit(X_all.loc[train_mask], y_train)

    score_col    = f"model_{spec.tag}_score"
    decision_col = f"model_{spec.tag}_decision"
    outcome_col  = f"model_{spec.tag}_observed_outcome"

    df[score_col]    = model.predict_proba(X_all)[:, 1].round(4)
    df[decision_col] = _apply_threshold(df[score_col])

    scrap_mask  = df[decision_col] == 1
    garage_mask = df[decision_col] == 0
    observed = pd.Series(np.nan, index=df.index)
    observed[scrap_mask]  = 1
    observed[garage_mask] = garage_outcome[garage_mask]
    df[outcome_col] = observed.astype("Int64")

    return df, X_all, imputer
