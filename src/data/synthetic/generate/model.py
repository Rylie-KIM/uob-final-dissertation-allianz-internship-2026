import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from generate.config import (
    SCRAP_THRESHOLD,
    SEED,
    MATURATION_BUFFER_MONTHS,
    OOT_MONTHS,
    MIN_TRAIN_ROWS,
    TARGET_PRECISION,
    VALIDATION_FRAC,
    ModelSpec,
    FEATURE_COLS,
    CATEGORICAL_COLS,
)
from generate.imputer import EnrichmentImputer
from generate.logger import get_logger

log = get_logger(__name__)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)


def _get_feature_matrix(df_encoded: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    dummy_cols = [c for c in df_encoded.columns if c not in base_cols]
    all_feature_cols = FEATURE_COLS + dummy_cols
    available = [c for c in all_feature_cols if c in df_encoded.columns]
    return df_encoded[available].astype(float)


def _apply_threshold(scores: pd.Series, tau: float) -> pd.Series:
    return (scores >= tau).astype(int)


def _tune_threshold(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    target: float = TARGET_PRECISION,
    fallback: float = SCRAP_THRESHOLD,
) -> float:
    """
    τ_v = the lowest (most permissive) score cutoff whose precision on the held-out
    validation slice first reaches `target`, scored against that version's
    (SFP-contaminated) training label. Falls back to SCRAP_THRESHOLD (0.872) when the
    target is unreachable.

    precision_recall_curve returns thresholds in increasing order with precision[i]
    aligned to threshold[i]; the first index clearing `target` is therefore the lowest
    threshold that still holds the precision constraint. This is the identical rule to
    evaluate.find_precision_threshold — the single source of truth for the scrap policy.
    """
    if len(np.unique(y_val)) < 2:
        return fallback
    prec, _rec, thresholds = precision_recall_curve(y_val, scores_val)
    for p, t in zip(prec, thresholds):
        if p >= target:
            return float(t)
    return fallback


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

    # Hold out a validation slice from the training window to tune τ_v out-of-sample.
    # The model is fit on the remaining rows and reused as-is for final scoring, so the
    # tuned threshold stays calibrated to the exact model that produces the scores.
    X_train = X_all.loc[train_mask]
    y_train = y_train.reindex(X_train.index)
    strat   = y_train if y_train.nunique() > 1 else None
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=VALIDATION_FRAC,
        random_state=SEED,
        stratify=strat,
    )

    model = XGBClassifier(n_estimators=100, random_state=SEED, eval_metric="logloss")
    model.fit(X_fit, y_fit)

    # τ_v: lowest cutoff holding precision >= TARGET_PRECISION on the validation slice
    # (scored against the contaminated training label); 0.872 fallback if unreachable.
    val_scores = model.predict_proba(X_val)[:, 1]
    tau_v      = _tune_threshold(y_val.to_numpy(), val_scores)
    log.info(
        f"[{spec.tag}] tuned tau_v = {tau_v:.4f} "
        f"(target precision {TARGET_PRECISION}, fallback {SCRAP_THRESHOLD})"
    )

    score_col    = f"model_{spec.tag}_score"
    decision_col = f"model_{spec.tag}_decision"
    outcome_col  = f"model_{spec.tag}_observed_outcome"

    df[score_col]    = model.predict_proba(X_all)[:, 1].round(4)
    df[decision_col] = _apply_threshold(df[score_col], tau_v)

    scrap_mask  = df[decision_col] == 1
    garage_mask = df[decision_col] == 0
    observed = pd.Series(np.nan, index=df.index)
    observed[scrap_mask]  = 1
    observed[garage_mask] = garage_outcome[garage_mask]
    df[outcome_col] = observed.astype("Int64")

    return df, X_all, imputer
