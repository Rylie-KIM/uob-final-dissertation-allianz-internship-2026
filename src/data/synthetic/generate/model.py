import numpy as np
import pandas as pd
from xgboost import XGBClassifier

FEATURE_COLS = [
    "vehicle_age_years", "mileage", "driver_age", "prior_claims_count",
    "customer_tenure_years", "repair_to_value_ratio", "vehicle_value",
    "repair_estimate_gbp", "report_delay_days",
    "is_weekend_claim", "has_prior_claims", "is_high_value_vehicle", "car_driveable",
]

CATEGORICAL_COLS = [
    "vehicle_make", "vehicle_model", "vehicle_type",
    "damage_type", "damage_location", "damage_severity",
    "agent_channel", "coverage_type",
]

THRESHOLD = 0.90
SEED = 42


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)


def _get_feature_matrix(df_encoded: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    dummy_cols = [c for c in df_encoded.columns if c not in base_cols]
    all_feature_cols = FEATURE_COLS + dummy_cols
    available = [c for c in all_feature_cols if c in df_encoded.columns]
    return df_encoded[available].astype(float)


def _apply_policy(scores: pd.Series) -> pd.Series:
    threshold = scores.quantile(THRESHOLD)
    return (scores >= threshold).astype(int)


def train_and_apply_v1(
    df: pd.DataFrame,
    garage_outcome: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train v1 on pre-ML labels (2016–2021) and apply policy. Returns (df, X_all)."""
    df_enc = _encode_categoricals(df)
    X_all  = _get_feature_matrix(df_enc, list(df.columns))

    # --- Step 4: Train v1 ---
    pre_ml_mask   = pd.to_datetime(df["claim_date"]).dt.year <= 2021
    train_mask_v1 = pre_ml_mask & df["pre_ml_label"].notna()

    model_v1 = XGBClassifier(n_estimators=100, random_state=SEED, eval_metric="logloss")
    model_v1.fit(X_all.loc[train_mask_v1], df.loc[train_mask_v1, "pre_ml_label"].astype(int))

    df["model_v1_score"]    = model_v1.predict_proba(X_all)[:, 1].round(4)
    df["model_v1_decision"] = _apply_policy(df["model_v1_score"])

    # --- Step 5: Apply v1 policy (SFP loop begins) ---
    v1_scrap_mask  = df["model_v1_decision"] == 1
    v1_garage_mask = df["model_v1_decision"] == 0

    observed = pd.Series(np.nan, index=df.index)
    observed[v1_scrap_mask]  = 1
    observed[v1_garage_mask] = garage_outcome[v1_garage_mask]
    df["model_v1_observed_outcome"] = observed.astype("Int64")

    post_ml_mask = ~pre_ml_mask
    df.loc[post_ml_mask & v1_scrap_mask,  "repair_decision"] = "scrapped_by_model"
    df.loc[post_ml_mask & v1_garage_mask, "repair_decision"] = "sent_to_garage"

    return df, X_all


def train_and_apply_v2(
    df: pd.DataFrame,
    X_all: pd.DataFrame,
    option: str = "A",
) -> pd.DataFrame:
    # option A: train on v1 log only (2022+) — full SFP bias
    # option B: train on pre-ML + v1 log combined — partial bias reduction

    if option == "A":
        # same as pre-COVID data dropped 
        v2_mask       = pd.to_datetime(df["claim_date"]).dt.year >= 2022
        train_mask_v2 = v2_mask & df["model_v1_observed_outcome"].notna()
        y_v2          = df.loc[train_mask_v2, "model_v1_observed_outcome"].astype(int)
    elif option == "B":
        pre_ml_mask   = pd.to_datetime(df["claim_date"]).dt.year <= 2021
        pre_ml_label  = df.loc[pre_ml_mask & df["pre_ml_label"].notna(), "pre_ml_label"].astype(int)
        v1_log_mask   = pd.to_datetime(df["claim_date"]).dt.year >= 2022
        v1_log_label  = df.loc[v1_log_mask & df["model_v1_observed_outcome"].notna(), "model_v1_observed_outcome"].astype(int)
        train_mask_v2 = pre_ml_mask & df["pre_ml_label"].notna() | v1_log_mask & df["model_v1_observed_outcome"].notna()
        y_v2          = pd.concat([pre_ml_label, v1_log_label]).sort_index()
    else:
        raise ValueError(f"option must be 'A' or 'B', got {option!r}")

    suffix = option.lower()
    model_v2 = XGBClassifier(n_estimators=100, random_state=SEED, eval_metric="logloss")
    model_v2.fit(X_all.loc[train_mask_v2], y_v2)

    score_col    = f"model_v2{suffix}_score"
    decision_col = f"model_v2{suffix}_decision"
    df[score_col]    = model_v2.predict_proba(X_all)[:, 1].round(4)
    df[decision_col] = _apply_policy(df[score_col])

    return df
