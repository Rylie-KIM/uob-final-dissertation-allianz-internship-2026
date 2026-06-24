# Step 3 — Simulate pre-ML era human decisions (2016–2021).
import numpy as np
import pandas as pd


def simulate_pre_ml_era(df: pd.DataFrame, garage_outcome: pd.Series) -> pd.DataFrame:
    is_pre_ml = pd.to_datetime(df["claim_date"]).dt.year <= 2021

    # Mimick human agent (handler) decision 
    rule_a = df["repair_to_value_ratio"] > 0.9
    rule_b = (df["damage_severity"] == "severe") & (df["vehicle_age_years"] > 15)
    scrapped_by_handler = rule_a | rule_b

    pre_ml_decision = pd.Series(np.nan, index=df.index)
    pre_ml_label    = pd.Series(np.nan, index=df.index)

    pre_ml_mask = is_pre_ml

    # Scrapped by handler 
    # label forced to 1 (self-fulfilling)
    scrap_mask = pre_ml_mask & scrapped_by_handler
    pre_ml_decision[scrap_mask] = 1
    pre_ml_label[scrap_mask]    = 1

    # Sent to garage 
    # label = actual garage outcome
    garage_mask = pre_ml_mask & ~scrapped_by_handler
    pre_ml_decision[garage_mask] = 0
    pre_ml_label[garage_mask]    = garage_outcome[garage_mask]

    df["pre_ml_decision"] = pre_ml_decision
    df["pre_ml_label"]    = pre_ml_label

    return df
