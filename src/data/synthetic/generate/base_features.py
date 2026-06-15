# Step 1 — Generate base claim features for all 10,000 rows.

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 10_000

VEHICLE_MODELS = {
    "Ford":       ["Fiesta", "Focus", "Mondeo", "Kuga", "Transit"],
    "BMW":        ["1 Series", "3 Series", "5 Series", "X3", "X5"],
    "Toyota":     ["Yaris", "Corolla", "RAV4", "Camry"],
    "Volkswagen": ["Polo", "Golf", "Passat", "Tiguan"],
    "Vauxhall":   ["Corsa", "Astra", "Mokka", "Insignia"],
    "Audi":       ["A3", "A4", "A6", "Q3", "Q5"],
    "Honda":      ["Jazz", "Civic", "HR-V", "CR-V"],
    "Mercedes":   ["A-Class", "C-Class", "E-Class", "GLC"],
    "Nissan":     ["Micra", "Juke", "Qashqai", "Leaf"],
    "Hyundai":    ["i10", "i20", "i30", "Tucson"],
}

DAMAGE_TYPES     = ["collision", "flood", "fire", "vandalism", "theft_damage"]
DAMAGE_LOCATIONS = ["front", "rear", "side", "roof", "multiple"]
DAMAGE_SEVERITIES = ["minor", "moderate", "severe"]
AGENT_CHANNELS   = ["online", "broker", "direct", "app", "phone"]
COVERAGE_TYPES   = ["third_party", "third_party_fire_theft", "comprehensive"]
VEHICLE_TYPES    = ["Sedan", "SUV", "Hatchback", "Estate", "Van", "Motorbike"]


def _generate_identifiers(rng: np.random.Generator) -> pd.DataFrame:
    claim_ids = [f"CLM{i:06d}" for i in range(1, N_ROWS + 1)]

    n_unique_policies = 7_000
    policy_pool = [f"POL{i:06d}" for i in range(1, n_unique_policies + 1)]
    # ~75% appear once, ~20% twice, ~5% three or more times
    weights = np.array([0.75] * int(n_unique_policies * 0.75)
                       + [0.20] * int(n_unique_policies * 0.20)
                       + [0.05] * (n_unique_policies - int(n_unique_policies * 0.75) - int(n_unique_policies * 0.20)))
    weights = weights[:n_unique_policies]
    weights /= weights.sum()
    policy_ids = rng.choice(policy_pool, size=N_ROWS, p=weights)

    return pd.DataFrame({"claim_id": claim_ids, "policy_id": policy_ids})


def _generate_dates(rng: np.random.Generator) -> pd.DataFrame:
    start = pd.Timestamp("2016-01-01")
    end   = pd.Timestamp("2024-12-31")
    total_days = (end - start).days


    day_offsets = rng.integers(0, total_days, size=N_ROWS)
    claim_dates = pd.to_datetime([start + pd.Timedelta(days=int(d)) for d in day_offsets])

    # incident_date <= claim_date; delay skewed toward 0
    delay_days = rng.integers(0, 31, size=N_ROWS)
    incident_dates = claim_dates - pd.to_timedelta(delay_days, unit="D")

    # policy_start_date: 2012-01-01 to claim_date
    policy_start_offsets = rng.integers(0, (claim_dates - pd.Timestamp("2012-01-01")).days.values.clip(min=1))
    policy_start_dates = pd.to_datetime([
        pd.Timestamp("2012-01-01") + pd.Timedelta(days=int(o))
        for o in policy_start_offsets
    ])
    policy_expiry_dates = policy_start_dates + pd.DateOffset(days=365)

    return pd.DataFrame({
        "claim_date":         claim_dates.strftime("%Y-%m-%d"),
        "incident_date":      incident_dates.strftime("%Y-%m-%d"),
        "policy_start_date":  policy_start_dates.strftime("%Y-%m-%d"),
        "policy_expiry_date": [d.strftime("%Y-%m-%d") for d in policy_expiry_dates],
        "report_delay_days":  delay_days,
    })


def _generate_categoricals(rng: np.random.Generator) -> pd.DataFrame:
    makes  = rng.choice(list(VEHICLE_MODELS.keys()), size=N_ROWS)
    models = np.array([
        rng.choice(VEHICLE_MODELS[make]) for make in makes
    ])

    return pd.DataFrame({
        "vehicle_make":     makes,
        "vehicle_model":    models,
        "vehicle_type":     rng.choice(VEHICLE_TYPES, size=N_ROWS),
        "damage_type":      rng.choice(DAMAGE_TYPES, size=N_ROWS),
        "damage_location":  rng.choice(DAMAGE_LOCATIONS, size=N_ROWS),
        "damage_severity":  rng.choice(DAMAGE_SEVERITIES, size=N_ROWS, p=[0.4, 0.4, 0.2]),
        "agent_channel":    rng.choice(AGENT_CHANNELS, size=N_ROWS),
        "coverage_type":    rng.choice(COVERAGE_TYPES, size=N_ROWS, p=[0.2, 0.2, 0.6]),
    })


def _generate_booleans(rng: np.random.Generator, claim_dates: pd.Series) -> pd.DataFrame:
    dates = pd.to_datetime(claim_dates)
    is_weekend = dates.dt.dayofweek >= 5

    prior_claims_count = rng.poisson(lam=0.8, size=N_ROWS)

    return pd.DataFrame({
        "is_weekend_claim":   is_weekend.values,
        "has_prior_claims":   prior_claims_count > 0,
        "car_driveable":      rng.random(N_ROWS) > 0.35,
        # is_high_value_vehicle is derived after enrichment join
    })


def _generate_numericals(rng: np.random.Generator) -> pd.DataFrame:
    vehicle_age = rng.integers(0, 21, size=N_ROWS) 
    mileage     = rng.integers(0, 200_001, size=N_ROWS)
    driver_age  = np.clip(rng.normal(loc=42, scale=15, size=N_ROWS).astype(int), 17, 85)
    prior_claims_count   = rng.poisson(lam=0.8, size=N_ROWS)
    customer_tenure      = rng.integers(0, 21, size=N_ROWS)

    return pd.DataFrame({
        "vehicle_age_years":    vehicle_age,
        "mileage":              mileage,
        "driver_age":           driver_age,
        "prior_claims_count":   prior_claims_count,
        "customer_tenure_years": customer_tenure,
    })


def generate_base_features() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    identifiers  = _generate_identifiers(rng)
    dates        = _generate_dates(rng)
    categoricals = _generate_categoricals(rng)
    booleans     = _generate_booleans(rng, dates["claim_date"])
    numericals   = _generate_numericals(rng)

    df = pd.concat([identifiers, dates, categoricals, booleans, numericals], axis=1)

    # manufacture_year derived here; used as enrichment join key
    df["manufacture_year"] = pd.to_datetime(df["claim_date"]).dt.year - df["vehicle_age_years"]

    return df
