from dataclasses import dataclass

SCRAP_THRESHOLD = 0.872   # absolute P cutoff
SEED = 42

N_ROWS = 70_000

# Real data retaiend year
RETAINED_DATA_START_YEAR = 2018

# project's synthetic choice (real dates unknown)
SYNTH_START_YEAR = 2016
SYNTH_END_YEAR   = 2024

PRE_ML_END_YEAR = 2021

# v1: trained on pre-ML era data
V1_DEPLOY_YEAR      = PRE_ML_END_YEAR + 1  # 2022
V1_TRAIN_START_YEAR = SYNTH_START_YEAR     # 2016
V1_TRAIN_END_YEAR   = PRE_ML_END_YEAR      # 2021

# v2
V2A_TRAIN_START_YEAR = V1_DEPLOY_YEAR   # 2022
V2A_TRAIN_END_YEAR   = SYNTH_END_YEAR   # 2024
V2B_TRAIN_START_YEAR = 2020             # includes pre-covid pre-ML data

# v3
V3_TRAIN_START_YEAR = 2023
V3_TRAIN_END_YEAR   = SYNTH_END_YEAR

# ML training window logic
MATURATION_BUFFER_MONTHS = 2   # exclude most recent N months
OOT_MONTHS               = 6   # hold out N months

# min dataset size for XGBoost model (chosen based on paper s1)
MIN_TRAIN_ROWS = 10_000

# Evaluation
TARGET_PRECISION = 0.985

# Features used for model training
FEATURE_COLS = [
    "vehicle_age_years", "mileage", "driver_age", "prior_claims_count",
    "customer_tenure_years", "repair_to_value_ratio", "vehicle_value",
    "repair_estimate_gbp", "report_delay_days", "used_car_price_index",
    "is_weekend_claim", "has_prior_claims", "is_high_value_vehicle", "car_driveable",
]

CATEGORICAL_COLS = [
    "vehicle_make", "vehicle_model", "vehicle_type",
    "damage_type", "damage_location", "damage_severity",
    "agent_channel", "coverage_type",
]


# Model version specs of training dataset 
@dataclass
class TrainingSegment:
    start_year: int
    target_col: str
    end_year_cap: int | None = None


@dataclass
class ModelSpec:
    tag: str                        # model_{tag}_score 
    cutoff_end_year: int            # year passed to _train_cutoff()
    segments: list[TrainingSegment]


V1_SPEC = ModelSpec(
    tag="v1",
    cutoff_end_year=V1_TRAIN_END_YEAR,
    segments=[
        TrainingSegment(V1_TRAIN_START_YEAR, "pre_ml_label"),
    ],
)

V2A_SPEC = ModelSpec(
    tag="v2a",
    cutoff_end_year=V2A_TRAIN_END_YEAR,
    segments=[
        TrainingSegment(V2A_TRAIN_START_YEAR, "model_v1_observed_outcome"),
    ],
)

# v2B mixes pre-ML data (2020-2021) with the v1 log (2022+) to dilute SFP.
V2B_SPEC = ModelSpec(
    tag="v2b",
    cutoff_end_year=V2A_TRAIN_END_YEAR,
    segments=[
        TrainingSegment(V2B_TRAIN_START_YEAR, "pre_ml_label", end_year_cap=PRE_ML_END_YEAR), #2021 
        TrainingSegment(V2A_TRAIN_START_YEAR, "model_v1_observed_outcome"),
    ],
)

V3A_SPEC = ModelSpec(
    tag="v3a",
    cutoff_end_year=V3_TRAIN_END_YEAR,
    segments=[
        TrainingSegment(V3_TRAIN_START_YEAR, "model_v2a_observed_outcome"),
    ],
)

V3B_SPEC = ModelSpec(
    tag="v3b",
    cutoff_end_year=V3_TRAIN_END_YEAR,
    segments=[
        TrainingSegment(V3_TRAIN_START_YEAR, "model_v2b_observed_outcome"),
    ],
)
