SCRAP_THRESHOLD = 0.872   # absolute P cutoff
SEED = 42

N_ROWS = 70_000

# Real data retaiend year
RETAINED_DATA_START_YEAR = 2018

# training window
SYNTH_START_YEAR = 2016
SYNTH_END_YEAR   = 2024

PRE_ML_END_YEAR = 2021

# v1: trained on pre-ML era data
V1_DEPLOY_YEAR  = PRE_ML_END_YEAR + 1  # 2022
V1_TRAIN_START_YEAR = SYNTH_START_YEAR   # = 2016 (our synthetic choice; real dates unknown)
V1_TRAIN_END_YEAR   = PRE_ML_END_YEAR    # = 2021

# v2

# option A - only trained on v1 logs
V2A_TRAIN_START_YEAR = V1_DEPLOY_YEAR     # = 2022
V2A_TRAIN_END_YEAR   = SYNTH_END_YEAR     # = 2024

# option B - research comparison only (includes pre-ml dataset + dropped pre-covid)
V2B_TRAIN_START_YEAR = 2020

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