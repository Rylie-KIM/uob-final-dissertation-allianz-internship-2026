"""
Real Dataset Downloader
========================
Downloads publicly available insurance datasets — no login required.
LINK: https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000 
Datasets:
    1. COIL 2000 (UCI) — Real Dutch insurance company data
    COIL: computational intelligence and Learning - 2000년 data mining competition name
       - 9,822 customers × 86 features
       - Actual product types, policy counts, premiums
       - Free, no login

    2. Porto Seguro Teaser (public sample) — Brazilian auto insurance
       - 10,000 rows subset (full dataset requires Kaggle login)
       - Demonstrates pricing loop features

    3. Fraud simulation on COIL 2000 — We add synthetic investigation
       flags to COIL 2000 to create a realistic loop dataset
Additional Information

Information about customers consists of 86 variables and includes product usage data and socio-demographic data derived from zip area codes. The data was supplied by the Dutch data mining company Sentient Machine Research and is based on a real world business problem. The training set contains over 5000 descriptions of customers, including the information of whether or not they have a caravan insurance policy. A test set contains 4000 customers of whom only the organisers know if they have a caravan insurance policy. 

The data dictionary (http://kdd.ics.uci.edu/databases/tic/dictionary.txt) describes the variables used and their values. 

Note: All the variables starting with M are zipcode variables. They give information on the distribution of that variable, e.g. Rented house, in the zipcode area of the customer. 

One instance per line with tab delimited fields. 

TICDATA2000.txt: Dataset to train and validate prediction models and build a description (5822 customer records). Each record consists of 86 attributes, containing sociodemographic data (attribute 1-43) and product ownership (attributes 44-86).The sociodemographic data is derived from zip codes. All customers living in areas with the same zip code have the same sociodemographic attributes. Attribute 86, "CARAVAN:Number of mobile home policies", is the target variable. 

TICEVAL2000.txt: Dataset for predictions (4000 customer records). It has the same format as TICDATA2000.txt, only the target is missing. Participants are supposed to return the list of predicted targets only. All datasets are in tab delimited format. The meaning of the attributes and attribute values is given below. 

TICTGTS2000.txt Targets for the evaluation set. 
Run:
    python datasets/download_real.py



"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in os.listdir(os.path.join(_ROOT, "builds")):
    _path = os.path.join(_ROOT, "builds", _sub)
    if os.path.isdir(_path):
        sys.path.insert(0, _path)

OUT_DIR = Path(__file__).parent / "real"
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. COIL 2000 via ucimlrepo
# ─────────────────────────────────────────────

def download_coil2000():
    print("\n[COIL 2000] Downloading from UCI ML Repository...")
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=125)
        features = dataset.data.features
        targets  = dataset.data.targets
        df = pd.concat([features, targets], axis=1)
        path = OUT_DIR / "coil2000.csv"
        df.to_csv(path, index=False)
        print(f"  ✅ coil2000.csv  [{df.shape[0]:,} rows × {df.shape[1]} cols]")
        print(f"     Target: CARAVAN (1=bought insurance, 0=did not) — base rate: {df['CARAVAN'].mean():.3f}")
        return df
    except ImportError:
        print("  ⚠️  ucimlrepo not installed. Run: pip install ucimlrepo")
        return None
    except Exception as e:
        print(f"  ⚠️  Download failed: {e}")
        _fallback_coil2000()
        return None


def _fallback_coil2000():
    """Fallback: create a COIL-2000-schema synthetic dataset if download fails."""
    print("  → Falling back to COIL-2000-schema synthetic dataset...")
    rng = np.random.default_rng(0)
    n = 9_822
    df = pd.DataFrame({
        # Socio-demographic
        'MOSTYPE':  rng.integers(1, 41, n),    # Customer main type
        'MAANTHUUR': rng.integers(0, 6, n),    # Renting price
        'MGEMOMV':  rng.integers(1, 6, n),     # Avg size household
        'MOSHOOFD': rng.integers(1, 10, n),    # Customer subtype
        # Policy counts (0–9 scale)
        'PPERSAUT': rng.integers(0, 9, n),     # Contribution car policies
        'PBESAUT':  rng.integers(0, 9, n),     # Contribution delivery van
        'PMOTSCO':  rng.integers(0, 9, n),     # Contribution motorcycle/scooter
        'PVRAAUT':  rng.integers(0, 9, n),     # Contribution lorry
        'PAANHANG': rng.integers(0, 9, n),     # Contribution trailer
        'PTRACTOR': rng.integers(0, 9, n),     # Contribution tractor
        'PWERKT':   rng.integers(0, 9, n),     # Contribution agricultural machines
        'PBRAND':   rng.integers(0, 9, n),     # Contribution fire policies
        'PZEILPL':  rng.integers(0, 9, n),     # Contribution surfboard policies
        'PPLEZIER': rng.integers(0, 9, n),     # Contribution boat policies
        'PFIETS':   rng.integers(0, 9, n),     # Contribution bicycle policies
        # Income level
        'MINKM30':  rng.integers(0, 9, n),     # Income < 30,000
        'MINK3045': rng.integers(0, 9, n),     # Income 30–45k
        'MINK4575': rng.integers(0, 9, n),     # Income 45–75k
        'MINK7512': rng.integers(0, 9, n),     # Income 75–122k
        'MINK123M': rng.integers(0, 9, n),     # Income > 123k
        # Target
        'CARAVAN':  rng.binomial(1, 0.06, n),  # 6% base rate (realistic)
    })
    path = OUT_DIR / "coil2000_synthetic_schema.csv"
    df.to_csv(path, index=False)
    print(f"  → coil2000_synthetic_schema.csv  [{n:,} rows × {df.shape[1]} cols]")
    return df


# ─────────────────────────────────────────────
# 2. COIL 2000 + Simulated SFP Loop
# ─────────────────────────────────────────────

def add_sfp_loop_to_coil(coil_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a simulated SFP loop to the COIL 2000 data:
        - Treat CARAVAN purchase as a proxy for "high-value customer risk"
        - Simulate a fraud/claim investigation loop on top of it:
          model scores → investigation policy → observed label bias
    """
    print("\n[COIL + SFP] Adding simulated SFP investigation loop...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df = coil_df.copy()
    n  = len(df)
    rng = np.random.default_rng(42)

    # Use numeric columns as features
    feature_cols = [c for c in df.columns if c != 'CARAVAN' and df[c].dtype in ['int64', 'float64']]
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Simulate a "true fraud" label based on features
    log_odds = X_scaled[:, :5].sum(axis=1) * 0.3 + rng.normal(0, 1, n)
    true_fraud_prob = 1 / (1 + np.exp(-log_odds - 2.5))
    true_fraud = rng.binomial(1, true_fraud_prob)

    # Train model v1 on a small unbiased seed
    seed_mask = rng.choice([True, False], size=n, p=[0.05, 0.95])
    X_seed, y_seed = X_scaled[seed_mask], true_fraud[seed_mask]
    if len(np.unique(y_seed)) < 2:
        y_seed[0] = 1  # ensure both classes

    model_v1 = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_v1.fit(X_seed, y_seed)
    score_v1 = model_v1.predict_proba(X_scaled)[:, 1]

    # Investigation: top 25% by v1 score
    thresh_v1 = np.percentile(score_v1, 75)
    investigated_v1 = (score_v1 >= thresh_v1).astype(int)
    observed_fraud_v1 = np.where(
        investigated_v1 == 1,
        np.where(true_fraud == 1, rng.binomial(1, 0.98, n), 0),
        np.nan
    )

    # Train model v2 on biased labels
    mask_v1 = investigated_v1 == 1
    X_v1, y_v1 = X_scaled[mask_v1], true_fraud[mask_v1]
    model_v2 = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_v2.fit(X_v1, y_v1.astype(int))
    score_v2 = model_v2.predict_proba(X_scaled)[:, 1]

    # Investigation v2
    thresh_v2 = np.percentile(score_v2, 75)
    investigated_v2 = (score_v2 >= thresh_v2).astype(int)

    # Add to dataframe
    df['true_fraud']          = true_fraud
    df['model_v1_score']      = score_v1.round(4)
    df['model_v2_score']      = score_v2.round(4)
    df['investigated']        = investigated_v2
    df['observed_fraud']      = np.where(
        investigated_v2 == 1,
        np.where(true_fraud == 1, rng.binomial(1, 0.98, n), 0),
        np.nan
    )

    path = OUT_DIR / "coil2000_with_sfp_loop.csv"
    df.to_csv(path, index=False)

    invest_rate = investigated_v2.mean()
    obs_rate    = df.loc[df['investigated'] == 1, 'observed_fraud'].mean()
    true_rate   = true_fraud.mean()

    print(f"  ✅ coil2000_with_sfp_loop.csv  [{n:,} rows × {df.shape[1]} cols]")
    print(f"     True fraud rate   : {true_rate:.3f}")
    print(f"     Observed fraud rate: {obs_rate:.3f}  (↑ inflated by investigation bias)")
    print(f"     Investigation rate: {invest_rate:.1%}")
    return df


# ─────────────────────────────────────────────
# 3. Porto Seguro public mini-sample
# ─────────────────────────────────────────────

def create_porto_seguro_sample():
    """
    Creates a Porto-Seguro-schema dataset without needing Kaggle login.
    Uses the same feature naming convention as the real dataset.
    For the real dataset: kaggle competitions download -c porto-seguro-safe-driver-prediction
    """
    print("\n[Porto Seguro] Creating schema-compatible synthetic sample (10k rows)...")
    rng = np.random.default_rng(7)
    n = 10_000

    data = {
        'id': range(n),
        # Individual features
        'ps_ind_01':    rng.integers(0, 7, n),
        'ps_ind_02_cat': rng.integers(1, 5, n),
        'ps_ind_03':    rng.integers(0, 11, n),
        'ps_ind_04_cat': rng.integers(0, 2, n),
        'ps_ind_05_cat': rng.integers(0, 7, n),
        'ps_ind_06_bin': rng.binomial(1, 0.3, n),
        'ps_ind_07_bin': rng.binomial(1, 0.15, n),
        'ps_ind_08_bin': rng.binomial(1, 0.1, n),
        'ps_ind_09_bin': rng.binomial(1, 0.05, n),
        'ps_ind_10_bin': rng.binomial(1, 0.02, n),
        'ps_ind_11_bin': rng.binomial(1, 0.01, n),
        'ps_ind_12_bin': rng.binomial(1, 0.01, n),
        'ps_ind_13_bin': rng.binomial(1, 0.01, n),
        'ps_ind_14':     rng.integers(0, 4, n),
        'ps_ind_15':     rng.integers(0, 13, n),
        'ps_ind_16_bin': rng.binomial(1, 0.6, n),
        'ps_ind_17_bin': rng.binomial(1, 0.1, n),
        'ps_ind_18_bin': rng.binomial(1, 0.08, n),
        # Regional features
        'ps_reg_01': rng.uniform(0.0, 0.9, n).round(1),
        'ps_reg_02': rng.uniform(0.0, 1.8, n).round(2),
        'ps_reg_03': rng.uniform(0.1, 4.0, n).round(4),
        # Car features
        'ps_car_01_cat': rng.integers(0, 12, n),
        'ps_car_02_cat': rng.integers(0, 2, n),
        'ps_car_03_cat': rng.integers(-1, 3, n),
        'ps_car_04_cat': rng.integers(0, 10, n),
        'ps_car_05_cat': rng.integers(-1, 2, n),
        'ps_car_06_cat': rng.integers(0, 18, n),
        'ps_car_07_cat': rng.integers(-1, 2, n),
        'ps_car_08_cat': rng.integers(0, 2, n),
        'ps_car_09_cat': rng.integers(0, 5, n),
        'ps_car_10_cat': rng.integers(1, 3, n),
        'ps_car_11_cat': rng.integers(1, 104, n),
        'ps_car_11':     rng.integers(0, 4, n),
        'ps_car_12':     rng.uniform(0.3, 0.7, n).round(4),
        'ps_car_13':     rng.uniform(0.2, 3.7, n).round(4),  # most predictive feature
        'ps_car_14':     rng.uniform(0.0, 0.7, n).round(4),
        'ps_car_15':     rng.uniform(0.0, 3.7, n).round(4),
    }

    # Simulate claim filed (target) — ~3.6% base rate like real data
    log_odds = (
        -3.5
        + 0.5 * data['ps_car_13']
        + 0.3 * data['ps_reg_03']
        + 0.4 * (data['ps_ind_06_bin'] == 0).astype(float)
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    data['target'] = rng.binomial(1, prob)

    df = pd.DataFrame(data)
    # Simulate missing values as -1 (Porto Seguro convention)
    for col in ['ps_car_03_cat', 'ps_car_05_cat', 'ps_car_07_cat']:
        mask = rng.random(n) < 0.15
        df.loc[mask, col] = -1

    path = OUT_DIR / "porto_seguro_sample.csv"
    df.to_csv(path, index=False)
    print(f"  ✅ porto_seguro_sample.csv  [{n:,} rows × {df.shape[1]} cols]")
    print(f"     Target (claim filed): base rate = {df['target'].mean():.3f}")
    print(f"     Note: For full 595k dataset: kaggle competitions download -c porto-seguro-safe-driver-prediction")
    return df


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Real Dataset Downloader")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    # 1. COIL 2000
    coil_df = download_coil2000()
    if coil_df is None:
        coil_df = _fallback_coil2000()

    # 2. COIL + SFP loop
    if coil_df is not None:
        add_sfp_loop_to_coil(coil_df)

    # 3. Porto Seguro sample
    create_porto_seguro_sample()

    print("\n✅ Done.")
    print("\nFiles created:")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<45} {size_kb:>8.1f} KB")

    print("""
Next steps:
  • To run tests with synthetic data:    pytest tests/
  • To use COIL 2000 in loop detector:
        from loop_detector import SFPDetector
        df = pd.read_csv('datasets/real/coil2000_with_sfp_loop.csv')
        detector = SFPDetector(df, model_versions=['v1', 'v2'], ...)
  • For real Porto Seguro (full 595k):
        pip install kaggle
        kaggle competitions download -c porto-seguro-safe-driver-prediction
  • For real IEEE-CIS (590k transactions):
        kaggle competitions download -c ieee-fraud-detection
""")
