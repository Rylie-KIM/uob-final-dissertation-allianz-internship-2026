"""
Export per-version threshold tuning candidates.

Where threshold tuning lives:
  * generate/model.py :: _tune_threshold()  — the canonical rule: τ_v = lowest score cutoff
    whose precision on the validation slice first reaches TARGET_PRECISION (0.985), else the
    0.872 fallback. train_and_apply() applies τ_v to build each version's decision column, but
    only *logs* the number — it is never persisted.
  * evaluate.py       — recovers the applied τ_v from the data and reports one row per version.

This script fills the gap: for every model version it records a *table of candidate
thresholds* with the precision / recall each would yield, so the precision–recall trade-off
around the scrap cutoff is auditable — not just the single chosen τ_v.

Basis (reproducible, no re-generation): each version's OOT holdout window scored against its
own (SFP-contaminated) observed label — the same slice and labels as model_evaluation.csv.
To keep the file small we keep ~20 candidates per version, centred on the target-precision
crossing (the region the tuner actually deliberates over).

Outputs (both to src/data/synthetic/csv/, per request):
  * model_threshold_candidates.csv — ~20 (threshold, precision, recall) rows per version
  * model_thresholds.csv           — one summary row per version (chosen + applied τ_v)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_curve, precision_score, recall_score

# Reuse the single source of truth for windows / columns / target (no duplication).
from evaluate import VERSION_COLS, OOT_WINDOWS, PARQUET_DIR
from generate.config import TARGET_PRECISION, SCRAP_THRESHOLD
from generate.logger import get_logger

log = get_logger(__name__)

OUT_DIR       = Path(__file__).parent / "csv"          # src/data/synthetic/csv  (requested target)
N_CANDIDATES  = 20                                     # rows kept per version
N_BELOW       = 8                                      # how many to keep just under the crossing


def _oot_slice(df: pd.DataFrame, version: str):
    """(y_true, y_score, decision) on the version's OOT window with a non-null label."""
    score_col, label_col = VERSION_COLS[version]
    decision_col         = f"model_{version}_decision"
    start, end           = OOT_WINDOWS[version]

    if score_col not in df.columns or label_col not in df.columns:
        return None
    dates = pd.to_datetime(df["claim_date"])
    mask  = (dates >= start) & (dates <= end) & df[label_col].notna()
    sub   = df[mask]
    if len(sub) == 0 or sub[label_col].nunique() < 2:
        return None
    return (sub[label_col].astype(int).to_numpy(),
            sub[score_col].to_numpy(),
            sub[decision_col].astype(int).to_numpy() if decision_col in sub else None)


def _applied_tau(df: pd.DataFrame, version: str) -> float:
    """The τ_v actually baked into the data = lowest score among scrapped rows (all years)."""
    score_col   = VERSION_COLS[version][0]
    decision_col = f"model_{version}_decision"
    if decision_col not in df.columns:
        return float("nan")
    scrapped = df.loc[df[decision_col] == 1, score_col]
    return float(scrapped.min()) if len(scrapped) else float("nan")


def candidate_table(y_true, y_score, target=TARGET_PRECISION,
                    n_keep=N_CANDIDATES, n_below=N_BELOW) -> tuple[pd.DataFrame, int]:
    """~n_keep candidate thresholds around the target-precision crossing, each with its
    precision, recall and #flagged. Returns (table, positional index of the chosen τ_v)."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    prec, rec = prec[:-1], rec[:-1]                    # drop the trailing (P=1,R=0) sentinel

    tbl = (pd.DataFrame({"threshold": thr, "precision": prec, "recall": rec})
           .sort_values("threshold").reset_index(drop=True))

    meets = (tbl["precision"] >= target).to_numpy()
    chosen = int(np.argmax(meets)) if meets.any() else int(tbl["precision"].to_numpy().argmax())
    #  tuning rule: lowest threshold clearing target ─┘   (else the highest-precision point)

    hi = min(len(tbl), max(chosen + (n_keep - n_below), chosen + 1))
    lo = max(0, hi - n_keep)
    hi = min(len(tbl), lo + n_keep)
    sel = tbl.iloc[lo:hi].copy()

    sel["n_flagged"]    = [int((y_score >= t).sum()) for t in sel["threshold"]]
    sel["meets_target"] = sel["precision"] >= target
    sel["is_chosen"]    = sel.index == chosen         # the τ_v the tuning rule would pick on OOT
    return sel, chosen


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    claims_all = PARQUET_DIR / "claims_all.parquet"
    if not claims_all.exists():
        raise FileNotFoundError(f"{claims_all} not found — run generation (run.py) first.")
    df = pd.read_parquet(claims_all)

    cand_frames, summary_rows = [], []
    for version in VERSION_COLS:
        sl = _oot_slice(df, version)
        if sl is None:
            log.warning(f"[{version}] skipped — no usable OOT slice")
            continue
        y_true, y_score, y_dec = sl
        applied = _applied_tau(df, version)

        # OOT performance at the ACTUALLY-APPLIED τ_v: read it off the version's decision
        # column directly (same basis as evaluate.py's *_at_applied), not a re-threshold.
        if y_dec is not None:
            prec_applied = float(precision_score(y_true, y_dec, zero_division=0))
            rec_applied  = float(recall_score(y_true, y_dec, zero_division=0))
        else:
            prec_applied = rec_applied = float("nan")

        sel, chosen = candidate_table(y_true, y_score)

        # flag the candidate nearest the τ_v actually applied in the data
        if np.isfinite(applied):
            near = (sel["threshold"] - applied).abs().idxmin()
            sel["is_applied"] = sel.index == near
        else:
            sel["is_applied"] = False

        sel.insert(0, "version", version)
        sel.insert(1, "rank", range(1, len(sel) + 1))     # 1 = lowest threshold in the window
        cand_frames.append(sel)

        chosen_row = sel.loc[sel["is_chosen"]]
        summary_rows.append({
            "version":            version,
            "oot_rows":           int(len(y_true)),
            "applied_tau":        round(applied, 4) if np.isfinite(applied) else None,
            "precision_at_applied": round(prec_applied, 4) if np.isfinite(prec_applied) else None,
            "recall_at_applied":  round(rec_applied, 4) if np.isfinite(rec_applied) else None,
            "chosen_tau_oot":     round(float(chosen_row["threshold"].iloc[0]), 4) if len(chosen_row) else None,
            "precision_at_chosen": round(float(chosen_row["precision"].iloc[0]), 4) if len(chosen_row) else None,
            "recall_at_chosen":   round(float(chosen_row["recall"].iloc[0]), 4) if len(chosen_row) else None,
            "target_precision":   TARGET_PRECISION,
            "target_reached":     bool(chosen_row["meets_target"].iloc[0]) if len(chosen_row) else False,
            "fallback_tau":       SCRAP_THRESHOLD,
            "n_candidates":       int(len(sel)),
        })

    candidates = pd.concat(cand_frames, ignore_index=True)
    candidates = candidates[["version", "rank", "threshold", "precision", "recall",
                             "n_flagged", "meets_target", "is_chosen", "is_applied"]]
    candidates[["threshold", "precision", "recall"]] = candidates[["threshold", "precision", "recall"]].round(4)
    summary = pd.DataFrame(summary_rows)
    return candidates, summary


def main() -> None:
    candidates, summary = build()
    OUT_DIR.mkdir(exist_ok=True)

    cand_path = OUT_DIR / "model_threshold_candidates.csv"
    summ_path = OUT_DIR / "model_thresholds.csv"
    candidates.to_csv(cand_path, index=False)
    summary.to_csv(summ_path, index=False)

    log.info("=== threshold tuning candidates (OOT, SFP-contaminated labels) ===")
    log.info(f"\n{summary.to_string(index=False)}")
    log.info(f"saved {len(candidates)} candidate rows across {summary.shape[0]} versions "
             f"→ {cand_path}")
    log.info(f"saved per-version summary → {summ_path}")


if __name__ == "__main__":
    main()
