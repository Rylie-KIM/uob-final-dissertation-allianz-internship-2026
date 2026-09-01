"""Reweighting corrector — notebook 03_02's four schemes (naive / A / B / AB), real-data port.

Column names below are the CANONICAL ones (src/schema.py), already translated at ingest.

THE CONTAMINATION, in PU vocabulary (Bekker & Davis 2020 — p28). Garaged claims (decision=0)
carry a verified outcome: P (observed=1) / N (observed=0). Model-scrapped claims (decision=1)
carry a FORCED observed=1 and no verification: U. Where IPSCorrector drops U and reweights what
is left, this corrector KEEPS every row and changes how much each one is trusted.

THE GRID. tau_i is the scrap cutoff in force for row i (see TAU below), h = band_h, and
high_i := (score_i >= tau_i - h) — the boundary band [tau-h, tau) plus everything above it.
A consistent U row has score > tau_i, so U always lands high. Cells (03_02's (1)-(4)):

    cell 1   high, observed=0    garage rows just below the cutoff — the rare verified edge
    cell 2   high, observed=1    the contaminated cell: all of U, plus verified band TLs
    cell 3   low,  observed=0    the common verified-repairable mass
    cell 4   low,  observed=1    rare verified low-score total losses

AXIS A — rarity weights (parameters: band_h, clip_lo, clip_hi). Labels are NOT touched:

    f_c = n_c / n                       cell frequency, on this split's rows
    m_c = 1 / (K * f_c)                 inverse frequency; K = number of non-empty cells, so
                                        equally-frequent cells would all get m_c = 1
    m_c = clip(m_c, clip_lo, clip_hi)   variance control (defaults 0.25 / 4.0)
    m_c = m_c / mean_c(m_c)             the K cell values normalised to mean 1
    w_i = m_{c_i}

AXIS B — soft split via the transport model. g(x) = P(observed=1 | x) is fitted on GARAGE rows
only (the verified labels) and evaluated on U. g is NOT the FTTL score (that model learned from
forced labels) and NOT the IPS propensity e(x) = P(scrap | x): it answers "was this car really
a total loss", learned where the answer is known and transported to where it is not. Each U row
is emitted TWICE — retrain.py's inner join duplicates the feature row to match:

    (label=1, weight g(x))   +   (label=0, weight 1 - g(x))

THE FOUR SCHEMES:

    scheme   garage rows            model-scrapped rows (U)
    naive    (observed, 1)          (1, 1)                            the contaminated baseline
    A        (observed, m_c)        (1, m_c)                          weights only, labels kept
    B        (observed, 1)          (1, g) + (0, 1-g)                 labels softened
    AB       (observed, m_c)        (1, g*m_c) + (0, (1-g)*m_c)      B's split x A's rarity

TAU. U is NEVER defined by tau — it is the recorded decision column. tau only shapes the band,
so `naive` and `B` are byte-identical under either mode; only A / AB depend on it.

    tau_mode="regime"   tau_i = the decider version's cutoff in force on row i's date
                        (config.threshold_on; decider defaults to "v2" — the v2 log generated
                        the labels for both the v2 and the v3 training sets)
    tau_mode="fixed"    one scalar for every row: `tau` if given, else read off the frame
                        (threshold.read_off = min score among decision=1 — 03_02's applied_tau)

WHICH COLUMNS g SEES: config.model_features(version), same rule and same reason as ips.py — the
exported matrix carries the target beside the inputs, so "everything except claim_id" would fit
g on the outcome it is trying to recover.

  PYTHONPATH=src .venv/bin/python -m mitigator.corrector.reweight --version v3 --scheme B \
      --features src/data/real/inputs/features_v3_train.parquet \
      --targets  src/data/real/inputs/targets_v3_train.parquet \
      --out      src/data/real/mitigation/v3_corrected_train_B.parquet

--targets must carry claim_id + decision + observed; A/AB also need score, plus date when
tau_mode="regime". A sibling <out stem>_meta.json records scheme / tau / cell table — none of
which are recoverable from the parquet itself.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
import threshold
from mitigator.corrector.base import TrainingDataCorrector

SCHEMES = ("naive", "A", "B", "AB")

CELL_NAMES = {
    1: "high-repairable (verified edge)",
    2: "high-TL (U + verified band TL)",
    3: "low-repairable",
    4: "low-TL",
}


class ReweightCorrector(TrainingDataCorrector):
    """03_02's reweighting schemes behind the TrainingDataCorrector interface."""

    def __init__(
        self,
        scheme: str = "B",
        tau_mode: str = "regime",
        decider: str = "v2",
        tau: float | None = None,
        band_h: float = 0.01,
        clip_lo: float = 0.25,
        clip_hi: float = 4.0,
        id_col: str = "claim_id",
        decision_col: str = "decision",
        outcome_col: str = "observed",
        score_col: str = "score",      # canonical names — see src/schema.py
        date_col: str = "date",
        seed: int = 42,
    ) -> None:
        if scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
        if tau_mode not in ("regime", "fixed"):
            raise ValueError(f"tau_mode must be 'regime' or 'fixed', got {tau_mode!r}")
        self.scheme = scheme
        self.tau_mode = tau_mode
        self.decider = decider
        self.tau = tau
        self.band_h = band_h
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.id_col = id_col
        self.decision_col = decision_col
        self.outcome_col = outcome_col
        self.score_col = score_col
        self.date_col = date_col
        self.seed = seed

    # -- column plumbing ---------------------------------------------------------------

    def _feature_cols(self, features: pd.DataFrame, given: list[str] | None) -> list[str]:
        """The model-input columns of `features` — supplied by the caller, never inferred.

        Same refusal as IPSCorrector, same reason: the exported matrix carries the target
        beside the inputs (v3's also its own predictions), so the silent fallback — every
        column but claim_id — would fit g(x) on the outcome it is trying to recover.
        """
        if not given:
            raise ValueError(
                "ReweightCorrector needs feature_cols: which columns of `features` are model "
                "inputs. Pass config.model_features(version) — they cannot be read off the frame."
            )
        missing = [c for c in given if c not in features.columns]
        if missing:
            raise ValueError(
                f"features is missing {len(missing)} model column(s), e.g. {missing[:8]}. "
                f"Point --features at that version's processed_inputs matrix."
            )
        extra = [c for c in features.columns if c not in given and c != self.id_col]
        if extra:
            print(f"  reweight: set aside {len(extra)} non-model column(s): {extra[:8]}"
                  + (" ..." if len(extra) > 8 else ""))
        return list(given)

    def _required_target_cols(self) -> list[str]:
        need = [self.id_col, self.decision_col, self.outcome_col]
        if self.scheme in ("A", "AB"):
            need.append(self.score_col)
            if self.tau_mode == "regime":
                need.append(self.date_col)
        return need

    # -- the two axes ------------------------------------------------------------------

    def _tau_per_row(self, m: pd.DataFrame) -> np.ndarray:
        """tau_i for every row — one scalar (fixed) or the decider's regime on that date."""
        if self.tau_mode == "fixed":
            t = self.tau
            if t is None:
                # 03_02's applied_tau convention: min score among decision=1 rows.
                t = threshold.read_off(m, self.score_col, self.decision_col)["tau"]
            return np.full(len(m), float(t))
        d = pd.to_datetime(m[self.date_col])
        if getattr(d.dt, "tz", None) is not None:
            d = d.dt.tz_localize(None)          # wall clock, matching threshold.apply()
        days = d.dt.normalize()
        lut = {day: config.threshold_on(self.decider, str(day.date())) for day in days.unique()}
        return days.map(lut).to_numpy(dtype=float)

    def _rarity(self, cell: np.ndarray) -> dict:
        """cell -> multiplier, per the Axis-A steps in the module docstring."""
        freq = pd.Series(cell).value_counts(normalize=True)
        mult = (1.0 / (freq.size * freq)).clip(self.clip_lo, self.clip_hi)
        mult = mult / mult.mean()
        return mult.to_dict()

    def _transport(
        self, m: pd.DataFrame, feat_cols: list[str],
        garage: np.ndarray, scrapped: np.ndarray, y: np.ndarray,
    ) -> np.ndarray:
        """g(x) = P(observed=1 | x): fit on garage rows only, score the U rows."""
        if len(np.unique(y[garage])) < 2:
            raise ValueError(
                "transport model g(x) needs both outcomes among garage rows; this split's "
                "verified labels are single-class."
            )
        num = m[feat_cols].select_dtypes("number")
        cat = m[feat_cols].select_dtypes(exclude="number")
        parts = [num.to_numpy(dtype=float)] if not num.empty else []
        if not cat.empty:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            parts.append(enc.fit_transform(cat))
        Xp = np.hstack(parts)
        g = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=1000, random_state=self.seed),
        )
        g.fit(Xp[garage], y[garage])
        return g.predict_proba(Xp[scrapped])[:, 1]

    # -- the corrector -----------------------------------------------------------------

    def correct(
        self,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        id_col = self.id_col
        feat_cols = self._feature_cols(features, feature_cols)

        need = self._required_target_cols()
        missing = [c for c in need if c not in labels.columns]
        if missing:
            raise ValueError(
                f"targets is missing {missing} (canonical names) — scheme {self.scheme!r} "
                f"with tau_mode={self.tau_mode!r} needs {need}."
            )
        for name, frame in (("features", features), ("targets", labels)):
            if frame[id_col].duplicated().any():
                raise ValueError(
                    f"{name} carries duplicate {id_col!r} rows — the log must be collapsed to "
                    f"one scoring event per claim before correction (see 01_export_v2_logs)."
                )

        m = features.merge(labels[need], on=id_col)
        if len(m) == 0:
            raise ValueError(
                f"features and targets share no {id_col} — they must describe the SAME split."
            )

        dec = m[self.decision_col].to_numpy().astype(int)
        y = m[self.outcome_col].to_numpy().astype(int)
        garage, scrapped = dec == 0, dec == 1
        n_garage, n_scrapped = int(garage.sum()), int(scrapped.sum())

        # forced-label sanity: decision=1 records observed=1 by construction (P1). A violation
        # means decision/observed disagree with the forcing mechanism — surface it, don't hide it.
        n_forced_zero = int((y[scrapped] == 0).sum())
        if n_forced_zero:
            print(f"  reweight: WARNING — {n_forced_zero} scrapped row(s) carry observed=0; "
                  f"the forced-label mechanism says that cannot happen. Check the export join.")

        diag: dict = {
            "scheme": self.scheme,
            "n_total": int(len(m)),
            "n_garage": n_garage,
            "n_scrapped": n_scrapped,
            "n_scrapped_observed_0": n_forced_zero,
        }

        # Axis A: per-row tau -> band -> cell -> rarity multiplier
        w_cell = None
        if self.scheme in ("A", "AB"):
            tau = self._tau_per_row(m)
            s = m[self.score_col].to_numpy(dtype=float)
            high = s >= (tau - self.band_h)
            cell = np.where(high, 1 + y, 3 + y)          # (1)(2)(3)(4) — see module docstring
            mult = self._rarity(cell)
            w_cell = pd.Series(cell).map(mult).to_numpy(dtype=float)
            freq = pd.Series(cell).value_counts(normalize=True)
            diag.update({
                "tau_mode": self.tau_mode,
                "decider": self.decider if self.tau_mode == "regime" else None,
                "tau_min": float(tau.min()),
                "tau_max": float(tau.max()),
                "band_h": self.band_h,
                "clip": [self.clip_lo, self.clip_hi],
                "cells": {
                    str(int(c)): {
                        "name": CELL_NAMES[int(c)],
                        "n": int((cell == c).sum()),
                        "freq": round(float(freq[c]), 6),
                        "multiplier": round(float(mult[c]), 4),
                    }
                    for c in sorted(mult)
                },
                # the two consistency counts: a strict rule (score > tau) implies both are 0 on a
                # clean single-regime slice; nonzero garage-above-tau = overlap rows (see read_off)
                "n_garage_above_tau": int((garage & (s > tau)).sum()),
                "n_scrapped_at_or_below_tau": int((scrapped & (s <= tau)).sum()),
            })

        # Axis B: transport split of the U rows
        g_u = None
        if self.scheme in ("B", "AB"):
            g_u = self._transport(m, feat_cols, garage, scrapped, y)
            diag["transport"] = {
                "estimator": "StandardScaler + LogisticRegression(max_iter=1000)",
                "n_fit_garage": n_garage,
                "n_scored_U": n_scrapped,
                "n_feature_cols": len(feat_cols),
                "g_mean_U": round(float(g_u.mean()), 4),
                "g_min_U": round(float(g_u.min()), 4),
                "g_max_U": round(float(g_u.max()), 4),
            }

        # assemble (claim_id, label, weight) per scheme
        ids = m[id_col].to_numpy()
        if self.scheme == "naive":
            out = pd.DataFrame({id_col: ids, "label": y, "weight": np.ones(len(m))})
        elif self.scheme == "A":
            out = pd.DataFrame({id_col: ids, "label": y, "weight": w_cell})
        else:  # B / AB — U rows duplicated into a (1, g) and a (0, 1-g) half
            w_gar = w_cell[garage] if self.scheme == "AB" else np.ones(n_garage)
            w_u = w_cell[scrapped] if self.scheme == "AB" else np.ones(n_scrapped)
            out = pd.concat(
                [
                    pd.DataFrame({id_col: ids[garage], "label": y[garage], "weight": w_gar}),
                    pd.DataFrame({id_col: ids[scrapped],
                                  "label": np.ones(n_scrapped, dtype=int),
                                  "weight": g_u * w_u}),
                    pd.DataFrame({id_col: ids[scrapped],
                                  "label": np.zeros(n_scrapped, dtype=int),
                                  "weight": (1.0 - g_u) * w_u}),
                ],
                ignore_index=True,
            )
        out["label"] = out["label"].astype(int)
        out["weight"] = out["weight"].astype(float)

        w = out["weight"].to_numpy()
        diag.update({
            "n_out_rows": int(len(out)),
            "weight_mean": round(float(w.mean()), 4),
            "weight_sum": round(float(w.sum()), 2),
            # the effective class balance the retrain will see (weighted share of label=1)
            "weighted_pos_share": round(float((w * out["label"]).sum() / w.sum()), 4),
        })
        return out[[id_col, "label", "weight"]], diag


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--features", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--version", required=True,
                   help="which version's registry names the model-input columns — the version "
                        "being retrained, e.g. v3")
    p.add_argument("--scheme", required=True, choices=SCHEMES)
    p.add_argument("--tau-mode", default="regime", choices=("regime", "fixed"),
                   help="A/AB only: per-row regime tau (default) or one scalar for every row")
    p.add_argument("--decider", default="v2",
                   help="regime mode: whose decision regimes define tau_i (the version whose "
                        "log produced the decisions — v2 for both training sets)")
    p.add_argument("--tau", type=float, default=None,
                   help="fixed mode: the single cutoff; omit to read it off the frame "
                        "(min score among decision=1)")
    p.add_argument("--band-h", type=float, default=0.01)
    p.add_argument("--clip-lo", type=float, default=0.25)
    p.add_argument("--clip-hi", type=float, default=4.0)
    p.add_argument("--id-col", default="claim_id")
    a = p.parse_args()

    corrector = ReweightCorrector(
        scheme=a.scheme, tau_mode=a.tau_mode, decider=a.decider, tau=a.tau,
        band_h=a.band_h, clip_lo=a.clip_lo, clip_hi=a.clip_hi, id_col=a.id_col,
    )
    corrected, diag = corrector.correct(
        pd.read_parquet(a.features), pd.read_parquet(a.targets),
        feature_cols=config.model_features(a.version),
    )

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_parquet(out, index=False)

    # sidecar meta — scheme/tau/cell table are not recoverable from the parquet itself
    meta = out.with_name(out.stem + "_meta.json")
    meta.write_text(
        json.dumps({"version": a.version, "features": a.features, "targets": a.targets, **diag},
                   indent=2),
        encoding="utf-8",
    )

    print(f"reweight[{a.scheme}]: {diag['n_total']} rows in "
          f"(garage {diag['n_garage']}, scrapped {diag['n_scrapped']}) -> "
          f"{diag['n_out_rows']} corrected rows, weighted pos share "
          f"{diag['weighted_pos_share']} -> {a.out}")
    print(f"  meta -> {meta}")


if __name__ == "__main__":
    main()
