"""The decision rule, in one place: tune / apply / read_off.

A decision is `score > tau` — STRICTLY greater, on every version (confirmed 2026-07-31). A car
scoring exactly tau is sent to the garage, not scrapped. This matters more than it looks: RDD is a
method for reading the discontinuity AT the cutoff, so which side the boundary rows fall on is not
a rounding detail. It is also why v1's overlap band is written `(0.75, 0.85]` — open on the left
(at 0.75 neither segment scraps), closed on the right (at 0.85 the immobile segment scraps and the
mobile one does not).

tau itself is NOT one number and NOT the same SHAPE across versions. `config.DECISION_RULES`
records what each version actually did:

    v1  segmented        two cutoffs keyed on vehicle mobility (0.75 immobile / 0.85 mobile)
    v2  piecewise_global one global cutoff that MOVED four times -> FIVE regimes, and it
                         alternates (0.8915 / 0.872 / 0.825 / 0.872 / 0.825)
    v3  global           one fixed cutoff (0.984)

So 0.872 is v2's cutoff during 2021-06-03..2024-06-02 and again 2026-02-25..2026-06-30, and
nothing else — it is not "v2's threshold", and the same value appearing in two eras does not make
them one deployment. (The old `config.SCRAP_THRESHOLD` constant that carried it was deleted
2026-08-09.) Using it for v1 or v3 silently mislabels rows. `apply()` exists to make that mistake
impossible: it takes the version and the frame, and dispatches on the rule's shape.

Because the shapes differ, `apply` needs the whole frame, not a score array — v1's rule reads the
mobility column, v2's reads the date. That is why this cannot be a scalar constant.

THE THREE FUNCTIONS ARE NOT INTERCHANGEABLE:

    read_off(df)          tau as an OBSERVED FACT — the boundary in the production log. Use this
                          for a model that RAN. A documented tau need not be the tau that produced
                          the decisions (04-01 already found this).
    apply(version, df)    reproduce the decisions from scores. Env-free, universal.
    tune(y, scores)       choose a tau for a model that NEVER ran (the mitigated one). Must be run
                          on a held-out slice inside the version env — see training/retrain.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config
import schema


def tune(
    y,
    scores,
    target: float | None = config.TARGET_PRECISION,
    fallback: float | None = None,
    *,
    grid=None,
    f1: bool = False,
    recall_min: float | None = None,
    recall_weight: float | None = None,
) -> float | None:
    """Choose tau on a grid — adapted from Allianz's own `select_best_threshold` (handed over
    2026-09-02; this replaces the provisional project rule, which survives as the default mode).

    Mode precedence follows the company function:

        f1=True           argmax of class-weighted F1 over the grid
        recall_weight=w   argmax of  w*recall + (1-w)*precision
        recall_min=r      HIGHEST grid point whose recall stays > r  (their constraint is strict)
        (default)         LOWEST grid point whose precision >= target — the precision-floor rule
                          (config.TARGET_PRECISION). The company signature carries `precision_min`
                          for this but the body as received never read it; this branch fills that
                          role and keeps our documented `>=`.

    Two deliberate departures from the code as received: the weighted combination uses `+` (its
    own comment says weighted AVERAGE of precision and recall — the received line multiplied,
    which cancels the weight out of the argmax), and an unsatisfiable constraint returns
    `fallback` instead of raising on an empty sequence.

    `grid` defaults to every observed score (np.unique(scores)) — what this function scanned
    before the handover; the company passes an explicit grid, and both work. The grid is sorted
    and deduplicated, so LOWEST/HIGHEST above are well defined.

    The decision is `score > t`, STRICT, matching apply(). Tuning on data the model was fitted
    on is in-sample, lands tau too low, and silently over-scraps — pass a held-out slice.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    thresholds = np.unique(s if grid is None else np.asarray(grid, dtype=float))

    n_pos = int((y == 1).sum())
    all_prec = np.zeros(len(thresholds))
    all_recall = np.zeros(len(thresholds))
    all_metric = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        pred = s > t                     # STRICT, to match apply() — see the module docstring
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        all_prec[i] = tp / (tp + fp) if tp + fp else 0.0   # == sklearn zero_division=0
        all_recall[i] = tp / n_pos if n_pos else 0.0
        if f1:
            all_metric[i] = f1_score(y, pred.astype(int), average="weighted")
        elif recall_weight is not None:
            all_metric[i] = recall_weight * all_recall[i] + (1.0 - recall_weight) * all_prec[i]

    if f1 or recall_weight is not None:
        return float(thresholds[int(np.argmax(all_metric))])
    if recall_min is not None:
        ok = np.flatnonzero(all_recall > recall_min)       # strict, as received
        return float(thresholds[ok.max()]) if ok.size else fallback
    if target is None:
        raise ValueError("no mode selected: give target, or one of f1 / recall_weight / recall_min")
    ok = np.flatnonzero(all_prec >= target)  # the PRECISION floor is >= ; the score rule is >
    return float(thresholds[ok.min()]) if ok.size else fallback


def apply(version: str, df: pd.DataFrame, score_col: str = schema.SCORE) -> np.ndarray:
    """Reproduce that version's scrap decisions from its scores. Returns 0/1.

    The comparison is STRICT (`score > tau`) on every version — a car scoring exactly tau is
    garaged. Dispatches on config.DECISION_RULES[version]["shape"], and raises rather than guessing
    when the frame lacks the column a rule needs: a segmented rule silently applied as a global one
    would mislabel every immobile vehicle in the (0.75, 0.85] band.
    """
    rule = config.DECISION_RULES[version]
    s = df[score_col].to_numpy(dtype=float)
    shape = rule["shape"]

    if shape == "global":
        return (s > rule["threshold"]).astype(int)

    if shape == "segmented":
        seg = rule["segment_by"]                      # canonical name, e.g. "mobility"
        if seg not in df.columns:
            raise KeyError(
                f"[{version}] rule is segmented on {seg!r}, which is not in the frame. "
                f"Declare config.VERSIONS['{version}']['columns']['{seg}'] and re-ingest — "
                f"applying a single cutoff instead would mislabel the overlap band "
                f"{rule['overlap_band']}."
            )
        mobile = df[seg].isin(rule["mobile_values"]).to_numpy()
        tau = np.where(mobile, rule["thresholds"]["mobile"], rule["thresholds"]["immobile"])
        return (s > tau).astype(int)

    if shape == "piecewise_global":
        if schema.DATE not in df.columns:
            raise KeyError(
                f"[{version}] rule is piecewise in time but {schema.DATE!r} is not in the frame. "
                f"Pooling the regimes conflates two different treatment assignments."
            )
        # WHOLE DAYS. Regime boundaries are dates, so any time-of-day on the column is discarded
        # and no timezone conversion happens. Two of the four breaks were mid-afternoon UK time
        # (recorded in config.DECISION_RULES['v2']['breaks']), so at most one morning of claims
        # per break sits in the wrong regime — a cost taken deliberately, since the log is not
        # guaranteed to carry timestamps or their timezone, and nothing here turns on one day.
        d = pd.to_datetime(df[schema.DATE])
        if getattr(d.dt, "tz", None) is not None:
            d = d.dt.tz_localize(None)      # drop the zone, keep the wall clock; never convert
        d = d.dt.normalize()                # midnight — time of day plays no part

        # Half-open [from, until) per regime. Both bounds are optional (the first regime has no
        # start, the last no end), so a middle regime carries BOTH — v2 declares five regimes and
        # three of them are middles. This is why the loop tests each bound independently: an
        # earlier version keyed on "until" in regime / else "from", which silently mis-assigned
        # every middle regime and was correct only while v2 had exactly two.
        tau = np.full(len(df), np.nan, dtype=float)
        for regime in rule["regimes"]:
            mask = np.ones(len(df), dtype=bool)
            if "from" in regime:
                mask &= (d >= pd.Timestamp(regime["from"])).to_numpy()
            if "until" in regime:
                mask &= (d < pd.Timestamp(regime["until"])).to_numpy()
            tau[mask] = regime["threshold"]

        if np.isnan(tau).any():
            # Regimes must tile the whole timeline. A gap means a row would be scored against no
            # policy at all — fail loudly rather than emit a decision nobody can attribute.
            n = int(np.isnan(tau).sum())
            raise ValueError(
                f"[{version}] {n} row(s) fall outside every declared regime "
                f"({d[np.isnan(tau)].min()} … {d[np.isnan(tau)].max()}). "
                f"config.DECISION_RULES[{version!r}]['regimes'] must tile the timeline."
            )
        return (s > tau).astype(int)

    raise ValueError(f"[{version}] unknown decision-rule shape {shape!r}")


def read_off(
    df: pd.DataFrame,
    score_col: str = schema.SCORE,
    decision_col: str = schema.DECISION,
) -> dict:
    """tau as an observed fact, read from the log's own score/decision columns.

    A LOG DOES NOT CONTAIN tau — it contains scores and decisions, and tau is inferred from where
    they separate. Because the rule is strict (`score > tau`), the log can only ever BRACKET it:

        max{score | decision=0}  <=  tau  <  min{score | decision=1}

    `tau` is reported as `min{score | decision=1}`, matching the convention already used in
    notebook 03_02 (`applied_tau`), and `tau_bracket` gives the interval the log actually pins
    down. Any value inside that bracket reproduces every decision in the log, so a documented tau
    sitting a rounding hair away from this one is not a contradiction — 04-01 already met that.

    `deterministic` — whether max{score | decision=0} < min{score | decision=1} — is the number to
    read first. True means treatment assignment is a hard cutoff, positivity above tau is exactly
    zero, and propensity-based correction is not identified there: identification has to come from
    continuity at the cutoff, PU relabelling, or injected randomness instead. Note it can be True
    only because the frame POOLS regimes or segments — run it per regime (v2) and per mobility
    segment (v1), not on the whole log at once.

    Run this FIRST on real data.
    """
    s = df[score_col].to_numpy(dtype=float)
    d = df[decision_col].to_numpy().astype(int)
    if d.sum() == 0 or (1 - d).sum() == 0:
        raise ValueError("read_off needs both decision=0 and decision=1 rows present")

    max_untreated = float(s[d == 0].max())
    min_treated = float(s[d == 1].min())
    return {
        "tau": min_treated,                       # by convention; see tau_bracket for what is pinned
        "tau_bracket": (max_untreated, min_treated),
        "max_score_decision_0": max_untreated,
        "min_score_decision_1": min_treated,
        "deterministic": bool(max_untreated < min_treated),
        # untreated rows sitting at or above the lowest treated score — the rows that make the
        # assignment non-deterministic. Zero of them is what "positivity is dead" looks like.
        "overlap_rows": int(((s >= min_treated) & (d == 0)).sum()),
        "n": int(len(df)),
    }
