"""ShapDiDEstimator — region(A/B) x version(before/after) difference-in-differences on SHAP
concentration. Promoted from `notebook/real/shap_did/04_02_shap_did_concentration.ipynb`
(`cross_version_estimate` -> `estimate()`, `local_cross_version_estimate` -> `estimate_local()`);
that notebook is left as the presentation layer (tables/figures) and should import this module
rather than redefine the logic — see `notebook/real/shap_did/04_02_shap_did_concentration.ipynb`.

**What it measures.** `region` is that row's OWN decision (A: score<=tau, garage-verified; B:
score>tau, model-scrapped, STRICT `>` — `src/threshold.py`). For two versions v_before -> v_after,
each on its OWN out-of-time holdout (`config.OOT_SPLIT`, never `train` — SHAP on training data can
reflect regularisation artefacts that have nothing to do with forced-label dose):

    DiD = (simpson(B) - simpson(A))|after - (simpson(B) - simpson(A))|before

simpson() is `estimator.concentration.simpson` — the Simpson concentration index of mean|phi| —
so this is a DiD on a concentration functional, not on an outcome mean.

**Three assumptions** (`assumptions()`), checked together by one `falsify()` call:

- **`parallel_trends`.** Absent the forced-label mechanism, region B's concentration would have
  drifted from region A's by the same amount in both versions. Untestable directly (no
  counterfactual version exists); `falsify()` gives the one observable proxy this repo has for it:
  `dose(v)` (a detection-layer metric, `detector.dose.DoseMeter` — moved out of this module
  2026-09-16, a model-scrap share is a detection quantity, not an estimation one), the share of v's OWN training
  claims whose label was FORCED by the prior version's scrap decision (never garage-verified).
  Two versions with negligible, similar dose but a large
  DiD argues the DiD is picking up something other than dose — most plausibly the v1->v2
  era/label-source jump, not a forced-label mechanism. Not auto-classified into pass/fail: an
  earlier version of this notebook tried a dose threshold + ratio rule and it broke, because v1's
  dose is 0 by construction (no prior model exists to have forced a v1 label) and a ratio rule
  divides by zero. Picking a threshold needs the real dose SCALE first, which is a
  company-laptop-only number, so `falsify()` reports the raw doses and leaves the read to
  whoever is looking at both numbers.
- **`measurement_comparability`.** Both versions must be attributed under the same SHAP backend —
  `falsify()` calls `concentration.require_comparable()`, which RAISES on a mismatch rather than
  degrading to a caveat (its own docstring: "a warning would be too weak, because the resulting
  numbers look perfectly reasonable").
- **`no_case_mix_confound`.** The two versions should be compared on a shared claim set, not each
  on its own per-version sample. `falsify()` reports whether they were (`shared_claim_set`) —
  never blocks on it, because it is a STANDING, usually-unmet limitation here: v2's window
  (2018-01..2020-09) and v3's (2023-06..2026-05) share no claim at all.

**Local (RDD) variant.** `estimate_local()` restricts to a narrow band `|score - tau| <= h` around
each row's OWN tau instead of the whole region A/B population — a sharp Regression Discontinuity
Design, the cross-version twin of the thesis's own boundary-gap construction (`eq:boundary-gap`,
`sec:bg-causal`). It is a robustness check on `estimate()`, not a replacement: if the local and
whole-population DiD agree in sign and rough size, the whole-population number is not simply an
artefact of region A/B having different covariate distributions. `select_local_h()` grid-selects
`h` on CELL COUNTS ONLY (never on a concentration number computed from those rows) — the same
power-gate discipline `mitigator/corrector/reweight.py`'s own `band_h` selection uses, re-derived
independently rather than importing that notebook's number.

v1 is excluded from the local estimator: its decision rule is mobility-segmented
(`threshold.apply`, not a scalar tau), and mobility is permanently unrecoverable
(`project_v1_mobility_not_a_feature`) — there is no single number to band `|score - tau|` around.

**Paired shared-claim variant.** `estimate()`/`estimate_local()` compare v2 and v3 on their OWN,
INDEPENDENT out-of-time samples — `no_case_mix_confound` is a STANDING, unmet assumption for that
pair (v2's window 2018-01..2020-09 and v3's 2023-06..2026-05 share no claim). But `corrector_targets`
(built for the dose/mitigation pipeline, `03_01_corrector_inputs.ipynb`) already links v3's OWN
claims to v2's PRODUCTION-LOG score/decision on the SAME claim_id — a claim v2 scored in production
and v3 later re-scored. That is a genuine panel (same units, two time points), so
`estimate_shared()`, `shared_claim_table()` and `shared_claim_paths()` below build the case-mix-free
complement: `region` is v2's OWN production `decision`, held FIXED across both arms (the same
treatment-set-once design `DoseMeter`/`dose()` already assumes for this exact prior-decision
relationship), and both arms are the IDENTICAL claim_id set.

This needs SHAP attributions that do not exist anywhere else in the pipeline: v2's own
`attributions` (from `00_SHAP.ipynb`/`attribute.py`) are drawn from v2's TRAIN/VAL/TEST splits —
the era v2 was FIT on (2018-2020) — never from `log_features`, v2's SERVING-time matrix, which is
the only place these shared claims' v2-side features live (the era v2 was actually DEPLOYED over).
So a dedicated `attribute.py` run against `log_features`, restricted via `--explain-ids` to exactly
`corrector_targets_v3_<split>.parquet`'s claim_ids, is required before `shared_claim_table()` can
read anything — see the recipe in `notebook/real/shap_did/04_02_shap_did_concentration.ipynb`'s own
markdown cell. `shared_claim_paths()` deliberately does not fall back to anything if that has not
been run: there is no substitute computable inside the analysis env.

`estimate_shared()` has its own local (RDD) twin, `estimate_shared_local()`, the same relationship
`estimate_local()` has to `estimate()` — see `shared_local_region()` / `select_shared_local_h()`.
It bands around v2's OWN tau specifically (the boundary that actually assigned the FIXED
treatment both variants use), never re-derived from v3's own score on the "after" side, so
`region_local` — and therefore the cell counts — are IDENTICAL across both arms, unlike the
whole-population local estimator where each side's own independent sample can carry different
counts.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import config
import schema
from detector.dose import DoseMeter
from estimator import concentration
from estimator.effect.base import Assumption, EffectEstimator, EffectReport, FalsificationReport
from loaders import load

ID_COL = schema.CLAIM_ID
TAG_COLS = [schema.DATE, "score", schema.DECISION, "region", "era", schema.OBSERVED]

#: local-band grid-selection defaults — see module docstring "Local (RDD) variant".
LOCAL_H_GRID = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1)
LOCAL_MIN_CELL_N = 100    # power-gate: RSE ~ 1/sqrt(n), ~10% at n=100 (same rationale as 03_02 SS2c)
LOCAL_H_MAX = 0.05        # locality cap -- otherwise the gate is met by de-localising "local"
LOCAL_H_FALLBACK = 0.01   # 03_02's own fallback constant, reused only as a last resort here too


def region_tag_inline(version: str, split: str) -> pd.DataFrame:
    """region(A/B) built in memory for a (version, split) with no persisted `shap_did_input` file
    yet — a defensive fallback: `04_01_shap_did_inputs.ipynb` is the normal path for all three
    versions (v1 included, since 2026-09-16), so this only fires if that notebook has not been
    (re)run for the (version, split) asked for here.

    Includes "score" (renamed from that version's own `d.score_col`, same rule
    `04_01_shap_did_inputs.ipynb::build_shap_did_input` uses) so `estimate_local()`'s band
    restriction has a column to band around even on this fallback path — the notebook's own
    original inline fallback omitted it, which only ever surfaced as a latent KeyError because
    `estimate()` (whole-population) never needed "score" and the local estimator was only ever
    called for v2->v3, which always had a `shap_did_input` file in practice.
    """
    d = load(version, split=split)
    df = d.frame.copy()
    df[schema.DECISION] = d.decisions
    df["region"] = np.where(df[schema.DECISION] == 1, "B", "A")
    df = df.rename(columns={d.score_col: "score"})
    return df[[ID_COL, schema.DATE, "score", "region"]]


def version_pair_table(version: str, split: str) -> pd.DataFrame:
    """region-tagged claims joined to that (version, split)'s attributions — via 04_01's
    `shap_did_input` file if it exists, else built inline."""
    if config.split_path("shap_did_input", version, split).is_file():
        tags = pd.read_parquet(config.split_path("shap_did_input", version, split))
    else:
        tags = region_tag_inline(version, split)
    attrs = load(version, split=split).attributions
    return tags.merge(attrs, on=ID_COL, how="inner")


#: v2<->v3 shared-claim SHAP artefacts (module docstring "Paired shared-claim variant") live
#: beside `shap_did_input` under `estimation/shap_did/` — the same "estimation/, not detection/"
#: rule config.py's FALLBACK comment states for that kind — rather than as a config.py KIND: this
#: pair exists only for this one Build-04 comparison, not for every (kind, version, split) config
#: already enumerates.
SHARED_CLAIM_DIR = config.ROOT / "src" / "data" / "real" / "estimation" / "shap_did"


def shared_claim_paths(v3_split: str) -> dict[str, pathlib.Path]:
    """Paths for one v3 split's shared-claim SHAP artefacts.

    `explain_ids` is `corrector_targets_v3_<split>.parquet` ITSELF — it already carries a
    `claim_id` column, so `attribute.py --explain-ids` can point at it directly; no separate id
    list is built. Passing this SAME file to both the v2-side and v3-side `attribute.py` runs is
    what makes the two attribution files a genuinely paired (case-mix-free) sample rather than two
    independent ones that happen to overlap.
    """
    return {
        "explain_ids": config.split_path("corrector_targets", "v3", v3_split),
        "v2_attributions": SHARED_CLAIM_DIR / f"v2_log_attributions_shared_v3_{v3_split}.parquet",
        "v3_attributions": SHARED_CLAIM_DIR / f"v3_attributions_shared_{v3_split}.parquet",
    }


def _meta_path(p: pathlib.Path) -> pathlib.Path:
    return p.with_name(p.stem + "_meta.json")


def shared_claim_table(v3_split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(before, after): v2's production-log SHAP vs v3's own SHAP, on the IDENTICAL claims.

    Each side stays in that version's OWN feature space (never merged into one frame — v2 and v3
    do not share a feature space, same rule `version_pair_table` and the per-feature tables above
    already follow). `region` is NOT re-derived per side: it is v2's ORIGINAL production
    `decision` (`corrector_targets`'s own column), held FIXED across both arms — see the module
    docstring "Paired shared-claim variant". Both returned frames carry claim_id + that version's
    phi columns + `_base_value` + `region` only.

    Raises FileNotFoundError naming whichever attribution file is missing — there is no inline
    fallback (unlike `region_tag_inline`): the whole point of this pair is a real,
    restricted-to-these-ids SHAP run that nothing in the analysis env can substitute for.
    """
    paths = shared_claim_paths(v3_split)
    missing = [k for k in ("v2_attributions", "v3_attributions") if not paths[k].is_file()]
    if missing:
        raise FileNotFoundError(
            f"shared-claim SHAP not built yet for v3/{v3_split}: missing {missing} under "
            f"{SHARED_CLAIM_DIR} — run the attribute.py recipe in this notebook's own markdown "
            f"cell first (company laptop: env-v2 against log_features, then env-v3 against "
            f"processed_inputs, both restricted to {paths['explain_ids'].name}'s claim_ids)."
        )

    ct = pd.read_parquet(paths["explain_ids"])[[ID_COL, schema.DECISION]]
    ct["region"] = np.where(ct[schema.DECISION] == 1, "B", "A")
    ct = ct[[ID_COL, "region"]]

    meta_before = json.loads(_meta_path(paths["v2_attributions"]).read_text(encoding="utf-8"))
    meta_after = json.loads(_meta_path(paths["v3_attributions"]).read_text(encoding="utf-8"))
    concentration.require_comparable({"v2": meta_before, "v3": meta_after})

    before = pd.read_parquet(paths["v2_attributions"]).merge(ct, on=ID_COL, how="inner")
    after = pd.read_parquet(paths["v3_attributions"]).merge(ct, on=ID_COL, how="inner")
    return before, after


def shared_local_region(v3_split: str, h: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The shared-claim twin of `local_region()` — narrows `shared_claim_table()`'s (before,
    after) to a band `|score - tau| <= h` around v2's OWN tau, at the date `corrector_targets`
    records for each claim, re-tagged `region_local` by the same sign (STRICT `>`, matching
    `src/threshold.py`).

    Banded around v2's tau SPECIFICALLY, never re-derived on the "after" side: v2's tau is the
    boundary that actually assigned the fixed treatment `estimate_shared()` uses, so that is what
    "local" means for this pair — not a fresh RDD cutoff on v3's own score. `n_before == n_after`
    by construction here (identical claim_id set and identical region_local both arms), unlike
    `local_region()`'s whole-population twin, where each version's own independent sample can
    carry different cell counts on each side.
    """
    before, after = shared_claim_table(v3_split)
    ct = pd.read_parquet(shared_claim_paths(v3_split)["explain_ids"])[[ID_COL, schema.DATE, "score"]]
    dist = ct["score"] - tau_by_date(ct, "v2")
    in_band = dist.abs() <= h
    band_ids = ct.loc[in_band, [ID_COL]].assign(region_local=np.where(dist[in_band] > 0, "B", "A"))

    tb = before.drop(columns=["region"]).merge(band_ids, on=ID_COL, how="inner")
    ta = after.drop(columns=["region"]).merge(band_ids, on=ID_COL, how="inner")
    return tb, ta


def cell_mabs(table: pd.DataFrame, drop_extra: list[str] | None = None, **filters: str) -> pd.Series:
    """mean|phi| per feature for the subset matching **filters."""
    sub = table
    for col, val in filters.items():
        sub = sub[sub[col] == val]
    drop_cols = [c for c in TAG_COLS + (drop_extra or []) if c in sub.columns]
    return concentration.mean_abs(sub.drop(columns=drop_cols, errors="ignore"), id_col=ID_COL)


def cell_simpson(table: pd.DataFrame, drop_extra: list[str] | None = None, **filters: str) -> float:
    """Simpson index of mean|phi|, restricted to the subset matching **filters."""
    return concentration.simpson(cell_mabs(table, drop_extra=drop_extra, **filters))


def tau_by_date(table: pd.DataFrame, version: str) -> pd.Series:
    """tau looked up per UNIQUE date, not per row — an OOT split alone can run to 10^5 claims,
    and `config.threshold_on` row-by-row would ask a handful of regimes the same question that
    many times."""
    dates = pd.to_datetime(table[schema.DATE])
    tau_by_date = {d: config.threshold_on(version, d) for d in dates.unique()}
    return dates.map(tau_by_date)


def local_region(table: pd.DataFrame, version: str, h: float) -> pd.DataFrame:
    """Re-tags rows into region_local A'/B' by distance to THAT ROW's OWN tau (regime-aware for
    v2, single global for v3), keeping only |score - tau| <= h. Strict '>' matches the project's
    region convention (`src/threshold.py`): exactly at the band edge above tau is B', not A'."""
    dist = table["score"] - tau_by_date(table, version)
    band = table[dist.abs() <= h].copy()
    band["region_local"] = np.where(dist[dist.abs() <= h] > 0, "B", "A")
    return band


class ShapDiDEstimator(EffectEstimator):
    """See the module docstring for the estimator, its assumption, and the local (RDD) variant.

    Loads its own inputs from canonical artefacts (`loaders.load`, `config.split_path`) rather
    than taking a pre-built frame — the same "reads its own artefacts" rule as `concentration.py`
    (`src/docs/STRUCTURE.md` § "Five layers").
    """

    def assumptions(self) -> list[Assumption]:
        return [
            Assumption(
                name="parallel_trends",
                description=(
                    "Absent the forced-label mechanism, region B's SHAP concentration would have "
                    "drifted from region A's by the same amount in both versions. Untestable "
                    "directly (no counterfactual version exists); falsify() reports dose(v) as "
                    "the one observable proxy."
                ),
            ),
            Assumption(
                name="measurement_comparability",
                description=(
                    "Both versions must be attributed under the SAME SHAP backend/settings "
                    "(interventional vs tree-path-dependent, same perturbation/model_output) — "
                    "mixing backends puts a distributional artefact into the DiD that has "
                    "nothing to do with the forced-label mechanism, and (per "
                    "concentration.require_comparable()'s own docstring) produces a number that "
                    "LOOKS entirely reasonable, so this is checked as a hard failure, not a "
                    "caveat: falsify() raises if it does not hold."
                ),
            ),
            Assumption(
                name="no_case_mix_confound",
                description=(
                    "Region A vs B, compared before vs after, should differ only in which side "
                    "of tau they fell on — not in which CLAIMS happen to populate each version's "
                    "sample. Violated whenever the two versions were explained on their own "
                    "per-version sample rather than one shared claim set. This is a STANDING, "
                    "usually-UNMET limitation here, not a rare failure: v2's window "
                    "(2018-01..2020-09) and v3's (2023-06..2026-05) share no claim at all, so "
                    "every v2->v3 estimate is case-mix confounded by construction. falsify() "
                    "reports it, never blocks on it."
                ),
            ),
        ]

    def falsify(self, v_before: str, split_before: str, v_after: str, split_after: str) -> FalsificationReport:
        """Three checks, not auto-combined into one verdict:

        1. dose(v_before) vs dose(v_after) — negligible and close together argues the DiD below
           is NOT tracking forced-label dose (a parallel-trends violation, most plausibly an
           era/label-source jump); a real, substantially different dose on at least one side
           supports reading the DiD as an estimate. Deliberately not auto-classified into
           pass/fail beyond data availability — see the module docstring.
        2. measurement_comparability — `concentration.require_comparable()`'s backend/perturbation
           check. RAISES (does not degrade to a note) on a genuine mismatch: silently comparing
           two backends produces a plausible-looking wrong number, which is exactly what this
           whole ABC exists to refuse.
        3. no_case_mix_confound — the same call's shared-claim-set check, captured as a
           diagnostic rather than printed and discarded.
        """
        meter = DoseMeter()
        dose_before = meter.measure(v_before, split_before)
        dose_after = meter.measure(v_after, split_after)

        meta_before = load(v_before, split=split_before).attribution_meta
        meta_after = load(v_after, split=split_after).attribution_meta
        concentration.require_comparable({v_before: meta_before, v_after: meta_after})

        ids_before = meta_before.get("explain_ids_file")
        ids_after = meta_after.get("explain_ids_file")
        shared_claim_set = ids_before is not None and ids_before == ids_after

        return FalsificationReport(
            passed=dose_before is not None and dose_after is not None,
            diagnostics={
                "dose_before": dose_before, "dose_after": dose_after,
                "shared_claim_set": shared_claim_set,
            },
            note=(
                "read dose_before/dose_after against the DiD by hand: both negligible for this "
                "business AND close to each other -> a non-zero DiD argues against parallel "
                "trends. A real, substantially different dose on at least one side -> read the "
                "DiD as an estimate. "
                + ("Shared claim set: case-mix is not a confound here."
                   if shared_claim_set else
                   "NOT a shared claim set: every difference below is also confounded with "
                   "case-mix (standing, not fixable for this pair — see no_case_mix_confound).")
            ),
        )

    def estimate(self, v_before: str, split_before: str, v_after: str, split_after: str) -> EffectReport:
        """region(A/B) x version(before/after) DiD, each version on its OWN OOT split.

        `falsify()` (below) raises before any number is computed if the two versions are not
        `measurement_comparable` (mismatched SHAP backends) — there is no separate comparability
        check here any more; it lives in the one place all three assumption checks belong.
        """
        falsification = self.falsify(v_before, split_before, v_after, split_after)

        table_before = version_pair_table(v_before, split_before)
        table_after = version_pair_table(v_after, split_after)

        sA_before = cell_simpson(table_before, region="A")
        sB_before = cell_simpson(table_before, region="B")
        sA_after = cell_simpson(table_after, region="A")
        sB_after = cell_simpson(table_after, region="B")
        did = (sB_after - sA_after) - (sB_before - sA_before)

        fmt = lambda x: f"{x:.4%}" if x is not None else "unmeasured"
        dose_before = falsification.diagnostics["dose_before"]
        dose_after = falsification.diagnostics["dose_after"]
        print(f"{v_before}/{split_before}: simpson(A)={sA_before:.4f}  simpson(B)={sB_before:.4f}  "
              f"(B-A)={sB_before - sA_before:+.4f}  n={len(table_before):,}  dose={fmt(dose_before)}")
        print(f"{v_after}/{split_after}: simpson(A)={sA_after:.4f}  simpson(B)={sB_after:.4f}  "
              f"(B-A)={sB_after - sA_after:+.4f}  n={len(table_after):,}  dose={fmt(dose_after)}")
        print(f"\nDiD({v_before}→{v_after}) = {did:+.4f}")
        print(falsification.note)

        return EffectReport(
            estimate=did,
            falsification=falsification,
            diagnostics={
                "pair": f"{v_before}({split_before})->{v_after}({split_after})",
                "simpson_A_before": sA_before, "simpson_B_before": sB_before,
                "simpson_A_after": sA_after, "simpson_B_after": sB_after,
                "n_before": len(table_before), "n_after": len(table_after),
            },
        )

    def select_local_h(self, v_before: str, split_before: str, v_after: str, split_after: str,
                        grid: tuple[float, ...] = LOCAL_H_GRID, min_n: int = LOCAL_MIN_CELL_N,
                        h_max: float = LOCAL_H_MAX, fallback: float = LOCAL_H_FALLBACK
                        ) -> tuple[float, pd.DataFrame]:
        """Grid search on CELL COUNTS ONLY — never on a concentration or precision number computed
        from these rows (the same discipline 03_02 SS2c uses for `band_h`/`clip_hi`)."""
        table_before = version_pair_table(v_before, split_before)
        table_after = version_pair_table(v_after, split_after)
        dist_before = table_before["score"] - tau_by_date(table_before, v_before)
        dist_after = table_after["score"] - tau_by_date(table_after, v_after)

        rows = []
        for h in sorted(grid):
            in_before, in_after = dist_before.abs() <= h, dist_after.abs() <= h
            n = {"A_before": int((dist_before[in_before] <= 0).sum()),
                 "B_before": int((dist_before[in_before] > 0).sum()),
                 "A_after": int((dist_after[in_after] <= 0).sum()),
                 "B_after": int((dist_after[in_after] > 0).sum())}
            rows.append({"h": h, **n, "all_pass": min(n.values()) >= min_n})
        table = pd.DataFrame(rows).set_index("h")

        ok = table.index[table["all_pass"] & (table.index <= h_max)]
        if len(ok):
            chosen = float(ok[0])
            print(f"selected h={chosen} (smallest with every region_local x version cell >= "
                  f"{min_n}, h<={h_max})")
        else:
            chosen = fallback
            print(f"!! no h <= {h_max} reaches {min_n} claims in every cell -- the near-boundary "
                  f"band is thin at every local width tried. Keeping fallback h={fallback}; read "
                  f"the local DiD below as LOW-POWER, not as a clean estimate.")
        return chosen, table

    def estimate_local(self, v_before: str, split_before: str, v_after: str, split_after: str,
                        h: float) -> EffectReport:
        """Same mechanism as `estimate()`, but `region_local` (band `[tau-h, tau+h]`) instead of
        the whole-population region — a sharp-RDD robustness check, not a replacement. See the
        module docstring "Local (RDD) variant" for why this is analogous to a LATE, not literal
        one, and why v1 has no local estimate.

        Does not call `falsify()` — it is the SAME parallel-trends assumption as `estimate()`
        (already reported there for this pair); re-running it would only repeat the identical
        dose numbers.
        """
        tb = local_region(version_pair_table(v_before, split_before), v_before, h)
        ta = local_region(version_pair_table(v_after, split_after), v_after, h)

        sA_before = cell_simpson(tb, drop_extra=["region_local"], region_local="A")
        sB_before = cell_simpson(tb, drop_extra=["region_local"], region_local="B")
        sA_after = cell_simpson(ta, drop_extra=["region_local"], region_local="A")
        sB_after = cell_simpson(ta, drop_extra=["region_local"], region_local="B")
        did_local = (sB_after - sA_after) - (sB_before - sA_before)

        n_before = {r: int((tb["region_local"] == r).sum()) for r in ("A", "B")}
        n_after = {r: int((ta["region_local"] == r).sum()) for r in ("A", "B")}
        print(f"{v_before}: simpson(A')={sA_before:.4f}  simpson(B')={sB_before:.4f}  "
              f"n=(A'={n_before['A']:,}, B'={n_before['B']:,})")
        print(f"{v_after}: simpson(A')={sA_after:.4f}  simpson(B')={sB_after:.4f}  "
              f"n=(A'={n_after['A']:,}, B'={n_after['B']:,})")
        print(f"\nlocal DiD({v_before}→{v_after}, h={h}) = {did_local:+.4f}")

        return EffectReport(
            estimate=did_local,
            falsification=self.falsify(v_before, split_before, v_after, split_after),
            diagnostics={
                "pair": f"{v_before}({split_before})->{v_after}({split_after})", "h": h,
                "simpson_A_before": sA_before, "simpson_B_before": sB_before,
                "simpson_A_after": sA_after, "simpson_B_after": sB_after,
                "n_before": sum(n_before.values()), "n_after": sum(n_after.values()),
            },
        )

    def estimate_shared(self, v3_split: str) -> EffectReport:
        """The case-mix-free complement to `estimate()` — see module docstring "Paired
        shared-claim variant". Same DiD mechanism (region(A/B) x before/after on Simpson
        concentration), but `before`/`after` are v2's production-log SHAP and v3's own SHAP on
        the IDENTICAL claim_id set, and `region` is FIXED at v2's original production decision
        rather than re-derived per side. `no_case_mix_confound` does not apply to this number —
        both arms are the same claims by construction — at the cost of a much smaller n (only
        the claims v2's log actually covers for this v3 split; see `corrector_targets`'s own
        coverage meta, `03_01_corrector_inputs.ipynb`).

        Does not call `falsify()`: `measurement_comparability` is already checked inside
        `shared_claim_table()` (`concentration.require_comparable`), `parallel_trends`'s dose
        proxy is the SAME dose(v2)/dose(v3) `estimate()` already reports for this pair, and
        `no_case_mix_confound` is true by construction here, not a diagnostic to re-derive.
        """
        before, after = shared_claim_table(v3_split)

        sA_before = cell_simpson(before, region="A")
        sB_before = cell_simpson(before, region="B")
        sA_after = cell_simpson(after, region="A")
        sB_after = cell_simpson(after, region="B")
        did = (sB_after - sA_after) - (sB_before - sA_before)

        print(f"v2(log, shared w/ v3/{v3_split}): simpson(A)={sA_before:.4f}  "
              f"simpson(B)={sB_before:.4f}  (B-A)={sB_before - sA_before:+.4f}  n={len(before):,}")
        print(f"v3({v3_split}, shared w/ v2 log): simpson(A)={sA_after:.4f}  "
              f"simpson(B)={sB_after:.4f}  (B-A)={sB_after - sA_after:+.4f}  n={len(after):,}")
        print(f"\nDiD_shared(v2→v3, v3/{v3_split}) = {did:+.4f}")
        print("region held FIXED at v2's OWN production decision, same claim_id set both arms — "
              "no_case_mix_confound does not apply to this number.")

        return EffectReport(
            estimate=did,
            falsification=FalsificationReport(
                passed=True,
                diagnostics={"shared_claim_set": True, "n_shared": len(before)},
                note="region fixed at v2's own production decision; both arms are the IDENTICAL "
                     "claim_id set by construction — no_case_mix_confound does not apply here.",
            ),
            diagnostics={
                "pair": f"v2(log)->v3({v3_split}) [shared claims, n={len(before)}]",
                "simpson_A_before": sA_before, "simpson_B_before": sB_before,
                "simpson_A_after": sA_after, "simpson_B_after": sB_after,
                "n_before": len(before), "n_after": len(after),
            },
        )

    def select_shared_local_h(self, v3_split: str, grid: tuple[float, ...] = LOCAL_H_GRID,
                               min_n: int = LOCAL_MIN_CELL_N, h_max: float = LOCAL_H_MAX,
                               fallback: float = LOCAL_H_FALLBACK) -> tuple[float, pd.DataFrame]:
        """Grid search on CELL COUNTS ONLY — same discipline as `select_local_h()`. Only two
        cells (A, B), not four: `shared_local_region()` bands around v2's OWN tau and applies the
        SAME region_local to both arms, so n_before == n_after by construction here — there is
        nothing to gate separately per arm, unlike the whole-population local estimator where
        each version's own independent sample can carry different cell counts."""
        ct = pd.read_parquet(shared_claim_paths(v3_split)["explain_ids"])[[ID_COL, schema.DATE, "score"]]
        dist = ct["score"] - tau_by_date(ct, "v2")

        rows = []
        for h in sorted(grid):
            in_band = dist.abs() <= h
            n = {"A": int((dist[in_band] <= 0).sum()), "B": int((dist[in_band] > 0).sum())}
            rows.append({"h": h, **n, "all_pass": min(n.values()) >= min_n})
        table = pd.DataFrame(rows).set_index("h")

        ok = table.index[table["all_pass"] & (table.index <= h_max)]
        if len(ok):
            chosen = float(ok[0])
            print(f"selected h={chosen} (smallest with both region_local cells >= {min_n}, "
                  f"h<={h_max})")
        else:
            chosen = fallback
            print(f"!! no h <= {h_max} reaches {min_n} claims in both cells -- the near-boundary "
                  f"band is thin at every local width tried. Keeping fallback h={fallback}; read "
                  f"the local shared-claim DiD below as LOW-POWER, not as a clean estimate.")
        return chosen, table

    def estimate_shared_local(self, v3_split: str, h: float) -> EffectReport:
        """The sharp-RDD twin of `estimate_shared()` — same relationship `estimate_local()` has
        to `estimate()`. Same case-mix-free design (identical claim_id set, region fixed at v2's
        own production decision), narrowed via `shared_local_region()` to a band around v2's OWN
        tau — the boundary that actually assigned that decision, never re-derived on the "after"
        side. A robustness check on `estimate_shared()`, not a replacement: if the local and
        whole-claim-set DiD agree in sign and rough size, `estimate_shared()`'s number is not
        simply an artefact of region A/B having different covariate distributions even within
        this already-paired sample.

        Does not call `falsify()` — same reasoning as `estimate_shared()`: nothing here is a
        fresh diagnostic to re-derive.
        """
        tb, ta = shared_local_region(v3_split, h)

        sA_before = cell_simpson(tb, drop_extra=["region_local"], region_local="A")
        sB_before = cell_simpson(tb, drop_extra=["region_local"], region_local="B")
        sA_after = cell_simpson(ta, drop_extra=["region_local"], region_local="A")
        sB_after = cell_simpson(ta, drop_extra=["region_local"], region_local="B")
        did_local = (sB_after - sA_after) - (sB_before - sA_before)

        n = {r: int((tb["region_local"] == r).sum()) for r in ("A", "B")}
        print(f"v2(log, shared w/ v3/{v3_split}): simpson(A')={sA_before:.4f}  "
              f"simpson(B')={sB_before:.4f}  n=(A'={n['A']:,}, B'={n['B']:,})")
        print(f"v3({v3_split}, shared w/ v2 log): simpson(A')={sA_after:.4f}  "
              f"simpson(B')={sB_after:.4f}  n=(A'={n['A']:,}, B'={n['B']:,})  [same claims both arms]")
        print(f"\nlocal DiD_shared(v2→v3, v3/{v3_split}, h={h}) = {did_local:+.4f}")

        return EffectReport(
            estimate=did_local,
            falsification=FalsificationReport(
                passed=True,
                diagnostics={"shared_claim_set": True, "n_shared": len(tb), "h": h},
                note="region_local fixed at v2's own production decision, banded around v2's own "
                     "tau; both arms are the IDENTICAL claim_id set by construction — "
                     "no_case_mix_confound does not apply here.",
            ),
            diagnostics={
                "pair": f"v2(log)->v3({v3_split}) [shared claims, local h={h}, n={len(tb)}]",
                "h": h,
                "simpson_A_before": sA_before, "simpson_B_before": sB_before,
                "simpson_A_after": sA_after, "simpson_B_after": sB_after,
                "n_before": len(tb), "n_after": len(ta),
            },
        )
