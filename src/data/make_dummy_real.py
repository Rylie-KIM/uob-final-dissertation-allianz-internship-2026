"""Write FAKE artefacts into src/data/real/ with the real shapes, so the chain runs off-site.

WHY THIS EXISTS. The real matrices, targets, scores and logs only exist on the company laptop.
Everything in `src/` and `notebook/real/` is written here, blind, and must be runnable on arrival
— so it needs something with the right SHAPE to run against locally: the right kinds, the right
per-split filenames, the right canonical columns, plausible score distributions either side of
each version's real cutoff. That is all this generates. It is scaffolding for the wiring, exactly
like `data/synthetic/`'s app dry-run, and it is NOT a dataset anyone analyses.

WHAT MAKES IT SAFE TO KEEP. Every value is drawn from a seeded RNG. No Allianz number, name or
claim is involved, and none could be: this file runs on a laptop that has never seen the data. A
`_DUMMY_DATA` marker is written beside the artefacts, and `loaders` prints nothing about it — the
marker is for a human who opens the directory and needs to know in one second what they are
looking at.

    python src/data/make_dummy_real.py              # write (refuses to touch a real tree)
    python src/data/make_dummy_real.py --force      # overwrite the previous dummy set
    python src/data/make_dummy_real.py --clean      # delete the dummy set, leave nothing else

⚠️ On the company laptop, do not run this. It refuses when the target directory holds files it did
not write, but the first line of defence is not running it where real exports live.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # noqa: E402
import schema  # noqa: E402

MARKER = "_DUMMY_DATA"          # written into src/data/real/; its presence is the licence to write
SEED = 20260818

# Split sizes. v1's are its REAL row counts (read off train.py's run, 2026-08-11) divided by 100,
# so the proportions a reader sees locally are the proportions the real run had. v2's and v3's are
# invented in the same spirit: a large train, a small val, a holdout tail.
ROWS: dict[str, dict[str, int]] = {
    "v1": {"train": 1738, "test": 435, "val1": 353, "val2": 512},
    "v2": {"train": 1600, "val": 400, "test": 500},
    "v3": {"train": 1500, "test": 375, "oot": 480},
}

# Feature names are each model's own ENCODED feature names and DELIBERATELY DIVERGENT across versions — the encoding divergence is a
# real finding of the mapping work (v1 carries 55 `make_*` columns, v2 41, and v2's top feature
# `location_Home` is absent from the other two). A dummy set that used one tidy shared schema
# would make every overlap and concentration figure look better than it can be.
FEATURES: dict[str, list[str]] = {
    "v1": (
        [f"make_{m}" for m in ("FORD", "VAUXHALL", "BMW", "AUDI", "VW", "NISSAN",
                               "TOYOTA", "MERCEDES", "PEUGEOT", "RENAULT", "KIA", "OTHER")]
        + ["age_of_vehicle", "estimated_repair_cost", "vehicle_value", "airbag_deployed",
           "damage_severity", "n_damaged_areas", "driveable", "vehicle_mobility_status_code"]
    ),
    "v2": (
        [f"make_{m}" for m in ("FORD", "VAUXHALL", "BMW", "AUDI", "VW", "NISSAN", "OTHER")]
        + ["location_Home", "location_Roadside", "location_Garage",
           "AgeOfVehicle", "EstimatedRepairCost", "VehicleValue", "AirbagsDeployed",
           "DamageSeverity", "NumberOfDamagedAreas", "Roadworthy", "ExtricationRequired"]
    ),
    "v3": (
        [f"make_{m}" for m in ("FORD", "VAUXHALL", "BMW", "AUDI", "VW", "OTHER")]
        + ["AgeOfVehicle_VEH", "EstimatedRepairCost_VEH", "VehicleValue_VEH",
           "AirbagsDeployed_VEH", "DamageSeverity_VEH", "NumberOfDamagedAreas_VEH",
           "Roadworthy_VEH", "ExtricationRequired_VEH", "TotalLossPropensity"]
    ),
}

# Each version's own date window, so a date filter written against the real windows behaves. v2:
# 2018-01-02 -> 2020-09-30; v3: 2023-06 -> 2026-05 (both user-confirmed). v1 is the pre-ML/early-ML
# era ahead of v2.
WINDOWS: dict[str, tuple[str, str]] = {
    "v1": ("2016-01-04", "2021-12-31"),
    "v2": ("2018-01-02", "2020-09-30"),
    "v3": ("2023-06-01", "2026-05-31"),
}


# ======================================================================================
# the pieces
# ======================================================================================

def _scores(rng: np.random.RandomState, n: int, tau: float) -> np.ndarray:
    """A right-skewed bulk, plus deliberate mass ABOVE tau and just BELOW it.

    Beta(2, 9) alone leaves both regions nearly empty, and then nothing downstream has anything to
    work on: an empty treated set cannot exercise a detector (the synthetic DGP's thin-scrap
    problem, in miniature), and an empty *boundary band* cannot exercise the analyses that live
    there — v1's mobility overlap band `(0.75, 0.85]`, the RDD window, `03_02`'s `[tau-h, tau)`
    corrector, and `02_error_inheritance`'s near-boundary bins. So both are drawn on purpose:

        ~8%  above tau        the fast-track region
        ~7%  in [tau-0.12, tau)   the boundary band

    These proportions are a convenience for exercising code, NOT a claim about the real score
    distribution — the real fast-track share is a measured quantity and is much smaller.
    """
    base = rng.beta(2.0, 9.0, size=n)
    picks = rng.choice(n, size=int(n * 0.15), replace=False)
    n_high = int(n * 0.08)
    high, band = picks[:n_high], picks[n_high:]
    base[high] = tau + (1.0 - tau) * rng.beta(1.6, 1.6, size=len(high))
    base[band] = rng.uniform(max(tau - 0.12, 0.0), tau, size=len(band))
    return np.clip(base, 1e-6, 1 - 1e-6)


def _tau_for(version: str) -> float:
    rule = config.DECISION_RULES[version]
    if rule["shape"] == "global":
        return float(rule["threshold"])
    if rule["shape"] == "piecewise_global":
        return float(rule["regimes"][0]["threshold"])
    return float(max(rule["thresholds"].values()))       # v1: the higher (mobile) cutoff


def _features(rng: np.random.RandomState, ids: np.ndarray, version: str) -> pd.DataFrame:
    cols = {schema.CLAIM_ID: ids}
    for name in FEATURES[version]:
        if name.startswith("make_") or name.startswith("location_"):
            cols[name] = rng.binomial(1, 0.12, size=len(ids))          # one-hot-ish
        elif any(k in name for k in ("Cost", "cost", "Value", "value")):
            cols[name] = np.round(rng.lognormal(8.0, 0.7, size=len(ids)), 2)
        elif any(k in name for k in ("Age", "age")):
            cols[name] = rng.randint(0, 22, size=len(ids))
        elif any(k in name for k in ("Airbag", "airbag", "Roadworthy", "driveable",
                                     "Extrication")):
            cols[name] = rng.binomial(1, 0.3, size=len(ids))
        else:
            cols[name] = np.round(rng.rand(len(ids)), 4)
    return pd.DataFrame(cols)


def _claim_ids(version: str, rng: np.random.RandomState) -> dict[str, tuple[np.ndarray, int]]:
    """Claim ids per split — and the OVERLAP between versions is the point.

    The real windows are v1 2016–2021, v2 2018-01→2020-09, v3 2023-06→2026-05. So v2's claims sit
    INSIDE v1's production era and largely rejoin them (which is what `02_error_inheritance`
    depends on: v1 band rows must be findable in v2's scores), while v3 is ~2¾ years later and
    shares nothing. A dummy set with three disjoint id spaces would make every cross-version join
    return zero rows and look like a bug in the notebook rather than the intended shape.

    v1 and v2 therefore draw from one id pool; v3 gets its own.
    """
    total = sum(ROWS[version].values())
    if version == "v3":
        pool = np.arange(30_000_000, 30_000_000 + total)          # its own era
    else:
        # v1 spans the pool; v2 samples inside it, so ~most v2 claims are also v1 claims.
        pool = rng.choice(np.arange(10_000_000, 10_000_000 + 6_000), size=total, replace=False)
    pool = np.sort(pool)

    out, cursor = {}, 0
    for split in config.SPLITS[version]:
        n = ROWS[version][split]
        out[split] = (pool[cursor:cursor + n], n)
        cursor += n
    return out


def _dates(rng: np.random.RandomState, n: int, version: str) -> pd.Series:
    lo, hi = (pd.Timestamp(d) for d in WINDOWS[version])
    offs = rng.randint(0, (hi - lo).days + 1, size=n)
    return pd.Series(lo + pd.to_timedelta(np.sort(offs), unit="D"))


# ======================================================================================
# writing
# ======================================================================================

def build(version: str, root: pathlib.Path, written: list[pathlib.Path]) -> None:
    rng = np.random.RandomState(SEED + int(version[1:]))
    tau = _tau_for(version)
    target_col = config.column(version, "observed")           # the version's OWN target name

    per_split = _claim_ids(version, rng)

    raw_parts = []
    for split, (ids, n) in per_split.items():
        feats = _features(rng, ids, version)
        dates = _dates(rng, n, version)
        scores = _scores(rng, n, tau)

        # `observed` is the CONTAMINATED target, and the contamination is the point: every row
        # above tau is a forced 1 (scrapped, never garage-verified). Below tau it is drawn with a
        # score-dependent rate, which is what an uncontaminated-ish garage outcome looks like.
        forced = scores >= tau
        observed = np.where(forced, 1, rng.binomial(1, np.clip(scores * 1.4, 0, 0.95)))

        # The matrix CARRIES THE TARGET, in every version — user-confirmed 2026-08-19, matching
        # what all three export notebooks do (v2 writes `[ID] + MODEL_FEATURES + [TARGET]`; v1
        # and v3 write the whole transformed frame). The dummy must reproduce that, because it is
        # the shape every consumer has to survive.
        #
        # ⚠️ So "every column except claim_id is a feature" is WRONG on real data. Anything that
        # selects features must name the non-feature columns or ask the fitted model which columns
        # it was trained on — see training/retrain.py, which does the latter.
        feats[target_col] = observed
        _write(feats, config.path("processed_inputs", version, "real", split=split), written)

        targets = pd.DataFrame({
            schema.CLAIM_ID: ids,
            schema.DATE: dates.values,
            schema.OBSERVED: observed,
        })
        _write(targets, config.path("targets", version, "real", split=split), written)

        _write(pd.DataFrame({schema.CLAIM_ID: ids, f"model_{version}_score": scores}),
               config.path("scores", version, "real", split=split), written)

        raw_parts.append(pd.DataFrame({
            schema.CLAIM_ID: ids,
            config.column(version, "date"): dates.values,
            target_col: observed,
            "postcode_area": rng.choice(list("ABCDEFGH"), size=n),
            "channel": rng.choice(["FNOL", "ENOL"], size=n, p=[0.62, 0.38]),
        }))

    raw = pd.concat(raw_parts, ignore_index=True)
    if version == "v1":                     # only v1's rule is segmented on mobility
        raw[config.column("v1", "mobility")] = rng.choice(
            ["Mobile", "Mobile Not Roadworthy", "Mobile Not Secure", "Immobile"],
            size=len(raw), p=[0.35, 0.1, 0.05, 0.5])
    _write(raw, config.path("raw_dataset", version, "real"), written)

    # The canonical production log — ONLY v2 has one. v1's was destroyed and v3 never deployed, so
    # generating those two would fake the single hardest constraint the project works under.
    if version == "v2":
        pooled = pd.concat(
            [pd.read_parquet(config.path("targets", "v2", "real", split=s))
             .merge(pd.read_parquet(config.path("scores", "v2", "real", split=s)),
                    on=schema.CLAIM_ID)
             for s in config.SPLITS["v2"]],
            ignore_index=True,
        ).rename(columns={"model_v2_score": schema.SCORE})
        pooled[schema.DECISION] = (pooled[schema.SCORE] >= tau).astype(int)
        _write(pooled[[schema.CLAIM_ID, schema.DATE, schema.SCORE,
                       schema.DECISION, schema.OBSERVED]],
               config.path("log", "v2", "real"), written)

        # The log's TRANSFORMED matrix (kind `log_features`) — the model's own feature columns for
        # the log's rows, no target. Written by 01_export_v2_logs.ipynb from what production
        # actually scored; here it is pooled off the split matrices instead, because the dummy
        # only owes the chain a SHAPE. On the real tree these rows are NOT the training splits:
        # the log spans the serving history, most of which postdates training.
        #
        # The other log kinds (`log_raw`, `log_scores`, `log_targets`) get no dummy: they carry
        # v2's own encoded column names and the correlation-id event key, neither of which this
        # generator knows, and nothing downstream reads them yet — `log` above already gives the
        # chain the shape it needs. Their real counterparts come out of 01_export_v2_logs.ipynb.
        log_feats = pd.concat(
            [pd.read_parquet(config.path("processed_inputs", "v2", "real", split=s))
             for s in config.SPLITS["v2"]],
            ignore_index=True,
        ).drop(columns=[target_col])
        log_feats = pooled[[schema.CLAIM_ID]].merge(log_feats, on=schema.CLAIM_ID, how="left")
        _write(log_feats, config.path("log_features", "v2", "real"), written)


def _write(df: pd.DataFrame, path: pathlib.Path, written: list[pathlib.Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    written.append(path)
    print(f"  {path.relative_to(config.ROOT)!s:<58} {len(df):>6} rows x {len(df.columns):>3} cols")


# ======================================================================================
# guards + entry point
# ======================================================================================

def _root() -> pathlib.Path:
    return config.ROOT / "src" / "data" / "real"


def _guard(root: pathlib.Path, force: bool) -> None:
    """Never write over files this script did not write.

    The marker is the whole check: a tree holding real exports has no marker, so the first run on
    the company laptop stops before touching anything. --force only overrides a tree that is
    ALREADY marked dummy.
    """
    marker = root / MARKER
    existing = [p for p in root.rglob("*") if p.is_file() and p.name != MARKER] if root.exists() else []
    if not existing:
        return
    if not marker.exists():
        raise SystemExit(
            f"\n{root.relative_to(config.ROOT)} already holds {len(existing)} file(s) and no "
            f"{MARKER} marker, so this is not a dummy tree — refusing to touch it.\n"
            f"If these really are disposable, delete them yourself first.\n"
        )
    if not force:
        raise SystemExit(
            f"\na dummy set is already there ({len(existing)} files). Re-run with --force to "
            f"overwrite it, or --clean to remove it.\n"
        )


def _clean(root: pathlib.Path) -> None:
    marker = root / MARKER
    if not marker.exists():
        raise SystemExit(f"\nno {MARKER} marker in {root} — nothing here is known to be dummy.\n")
    listed = json.loads(marker.read_text(encoding="utf-8")).get("files", [])
    sidecars = 0
    for rel in listed:
        target = config.ROOT / rel
        target.unlink(missing_ok=True)
        # v1_parquet_to_csv.py writes a CSV beside every v1 parquet (env-v1 has no parquet
        # engine), and those are not in the marker's list because that script also runs on the
        # real tree, where there is no marker to write into. Left behind, they would sit in a
        # directory this clean is meant to empty — and the next --force would then find files
        # with no marker and refuse. So take the sibling with the parquet.
        csv = target.with_suffix(".csv")
        if target.suffix == ".parquet" and csv.exists():
            csv.unlink()
            sidecars += 1
    marker.unlink()
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    print(f"removed {len(listed)} dummy files"
          + (f" (+{sidecars} CSV sidecars)" if sidecars else "") + " and the marker")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--versions", nargs="+", default=list(config.VERSION_LABELS))
    p.add_argument("--force", action="store_true", help="overwrite an existing DUMMY set")
    p.add_argument("--clean", action="store_true", help="delete the dummy set and stop")
    a = p.parse_args()

    root = _root()
    if a.clean:
        _clean(root)
        return

    _guard(root, a.force)
    root.mkdir(parents=True, exist_ok=True)

    written: list[pathlib.Path] = []
    for v in a.versions:
        print(f"\n[{v}]  splits {config.SPLITS[v]}  ·  tau {_tau_for(v)}  ·  "
              f"{len(FEATURES[v])} features")
        build(v, root, written)

    (root / MARKER).write_text(json.dumps({
        "what": "FAKE data with the real shapes — generated by src/data/make_dummy_real.py",
        "why": "the real artefacts exist only on the company laptop; this exercises the wiring",
        "not": "no Allianz value, name or claim is present; every number is from a seeded RNG",
        "seed": SEED,
        "versions": a.versions,
        "files": sorted(str(p.relative_to(config.ROOT)) for p in written),
    }, indent=2), encoding="utf-8")

    print(f"\nwrote {len(written)} files + {MARKER}")
    print("delete with: python src/data/make_dummy_real.py --clean")


if __name__ == "__main__":
    main()
