"""v1 CSV → parquet. Runs in the analysis `.venv`, NOT in env-v1.

    uv run python v1_csv_to_parquet.py                 # convert everything v1 exported
    uv run python v1_csv_to_parquet.py --dry-run       # list what would be converted
    uv run python v1_csv_to_parquet.py --splits train  # just one split

WHY THIS STEP EXISTS AT ALL. env-v1 has **no parquet engine** — `src/envs/v1/requirements.txt`
pins scikit-learn / pandas / pyodbc / matplotlib 2.2.5 / ipykernel 4.10.1 / joblib 0.14.1 and
nothing else, and neither pyarrow nor fastparquet can be added on Python 3.5.6. So
`notebook/real/01_export_v1.ipynb` writes **CSV**, and this script converts it where an engine
exists. v2 and v3 have modern envs and write parquet directly; this is a v1-only stage.

PATHS COME FROM `config`, NEVER FROM A LIST HERE. The destination is whatever
`config.path(kind, "v1", split=...)` resolves to and the source is that same path with a `.csv`
suffix — which is exactly the naming `01_export_v1.ipynb` mirrors. An earlier version of this file
carried a hand-written `PAIRS` list of UNSPLIT names (`features_v1.csv` → `features_v1.parquet`)
and had already drifted out of step with the notebook, which writes one file per split. Deriving
the names removes the class of bug rather than fixing one instance of it.

The parquet destinations are config's FALLBACK locations, so once this has run
`loaders.load("v1", split=...)` resolves everything with no config declaration.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import config  # noqa: E402

VERSION = "v1"

#: The per-split kinds 01_export_v1 writes, in the order it writes them.
SPLIT_KINDS = ("processed_inputs", "targets", "scores")

#: Single-file kinds. `raw_dataset` is written by a line that is COMMENTED OUT in the notebook as
#: committed, so it is normal for it to be absent — missing sources are reported, never fatal.
SINGLE_KINDS = ("raw_dataset",)


def to_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns off a CSV mix str with float NaN, which Arrow cannot type.

    Cast them to pandas' nullable string dtype so NaN becomes <NA>. Values are preserved: this
    only fixes the *dtype*, never the content. In particular ID-ish code columns stay strings on
    purpose — see the notes at the bottom of this file.
    """
    for c in df.columns[df.dtypes == "object"]:
        df[c] = df[c].astype("string")
    return df


def convert(csv_path: pathlib.Path, parquet_path: pathlib.Path, dry_run: bool) -> tuple[int, int]:
    """One CSV → one parquet, with a row-count check. Returns (rows, cols)."""
    df = pd.read_csv(csv_path, low_memory=False)
    mixed = [c for c in df.columns[df.dtypes == "object"]
             if df[c].map(type).nunique(dropna=True) > 1]
    if mixed:
        print(f"      mixed-type object columns cast to string: {mixed[:8]}"
              f"{' …' if len(mixed) > 8 else ''}")

    if not dry_run:
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        to_arrow_safe(df).to_parquet(parquet_path, index=False)

        # Re-read the metadata rather than trust the write — a truncated parquet is silent, and
        # everything downstream joins on these row sets.
        import pyarrow.parquet as pq

        n = pq.read_metadata(parquet_path).num_rows
        if n != len(df):
            raise SystemExit(f"\n{parquet_path.name}: wrote {n} rows but the CSV has {len(df)}\n")

    return len(df), df.shape[1]


def jobs(splits: list[str], source: str) -> list[tuple[str, pathlib.Path, pathlib.Path]]:
    """(label, csv, parquet) for every artefact, derived from config — never hand-listed."""
    out = []
    for split in splits:
        for kind in SPLIT_KINDS:
            dst = config.path(kind, VERSION, source, split=split)
            out.append((f"{kind}/{split}", dst.with_suffix(".csv"), dst))
    for kind in SINGLE_KINDS:
        dst = config.path(kind, VERSION, source)
        out.append((kind, dst.with_suffix(".csv"), dst))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="real", choices=("real", "synthetic"))
    p.add_argument("--splits", nargs="+", default=list(config.SPLITS[VERSION]),
                   help=f"v1 splits to convert (default: all — {config.SPLITS[VERSION]})")
    p.add_argument("--dry-run", action="store_true", help="list what would be converted")
    a = p.parse_args()

    for s in a.splits:
        if s not in config.SPLITS[VERSION]:
            raise SystemExit(f"\nunknown v1 split {s!r}; expected one of "
                             f"{config.SPLITS[VERSION]}\n")

    done, missing, rows_total = [], [], 0
    for label, csv_path, parquet_path in jobs(a.splits, a.source):
        if not csv_path.exists():
            missing.append((label, csv_path))
            continue
        rows, cols = convert(csv_path, parquet_path, a.dry_run)
        rows_total += rows
        done.append(label)
        arrow = "would ->" if a.dry_run else "->"
        print(f"  {label:<24} {csv_path.name:<30} {arrow} {parquet_path.name:<34} "
              f"{rows:>9,} rows x {cols:>4} cols")

    if missing:
        print("\nno CSV found (fine when that artefact was not exported — `raw_v1.csv` is written "
              "by a line that is commented out in 01_export_v1.ipynb):")
        for label, csv_path in missing:
            print(f"  {label:<24} {csv_path.relative_to(config.ROOT)}")

    verb = "would convert" if a.dry_run else "converted"
    print(f"\n{verb} {len(done)} file(s), {rows_total:,} rows total"
          f"{' (dry run — nothing written)' if a.dry_run else ''}")
    if done and not a.dry_run:
        print("These parquet paths ARE config's fallback locations, so "
              "`loaders.load(\"v1\", split=...)` now resolves with no config declaration.")


if __name__ == "__main__":
    main()


# Notes:
# - The train split's scores are **in-sample** (predictions.pkl is train-time output); carry that
#   caveat into anything that uses them.
# - Split membership is reconstructable from `date` against 2017-04-01 / 2017-06-30 boundaries
#   (the 80/20 `train_test_split` within the first window was random — NOT reconstructable).
# - `cc_fttl`-flagged rows (~2.5%) were EXCLUDED from training; the raw table still has them.
# - Columns like `abicode_ext` are ID-ish codes, not numbers: they stay strings on purpose.
#   Do NOT "fix" them with pd.to_numeric — leading zeros and non-numeric suffixes would be lost.
