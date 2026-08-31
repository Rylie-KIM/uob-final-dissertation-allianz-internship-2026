"""v1 parquet -> CSV. The mirror image of `v1_csv_to_parquet.py`, and it runs in the same place.

    uv run python v1_parquet_to_csv.py                 # every v1 parquet under src/data/real
    uv run python v1_parquet_to_csv.py --dry-run       # list what would be written
    uv run python v1_parquet_to_csv.py --splits train  # only files whose name carries that split

WHY BOTH DIRECTIONS EXIST. env-v1 is Python 3.5.6 with **no parquet engine** — neither pyarrow nor
fastparquet installs there (`src/envs/v1/requirements.txt`). So anything that has to be READ inside
env-v1 must be a CSV:

    export       01_export_v1.ipynb (env-v1)  writes CSV  ->  v1_csv_to_parquet.py  ->  parquet
    consume back  parquet  ->  THIS SCRIPT  ->  CSV  ->  attribute/predict running in env-v1

v2 and v3 have modern envs and read parquet directly; this is a v1-only stage in both directions.
Both formats are kept on disk on purpose — parquet is what the analysis `.venv` and every notebook
under `notebook/real/` load through `loaders.load()`, CSV is what env-v1 can open. Neither is a
copy to be cleaned up.

WHICH FILES. Every `*.parquet` under `src/data/real/` whose path carries `v1` as a whole token —
`features_v1_train.parquet`, `raw_v1.parquet`, `v1_scores_val2.parquet`,
`detection/shap/v1/v1_attributions_train_native.parquet`. A glob rather than a config-derived list
(which is what `v1_csv_to_parquet.py` uses) because the direction is different: that script must
know which artefacts it EXPECTS so it can report a missing export, while this one converts
whatever is actually there — including kinds config gained after this file was written, and the
`--out-suffix` attribution variants config.path() cannot name.

The CSV lands beside the parquet under the same stem, which is exactly the pairing
`v1_csv_to_parquet.py` reads back.

⚠️ CSV IS LOSSY ABOUT DTYPES, and only the dtypes. Values survive; what does not is the *type* a
reader infers. A date column comes back as a string unless parsed, and a zero-padded code column
comes back as an int. That asymmetry already exists in the other direction and is why
`v1_csv_to_parquet.to_arrow_safe` casts object columns to string. Anything reading these CSVs in
env-v1 should name its dtypes rather than trust inference.

⚠️ AND READ FLOATS WITH `float_precision="round_trip"`. The values written here are exact —
`to_csv` uses repr() — but `read_csv`'s default parser is a fast, not correctly-rounded,
string->double conversion and returns them ~1e-16 off. Harmless in a score column; not harmless
in a SHAP additivity check, which is the main reason v1 reads a CSV at all. The argument exists
in pandas 0.24, so env-v1 can pass it.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
import config  # noqa: E402

VERSION = "v1"


def targets(root: pathlib.Path, version: str, splits: list[str] | None) -> list[pathlib.Path]:
    """Every parquet under `root` belonging to `version`, optionally filtered by split name.

    The token test is deliberate: a substring test would match `v10` or `rev1_`, and the split
    filter is applied to the STEM's tokens for the same reason (`train` must not match
    `retrain`).
    """
    token = re.compile(r"(?<![0-9a-zA-Z])%s(?![0-9a-zA-Z])" % re.escape(version))
    out = []
    for p in sorted(root.rglob("*.parquet")):
        rel = str(p.relative_to(root))
        if not token.search(rel):
            continue
        if splits and not any(re.search(r"(?<![0-9a-zA-Z])%s(?![0-9a-zA-Z])" % re.escape(s), p.stem)
                              for s in splits):
            continue
        out.append(p)
    return out


def convert(parquet_path: pathlib.Path, csv_path: pathlib.Path, dry_run: bool) -> tuple[int, int, str]:
    """One parquet -> one CSV, read back and checked. Returns (rows, cols, note)."""
    df = pd.read_parquet(parquet_path)
    if dry_run:
        return len(df), df.shape[1], ""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Read the CSV back rather than trust the write. A truncated file is silent, and these row
    # sets are what every downstream join keys on. Floats are written with repr(), which
    # round-trips exactly for float64 — so this checks the values too, not just the shape.
    #
    # `float_precision="round_trip"` is load-bearing, NOT belt-and-braces: read_csv's default C
    # parser uses a fast, not correctly-rounded, string->double conversion and comes back ~1e-16
    # off. That is invisible in a score column and fatal in an additivity check, which is exactly
    # what env-v1 opens these attributions to do. Whatever reads them there needs the same
    # argument (it exists in pandas 0.24).
    back = pd.read_csv(csv_path, low_memory=False, float_precision="round_trip")
    if len(back) != len(df) or list(back.columns) != list(df.columns):
        raise SystemExit(
            f"\n{csv_path.name}: wrote {len(back)} rows x {back.shape[1]} cols but the parquet "
            f"has {len(df)} x {df.shape[1]}\n")

    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    gap = max((float(np.nanmax(np.abs(back[c].to_numpy(float) - df[c].to_numpy(float))))
               for c in num), default=0.0)
    if gap > 0:
        raise SystemExit(f"\n{csv_path.name}: numeric round-trip differs by {gap:.3e}\n")

    lost = [c for c in df.columns if str(df[c].dtype) != str(back[c].dtype)]
    return len(df), df.shape[1], (f"dtype not preserved: {lost[:4]}" if lost else "")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="real", choices=("real", "synthetic"))
    p.add_argument("--version", default=VERSION, choices=list(config.VERSION_LABELS),
                   help="v2/v3 envs read parquet directly and do not need this")
    p.add_argument("--splits", nargs="+", default=None,
                   help=f"only files naming these splits (v1's are {config.SPLITS[VERSION]})")
    p.add_argument("--dry-run", action="store_true", help="list what would be written")
    a = p.parse_args()

    if a.splits:
        for s in a.splits:
            if s not in config.SPLITS[a.version]:
                raise SystemExit(f"\nunknown {a.version} split {s!r}; expected one of "
                                 f"{config.SPLITS[a.version]}\n")

    root = config.ROOT / "src" / "data" / a.source
    found = targets(root, a.version, a.splits)
    if not found:
        raise SystemExit(f"\nno {a.version} parquet under {root.relative_to(config.ROOT)}"
                         f"{' for splits ' + str(a.splits) if a.splits else ''}\n")

    rows_total, notes = 0, []
    for parquet_path in found:
        csv_path = parquet_path.with_suffix(".csv")
        rows, cols, note = convert(parquet_path, csv_path, a.dry_run)
        rows_total += rows
        arrow = "would ->" if a.dry_run else "->"
        print(f"  {str(parquet_path.relative_to(root)):<52} {arrow} {csv_path.name:<48} "
              f"{rows:>7,} rows x {cols:>3} cols")
        if note:
            notes.append(f"    {csv_path.name}: {note}")

    verb = "would write" if a.dry_run else "wrote"
    print(f"\n{verb} {len(found)} CSV file(s), {rows_total:,} rows total, beside their parquets.")
    if notes:
        print("\ndtypes a plain read_csv will NOT recover (values are intact; name them on read):")
        print("\n".join(notes))
    print("\nreverse direction: uv run python v1_csv_to_parquet.py")


if __name__ == "__main__":
    main()
