"""Re-key the v3 artefacts already on disk: `claim_id` ID_CLAIM -> Claimnumber_CLAIM. One-off.

WHY. 01_export_v3, as first run on the company laptop, took config.column("v3", "claim_id") =
"ID_CLAIM" -- the v3 repo's `project_params.KEY`, a database row id -- and wrote it out as
`claim_id` in every v3 file. Every other artefact keys on the BUSINESS claim number (v1
`claimnumber`, v2 `Claimnumber_CLAIM`, the v2 serving log `ClaimNumber`), so every cross-version
join that touched a v3 file (03_01 corrector inputs; 02_error_inheritance v2->v3) compared two
unrelated integer sequences and passed on accidental collisions. Re-exporting from Z: is the
clean fix; this script is the fast one. raw_v3.parquet was exported as-is and so carries BOTH
columns -- it is the bridge -- and nothing has to be re-read from Z: or re-run through SHAP.

WHAT IT DOES   (analysis .venv, company laptop:  uv run python src/data/rekey_v3_claim_id.py)
  1. raw_v3.parquet -> bridge {ID_CLAIM -> Claimnumber_CLAIM}. Both must be non-null and unique;
     a claim number that maps to several rows STOPS the run (that is a collapse decision, not
     something to guess here).
  2. features / targets / scores per split and every attributions parquet under
     detection/shap/v3/: `claim_id` (ID_CLAIM) is replaced by the mapped claim number, 100 %
     coverage asserted, uniqueness asserted. An existing Claimnumber_CLAIM column (the features
     files may carry one) is checked for equality and removed. The old id is DROPPED from these
     files -- raw keeps it, and raw is the only place it is needed.
  3. raw_v3.parquet itself: `claim_id` -> ID_CLAIM (its real name back), Claimnumber_CLAIM ->
     `claim_id`, key first. Done LAST, so a failure part-way leaves the bridge readable.
  4. Each attributions `_meta.json` gets a `claim_id_rekeyed` record.
  5. Files DERIVED from the wrong join are stale -- corrector_targets_v3_* (+meta),
     v3_corrected_*, the mitigated v3 model, v3 reeval scores. They are listed, deleted only with
     --delete-stale, and rebuilt by 03_01 -> 03_05.

SAFETY. Every file overwritten is first copied to src/data/<source>/_rekey_backup_v3/ (same
relative path; --no-backup skips). Writes go to a temp file, then replace. A manifest
(src/data/<source>/_REKEY_V3_CLAIM_ID.json) records what was done, so a re-run skips finished
files; a file whose ids are already claim numbers is recognised and skipped even without it.
--dry-run reports everything and writes nothing.

AFTER. config.VERSIONS["v3"]["columns"]["claim_id"] is "Claimnumber_CLAIM" from 2026-09-03 and
01_export_v3 writes this layout directly, so a fresh export never needs this script again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
import config  # noqa: E402

VERSION = "v3"
ID = "claim_id"
OLD_KEY = "ID_CLAIM"            # what the first export renamed to claim_id (a DB row id)
NEW_KEY = "Claimnumber_CLAIM"   # the business claim number = config.column("v3", "claim_id") now
REKEY_KINDS = ("processed_inputs", "targets", "scores")
STALE_KINDS = ("corrector_targets", "corrected", "mitigated", "reeval_scores")


# ---------------------------------------------------------------------------- locations --
def data_dir(source: str) -> Path:
    """src/data/<source> -- derived from a config path so the layout is never spelled twice."""
    return config.path("raw_dataset", VERSION, source).parent.parent


def manifest_path(source: str) -> Path:
    return data_dir(source) / "_REKEY_V3_CLAIM_ID.json"


def backup_dir(source: str) -> Path:
    return data_dir(source) / "_rekey_backup_v3"


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(config.ROOT))
    except ValueError:
        return str(p)


def rekey_files(source: str) -> list[tuple[str, Path]]:
    """(label, path) for every v3 file whose claim_id must move to the claim number."""
    files = []
    for s in config.SPLITS[VERSION]:
        for kind in REKEY_KINDS:
            files.append((f"{kind}/{s}", config.split_path(kind, VERSION, s, source)))
    shap_dir = config.path("attributions", VERSION, source, split=config.SPLITS[VERSION][0]).parent
    for p in sorted(shap_dir.glob(f"{VERSION}_attributions_*.parquet")):
        files.append(("attributions", p))
    return files


def stale_files(source: str) -> list[Path]:
    """Artefacts computed FROM the mis-keyed join; nothing here can be re-keyed, only rebuilt."""
    out = []
    for s in config.SPLITS[VERSION]:
        for kind in STALE_KINDS:
            p = config.split_path(kind, VERSION, s, source)
            for q in (p, p.with_name(p.stem + "_meta.json")):
                if q.exists():
                    out.append(q)
    return out


# ------------------------------------------------------------------------------ bridge --
def _tidy(s: pd.Series) -> pd.Series:
    """A float-typed claim number (a NaN somewhere upstream) becomes int64 when it is integral."""
    if pd.api.types.is_float_dtype(s) and s.notna().all() and (s == s.round()).all():
        return s.astype("int64")
    return s


def load_bridge(raw_path: Path) -> tuple[pd.Series, bool]:
    """{ID_CLAIM -> Claimnumber_CLAIM} from raw_v3.parquet, plus whether raw is already re-keyed.

    Both layouts of raw are readable: the first export's (claim_id == ID_CLAIM, Claimnumber_CLAIM
    beside it) and this script's own output (claim_id == Claimnumber_CLAIM, ID_CLAIM beside it).
    """
    if not raw_path.is_file():
        raise SystemExit(f"{rel(raw_path)} is missing -- it is the only bridge between the two "
                         f"keys, nothing can be re-keyed without it")
    raw = pd.read_parquet(raw_path)
    cols = set(raw.columns)
    if ID in cols and NEW_KEY in cols:
        old, new, raw_done = raw[ID], raw[NEW_KEY], False
    elif ID in cols and OLD_KEY in cols:
        old, new, raw_done = raw[OLD_KEY], raw[ID], True
    else:
        hint = ""
        if (raw_path.parent.parent / "_DUMMY_DATA").exists():
            hint = ("\n  (this tree is the LOCAL DUMMY -- make_dummy_real.py -- which has no "
                    "claim-number column; the script is for the company laptop)")
        raise SystemExit(
            f"{rel(raw_path)} carries neither ({ID!r} + {NEW_KEY!r}) nor ({ID!r} + {OLD_KEY!r}); "
            f"columns start {list(raw.columns)[:8]}{hint}")

    label = "already re-keyed" if raw_done else "first-export layout"
    print(f"raw: {rel(raw_path)}  {len(raw):,} rows  ({label})")
    if old.isna().any():
        raise SystemExit(f"raw: {int(old.isna().sum()):,} rows have a null {OLD_KEY}")
    if not old.is_unique:
        raise SystemExit(f"raw: {OLD_KEY} is not unique ({int(old.duplicated().sum()):,} dups)")
    n_null = int(new.isna().sum())
    if n_null:
        raise SystemExit(f"raw: {n_null:,} rows have no {NEW_KEY} -- those rows cannot be keyed "
                         f"by claim number; decide what to do with them first, nothing written")
    dup = new.duplicated(keep=False)
    if dup.any():
        ex = new[dup].value_counts().head(5)
        raise SystemExit(
            f"raw: {NEW_KEY} is NOT one row per claim -- {int(dup.sum()):,} rows share "
            f"{int(new[dup].nunique()):,} claim numbers (worst: {ex.to_dict()}). One claim number "
            f"-> several {OLD_KEY} rows is a collapse decision (which row is the claim?), not "
            f"something to guess here. Nothing written.")
    new = _tidy(new)
    print(f"     bridge {OLD_KEY} -> {NEW_KEY}: {len(new):,} pairs, "
          f"{OLD_KEY} dtype {old.dtype}, {NEW_KEY} dtype {new.dtype}")
    return pd.Series(new.values, index=old.values), raw_done


# ------------------------------------------------------------------------------- rekey --
def classify(ids: pd.Series, bridge: pd.Series) -> tuple[str, int, int]:
    """'old' (ids are ID_CLAIM), 'new' (already claim numbers), or 'mixed' (neither fully)."""
    in_old = int(ids.isin(bridge.index).sum())
    in_new = int(ids.isin(bridge.values).sum())
    n = len(ids)
    if in_old == n:
        return "old", in_old, in_new
    if in_new == n:
        return "new", in_old, in_new
    return "mixed", in_old, in_new


def rekey_frame(df: pd.DataFrame, bridge: pd.Series, name: str) -> pd.DataFrame:
    """Replace claim_id by the bridged claim number; drop every trace of the old key."""
    new_ids = df[ID].map(bridge)
    if new_ids.isna().any():                       # classify() already ruled this out
        raise SystemExit(f"{name}: {int(new_ids.isna().sum()):,} ids not in the bridge")
    if NEW_KEY in df.columns:
        same = df[NEW_KEY].astype("string").values == new_ids.astype("string").values
        if not same.all():
            raise SystemExit(f"{name}: carries its own {NEW_KEY} and it disagrees with raw's on "
                             f"{int((~same).sum()):,} rows -- different extracts? Nothing written")
        df = df.drop(columns=[NEW_KEY])
    df = df.drop(columns=[ID] + ([OLD_KEY] if OLD_KEY in df.columns else []))
    df.insert(0, ID, new_ids.values)
    if not df[ID].is_unique:
        raise SystemExit(f"{name}: claim_id not unique after re-keying")
    return df


def write_atomic(df: pd.DataFrame, p: Path, source: str, backup: bool) -> None:
    if backup:
        b = backup_dir(source) / p.relative_to(data_dir(source))
        if not b.exists():
            b.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, b)
    tmp = p.with_name(p.name + ".rekey.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)


def stamp_meta(p: Path, dry_run: bool) -> None:
    """Attributions carry a sidecar; record the key change there (the parquet cannot say it)."""
    mp = p.with_name(p.stem + "_meta.json")
    if not mp.is_file():
        print(f"     (no sidecar {mp.name} -- meta not stamped)")
        return
    if dry_run:
        return
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta["claim_id_rekeyed"] = {"from": OLD_KEY, "to": NEW_KEY,
                                "via": "inputs/raw_v3.parquet",
                                "date": dt.date.today().isoformat()}
    mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# -------------------------------------------------------------------------------- main --
def run(source: str = "real", dry_run: bool = False, backup: bool = True,
        delete_stale: bool = False) -> dict:
    raw_path = config.path("raw_dataset", VERSION, source)
    bridge, raw_done = load_bridge(raw_path)

    mp = manifest_path(source)
    manifest = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {
        "what": f"{VERSION} claim_id re-keyed {OLD_KEY} -> {NEW_KEY} by src/data/rekey_v3_claim_id.py",
        "done": {}}
    done: dict = manifest["done"]
    verb = "would re-key" if dry_run else "re-keyed"
    summary = {"rekeyed": [], "skipped": [], "absent": []}

    for label, p in rekey_files(source):
        key = rel(p)
        if not p.is_file():
            print(f"-- {label:<22} {key}  (absent)")
            summary["absent"].append(key)
            continue
        df = pd.read_parquet(p)
        if ID not in df.columns:
            raise SystemExit(f"{key}: no {ID} column ({list(df.columns)[:6]}...)")
        state, in_old, in_new = classify(df[ID], bridge)
        if key in done or state == "new":
            why = "manifest" if key in done else "ids are already claim numbers"
            print(f"-- {label:<22} {key}  {len(df):,} rows  skipped ({why})")
            summary["skipped"].append(key)
            continue
        if state == "mixed":
            raise SystemExit(
                f"{key}: {len(df):,} rows, {in_old:,} ids are {OLD_KEY}s, {in_new:,} are claim "
                f"numbers, the rest are in neither -- this file and raw_v3 do not come from the "
                f"same extract. Nothing written.")
        out = rekey_frame(df, bridge, key)
        print(f"OK {label:<22} {key}  {len(out):,} rows  {verb}  "
              f"(cols {len(df.columns)} -> {len(out.columns)})")
        if not dry_run:
            write_atomic(out, p, source, backup)
            done[key] = {"rows": int(len(out)), "date": dt.date.today().isoformat()}
            mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if label == "attributions":
            stamp_meta(p, dry_run)
        summary["rekeyed"].append(key)

    # raw last: swap the two key columns so claim_id is the claim number, ID_CLAIM keeps its name
    key = rel(raw_path)
    if raw_done:
        print(f"-- raw                    {key}  skipped (already {ID} == {NEW_KEY})")
        summary["skipped"].append(key)
    else:
        raw = pd.read_parquet(raw_path).rename(columns={ID: OLD_KEY, NEW_KEY: ID})
        raw = raw[[ID] + [c for c in raw.columns if c != ID]]
        raw[ID] = _tidy(raw[ID])
        print(f"OK raw                    {key}  {len(raw):,} rows  {verb}  "
              f"({ID} <- {NEW_KEY}; {OLD_KEY} kept under its own name)")
        if not dry_run:
            write_atomic(raw, raw_path, source, backup)
            done[key] = {"rows": int(len(raw)), "date": dt.date.today().isoformat(),
                         "note": f"{ID} is now {NEW_KEY}; {OLD_KEY} kept"}
            mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        summary["rekeyed"].append(key)

    stale = stale_files(source)
    summary["stale"] = [rel(q) for q in stale]
    if stale:
        print(f"\nstale (built from the mis-keyed join; rebuild via 03_01 -> 03_05):")
        for q in stale:
            action = "deleted" if (delete_stale and not dry_run) else \
                     ("would delete" if delete_stale else "left in place (--delete-stale removes)")
            print(f"   {rel(q)}  {action}")
            if delete_stale and not dry_run:
                q.unlink()

    print(f"\n{len(summary['rekeyed'])} {verb}, {len(summary['skipped'])} skipped, "
          f"{len(summary['absent'])} absent, {len(stale)} stale"
          + ("" if dry_run else f"\nbackups: {rel(backup_dir(source))}" if backup else
             "\n(no backups -- --no-backup)")
          + ("" if dry_run else f"\nmanifest: {rel(mp)}"))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="real")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--no-backup", action="store_true",
                    help="do not copy originals to _rekey_backup_v3/ first")
    ap.add_argument("--delete-stale", action="store_true",
                    help="remove artefacts derived from the mis-keyed join (03_01 -> 03_05 outputs)")
    a = ap.parse_args(argv)
    run(a.source, dry_run=a.dry_run, backup=not a.no_backup, delete_stale=a.delete_stale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
