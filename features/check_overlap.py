"""Build features/feature_overlap.json from the hand-confirmed Excel mapping.

The automated L0/L1/L2 canonicalisation that used to live here is retired (2026-08-04): the
cross-version feature correspondence was confirmed BY HAND on the company laptop, in
features/common_features_260804.xlsx (sheet "ALL_SORTED"). A hand-confirmed table supersedes any
string-based matching, so this script no longer matches anything — it only:

  1. reads the sheet,
  2. cleans the cell noise (wrapping quotes, trailing commas, stray blanks),
  3. validates every name against features/registry/<v>.json where that registry exists
     (the registry is extracted from the pickles by extract_features.py, so this is the
      typo check — names are never guessed), and
  4. writes features/feature_overlap.json:

        {
          "1": {"v1": "...", "v2": "...", "v3": "..."},
          "2": {"v2": "...", "v3": "..."},          <- a pair mapped in only two versions
          ...
        }

Sheet layout (columns A..F): A index, B v3 name | C index, D v2 name | E index, F v1 name.
The same integer index across the three column-pairs links the same underlying feature; an
index absent from a version's columns means that version has no counterpart.

Run in the analysis .venv (needs pandas + openpyxl):

    python features/check_overlap.py
    python features/check_overlap.py --excel features/common_features_260804.xlsx --sheet ALL_SORTED

If validation problems are found, nothing is written unless --force is given — a wrong name in
the mapping silently corrupts every cross-version SHAP share downstream, with nothing to catch it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REGISTRY = HERE / "registry"

VERSIONS = ("v1", "v2", "v3")

# (index column, name column) per version — 0-based positions of Excel columns A..F.
LAYOUT: dict[str, tuple[int, int]] = {"v3": (0, 1), "v2": (2, 3), "v1": (4, 5)}

DEFAULT_EXCEL = HERE / "common_features_260804.xlsx"
DEFAULT_SHEET = "ALL_SORTED"
DEFAULT_OUT = HERE / "feature_overlap.json"


def clean_name(cell) -> str | None:
    """Strip the sheet's cell noise: blanks, trailing commas, wrapping quotes — repeatedly,
    so mixtures like  ' "name", '  come out as  name."""
    if cell is None or pd.isna(cell):
        return None
    s = str(cell)
    prev = None
    while s != prev:
        prev = s
        s = s.strip().rstrip(",").strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1]
    return s or None


def parse_index(cell) -> int | None:
    """Excel hands integers back as floats ('12.0') and headers as text — both handled."""
    if cell is None or pd.isna(cell):
        return None
    try:
        return int(float(str(cell).strip().rstrip(",")))
    except ValueError:
        return None


def read_mapping(excel: Path, sheet: str):
    df = pd.read_excel(excel, sheet_name=sheet, header=None)
    mapping: dict[int, dict[str, str]] = {}
    problems: list[str] = []
    skipped: list[str] = []
    warnings: list[str] = []

    for version, (icol, ncol) in LAYOUT.items():
        if max(icol, ncol) >= df.shape[1]:
            problems.append(
                f"{version}: sheet has only {df.shape[1]} columns, expected name in column "
                f"{ncol + 1} — layout mismatch, check LAYOUT at the top of this script"
            )
            continue
        for row in range(len(df)):
            idx = parse_index(df.iat[row, icol])
            name = clean_name(df.iat[row, ncol])
            if idx is None and name is None:
                continue                      # empty region of the sheet
            if idx is None:
                # header text or a stray label in the index column — skip, but say so
                skipped.append(f"{version} row {row + 1}: {name!r} has no integer index")
                continue
            if name is None:
                problems.append(f"{version} row {row + 1}: index {idx} has an empty name cell")
                continue
            slot = mapping.setdefault(idx, {})
            if version in slot and slot[version] != name:
                problems.append(
                    f"{version}: index {idx} mapped twice — {slot[version]!r} and {name!r}"
                )
                continue
            slot[version] = name

    # the same name under two different indices within one version = two "common features"
    # claiming the same column — one of them is wrong
    for version in LAYOUT:
        seen: dict[str, int] = {}
        for idx in sorted(mapping):
            name = mapping[idx].get(version)
            if name is None:
                continue
            if name in seen:
                problems.append(
                    f"{version}: {name!r} appears under indices {seen[name]} and {idx}"
                )
            else:
                seen[name] = idx

    # An index present in a single version only is not a *common* feature. Downgraded from a
    # write-blocking problem to a WARNING (2026-08-09, decided on the company laptop): the real
    # sheet legitimately carries a few indexed version-unique rows (7 v2-only, 2 v3-only at last
    # run). They are excluded from the written JSON in main(). ⚠️ Still eyeball the sheet around
    # these indices when the count changes: a slipped row shows up here as orphans, and the
    # dangerous half of a slip — neighbouring indices paired to the WRONG counterpart — is not
    # detectable mechanically.
    for idx in sorted(mapping):
        if len(mapping[idx]) == 1:
            (only,) = mapping[idx]
            warnings.append(
                f"index {idx} maps only {only} ({mapping[idx][only]!r}) — not common to any pair"
            )

    return mapping, problems, skipped, warnings


def validate_against_registry(mapping: dict[int, dict[str, str]]):
    """Every mapped name must exist in that version's pickle-extracted feature list."""
    notes: list[str] = []
    problems: list[str] = []
    for version in VERSIONS:
        reg_path = REGISTRY / f"{version}.json"
        if not reg_path.exists():
            notes.append(
                f"{version}: no registry ({reg_path}) — names UNVERIFIED. Run "
                f"extract_features.py inside env-{version} to enable the typo check."
            )
            continue
        known = set(json.loads(reg_path.read_text(encoding="utf-8")).get("model_features", []))
        if not known:
            notes.append(f"{version}: registry exists but lists no model_features — UNVERIFIED")
            continue
        bad = sorted(
            {names[version] for names in mapping.values() if version in names}
            - known
        )
        if bad:
            head = ", ".join(repr(b) for b in bad[:10])
            more = f" … and {len(bad) - 10} more" if len(bad) > 10 else ""
            problems.append(
                f"{version}: {len(bad)} mapped name(s) not in the pickle's own feature list: "
                f"{head}{more}"
            )
        else:
            notes.append(f"{version}: all mapped names confirmed against the registry")
    return notes, problems


def summarise(mapping: dict[int, dict[str, str]]) -> None:
    combos: dict[str, int] = {}
    for names in mapping.values():
        key = "+".join(v for v in VERSIONS if v in names)
        combos[key] = combos.get(key, 0) + 1
    print(f"\ncommon features mapped: {len(mapping)}")
    for key in sorted(combos, key=lambda k: (-k.count("+"), k)):
        print(f"  {key:<10} {combos[key]}")
    for version in VERSIONS:
        n = sum(1 for names in mapping.values() if version in names)
        print(f"  {version} appears in {n} mappings")


def main() -> None:
    # NOT `__doc__.split(...)`: __doc__ is None under `python -OO` (and in any copy whose
    # docstring was stripped), which turns --help into an AttributeError at startup.
    p = argparse.ArgumentParser(
        description="Build features/feature_overlap.json from the hand-confirmed Excel mapping."
    )
    p.add_argument("--excel", default=str(DEFAULT_EXCEL))
    p.add_argument("--sheet", default=DEFAULT_SHEET)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--force", action="store_true",
                   help="write the JSON even if validation problems were found")
    a = p.parse_args()

    excel = Path(a.excel)
    if not excel.exists():
        raise SystemExit(
            f"{excel} not found. The hand-confirmed mapping lives on the company laptop — "
            f"point --excel at it."
        )

    mapping, problems, skipped, warnings = read_mapping(excel, a.sheet)
    notes, reg_problems = validate_against_registry(mapping)
    problems += reg_problems

    if skipped:
        print("skipped (no integer index — header rows are expected here):")
        for s in skipped:
            print(f"  - {s}")

    if warnings:
        print(f"\n{len(warnings)} warning(s) — single-version rows, excluded from the JSON:")
        for w in warnings:
            print(f"  ⚠ {w}")

    for n in notes:
        print(f"note: {n}")

    summarise(mapping)

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for prob in problems:
            print(f"  ✗ {prob}")
        if not a.force:
            raise SystemExit(
                "\nNothing written. Fix the sheet (or the registry) and rerun — or rerun "
                "with --force to write anyway."
            )
        print("\n--force given: writing despite the problems above.")

    out = Path(a.out)
    payload = {str(idx): {v: mapping[idx][v] for v in VERSIONS if v in mapping[idx]}
               for idx in sorted(mapping)
               if len(mapping[idx]) > 1}       # single-version rows: warned above, not written
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}  ({len(payload)} entries)")


if __name__ == "__main__":
    main()
