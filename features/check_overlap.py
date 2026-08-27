"""Build features/feature_overlap.json from the hand-confirmed Excel mapping.

The automated L0/L1/L2 canonicalisation that used to live here is retired (2026-08-04): the
cross-version feature correspondence was confirmed BY HAND on the company laptop, in
features/common_features_260804.xlsx (sheet "ALL_SORTED"). A hand-confirmed table supersedes any
string-based matching, so this script no longer matches anything — it only:

  1. reads the sheet,
  2. cleans the cell noise (wrapping quotes, trailing commas, stray blanks),
  3. restores a name's exact pickle spelling when the sheet and the registry differ by
     WHITESPACE ONLY (v2 carries two feature names trained with a blank in them — a
     spreadsheet cannot hold a trailing blank reliably, so it is taken from the registry
     rather than typed into the sheet; see restore_registry_spelling),
  4. validates every name against features/registry/<v>.json where that registry exists
     (the registry is extracted from the pickles by extract_features.py, so this is the
      typo check — names are never guessed), and
  5. writes features/feature_overlap.json:

        {
          "1": {"v1": "...", "v2": "...", "v3": "..."},
          "2": {"v2": "...", "v3": "..."},          <- a pair mapped in only two versions
          ...
        }

  6. writes features/feature_overlap_report.json — the same mapping counted per version
     SUBSET, which is what the analysis actually restricts on. Two different questions live
     here and they are kept apart on purpose:

       "subsets"            AT LEAST these versions. `v2+v3` therefore includes rows that
                            also map v1. This is the one that matches a comparison basis:
                            restricting a v2-vs-v3 comparison has no reason to drop a feature
                            merely because v1 happens to have it too.
       "exact_combinations" EXACTLY these versions and no others — how the sheet's rows
                            partition. Use it to see the shape of the mapping, never to size
                            a comparison.

     Both carry the mapped INDICES (the sheet's own numbering, so a row is findable again in
     the Excel) and each version's names, plus a `n` count.

Sheet layout (columns A..F): A index, B v3 name | C index, D v2 name | E index, F v1 name.
The same integer index across the three column-pairs links the same underlying feature; an
index absent from a version's columns means that version has no counterpart.

Run in the analysis .venv (needs pandas + openpyxl):

    python features/check_overlap.py
    python features/check_overlap.py --excel features/common_features_260804.xlsx --sheet ALL_SORTED
    python features/check_overlap.py --from-json      # rebuild ONLY the report, no Excel needed

If validation problems are found, nothing is written unless --force is given — a wrong name in
the mapping silently corrupts every cross-version SHAP share downstream, with nothing to catch it.
"""

from __future__ import annotations

import argparse
import itertools
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
DEFAULT_REPORT = HERE / "feature_overlap_report.json"


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


def ws_key(s: str) -> str:
    """Whitespace-insensitive key: edges stripped, internal runs collapsed to a single space.
    A non-breaking space (the usual copy-paste-into-Excel artefact) counts as whitespace."""
    return " ".join(s.replace("\u00a0", " ").split())


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


def load_registry(version: str):
    """(names, note). `names` is None when the registry cannot be used — `note` says why."""
    reg_path = REGISTRY / f"{version}.json"
    if not reg_path.exists():
        return None, (
            f"{version}: no registry ({reg_path}) — names UNVERIFIED. Run "
            f"extract_features.py inside env-{version} to enable the typo check."
        )
    known = json.loads(reg_path.read_text(encoding="utf-8")).get("model_features", [])
    if not known:
        return None, f"{version}: registry exists but lists no model_features — UNVERIFIED"
    return known, None


def restore_registry_spelling(mapping: dict[int, dict[str, str]]):
    """Give a mapped name back the pickle's exact spelling when the two differ by WHITESPACE ONLY.

    v2 was trained with two feature names that carry a blank in them (confirmed on the company
    laptop, 2026-08-27). A spreadsheet cannot carry that blank: Excel renders it invisibly, any
    later edit silently drops it, and clean_name() strips it along with the real cell noise. So
    the blank is restored from the registry — the pickle's own feature list — instead of being
    typed into the sheet, where it could not survive anyway.

    This is NOT fuzzy matching and does not relax the never-guess rule: the two strings must be
    identical once whitespace is normalised. Anything else is left untouched for
    validate_against_registry() to report as a problem.

    Assumes no version's registry holds two names that differ from each other by whitespace only
    (confirmed 2026-08-27 — it does not happen). That assumption is what makes the ws_key lookup
    a lookup rather than a choice; if it ever stopped holding, this would silently pick one.
    """
    notes: list[str] = []
    for version in VERSIONS:
        known, _ = load_registry(version)
        if known is None:
            continue                          # unverifiable here; the validator says so
        exact = set(known)
        by_key = {ws_key(name): name for name in known}
        for idx in sorted(mapping):
            name = mapping[idx].get(version)
            if name is None or name in exact:
                continue
            literal = by_key.get(ws_key(name))
            if literal is not None:
                mapping[idx][version] = literal
                notes.append(
                    f"{version}: index {idx} {name!r} -> {literal!r} "
                    f"(registry spelling; the two differ by whitespace only)"
                )
            # no whitespace-only counterpart: leave it alone — validate_against_registry() reports
            # it as a name the pickle does not have, which is what it is
    return notes


def validate_against_registry(mapping: dict[int, dict[str, str]]):
    """Every mapped name must exist in that version's pickle-extracted feature list."""
    notes: list[str] = []
    problems: list[str] = []
    for version in VERSIONS:
        known_list, note = load_registry(version)
        if known_list is None:
            notes.append(note)
            continue
        known = set(known_list)
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


def subsets_of(versions: tuple[str, ...]):
    """Every version subset of size >= 2, widest first. A single version is not an overlap."""
    return [combo
            for size in range(len(versions), 1, -1)
            for combo in itertools.combinations(versions, size)]


def build_report(mapping: dict[int, dict[str, str]], excel: Path, sheet: str) -> dict:
    """Per-subset overlap counts and indices — see the docstring for what the two views mean.

    Counted off `mapping`, i.e. the sheet as read, NOT off the written JSON: single-version rows
    are excluded from the written file but still belong in `version_unique` here, which is where
    a slipped sheet row shows up first.

    The counts are an UPPER BOUND on what any analysis can use. 00_shap_attribution.ipynb
    additionally drops a mapped name that is absent from the attributions actually on disk, so
    its basis can be smaller than `n` here — never larger. A gap between the two means a stale
    mapping or the wrong feature matrix.
    """
    report = {
        "source": {"excel": str(excel), "sheet": sheet},
        "versions": list(VERSIONS),
        "n_mapped_rows": len(mapping),
        "subsets": {},
        "exact_combinations": {},
        "version_unique": {},
    }

    for combo in subsets_of(VERSIONS):
        key = "+".join(combo)
        at_least = sorted(i for i, names in mapping.items()
                          if all(v in names for v in combo))
        exact = sorted(i for i in at_least if len(mapping[i]) == len(combo))
        report["subsets"][key] = {
            "versions": list(combo),
            "n": len(at_least),
            "indices": at_least,
            "names": {v: [mapping[i][v] for i in at_least] for v in combo},
        }
        report["exact_combinations"][key] = {"n": len(exact), "indices": exact}

    for v in VERSIONS:
        only = sorted(i for i, names in mapping.items() if list(names) == [v])
        report["version_unique"][v] = {
            "n": len(only),
            "indices": only,
            "names": [mapping[i][v] for i in only],
        }

    # The headline the analysis needs in one line: the widest subset is not always the most
    # useful one. An intersection is a MIN, so a version that shares little starves the pair
    # that shares a lot — and the SFP argument rests on the v2->v3 pair, not on all three.
    widest = "+".join(VERSIONS)
    report["largest_subset"] = max(report["subsets"],
                                   key=lambda k: (report["subsets"][k]["n"], k.count("+")))
    report["all_versions_subset"] = widest
    report["pairs_wider_than_all_versions"] = sorted(
        k for k, d in report["subsets"].items()
        if k != widest and d["n"] > report["subsets"].get(widest, {}).get("n", 0))
    return report


def print_report(report: dict) -> None:
    print("\noverlap per version subset (AT LEAST these versions):")
    for key, d in report["subsets"].items():
        exact = report["exact_combinations"][key]["n"]
        print(f"  {key:<10} {d['n']:>4} mapped   ({exact} of them map ONLY these)")
    uniq = {v: d["n"] for v, d in report["version_unique"].items() if d["n"]}
    if uniq:
        print("  version-unique rows (not an overlap, excluded from feature_overlap.json): "
              + ", ".join(f"{v}={n}" for v, n in uniq.items()))
    if report["pairs_wider_than_all_versions"]:
        print(f"\n  ⚠ wider than the all-versions subset: "
              + ", ".join(f"{k} ({report['subsets'][k]['n']})"
                          for k in report["pairs_wider_than_all_versions"])
              + f" vs {report['all_versions_subset']} "
                f"({report['subsets'][report['all_versions_subset']]['n']}).\n"
                "    Restricting every comparison to the all-versions set would drop features "
                "the pair DOES share.\n    Report the pair the claim rests on, on its own basis, "
                "and name the basis beside the number.")


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
    p.add_argument("--report", default=str(DEFAULT_REPORT),
                   help="per-subset overlap counts and indices")
    p.add_argument("--from-json", action="store_true",
                   help="rebuild ONLY the report from an existing --out, without the Excel. "
                        "The report is derived, so this cannot invent anything the mapping "
                        "does not already say — but it also cannot re-run the registry typo "
                        "check. `version_unique` reflects whatever the JSON holds: this "
                        "script's own main() excludes single-version rows, so it comes back "
                        "empty for a file it wrote — a non-empty one means the JSON came from "
                        "somewhere else.")
    p.add_argument("--force", action="store_true",
                   help="write the JSON even if validation problems were found")
    a = p.parse_args()

    if a.from_json:
        src = Path(a.out)
        if not src.exists():
            raise SystemExit(f"{src} not found — build it from the Excel first.")
        mapping = {int(k): row for k, row in json.loads(src.read_text(encoding="utf-8")).items()}
        report = build_report(mapping, src, "(rebuilt from JSON — no Excel read)")
        print_report(report)
        Path(a.report).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
        print(f"\nwrote {a.report}  (report only; {src.name} untouched)")
        return

    excel = Path(a.excel)
    if not excel.exists():
        raise SystemExit(
            f"{excel} not found. The hand-confirmed mapping lives on the company laptop — "
            f"point --excel at it."
        )

    mapping, problems, skipped, warnings = read_mapping(excel, a.sheet)
    # Restore the pickle's whitespace before validating, so the validator is not handed a name
    # the sheet was structurally unable to spell.
    ws_notes = restore_registry_spelling(mapping)
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

    if ws_notes:
        print(f"\n{len(ws_notes)} name(s) respelled from the registry (whitespace only) — the "
              f"written JSON carries the blank, so index columns by the exact string:")
        for n in ws_notes:
            print(f"  ~ {n}")

    for n in notes:
        print(f"note: {n}")

    summarise(mapping)

    report = build_report(mapping, excel, a.sheet)
    print_report(report)

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

    # Written together, always. A report that can disagree with the mapping beside it is worse
    # than no report — and the report is the file the write-up quotes counts from.
    report["n_entries_written"] = len(payload)
    report_path = Path(a.report)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {report_path}  ({len(report['subsets'])} subsets)")


if __name__ == "__main__":
    main()
