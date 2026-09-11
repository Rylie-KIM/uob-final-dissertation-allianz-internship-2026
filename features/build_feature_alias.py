"""Build the real-name -> anonymised-alias map used to label figures — run in the analysis env.

Real Allianz column names (``EstimatedRepairCost_VEH``, ``location_Home``, ...) must never appear
on a figure that could land in the (NDA-bound) thesis. This script assigns each version's model
features a stand-in name — ``v1_feat_01``, ``v1_feat_02``, ... — in TRAINED order, and writes one
combined JSON that ``src/feature_alias.py`` reads at plot time.

Trained order, not alphabetical: it is already the order every other real-data build treats as
canonical (features/extract_features.py's ladder, the SHAP attribution parquets), so the alias
numbering lines up with everything else instead of adding a second ordering.

Versions are kept SEPARATE on purpose — never merge the same real name across versions into one
alias. CLAUDE.md is explicit that SHAP/concentration work uses each model's own encoded names
uncollapsed (v1's 55 ``make_*`` columns are not v2's 41); giving two versions' features the same
alias would silently re-introduce that collapse.

Needs only ``features/registry/<v>.json`` (built by extract_features.py / extract_features_v1.py)
to already exist — no pickle is opened here, so this runs in the analysis .venv, not env-vX:

    uv run python features/build_feature_alias.py

A version whose registry is missing, or is still the local DUMMY stand-in, is skipped with a
warning rather than failing the whole build — useful while registries are built one at a time.

Writes features/registry/feature_alias_map.json (config.alias_map_path()). That path matches
features/registry/*.json in .gitignore, so the real names inside it never reach git.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402


def _is_dummy(version: str) -> bool:
    payload = json.loads(config.registry_path(version).read_text(encoding="utf-8"))
    return "DUMMY" in str(payload.get("model_path", ""))


def build_one(version: str) -> dict | None:
    p = config.registry_path(version)
    if not p.is_file():
        print(f"[{version}] SKIP — {p} does not exist yet.")
        return None
    if _is_dummy(version):
        print(f"[{version}] SKIP — {p} is the local DUMMY stand-in registry, not the real one.")
        return None

    names = config.model_features(version)
    real_to_alias = {name: f"{version}_feat_{i:02d}" for i, name in enumerate(names, start=1)}
    alias_to_real = {alias: real for real, alias in real_to_alias.items()}
    print(f"[{version}] {len(names)} features aliased (trained order).")
    return {"order": names, "real_to_alias": real_to_alias, "alias_to_real": alias_to_real}


def main() -> None:
    payload: dict[str, dict] = {}
    for version in config.VERSION_LABELS:
        built = build_one(version)
        if built is not None:
            payload[version] = built

    if not payload:
        raise SystemExit(
            "\nNo real registries found — nothing to alias. Build registries first:\n"
            "    <env-v1 python> features/extract_features_v1.py --model <v1 pkl>\n"
            "    <env-v2 python> features/extract_features.py --version v2\n"
            "    <env-v3 python> features/extract_features.py --version v3\n"
        )

    out = config.alias_map_path()
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"-> {out} ({', '.join(payload)})")


if __name__ == "__main__":
    main()
