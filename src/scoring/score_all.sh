#!/usr/bin/env bash
# Score every model version, each in its OWN env, on its OWN feature file.
# Version-agnostic: the active env + CLI args decide which version is scored. Run from repo root.
set -euo pipefail

SOURCE="${1:-synthetic}"                        # synthetic | real
ENVDIR="src/envs"
INPUTS="src/data/$SOURCE/inputs"                # per-version features_<v>.parquet
MODELS="src/models/$SOURCE/baseline"           # baseline pkls (production reproduction)
OUTDIR="src/data/$SOURCE/detection"            # baseline scores → detector
mkdir -p "$OUTDIR"

for V in v1 v2 v3; do
    "$ENVDIR/$V/.venv/bin/python" src/scoring/predict.py \
        --model    "$MODELS/$V.pkl" \
        --features "$INPUTS/features_$V.parquet" \
        --version  "$V" \
        --out      "$OUTDIR/${V}_scores.parquet"
done

echo "All versions scored -> $OUTDIR"
