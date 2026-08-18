"""Single source of truth for real-data paths, per-version names, and decision rules.

HOW TO USE THIS FILE
--------------------
Everything marked ``PLACEHOLDER`` is unknown on the local laptop and must be filled in **once**
on the company laptop, where ``model_repos/real/`` holds the three cloned version repos. No other
file should ever hard-code a repo name, a pickle filename, a column name or an interpreter path —
they all read from ``VERSIONS`` here.

To see what still needs filling in, run::

    python src/config.py                 # report of remaining placeholders
    python src/config.py --paths         # every (kind, version) resolved path
    python src/config.py --columns       # every their-name -> our-name mapping

Three things that are easy to get wrong:

* **The repo names are NOT ``fttl-v1/v2/v3``.** All three real repos are named differently from
  each other and differently from our internal ``v1``/``v2``/``v3`` labels. ``repo_dir`` is the
  translation point — our label on the left, their real folder name on the right.
* **The decision rule differs in *shape*, not just value.** v1 has two cutoffs keyed on vehicle
  mobility; v2 has one cutoff that *changed over time*; v3 has one fixed cutoff. Code must branch
  on the rule's shape, never assume a single scalar tau. See ``DECISION_RULES``.
* **A declared path always beats the fallback template.** See ``path()`` below — this is what
  lets the same ``src/`` code run against the real repos' own artefacts and against artefacts we
  generate ourselves, with no code change.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_REPOS = ROOT / "model_repos" / "real"

PLACEHOLDER = "<<FILL IN>>"

TARGET_PRECISION = 0.985

VERSION_LABELS = ("v1", "v2", "v3")


# ======================================================================================
# 1. Artefact kinds  —  the vocabulary every other file uses
# ======================================================================================
#
# A "kind" is WHAT an artefact is, independent of where it lives or what its file is called.
# Application code asks for a kind; this module answers with a real path. Two groups:
#
#   READ_KINDS   things that ALREADY EXIST in the real version repo. Each version names and
#                places them differently, so each must be declared in VERSIONS[v]["paths"].
#
#   WRITE_KINDS  things THIS project produces. The real repos have no counterpart, so they fall
#                back to our own tree. Still declarable — see the note under FALLBACK.

# What ACTUALLY exists per version — read off the real training code, 2026-08-08:
#
#                     v1                          v2                          v3
#   model             outputs/fasstacker_xgb.pkl  outputs/model.pkl           outputs/p146_model.pkl
#   preproc           outputs/fttl_pipeline.pkl   outputs/pipeline.pkl        outputs/p146_pipeline.pkl
#   raw_dataset       Z: inputs.pkl               Z: Data/clean_dataset.pkl   Z: (exists — 2026-08-10)
#   processed_inputs  Z: inputs_transformed.pkl   Z: Data/*_transf_{ts}.pkl   Z: (exists — 2026-08-10)
#   targets           a COLUMN in the data (veh_total_loss / veh_total_loss / Fttl), never its own file
#   prod log          DESTROYED                   exists (live model)         NEVER EXISTED (confirmed)
#
# NB v3: the training SCRIPT persists nothing (as transcribed 2026-08-08), but its raw extract and
# transformed data DO sit on the Z: drive (user-confirmed 2026-08-10) — so v3 exports like v1/v2,
# and the DB-regeneration chain (sql -> enrichments -> pipeline) is only a FALLBACK, with its
# as-of-now enrichment caveat applying to that route alone.
#
# So: model + preprocessor live IN the repo (two separate pickles, all three versions);
# data lives on the Z: network drive as pandas pickles (v1/v2) or in the database (v3).
# A Z-drive pandas pickle only opens under that version's own pandas — read it inside env-vX
# and re-export parquet for the analysis env.

READ_KINDS = (
    "model",         # the fitted ESTIMATOR pickle, in-repo. Pickled SEPARATELY from the
                     #   preprocessing (confirmed 2026-07-31 and again off the code 2026-08-08);
                     #   its predict_proba takes the ALREADY PREPROCESSED matrix.
    "preprocessor",  # the fitted preprocessing pipeline, its own in-repo pickle.
                     #   v2's is built from `tubular` classes; v3 additionally has a STATELESS
                     #   stage applied before the fitted one (see README training-flow section).
    "log_source",         # the production scored log AS THAT VERSION EMITTED IT — their column
                          #   names. ONLY v2 has one: v1's was destroyed, v3 was never deployed.
    "processed_inputs",   # POST-preprocessing matrix: what the model consumes. v1 ships one
                          #   appended file (train+test+val1+val2), v2 ships three TIMESTAMPED
                          #   files, v3 none. (Renamed from "features", 2026-08-09 — pairs with
                          #   the registry's raw_features/model_features vocabulary.)
    "raw_dataset",        # the raw claim table, PRE-preprocessing. v1/v2: Z-drive pickle.
                          #   v3: a database table, not a file — export first (see VERSIONS["v3"]).
)

WRITE_KINDS = (
    "log",            # log_source with its columns RENAMED to ours — 01_export_v2.ipynb writes
                      #   it (only v2 has a production log), and it is the only log anything
                      #   downstream reads. (scoring/ingest.py did this; absorbed 2026-08-09.)
    "targets",        # claim_id + date + observed, DERIVED at ingest. No version ships this as
                      #   its own artefact — the target is a column inside raw/features
                      #   (v1/v2 `veh_total_loss`, v3 `Fttl`), so we extract it ourselves.
    "scores",         # baseline scores we recompute       -> detector/
    "attributions",   # per-row SHAP values                 -> estimator/
    "corrected",      # de-contaminated target              <- mitigator/
    "mitigated",      # model retrained on corrected target
    "reeval_scores",  # scores from the mitigated model     -> reeval/
)

KINDS = READ_KINDS + WRITE_KINDS

# Per-version split names, in training order, under each version's OWN naming — v1 and v2
# INVERT what "test" means (v2's "test" is the OOT-style tail), so the names are never
# unified. Per-split artefacts live beside the base kind with a `_{split}` suffix before
# the extension — resolve them by passing `split=` to path(), never by hand.
SPLITS: dict[str, tuple[str, ...]] = {
    "v1": ("train", "test", "val1", "val2"),   # one merged pkl, sliced by known row counts
    "v2": ("train", "val", "test"),            # three separate Z: files, one per split
    "v3": ("train", "test", "oot"),            # three separate Z: files (user-confirmed
                                               #   2026-08-12); OOT is its own file
}

# WHICH split of each version is the out-of-time holdout — the temporally-later block the
# README's training timeline calls OOT. The three versions spell it differently and one of
# them spells it misleadingly, which is exactly why this mapping exists rather than a rule:
#
#   v1  val2   (user-confirmed 2026-08-18) — NOT "test". v1's "test" is an in-period split.
#   v2  test   its "test" IS the tail, inverting v1's meaning of the same word.
#   v3  oot    the only version that says so in the name.
#
# Use it wherever an analysis wants "the holdout" across versions — a cross-version SHAP
# comparison must put every version on the same KIND of split, or the difference is partly
# in-sample-vs-out-of-sample rather than in the fitted functions.
OOT_SPLIT: dict[str, str] = {"v1": "val2", "v2": "test", "v3": "oot"}

# Kinds that exist ONLY per split. The export notebooks (notebook/real/0{1,2,3}_export_v*)
# write one file per SPLITS[version] and NEVER an unsplit base file, so asking for one of
# these without a split is a bug, not a default — path() raises rather than resolving to a
# name nothing produces. Everything outside this tuple (model, preprocessor, raw_dataset,
# log, log_source) is a single artefact per version.
#
# The derived kinds are here for the same reason the produced ones are: attributions are
# computed FROM one split's matrix, so "v2 SHAP concentration" is only a statement once the
# split is named. Losing that would put an unrecorded case-mix choice under the thesis's
# central number.
SPLIT_KINDS = (
    "processed_inputs",   # ┐ written per split by the export notebooks
    "targets",            # │
    "scores",             # ┘
    "attributions",       # ┐ derived from one of the above, so they inherit its split
    "corrected",          # │
    "mitigated",          # │
    "reeval_scores",      # ┘
)

# Analysis code may pass this instead of a split name to mean "every split, pooled, with a
# `split` column recording where each row came from". It is deliberately a WORD, not the
# default: pooling v1's 4-way split against v3's 3-way one mixes different train/holdout
# proportions into a cross-version comparison, and that has to be a choice someone typed.
ALL_SPLITS = "all"

# Fallback location for a kind that is NOT declared for a version. Relative to ROOT.
# ``{source}`` is "real" or "synthetic"; ``{v}`` is our version label.
#
# NOTE on "model": if VERSIONS[v]["paths"]["model"] is declared, we LOAD that version's own
# production pickle. If it is left blank, we fall back to a baseline pickle WE retrain into
# src/models/<source>/baseline/<v>.pkl. Both routes are supported on purpose — v1's training data
# was destroyed, so v1 can only ever be loaded, while v2/v3 could go either way.
FALLBACK: dict[str, str | None] = {
    "model":            "src/models/{source}/baseline/{v}.pkl",
    "preprocessor":     "src/models/{source}/baseline/{v}_prep.pkl",
    "log_source":       None,   # no sensible default — must be declared
    "log":              "src/data/{source}/logs/{v}.parquet",
    "processed_inputs": "src/data/{source}/inputs/features_{v}.parquet",   # filename kept: docs
                                                                           #   reference it
    "targets":          "src/data/{source}/inputs/targets_{v}.parquet",
    "raw_dataset":      "src/data/{source}/inputs/raw_{v}.parquet",   # the export notebooks'
                                                                      #   frozen snapshot
    "scores":        "src/data/{source}/detection/{v}_scores.parquet",
    "attributions":  "src/data/{source}/detection/shap/{v}_attributions.parquet",
    "corrected":     "src/data/{source}/mitigation/{v}_corrected.parquet",
    "mitigated":     "src/models/{source}/mitigated/{v}.pkl",
    "reeval_scores": "src/data/{source}/reeval/{v}_mitigated_scores.parquet",
}


# ======================================================================================
# 2. Canonical column names  —  what src/ code is allowed to say out loud
# ======================================================================================
#
# Application code refers to columns ONLY by these canonical names. VERSIONS[v]["columns"]
# translates each one to that version's real column name. The translation is applied once, at
# ingest, so nothing downstream ever learns a version's real column names.

CANONICAL_COLUMNS = (
    "claim_id",   # row key, joins every artefact together
    "date",       # v1 lossdate / v2 ReportedDate / v3 ReportedDate_CLAIM
    "score",      # the model's output in production
    "decision",   # 1 = fast-tracked to scrap, 0 = sent to garage
    "observed",   # the outcome label as recorded (SFP-contaminated by construction)
    "mobility",   # vehicle mobility status — v1's decision rule is segmented on this
)


# ======================================================================================
# 3. Per-version declarations  —  the only place real names may appear
# ======================================================================================

VERSIONS: dict[str, dict] = {
    "v1": {
        # -- identity -------------------------------------------------------------------
        # Folder name inside model_repos/real/. NOT "fttl-v1" — use the real clone's name.
        "repo_dir": PLACEHOLDER,
        # ("package" key removed 2026-08-09: nothing read it, and the concept does not match
        # the real envs — v1/v2 are registered via a hand-written fttl.pth (bare module dirs,
        # no installable package name), v3 via an editable install. See SETUP.md.)
        # The xgboost release this version's model was serialised with. Read off env-v1 itself on
        # 2026-08-06: `xgboost.__version__` prints "0.72" (Python 3.5.6) — it comes from the repo's
        # vendored cp35 wheel, not from PyPI, so it carries no patch component. Transcribe it
        # EXACTLY: src/shap_kit.py compares this string for equality, and "0.72.1" would raise.
        "xgboost": "0.72",
        # Interpreter for THIS version's env. Loading the pickle re-imports the repo's classes
        # and its exact library stack, so it must be that version's own env — see README
        # "Model artefacts and reproduction". Absolute, or relative to repo root.
        # The ENV DIRECTORY, not an interpreter path — the interpreter's location differs per
        # OS/layout (bin/python on POSIX, Scripts\python.exe in a Windows venv, python.exe at
        # the root of a conda env). python_bin() resolves it at call time, so this declaration
        # is true on every machine.
        "python": "src/envs/v1/.venv",

        # ("builder"/"adapter" hooks removed 2026-08-09 — never used. The fitted preprocessing
        # is always LOADED from paths.preprocessor, and all three versions expose predict_proba,
        # confirmed 2026-07-31, so no per-version handling override was ever needed.)

        # ("train_scores_source" key removed 2026-08-13: no code ever resolved it, and like
        # every version's Z: sources the path lives in the export notebook's SOURCES cell —
        # here notebook/real/01_export_v1.ipynb (PREDICTIONS). What it pointed at: v1's
        # surviving TRAIN-TIME scored file, [claimnumber, predictions], all four splits,
        # 2016-01-01 -> 2018-02-09 — not the destroyed production log. The export lands in
        # the "scores" parquet, in-sample for the train split.)

        # -- artefact paths (read off the training code 2026-08-08) ---------------------
        # Absolute, or relative to repo_dir. Blank/PLACEHOLDER -> FALLBACK template.
        #
        # None here means: config does NOT resolve this from a declared file. Either the
        # artefact never existed, or its Z:/DB source path lives in the SOURCES cell of
        # notebook/real/01_export_<v>.ipynb (decided 2026-08-09) — that notebook converts the
        # source to canonical parquet/CSV in our tree, and analysis reads the FALLBACK path.
        # Z: paths are deliberately NOT declared here: a declared path always wins, and it
        # would point the analysis .venv at a version-bound pandas pickle it cannot open.
        "paths": {
            # As transcribed from the training script. VERIFY the spelling against
            # `dir outputs` before first use ("fasstacker" vs "fasttracker") — a typo just
            # fails loudly with FileNotFoundError, so this is safe to try as-is.
            "model": "outputs/fasstacker_xgb.pkl",
            "preprocessor": "outputs/fttl_pipeline.pkl",
            # Production log destroyed — None for good. The surviving train-time scored file
            # is a different thing — its Z: path lives in 01_export_v1's SOURCES cell.
            "log_source": None,
            # Z:/.../inputs_transformed.pkl (one appended file, train+test+val1+val2) —
            # read and exported by 01_export_v1; analysis reads the fallback.
            "processed_inputs": None,
            # Z:/P10_.../inputs.pkl — read and exported by 01_export_v1.
            "raw_dataset": None,
        },

        # -- column names: OUR canonical name -> THEIR real column ----------------------
        "columns": {
            "claim_id": "claimnumber",   # confirmed 2026-08-08 (predictions file carries it)
            "date": "lossdate",          # accident date; v1 splits on it (confirmed)
            "score": "predictions",      # in the surviving train-time scored file. NOT a
                                         #   production log score — that log is gone.
            "decision": None,            # no surviving artefact records production decisions
                                         #   (the destroyed log's field was named
                                         #   `fasttracker_decision` per the serving code,
                                         #   2026-08-10); reconstruct via DECISION_RULES["v1"]
            "observed": "veh_total_loss",  # the training target; veh_fast_track=1 forces it to 1
                                           #   (README "v1 target construction")
            "mobility": "vehicle_mobility_status",   # confirmed 2026-08-10, read off v1's
                                                     #   serving code (the segmented rule keys
                                                     #   on it — see DECISION_RULES["v1"])
        },

        # Which era's label this version learned from. A CONCEPT, not a column name — the real
        # column goes in columns.observed. This is the contamination chain: pre_ml -> v1 -> v2.
        "trained_on": "pre_ml",
    },
    "v2": {
        "repo_dir": PLACEHOLDER,
        "xgboost": "1.4.2",       # user-confirmed 2026-08-01
        "python": "src/envs/v2/.venv",   # env dir; interpreter resolved by python_bin()
        "paths": {
            "model": "outputs/model.pkl",           # read off the training code 2026-08-08
            "preprocessor": "outputs/pipeline.pkl",  # tubular-based; joblib.dump'd
            # v2 is LIVE — a production scored log exists somewhere. Location unknown; once
            # found, 01_export_v2 reads it from here and writes the canonical "log" parquet.
            "log_source": PLACEHOLDER,
            # THREE timestamped Z: files (Data/{train,val,test}_transf_{ts}.pkl) — their paths
            # live in 01_export_v2's SOURCES cell; analysis reads the exported fallback parquet.
            "processed_inputs": None,
            # {DATA_FOLDER_PROD}/Data/clean_dataset.pkl on Z: — in 01_export_v2's SOURCES cell.
            "raw_dataset": None,
        },
        "columns": {
            "claim_id": PLACEHOLDER,     # not read off yet — check clean_dataset.pkl
            "date": "ReportedDate",      # confirmed 2026-07-29
            "score": PLACEHOLDER,        # whatever the live log calls the model output
            "decision": PLACEHOLDER,     # whatever the live log calls the fast-track flag
            "observed": "veh_total_loss",  # par.TARGET in training (confirmed); the LIVE log's
                                           #   outcome column name may differ — verify at ingest
            "mobility": None,            # v2's rule is global — not needed
        },
        "trained_on": "v1",           # concept, not a column — see the note on v1
    },
    "v3": {
        "repo_dir": PLACEHOLDER,
        "xgboost": "3.2.0",       # user-confirmed 2026-08-01
        "python": "src/envs/v3/.venv",   # env dir; interpreter resolved by python_bin()
        # v3's raw data is NOT a file: the training script pulls it straight from the database
        # (polars). To make it a raw_input artefact, export once on the company laptop and
        # declare the export path below.
        "sql": "SELECT * FROM datascience_lab.prod.p146_extract_v3",
        # In-repo lookup tables the training flow joins in BEFORE the pipeline
        # (hpi + thatcham parquets on ENRICHMENT_KEY, cc_rule_lookup.csv):
        "dependencies_dir": "fttl/dependencies",
        "paths": {
            "model": "outputs/p146_model.pkl",           # read off the training code 2026-08-08
            "preprocessor": "outputs/p146_pipeline.pkl",  # the STATEFUL pipeline. A stateless
                                                          #   stage runs before it and is code,
                                                          #   not a pickle — rerun it from the repo
            # Never deployed -> no production log exists, and never will (user-reconfirmed
            # 2026-08-10). None = "this version has no such artefact".
            "log_source": None,
            # EXISTS on Z: (confirmed 2026-08-10 — the training script saves nothing, but the
            # data is there anyway). Path lives in 01_export_v3's SOURCES cell; analysis reads
            # the exported fallback parquet. Being the data training ACTUALLY saw, it carries
            # no as-of-now enrichment caveat — unlike the DB-regeneration fallback.
            "processed_inputs": None,
            # Same: exists on Z:, read by 01_export_v3, analysis reads the fallback snapshot.
            "raw_dataset": None,
        },
        "columns": {
            "claim_id": "ID_CLAIM",    # `project_params.KEY` in the repo — read the actual string
            # Same underlying field as v2's ReportedDate, renamed (confirmed 2026-07-29).
            "date": "ReportedDate_CLAIM",
            # TRAIN-TIME score, saved as a column inside the Z: transformed data (confirmed
            # 2026-08-10) — same status as v1's "predictions": the fitted model's own output,
            # in-sample for the train split. NOT a production score (never deployed).
            "score": "fttl_predicted_prob",
            # The sibling column `fttl_predicted_label` (score > tuned threshold) also exists,
            # but it is deliberately NOT declared as our canonical `decision`: canonical
            # decision means the TREATMENT (a car actually fast-tracked to scrap), and v3
            # never actioned anything. A predicted label is not a treatment.
            "decision": None,
            "observed": "Fttl",         # add_target(): vehicle status in {fttl, total_loss,
                                        #   unrecovered} -> Fttl=1. Same label as veh_total_loss
                                        #   renamed (confirmed 2026-08-04).
            "mobility": None,
        },
        "trained_on": "v2",           # concept, not a column — see the note on v1
    },
}


# ======================================================================================
# 4. Accessors  —  the only API src/ code should use
# ======================================================================================


def _version(version: str) -> dict:
    if version not in VERSIONS:
        raise KeyError(f"unknown version {version!r}; expected one of {VERSION_LABELS}")
    return VERSIONS[version]


def _filled(value) -> bool:
    """A declaration counts as filled in if it is a non-empty, non-placeholder string."""
    return isinstance(value, str) and value not in ("", PLACEHOLDER)


def _require(version: str, key: str) -> str:
    value = _version(version).get(key)
    if not _filled(value):
        raise ValueError(
            f"config.VERSIONS['{version}']['{key}'] is still a placeholder. "
            f"Fill it in on the company laptop (run `python src/config.py` to list what is missing)."
        )
    return value


def repo(version: str) -> pathlib.Path:
    """That version's cloned repo directory."""
    return MODEL_REPOS / _require(version, "repo_dir")


def path(kind: str, version: str, source: str = "real", split: str | None = None) -> pathlib.Path:
    """Resolve an artefact by KIND, never by filename.

    A declared path in ``VERSIONS[version]["paths"][kind]`` always wins; absolute paths are used
    as-is (real artefacts may live off-repo, e.g. on a network drive), relative ones resolve
    against ``repo_dir``. If nothing is declared, the FALLBACK template under our own tree is
    used. So a kind moves from "ours" to "theirs" by filling in one line here — no code change.

    ``split`` is REQUIRED for every kind in SPLIT_KINDS and rejected for the rest. It appends
    ``_{split}`` before the extension, which is exactly what the export notebooks write. The
    unsplit name (``features_v2.parquet``) is not a default — nothing produces it — so asking
    for one without a split raises here instead of failing later as a missing file.
    """
    if kind not in KINDS:
        raise KeyError(f"unknown kind {kind!r}; expected one of {KINDS}")

    if kind in SPLIT_KINDS and split is None:
        raise ValueError(
            f"{kind!r} exists only per split — the export notebooks write one file per split "
            f"and never an unsplit base file. Pass split=<one of {SPLITS.get(version, ())}>. "
            f"(To pool every split, analysis code uses loaders.load(..., split={ALL_SPLITS!r}); "
            f"a single path cannot name more than one file.)"
        )
    if kind not in SPLIT_KINDS and split is not None:
        raise ValueError(
            f"{kind!r} is a single artefact per version, so split={split!r} is meaningless. "
            f"Split kinds are {SPLIT_KINDS}."
        )
    if split is not None:
        if version not in SPLITS:
            raise KeyError(f"no splits declared for {version!r}; declare them in config.SPLITS")
        if split not in SPLITS[version]:
            raise KeyError(
                f"unknown split {split!r} for {version}; expected one of {SPLITS[version]}. "
                f"Split names are each version's OWN — v1 and v2 invert what 'test' means, and "
                f"only v3 has 'oot', so they are never unified."
            )

    declared = _version(version).get("paths", {}).get(kind)
    if _filled(declared):
        p = pathlib.Path(declared)
        base = p if p.is_absolute() else repo(version) / p
    else:
        template = FALLBACK[kind]
        if template is None:
            raise ValueError(
                f"config.VERSIONS['{version}']['paths']['{kind}'] is not declared and '{kind}' "
                f"has no fallback location. Fill it in on the company laptop — unless it is "
                f"declared None on purpose because the artefact does not exist (v1's production "
                f"log was destroyed; v3 was never deployed), in which case nothing may ask for it."
            )
        base = ROOT / template.format(source=source, v=version)

    if split is None:
        return base
    return base.with_name(f"{base.stem}_{split}{base.suffix}")


def split_path(kind: str, version: str, split: str, source: str = "real") -> pathlib.Path:
    """Per-split variant of path(), kept as the name the export notebooks call.

    E.g. split_path("processed_inputs", "v1", "train") ->
    .../inputs/features_v1_train.parquet. Valid splits are config.SPLITS[version].
    """
    return path(kind, version, source, split=split)


def resolve_splits(spec: list[str] | None, versions: list[str]) -> dict[str, str]:
    """Turn a CLI --split argument into {version: split}.

    Two forms, because split names are per-version and only sometimes shared:

        --split test              one name for every version named in --versions
        --split v2=test v3=oot    a name per version (the only way to reach v3's 'oot')

    Every name is validated against SPLITS[version] here, so a typo fails before any
    subprocess is launched rather than inside a version env on the company laptop.
    """
    if not spec:
        raise SystemExit(
            "\n--split is required: processed_inputs / targets / scores exist only per split.\n"
            "  --split test               (same name for every version)\n"
            "  --split v2=test v3=oot     (per version)\n"
            + "\n".join(f"  {v}: {SPLITS.get(v, ())}" for v in versions) + "\n"
        )

    if len(spec) == 1 and "=" not in spec[0]:
        chosen = {v: spec[0] for v in versions}
    else:
        chosen = {}
        for item in spec:
            if "=" not in item:
                raise SystemExit(
                    f"\n--split: mixed forms. Use ONE bare name for all versions, or "
                    f"'version=split' for every one. Got {item!r}.\n"
                )
            v, _, s = item.partition("=")
            chosen[v] = s
        missing = [v for v in versions if v not in chosen]
        if missing:
            raise SystemExit(f"\n--split: no split given for {', '.join(missing)}\n")

    for v, s in chosen.items():
        if v not in SPLITS:
            raise SystemExit(f"\n--split: no splits declared for {v!r}\n")
        if s not in SPLITS[v]:
            raise SystemExit(
                f"\n--split: {s!r} is not a split of {v} — expected one of {SPLITS[v]}.\n"
                f"Split names are each version's own; v1 and v2 invert 'test'.\n"
            )
    return chosen


def python_bin(version: str) -> pathlib.Path:
    """Interpreter of that version's isolated env — the only one its pickle unpickles in.

    The declaration is the ENV DIRECTORY (``src/envs/v2/.venv``), deliberately OS-neutral —
    there is no interpreter path that exists on every machine. The three real layouts are
    tried in order:

    * ``<env>/bin/python``          — venv/conda on POSIX (local laptop)
    * ``<env>\\Scripts\\python.exe``  — venv on Windows (env-v2, env-v3 on the company laptop)
    * ``<env>\\python.exe``          — conda env created with ``-p`` on Windows (env-v1:
      Python 3.5 comes from conda because uv does not work below 3.6 — see SETUP.md)

    A declared FILE path (an explicit interpreter) is also accepted and returned as-is.
    """
    p = pathlib.Path(_require(version, "python"))
    p = p if p.is_absolute() else ROOT / p
    if p.is_file():
        return p
    for alt in (p / "bin" / "python", p / "Scripts" / "python.exe", p / "python.exe"):
        if alt.exists():
            return alt
    raise FileNotFoundError(
        f"[{version}] no interpreter found under {p} (tried bin/python, Scripts/python.exe, "
        f"python.exe). Is the env built? See SETUP.md."
    )


def column(version: str, canonical: str) -> str:
    """That version's real name for one of our canonical columns."""
    if canonical not in CANONICAL_COLUMNS:
        raise KeyError(f"unknown column {canonical!r}; expected one of {CANONICAL_COLUMNS}")
    value = _version(version).get("columns", {}).get(canonical)
    if value is None:
        raise ValueError(
            f"config.VERSIONS['{version}']['columns']['{canonical}'] is None — this version "
            f"declares that it has no such column. Guard for it before asking."
        )
    if not _filled(value):
        raise ValueError(
            f"config.VERSIONS['{version}']['columns']['{canonical}'] is still a placeholder."
        )
    return value


def rename_map(version: str) -> dict[str, str]:
    """THEIR name -> OUR name, ready for ``df.rename(columns=...)`` at ingest.

    Skips columns this version does not have (declared None) and ones not yet filled in, so it is
    usable before every placeholder is resolved.
    """
    cols = _version(version).get("columns", {})
    return {real: ours for ours, real in cols.items() if _filled(real)}


# ======================================================================================
# 5. Decision rules  —  confirmed from production code, do not edit without a source
# ======================================================================================
#
# The *arity* of the cutoff differs by version. Only two things are invariant: the cutoff lives
# in score space (never a percentile), and it exists to hold precision >= 0.985.

DECISION_RULES: dict[str, dict] = {
    # v1 alone is SEGMENTED — two cutoffs keyed on vehicle mobility (confirmed 2026-07-29;
    # RE-CONFIRMED verbatim from the serving code 2026-08-10: column `vehicle_mobility_status`,
    # mobile set exactly as below, STRICT `>` comparisons — matching threshold.py).
    # The immobile (undrivable, more damaged) car is scrapped on a lower bar.
    "v1": {
        "shape": "segmented",
        "segment_by": "mobility",           # canonical column name — see CANONICAL_COLUMNS
        "thresholds": {"immobile": 0.75, "mobile": 0.85},
        # MOBILE is this set; IMMOBILE is its complement.
        "mobile_values": ("Mobile", "Mobile Not Roadworthy", "Mobile Not Secure"),
        # Consequence: scores in (0.75, 0.85] can be either scrapped or sent to garage depending
        # on mobility — a confounded but usable overlap band. Above 0.85 there are no garage rows.
        "overlap_band": (0.75, 0.85),
    },
    # v2 is a SINGLE global cutoff, but it CHANGED mid-production and was not reverted.
    # Piecewise-constant: 0.872 up to the break, 0.825 after (read from v2 score.py, 2026-07-29).
    #
    # BREAK = 2026-06-30 14:30 UK local time (BST, UTC+1) = 2026-06-30 13:30Z (user-confirmed
    # 2026-07-31). It is an INTRA-DAY instant, not a date. 2026-06-30 therefore spans BOTH
    # regimes, so it cannot be assigned to either as a whole day.
    #
    # -----------------------------------------------------------------------------------------
    # ⚠️ NOTE, DELIBERATELY NOT ENCODED (2026-07-31): an EARLIER change is known to exist —
    #     2026-02-25 16:26 UK local (GMT, UTC+0 -> 16:26Z):  0.825 -> 0.872
    # so 0.825 was in force for some period BEFORE that date too. How far back is UNKNOWN: the
    # deployment/change record is broken before this point and cannot be recovered.
    #
    # It is left out of `regimes` on purpose. Encoding it would require a start date for the
    # first 0.825 era, and inventing one would silently relabel years of v2 log rows on a guess —
    # worse than the known, documented gap. `regimes` below therefore describes the period the
    # record actually supports, and the first entry should be read as "0.872 as documented, for
    # the span the record covers", NOT as a claim about all history.
    #
    # Consequence carried in the docs as a LIMITATION (README + paper §4.6), not as a retraction:
    # v3's window (2023-06-01 -> 2026-05-01) and its OOT slice (2025-12-01 -> 2026-05-01) both
    # CONTAIN 2026-02-25, so "v3 trains on a policy-homogeneous log" may not hold. It stands as
    # the documented reading until the change record can be reconstructed.
    # -----------------------------------------------------------------------------------------
    "v2": {
        "shape": "piecewise_global",
        # DATE-GRANULAR ON PURPOSE. The break really happened at 2026-06-30 14:30 UK local
        # (= 13:30Z, BST on that date), and the earlier one at 2026-02-25 16:26 UK local
        # (= 16:26Z, GMT). Those instants are recorded here as fact, but the code works in whole
        # days: the whole of 2026-06-30 is assigned to the POST regime. The cost is at most one
        # day of morning claims labelled 0.825 when they were decided at 0.872, which is not a
        # difference any conclusion in this thesis turns on — and modelling it properly would
        # require the log to carry timestamps AND their timezone, neither of which is guaranteed.
        "regimes": [
            {"until": "2026-06-30", "threshold": 0.872},
            {"from": "2026-06-30", "threshold": 0.825},
        ],
        "break_date": "2026-06-30",      # 14:30 UK local; see the note above
        "break_date_confirmed": True,
        # The earlier change, recorded but NOT applied — see the block comment above.
        "known_unmodelled_break": {
            "date": "2026-02-25",        # 16:26 UK local
            "from_threshold": 0.825,
            "to_threshold": 0.872,
            "prior_era_start": None,     # unknown — deployment record broken, not recoverable
            "reason_not_applied": "no start date for the prior 0.825 era; guessing one would "
                                  "relabel years of log rows",
        },
        # Any per-row analysis on the v2 log must either restrict to one regime or carry the
        # regime as a covariate. Silently pooling conflates two treatment assignments — and it
        # shows up as read_off() reporting deterministic=False for a rule that is perfectly
        # deterministic inside each regime.
    },
    # v3 is a single global cutoff (confirmed 2026-07-29). Never deployed.
    "v3": {
        "shape": "global",
        "threshold": 0.984,
    },
}


# ======================================================================================
# 5b. Training configuration  —  read off each repo's own training call
# ======================================================================================
#
# Recorded here so a notebook can print the configuration BESIDE the SHAP figures without opening
# anything. That is not a nicety: problem.md §1.4c establishes that no adjacent version pair shares
# a configuration, and L1 regularisation concentrates feature importance BY CONSTRUCTION — so a
# concentration difference has a competing mechanical explanation. The rule adopted there is that
# the configuration is always shown next to the number.
#
# ⚠️ The xgboost RELEASE differs too (0.72 / 1.4.2 / 3.2.0 — see VERSIONS[v]["xgboost"]), which
# means the DEFAULTS differ as well. v1 sets almost nothing, so nearly all of v1's configuration is
# "whatever xgboost 0.72's XGBClassifier defaulted to" (max_depth=3, n_estimators=100,
# learning_rate=0.1 in that era's sklearn wrapper) — NOT the same defaults a modern reader assumes.
# Any "v1 is a simpler model" claim is therefore about the 2018 library as much as about v1.

TRAINING_CONFIG: dict[str, dict] = {
    # Everything else is the 0.72 default. `silent=False` and `eval_set` are old-API and gone
    # from later releases. `mlogloss` on a binary target is a MULTI-class metric — it still trains
    # a binary:logistic model (objective is untouched), it only makes the eval printout odd.
    "v1": {
        "eval_metric": "mlogloss",
        "n_jobs": 20,
        "silent": False,
        "_note": "all other hyperparameters are xgboost 0.72 defaults; eval_set was passed",
    },
    # ⚠️ eta AND learning_rate are both set — they are the SAME knob under two names, so one of
    # them silently loses. Which one wins depends on the release (problem.md §1.4b, open item).
    "v2": {
        "objective": "binary:logistic",
        "eval_metric": "auc",          # ⚠️ misaligned with the precision >= 0.985 constraint
        "colsample_bytree": 0.6,
        "eta": 0.147686,
        "learning_rate": 0.0887667,    # ⚠️ clashes with eta above
        "gamma": 15.0,
        "grow_policy": "depthwise",
        "max_delta_step": 10,
        "max_depth": 10,
        "min_child_weight": 1,
        "n_estimators": 450,
        "random_state": 42,
        "reg_alpha": 20.0,             # heavy L1 — the concentration confound of §1.4b
        "reg_lambda": 0.0123626,
        "scale_pos_weight": 4.5,       # => the score is NOT a calibrated probability
        "subsample": 1.0,
        "n_jobs": -1,
    },
    # Regularises by TREE STRUCTURE where v2 regularised by PENALTY — opposite mechanisms.
    # reg_alpha is absent from the call as read; problem.md's open item asks whether it is
    # genuinely unset (=> 0) rather than merely omitted from the transcription.
    "v3": {
        "colsample_bytree": 0.887008,
        "subsample": 0.980460,
        "max_depth": 3,
        "max_leaves": 18,
        "grow_policy": "depthwise",
        "gamma": 0.0004847861,
        "learning_rate": 0.09973689,
        "min_child_weight": 44,
        "n_estimators": 802,
        "reg_lambda": 1.182373785,
        "scale_pos_weight": 5.552301,
        "max_delta_step": 61,
        "random_state": 123,
        "n_jobs": -1,
    },
}


def xgboost_pin(version: str) -> str | None:
    """The xgboost release this version's pickle was serialised with, or None if undeclared."""
    return _version(version).get("xgboost")


def training_config(version: str) -> dict:
    """That version's training call, as read off its repo. Keys starting with '_' are notes."""
    return {k: v for k, v in TRAINING_CONFIG.get(version, {}).items() if not k.startswith("_")}


# (Section 6, legacy shims, removed 2026-08-09: SCRAP_THRESHOLD — v2's pre-break 0.872, never a
# universal constant — is now derived from DECISION_RULES by the legacy synthetic notebooks that
# still want it; model_path() is gone, extract_features.py calls path("model", v) directly.)


# ======================================================================================
# 6. Report  —  what is still missing
# ======================================================================================


def missing() -> dict[str, list[str]]:
    """Which declarations are still placeholders, per version, as dotted keys."""
    gaps: dict[str, list[str]] = {}
    for v, fields in VERSIONS.items():
        out: list[str] = []
        for key in ("repo_dir", "python"):
            if not _filled(fields.get(key)):
                out.append(key)
        for kind, value in fields.get("paths", {}).items():
            if value is None:          # explicitly "this version has no such artefact"
                continue
            if not _filled(value):
                out.append(f"paths.{kind}")
        for canonical, value in fields.get("columns", {}).items():
            if value is None:          # explicitly "this version has no such column"
                continue
            if not _filled(value):
                out.append(f"columns.{canonical}")
        gaps[v] = out
    return gaps


def _print_missing() -> None:
    gaps = missing()
    print(f"\nmodel_repos/real -> {MODEL_REPOS}")
    print(f"exists: {MODEL_REPOS.exists()}\n")

    if not any(gaps.values()):
        print("All declarations are filled in.\n")
        return

    print("Still to fill in (edit VERSIONS in this file):\n")
    for v, keys in gaps.items():
        if keys:
            print(f"  {v}:")
            for k in keys:
                print(f"      {k}")
    print("\nA 'paths.<kind>' left blank is not an error — it falls back to our own tree.")
    print("Fill it in only when that artefact already exists in the real repo.")
    print("The 'confirmed' entries (columns.date, columns.observed, trained_on) are correct — leave them.\n")


def _print_paths(source: str = "real") -> None:
    print(f"\nResolved paths (source={source}). 'declared' = read from the real repo,")
    print("'fallback' = our own tree.\n")
    for v in VERSION_LABELS:
        print(f"  {v}")
        for kind in KINDS:
            declared = VERSIONS[v].get("paths", {}).get(kind)
            origin = "declared" if _filled(declared) else "fallback"
            try:
                resolved = path(kind, v, source)
            except ValueError as exc:
                resolved = f"UNRESOLVED ({exc.args[0].split('.')[0]}...)"
            print(f"      {kind:<14} [{origin}]  {resolved}")
        print()


def _print_columns() -> None:
    print("\nColumn mapping — THEIR name -> OUR canonical name (applied once, at ingest).\n")
    for v in VERSION_LABELS:
        print(f"  {v}")
        for ours in CANONICAL_COLUMNS:
            real = VERSIONS[v].get("columns", {}).get(ours)
            shown = "(this version has no such column)" if real is None else real
            print(f"      {shown:<32} -> {ours}")
        print()


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if "--paths" in args:
        _print_paths("synthetic" if "--synthetic" in args else "real")
    elif "--columns" in args:
        _print_columns()
    else:
        _print_missing()
