# auto_research Results Archive (2026-07-28)

This folder is a self-contained snapshot of the `Compitetion/auto_research`
experimental results and the architecture map needed to resume work without
re-searching the subsystem.

It does **not** contain new experiments. It only preserves what already existed
on disk at the snapshot time, with checksums so every file can be verified.

---

## Why this folder exists

The auto_research subsystem was in the middle of a protocol transition:

- Older runs used a **legacy full-prior** or a **seeded 5-point** prior.
- The current runtime code (`engine.py`, `loop.py`, `agent_step.py`) has been
  switched to the **competition fixed-prior** protocol, but no
  `*_fixed_prior.json` result file existed yet when this archive was taken.
- During that transition, two result files were at risk of being lost:
  - `lgbo_mean_shift_seeded.json` — tracked by Git but expanded on disk from
    8 committed records to the full 40.
  - `lgbo_cake_seeded.json` — fully complete on disk but not yet tracked.

This archive freezes those results, plus the full experiment catalog, so the
historical record cannot be silently lost or overwritten.

---

## Folder contents

```
results_2026-07-28/
├── README.md                    # this file — entry point
├── METHODS.md                   # methods used, n_initial=5, protocol, metrics
├── ARCHITECTURE.md              # module / data-flow / resume checklist
├── EXPERIMENT_MANIFEST.md       # 14 experiment JSONs: counts, protocol,
│                                 #   per-dataset metrics, SHA-256, git state
└── snapshots/
    ├── README.md                # snapshot provenance and source state
    ├── SHA256SUMS               # checksums for the two at-risk result copies
    ├── experiments_sha256.txt   # checksums for all 14 experiment JSONs
    ├── lgbo_mean_shift_seeded_40.json   # byte copy of source (40 records)
    └── lgbo_cake_seeded_40.json         # byte copy of source (40 records)
```

The two `.json` copies under `snapshots/` are byte-for-byte identical to the
files in `../history/experiments/`. `cmp` and `sha256sum` both confirm this.

---

## The original (authoritative) results live elsewhere

This folder holds **copies and documentation only**. The authoritative raw
result files are untouched and remain at:

```
Compitetion/auto_research/history/experiments/*.json   (14 files, 40 records each)
Compitetion/auto_research/history/full_*.json         (earlier tier outputs)
Compitetion/auto_research/history/ledger.jsonl         (append-only research log)
Compitetion/auto_research/history/champion.json
```

Do not edit the archive copies to "update" results — re-deriving provenance
from a copied file is unsafe. If a result must be rerun, write a new file with
a `_fixed_prior.json` suffix and record the decision in `ledger.jsonl`.

---

## Methods and protocol at a glance

**Read `METHODS.md` first** for the full method × component table.

The 7 completed seeded results all used:

| Setting | Value |
|---|---|
| Protocol | `prior_protocol="seeded_subsample"` |
| Initial prior size | **`n_initial = 5`** |
| Prior sampling | 5 rows drawn from train set via local `RandomState(seed)`; indices stored as `initial_indices` |
| Query budget | 40 steps (test-pool only) |
| Seeds | `100, 200, …, 2000` (20) |
| Datasets | `buchwald_sub4`, `suzuki` |
| Status | 40/40 `status="ok"` per file |

| Method | Surrogate | LLM | Result file |
|---|---|---|---|
| `gpbo_ei` | Matérn GP | none | `gpbo_ei_seeded.json` |
| `gpbo_manifold` | Kernel Manifold | none | `gpbo_manifold_seeded.json` |
| `gpbo_alas` | ALAS α-stable | none | `gpbo_alas_seeded.json` |
| `gpbo_dkl` | Deep Kernel Learning | none | `gpbo_dkl_seeded.json` |
| `gpbo_cake` | CAKE kernel pop. | kernel-only | `gpbo_cake_seeded.json` |
| `lgbo_mean_shift` | Matérn GP | mean-shift | `lgbo_mean_shift_seeded.json` |
| `lgbo_cake` | CAKE + mean-shift | both | `lgbo_cake_seeded.json` |

### Three protocol families — never aggregate across them

| Family | How to recognize | Result files |
|---|---|---|
| Legacy full-prior (unlabeled) | No `prior_protocol` field in JSON; provenance from `roadmap.md` / `ledger.jsonl` | `H1_*`–`H4_*`, early `full_*` |
| Seeded 5-point subsample | `prior_protocol="seeded_subsample"`, **`n_initial=5`**, five `initial_indices` | `*_seeded.json` (7 complete files) |
| Competition fixed-prior (current code) | `prior_protocol="fixed_train_prior"`, full `train.csv` prior | intended `*_fixed_prior.json` — **none existed at archive time** |

Per-dataset metric means and SHA-256 checksums are in `EXPERIMENT_MANIFEST.md`.
Full component / param tables are in `METHODS.md`.

---

## How to verify this archive

```bash
cd Compitetion/auto_research/results_2026-07-28

# 1. Snapshot copies match their checksums.
cd snapshots && sha256sum -c SHA256SUMS && cd ..

# 2. Snapshot copies are byte-identical to the authoritative sources.
cmp ../history/experiments/lgbo_mean_shift_seeded.json \
    snapshots/lgbo_mean_shift_seeded_40.json
cmp ../history/experiments/lgbo_cake_seeded.json \
    snapshots/lgbo_cake_seeded_40.json

# 3. Every experiment JSON still matches its recorded checksum.
cd ../history/experiments && sha256sum -c \
    ../../results_2026-07-28/snapshots/experiments_sha256.txt
```

All three checks pass as of the archive creation time.

---

## Resume checklist

1. Read `METHODS.md` for which methods ran, with what components, and under
   which protocol (`n_initial=5` seeded vs legacy full-prior vs fixed-prior).
2. Read `ARCHITECTURE.md` for the module map and data flow.
3. Read `EXPERIMENT_MANIFEST.md` for the full result catalog and checksums.
4. Confirm the intended active protocol before running anything — do not infer
   it from a filename.
5. Check `../history/experiments/` for an existing output before starting a run.
6. Use distinct filenames per protocol (`*_seeded.json` vs `*_fixed_prior.json`).
7. Run the seed-completeness and mixed-protocol guards before comparing methods.
8. Append conclusions to `../history/ledger.jsonl`; do not rewrite raw result
   JSON to change provenance.

---

## Provenance

- Archive created: 2026-07-28
- Source commit: `313c86cca68fd62adeea73734285aabaa12db2a5`
- Source working tree: uncommitted fixed-prior protocol changes (engine/loop/
  agent_step) plus the completed 40-record seeded results on disk.
- No original experiment JSON was modified to build this archive.
```
