# Auto-Research Architecture and Recovery Map

Snapshot date: 2026-07-28

This document is the entry point for resuming `Compitetion/auto_research` without rediscovering the subsystem.

## Control Plane

| File | Responsibility |
|---|---|
| `GOAL.md` | Research objective, scope, budgets, failure handling, and completion criteria. |
| `roadmap.md` | Human-owned hypotheses and evidence. Agents update status/evidence only. |
| `status.json` | Current loop state, active hypothesis, budget, and champions. |
| `agent_step.py` | One resumable roadmap step: load status, resolve compositions, run queued experiments, update ledger/status/report. |
| `loop.py` | Tiered batch runner (`smoke`, `confirm`, `full`) and leaderboard/report generation. |
| `rerun_llm_resumable.py` | Checkpoint-per-run rerunner for `gpbo_cake`, `lgbo_mean_shift`, and `lgbo_cake` under the current fixed-prior protocol. |

## Algorithm Plane

| File | Responsibility |
|---|---|
| `components/protocol.py` | Four component registries and contracts: surrogate, acquisition, selector, LLM strategy. Defines `Composition` and per-step `StepContext`. |
| `components/library.py` | Registers standard surrogate/acquisition/selector/LLM implementations. |
| `components/cake.py` | CAKE kernel population, fitting, LLM crossover/mutation, ensemble prediction, and diagnostics. |
| `components/{kernel_manifold,alas,dkl}.py` | Alternative kernel/surrogate implementations used by roadmap hypotheses. |
| `compositions/base.py` | Named method configurations such as `gpbo_ei`, `gpbo_manifold`, `gpbo_alas`, `gpbo_dkl`, `gpbo_cake`, `lgbo_mean_shift`, and `lgbo_cake`. |
| `engine.py` | `HybridEngine`: loads the dataset, builds canonical categorical encoding, instantiates components, executes 40 BO steps, and records runtime diagnostics. |
| `analyze.py` | Seed completeness, protocol isolation, metric aggregation, normalized composite scoring, trajectory analysis, and Markdown report generation. |

`engine.py` prepends `Compitetion/submission/code` to `sys.path`, so its data loader and categorical encoder come from the submission copy of `bo_core`.

## Runtime Data Flow

```text
GOAL.md + roadmap.md + status.json
              |
              v
      compositions/base.py
              |
              v
components registries -----> HybridEngine(engine.py)
                              |  load train/test/options
                              |  fit surrogate
                              |  score acquisition
                              |  optional LLM decision/mean shift
                              |  select unqueried test candidate
                              v
                   trajectory + diagnostics
                              |
                              v
                 compute_metrics / run_one
                              |
              +---------------+----------------+
              |                                |
              v                                v
 history/*.json or                  history/failures.jsonl
 history/experiments/*.json
              |
              v
 assert_seed_completeness -> aggregate_results -> composite_score
              |
              v
 reports/*.md + history/ledger.jsonl + status/champion updates
```

## Artifact Map

| Path | Meaning |
|---|---|
| `history/experiments/*.json` | Per-hypothesis or per-composition 20-seed result sets. See `history/EXPERIMENT_MANIFEST.md`. |
| `history/snapshots/` | Exact copies of vulnerable uncommitted result files, with checksums. |
| `history/ledger.jsonl` | Append-only research decisions, experiment conclusions, audits, and protocol changes. |
| `history/full_*.json`, `confirm_*.json`, `smoke_*.json` | Earlier tier outputs from `loop.py`. |
| `history/champion.json` | Last persisted champion summary. |
| `history/phase1/`, `history/phase2/` | Archived loop phases and their status. |
| `reports/` | Generated comparison and roadmap reports when present. |
| `baselines/seed_baseline.json` | Baseline reference artifact. |

## Protocol Families

Never aggregate across these families.

1. **Legacy full-prior, unlabeled JSON**
   - Files `H1_*` through `H4_*` and early `full_*` artifacts.
   - `roadmap.md` identifies these as legacy full-train-prior runs with deterministic seeds.
   - The JSON records do not carry `prior_protocol`; provenance comes from the roadmap/ledger.

2. **Seeded subsample**
   - Files ending `_seeded.json`.
   - Explicit `prior_protocol="seeded_subsample"`, `n_initial=5`, and five `initial_indices` per record.
   - Seven compositions have complete 2-dataset x 20-seed outputs.

3. **Competition fixed prior (current code)**
   - Complete `train.csv` prior: Buchwald 35 rows, Suzuki 29 rows.
   - Buchwald encoder is the 32-dimensional `options.json` decision schema; training-only unknown `Reactant2` values use a zero block. Candidate and LLM encodes remain strict.
   - Intended output names are `*_fixed_prior.json`.
   - No fixed-prior result file existed at the snapshot time.

## Current Transition State

The code and research control files are temporarily inconsistent:

- `engine.py`, `loop.py`, and `agent_step.py` now implement/report `fixed_train_prior`.
- `roadmap.md` lines 8-15 and `status.json.last_event` still describe a five-point `seeded_subsample` rerun plan (`H1b-H6b`).
- The stale `rerun_llm.pid` does not identify a live process.
- The historical seeded results are preserved and must remain historical evidence, not be relabeled as fixed-prior results.

Before restarting auto-research, choose one protocol as the active roadmap protocol and update `GOAL.md`, `roadmap.md`, and `status.json` together. For competition work, the fixed-prior protocol is the implemented path.

## Resume Checklist

1. Read this file, `history/EXPERIMENT_MANIFEST.md`, `GOAL.md`, `roadmap.md`, and `status.json`.
2. Verify the intended active protocol; do not infer it from filename alone.
3. Check `history/experiments/` for an existing output before starting a run.
4. Use distinct filenames per protocol (`*_seeded.json` vs `*_fixed_prior.json`).
5. Run seed-completeness and mixed-protocol guards before comparing methods.
6. Append conclusions to `history/ledger.jsonl`; do not rewrite raw result JSON to change provenance.
