# Journal - Zorya (Part 1)

> AI development session journal
> Started: 2026-07-24

---



## Session 1: BoTorch-first GP backend migration

**Date**: 2026-07-25
**Task**: BoTorch-first GP backend migration
**Package**: bo-core
**Branch**: `main`

### Summary

Implemented shared sklearn/BoTorch surrogate layer (surrogate.py), switched default backend to BoTorch/GPyTorch across BayesianOptimizer, LGBOEngine, benchmark runners and CLI. sklearn moved to optional dependency; Python minimum raised to 3.11. Fixed BoTorch L-BFGS-B ABNORMAL convergence via maxls=80 and inference jitter reuse. Added backend_benchmark.py with formal H365 benchmark (0/960 OptimizationWarnings). 98 tests pass. Merged as PR #1.

### Main Changes

- Detailed change bullets were not supplied; see the summary above.

### Git Commits

| Hash | Message |
|------|---------|
| `300c3bb` | (see git log) |
| `37230e0` | (see git log) |

### Testing

- Validation was not recorded for this session.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 2: BoTorch-first migration + 20-seed LGBO benchmark

**Date**: 2026-07-25
**Task**: BoTorch-first migration + 20-seed LGBO benchmark
**Package**: bo-core
**Branch**: `main`

### Summary

Migrated GP backend to BoTorch-first (sklearn optional, Python>=3.11). Implemented shared SurrogateModel layer with SklearnSurrogate/BoTorchSurrogate, L-BFGS-B maxls=80, inference jitter reuse (0/960 OptimizationWarnings). Switched all defaults to botorch across optimizer, LGBOEngine, benchmark runners, CLI. Merged PR #1. Ran 3-seed validation and full 20-seed LGBO/GPBO sweep (80 configs, 5940s, 8 workers): Buchwald GPBO 86.60/LGBO 86.46; Suzuki GPBO 94.33/LGBO 95.52 (LGBO +1.19%, t95 23.7 vs 41). Identified next improvement: optimize_acqf multi-start L-BFGS-B with snap-to-pool for Suzuki t95 from 23.7 towards paper's 15.7.

### Main Changes

- Detailed change bullets were not supplied; see the summary above.

### Git Commits

| Hash | Message |
|------|---------|
| `300c3bb` | (see git log) |
| `37230e0` | (see git log) |

### Testing

- Validation was not recorded for this session.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 3: Chemistry BO fixed-prior protocol + result archive

**Date**: 2026-07-29
**Task**: Chemistry BO fixed-prior protocol + result archive
**Package**: bo-core
**Branch**: `main`

### Summary

Completed the chemistry BO protocol work and archived task 07-24-chemical-bo-run.

Done:
- Competition fixed-train prior: Buchwald 35 / Suzuki 29 rows; Buchwald 32-dim options.json encoder with zero-block for train-only unknowns; candidates/LLM remain strict.
- Removed random n_initial from auto_research; submission routes through LGBO runner; prior_protocol metadata + diagnostics.
- Archived seeded-subsample results (n_initial=5): full 40-record lgbo_mean_shift/lgbo_cake plus results_2026-07-28 catalog (METHODS/ARCHITECTURE/MANIFEST/checksums).
- Project ECC config and hardware docs committed.

Deferred / follow-up (not blocking archive):
- Full fixed-prior 20-seed LGBO/GPBO reruns (*_fixed_prior.json) — none produced yet.
- formal scientific-reviewer pass on mean-shift math still open in original PRD AC.
- 00-bootstrap-guidelines left active (spec still incomplete).

### Main Changes

- Detailed change bullets were not supplied; see the summary above.

### Git Commits

| Hash | Message |
|------|---------|
| `0ed3495` | (see git log) |
| `6f7b893` | (see git log) |
| `c90490f` | (see git log) |

### Testing

- Validation was not recorded for this session.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 4: Archive bootstrap guidelines (specs already in place)

**Date**: 2026-07-29
**Task**: Archive bootstrap guidelines (specs already in place)
**Package**: bo-core
**Branch**: `main`

### Summary

Archived 00-bootstrap-guidelines as complete for practical purposes.

Project conventions are already established:
- Shared: root CLAUDE.md, AGENTS.md, .claude ECC rules/hooks, docs/
- Local Trellis specs: .trellis/spec/{api,bo-core,web,guides}

The bootstrap PRD checklist (PVK-LLM / faithful-bo / cake) was outdated
template scaffolding and does not match the current package layout.
No further bootstrap work needed; .trellis remains gitignored and local-only.

### Main Changes

- Detailed change bullets were not supplied; see the summary above.

### Git Commits

| Hash | Message |
|------|---------|
| `c90490f` | (see git log) |

### Testing

- Validation was not recorded for this session.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 5: 80-run hybrid matrix: validation-gate fix and full completion

**Date**: 2026-07-30
**Task**: 80-run hybrid matrix: validation-gate fix and full completion
**Package**: bo-core
**Branch**: `main`

### Summary

Completed the frozen 2x2x20x40=80 hybrid comparison matrix (lgbo_manifold/dkl x buchwald/suzuki x seeds 100-2000). Fixed the last failing unit (lgbo_dkl/buchwald_sub4/seed1800): root cause was the validation gate's absolute 1e-12 degeneracy threshold misjudging a sparse-but-valid EI spike (single argmax-meaningful candidate among 782 zeros), NOT DKL numerics or EI underflow — verified by log-space EI giving identical std. Switched is_constant to a relative threshold (std<=1e-9*max|scores|). Also added LLM client retry/backoff for the 30 req/min relay, resolved the hardcoded model-name override, and recorded the fingerprint-encoding experiment. Final: 80/80 ok, all gates zero, LLM failure rate 2.09%, lgbo_manifold champion (composite 0.9053) with both datasets touching global_best in single seeds.

### Main Changes

- Detailed change bullets were not supplied; see the summary above.

### Git Commits

| Hash | Message |
|------|---------|
| `d5eed94` | (see git log) |
| `bc6c868` | (see git log) |
| `c699124` | (see git log) |
| `6f27a13` | (see git log) |

### Testing

- Validation was not recorded for this session.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 6: Chem-LGBO Tool Call and ReAct

**Date**: 2026-08-03
**Task**: Chem-LGBO Tool Call and ReAct
**Package**: bo-core
**Branch**: `feature/chem-lgbo-v1`

### Summary

Implemented forced Tool Calling for Chem-LGBO with one retry, typed client payloads, parser and fallback telemetry, package/submission parity, provenance safeguards, experiment protocol, and regression coverage. Verified bo-core 187 tests, experiment 91 tests, submission 5 tests, changed-file Ruff, and mirror equality.

### Main Changes

- Detailed change bullets were not supplied; see the summary above.

### Git Commits

| Hash | Message |
|------|---------|
| `7c7c73a00234d6f3605ce902c8e372aaa079d5a1` | (see git log) |
| `3032597dcc4c52bf2946560a28ff44bf81e7482f` | (see git log) |

### Testing

- Validation was not recorded for this session.

### Status

[OK] **Completed**

### Next Steps

- None - task complete
