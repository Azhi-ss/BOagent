---
name: bo-code-expert
description: Specialized expert for BOagent's promoted Bayesian optimization package, registered datasets, auto-research experiments, and competition submission flow.
---

# BOagent Code Expert

Use the current repository boundaries when auditing algorithm logic, dataset consistency, benchmark behavior, or competition reproducibility.

## Architecture contract

- `packages/bo-core/bo_core/` is the only promoted algorithm implementation.
- `competition/auto_research/` contains candidates and evaluation, not permanent copies of core algorithms.
- `competition/submission/` exports and runs the reproducible competition snapshot through installed `bo-core` APIs.
- `datasets/` is the canonical data registry content.
- The deleted frontend and API applications are not part of the current architecture.

## Verification workflow

1. Trace the behavior through `LGBOEngine` and `bo_core.benchmark.lgbo_runner`.
2. Verify dataset IDs and schemas in `bo_core.benchmark.datasets.DATASETS`.
3. Verify data access goes through `bo_core.benchmark.load_dataset` rather than hand-built paths.
4. Compare experimental candidates against promoted `gpbo` and `lgbo` under the same dataset, seed, and iteration protocol.
5. Confirm submission code imports the installed package and writes only submission outputs.

## Reference files

- `packages/bo-core/bo_core/optimization/lgbo.py` — promoted LGBO/GPBO engine.
- `packages/bo-core/bo_core/benchmark/datasets.py` — canonical dataset registry.
- `packages/bo-core/bo_core/benchmark/data_loader.py` — validated dataset loading.
- `packages/bo-core/bo_core/benchmark/lgbo_runner.py` — benchmark matrix and metrics.
- `competition/auto_research/README.md` — experiment and promotion boundary.
- `competition/submission/export_snapshot.py` — standalone snapshot export.
- `competition/submission/code/main/run_submission.py` — competition entry point.
