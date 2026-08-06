# BOagent AI Developer Guide

BOagent is a Bayesian optimization research repository. The promoted algorithm implementation lives only in `packages/bo-core`; competition experiments consume that package rather than maintaining parallel implementations.

## Repository contract

| Path | Responsibility |
|---|---|
| `packages/bo-core/bo_core/` | Promoted algorithms, dataset interfaces, and benchmark runners |
| `packages/bo-core/tests/` | Contract and regression tests for promoted code |
| `competition/auto_research/` | Experimental candidates and evaluation; never the permanent algorithm source |
| `competition/submission/` | Reproducible competition snapshot export and submission execution |
| `datasets/` | Canonical dataset registry content |

There is no supported frontend or API service in the current tree. Do not restore or reference the deleted `apps/api` or `apps/web` paths.

## Algorithm and data flow

1. `bo_core.benchmark.datasets.DATASETS` defines each dataset ID, category, feature columns, target, and objective.
2. `bo_core.benchmark.load_dataset(dataset_id)` loads the registered files from `datasets/`.
3. `bo_core.optimization.lgbo.LGBOEngine` owns the promoted LGBO loop; `gpbo` uses the same engine with LLM guidance disabled.
4. `bo_core.benchmark.lgbo_runner` runs dataset × method × seed configurations and writes trajectories plus summary metrics.
5. `competition/auto_research` evaluates candidates against the promoted `gpbo` and `lgbo` contracts. A candidate moves to `packages/bo-core` only after its declared promotion gate passes, and its experimental duplicate is then removed.
6. `competition/submission/code/main/run_submission.py` calls the installed `bo-core` runner. `competition/submission/export_snapshot.py` exports the package, submission code, and competition datasets as a standalone snapshot.

Dataset code must use the registry and loader. Do not assemble dataset paths, repeat feature schemas, or hard-code `global_best` in algorithms or experiments.

## Supported commands

Run commands from the repository root:

```bash
uv sync
uv run pytest packages/bo-core/tests competition/auto_research/tests competition/submission/test_export_snapshot.py
uv run ruff check packages/bo-core competition competition/submission
```

Quick GPBO smoke run:

```bash
uv run python -m bo_core.benchmark.lgbo_runner \
  --datasets buchwald_sub4 \
  --methods gpbo \
  --seeds 100 \
  --n_iters 1 \
  --workers 1 \
  --backend sklearn \
  --output_dir /tmp/boagent-smoke
```

Run the competition submission:

```bash
uv run python competition/submission/code/main/run_submission.py
```

Export a standalone competition snapshot:

```bash
uv run python competition/submission/export_snapshot.py /tmp/boagent-submission
```

## Change rules

- Keep one promoted implementation in `packages/bo-core`; experiments import it.
- Add or update observable contract tests when promoted behavior changes.
- Preserve local RNG isolation in optimization and benchmark paths.
- Load secrets from environment variables; never hard-code API credentials.
- Do not modify `.env` or generated result files without explicit user approval.
- Consult relevant `docs/` references for numerical, hardware, or library constraints before changing those areas.
