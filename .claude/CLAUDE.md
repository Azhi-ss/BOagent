# ECC Project Setup

Initialized for **BOagent** on 2026-07-28.

> TODO: read `README.md` and replace this with a one-line project description.

**Dev tools:** pytest

## Dev Tooling

Detected tools (linters, formatters, static analysis, coverage):

- `pytest`

## Toolchain

Read these manifests to identify the data/ML/coverage toolchain (DataOps, MLOps, coverage tools, etc.):
- `pyproject.toml`

## Rules

Global ECC common rules are used from `~/.claude/rules/`.
Project-local language/framework rules:

- .claude/rules/ecc/python/coding-style.md
- .claude/rules/ecc/python/fastapi.md
- .claude/rules/ecc/python/hooks.md
- .claude/rules/ecc/python/patterns.md
- .claude/rules/ecc/python/security.md
- .claude/rules/ecc/python/testing.md

## Commands

- **test**: `pytest`
- **lint**: `ruff check .`

## Auto-Verification Rules

- **Mandatory Test Execution**: After any code edit or write operation, ALWAYS run the project test and coverage commands to verify correctness.
- **Fail-Fast & Auto-Fix**: If tests fail or coverage drops, analyze the error traceback and fix the code before completing the turn.

## Data & Model Weight Protection (Safety)

- **Never Direct Read Large Data/Weights**: NEVER use `Read` or `Grep` directly on large dataset or model checkpoint files (`*.pt`, `*.pth`, `*.bin`, `*.safetensors`, `*.parquet`, `*.h5`, `*.csv`, `*.feather`, `*.ckpt`).
- **Schema Inspection Only**: Use Python one-liners (`pandas`/`polars` schema inspection) or CLI tools to inspect data structures without loading multi-megabyte contents into context.

## Environment & Resource Guidelines

- **Virtual Environment First**: Always execute Python commands using the project virtual environment (e.g. `uv run` or active `.venv`).
- **GPU/CUDA Memory Isolation**: Keep single-test GPU memory allocation lightweight and release CUDA memory explicitly in benchmarks.

## Hooks

- pre:bash:block-no-verify → `.claude/scripts/hooks/block-no-verify.js`
- pre:bash:git-push-reminder → `.claude/scripts/hooks/pre-bash-git-push-reminder.js`
- pre:bash:tmux-reminder → `.claude/scripts/hooks/pre-bash-tmux-reminder.js`
- pre:write:doc-file-warning → `.claude/scripts/hooks/doc-file-warning.js`

## Notes

- Do not edit linter/formatter configs via agent without explicit approval.
- Project-specific conventions should be added to this file.
