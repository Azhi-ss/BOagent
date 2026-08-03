# Execution Plan: Chem-LGBO Tool Call + ReAct

## Implementation Order

### 1. Client response contract

Files:
- `packages/bo-core/bo_core/llm_client.py`
- `packages/bo-core/tests/test_llm_client.py` or existing nearest client test

Changes:
- widen message typing to nested OpenAI-compatible objects;
- add defaulted `LlmCallResult.tool_calls`;
- add keyword-only `temperature=0.0`;
- accept empty content when tool calls exist;
- preserve protected payload-key behavior.

Verify:
- payload temperature default/override;
- empty content + tool call returns success;
- text-only existing response unchanged.

### 2. Base LGBO plumbing

Files:
- `packages/bo-core/bo_core/optimization/lgbo.py`
- `packages/bo-core/tests/test_lgbo.py`

Changes:
- add `llm_temperature=0.0` constructor parameter;
- add optional tools/tool_choice to `_call_llm`;
- pass explicit temperature and tool payload to client.

Verify:
- existing point LGBO path remains text-based;
- default temperature remains 0.0;
- client errors retain existing fallback reasons.

### 3. Chem Tool Call + semantic ReAct

Files:
- `packages/bo-core/bo_core/optimization/chem_lgbo.py`
- `packages/bo-core/tests/test_chem_lgbo.py`

Changes:
- add module-level tool schema and forced choice;
- set Chem default temperature to 0.2;
- evaluate target tool arguments through existing Parser;
- evaluate accepted subspace against remaining pool before accepting;
- retry once for structural, dictionary, and semantic reasons;
- construct assistant/tool messages with matching call id;
- keep retry conversation local;
- record final telemetry in guidance artifacts.

Verify:
- `unknown_value -> accepted`;
- `already_queried_only -> accepted`;
- `uninformative_full_pool -> accepted`;
- two failures -> one final fallback artifact;
- retry messages absent from trajectory and next step prompt.

### 4. Experiment provenance

Files:
- `Compitetion/auto_research/chem_lgbo_experiment.py`
- `Compitetion/auto_research/chem_lgbo_prompt_ablation.py`
- nearest experiment tests

Changes:
- thread real Chem temperature where engines are constructed;
- record temperature, response mode, and retry limit in config/model_config;
- remove hardcoded `temperature: 0.0` for new runs.

Verify:
- output artifact reports the actual runtime protocol;
- report/preflight modes do not invoke LLM unexpectedly;
- old artifacts remain readable.

### 5. Submission parity

Files:
- `Compitetion/submission/code/bo_core/llm_client.py`
- `Compitetion/submission/code/bo_core/optimization/lgbo.py`
- `Compitetion/submission/code/bo_core/optimization/chem_lgbo.py`
- submission contract tests

Changes:
- mirror the three package behavior changes exactly;
- do not modify parser or mean-shift mathematics.

Verify:
- package and submission accept/reject the same scripted responses;
- submission smoke test imports its vendored `bo_core`, not the package copy.

### 6. Static and behavioral verification

Run from repository root:

```bash
uv run ruff check packages/bo-core/bo_core packages/bo-core/tests Compitetion/auto_research Compitetion/submission/code/bo_core
uv run mypy packages/bo-core/bo_core
uv run pytest packages/bo-core/tests -q
uv run pytest Compitetion/auto_research/tests/test_chem_lgbo_experiment.py Compitetion/auto_research/tests/test_chem_lgbo_prompt_ablation.py -q
uv run pytest Compitetion/submission/code/tests/test_run_submission.py -q
```

Then run one saved-state real-client smoke at temperature 0.2. Report:

- initial calls;
- structurally valid tool calls;
- semantically accepted subspaces;
- retries attempted;
- retries recovered;
- final fallbacks by reason.

## Stop Conditions

Do not run a new full 226-state experiment in this task. Stop after contracts, tests, mirror parity, and bounded real-client smoke pass. A new comparative experiment is a separate protocol/versioned artifact.
