# BOagent API Contract

## 1. Design Reference

This API follows the Claw Code architectural idea of separating:

- message/runtime orchestration
- tool or stage execution
- session/run persistence
- structured response envelopes

The P0 implementation is intentionally deterministic. It does not call a real LLM provider yet, but the API shape leaves room for Anthropic/OpenAI-compatible providers later.

## 2. Response Envelope

Success:

```json
{
  "data": {}
}
```

Error:

```json
{
  "error": {
    "code": "not_found",
    "message": "Agent run 'run_x' was not found."
  }
}
```

## 3. Endpoints

### `GET /api/v1/health`

Returns API status.

### `POST /api/v1/agent-runs`

Creates and executes a deterministic agent run.

Request:

```json
{
  "task_text": "优化钙钛矿钝化配方，提高 PCE",
  "recommendation_count": 3,
  "language": "zh",
  "use_llm": true
}
```

Response `data`:

```json
{
  "run_id": "run_abc123",
  "status": "completed",
  "stage_results": [],
  "data_summary": {},
  "recommendations": [],
  "feedback": {},
  "llm_notes": {
    "status": "success",
    "provider": "deepseek",
    "model": "deepseek-v4-flash",
    "content": "..."
  },
  "guardrails": {},
  "artifacts": ["data-summary", "recommendations", "simulated-feedback", "agent-stages", "llm-notes"]
}
```

Set `use_llm` to `false` or omit it to run the deterministic local pipeline without spending tokens. Set `use_llm` to `true` to call DeepSeek using `.env`.

### `GET /api/v1/agent-runs/{run_id}`

Reads a completed run.

### `GET /api/v1/agent-runs/{run_id}/artifacts/{artifact_name}`

Reads an artifact from a run.

Supported artifact names:

- `data-summary`
- `recommendations`
- `simulated-feedback`
- `agent-stages`
- `llm-notes` when `use_llm=true`

## 4. Agent Runtime Mapping

| Claw / AutoPolyAgent concept | BOagent API mapping |
|---|---|
| `ConversationRuntime` / `PipelineRuntime` | `AgentRunStore.create_run()` |
| `Session` | `AgentRun` |
| `ContentBlock` / message output | stage result + artifact |
| `ToolResult` | stage artifact |
| `Guardrail` | run `guardrails` object |
| `ArtifactStore` | in-memory `artifacts` map for P0 |

## 5. Run Commands

```bash
python -m uvicorn api:app --reload --port 8000
```

OpenAPI docs:

```text
http://localhost:8000/docs
```
