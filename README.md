# BOagent Demo

FastAPI + Vite/React demo for the PVK-BO Agent workflow.

## 图中 MVP v0.2

图中 MVP v0.2 展示三步 agent 流程：

1. **Initialization**: 加载 PVK-LLM 派生的 demo 数据源，并初始化优化任务上下文。
2. **Screening**: 筛选 passivation strategies/combinations，并在推荐前标注高风险或低置信候选。
3. **Optimization**: 展示 BO-style 优化轨迹和下一轮 demo 推荐候选。

该 MVP 用于说明图中的工作流形态，不用于建立科学 benchmark 结果。

## Current Agent Architecture

The current app is an LLM-first PVK BO research-agent demo:

```text
React chat UI
  -> POST /api/v1/chat
  -> ChatAgent LLM planner
  -> hidden backend gate
  -> real PVKBO runtime
  -> result interpreter
  -> natural-language reply + evidence panel
```

User-visible replies should come from the LLM or result interpreter. The backend gate stays hidden and only enforces safety: demo/reference data boundaries, fixed chat demo task settings, and "workbook lookup is not wet-lab validation" claims.

Useful local URLs after startup:

- Frontend: `http://127.0.0.1:5175`
- FastAPI docs: `http://127.0.0.1:8010/docs`
- Live backend logs: `http://127.0.0.1:8010/logs`

## Real PVKBO Status

This repo uses the original PVK-LLM implementation as a read-only reference and exposes real PVKBO session tasks through the FastAPI session API:

- `band_alignment`: expects `bandAlignment.xlsx`
- `defects_doping`: expects `defectsAndDoping.xlsx`

By default, BOagent looks for a sibling checkout:

```text
../PVK-LLM/
../PVK-LLM/custom_perovskite_dataset/
```

Override these paths when your local layout differs:

```bash
PVK_LLM_ROOT=/path/to/PVK-LLM
PVK_DATA_ROOT=/path/to/PVK-LLM/custom_perovskite_dataset
```

If the workbook, API key, or PVK-LLM dependencies are missing, real tasks fail fast with a visible API error. `demo_optimization_table.csv` remains available for `passivation_demo`, but it is not used as a substitute for real PVKBO tasks.

## Scientific Boundaries

- 真实 `band_alignment` / `defects_doping` 任务使用 PVKBO 的 `LLM_ACQ`、LLM surrogate 和 Excel black-box lookup。
- MVP 中展示的 `passivation ratio` 是 strategy/combination 标签，不是真实 molar ratio。
- PipDI 应被视为高风险候选，不应呈现为已验证推荐。
- 真实任务的 BO 曲线只展示当前 session best-so-far；它仍不是湿实验验证或正式 benchmark。
- `/api/v1/sessions/{session_id}/bo-curve` does not include fabricated `general_llm` or `bo_baseline` benchmark curves.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional LLM notes use DeepSeek env vars. Do not put real keys in git; copy `.env.example` to `.env` locally if needed.

If a real API key was ever stored in `.env`, rotate it before publishing this repository. `.env` is ignored, but local secrets can still leak through shell history, screenshots, logs, or accidental commits.

For real PVKBO LLM calls, configure either OpenAI-compatible vars or the existing DeepSeek vars:

```bash
OPENAI_API_KEY=...
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
```

or:

```bash
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

## Backend Test

```bash
python -m pytest -q tests/test_api.py tests/test_pvk_demo.py tests/test_pvk_session_runtime.py tests/test_pvk_mvp.py tests/test_pvk_llm_bo_runtime.py
```

This is the fastest demo smoke test path for the API, session runtime, PVK demo modules, and MVP v0.2 endpoints. Run `python -m pytest -q` for the full suite.

## Start FastAPI

```bash
python -m uvicorn api:app --reload --port 8000
```

If port `8000` is already occupied, use `8010`:

```bash
python -m uvicorn api:app --reload --port 8010
```

OpenAPI docs:

```text
http://localhost:8000/docs
```

When using port `8010`, open `http://localhost:8010/docs` instead.

## Start Frontend

Run the Vite frontend from `frontend/` and point it at the backend with `VITE_API_BASE_URL`:

```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

If the backend runs on `8010`, use:

```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8010 npm run dev
```

The legacy Streamlit demo can still be started with `python -m streamlit run app.py` if needed.

## API Examples

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Create a demo agent run:

```bash
curl -X POST http://localhost:8000/api/v1/agent-runs \
  -H "Content-Type: application/json" \
  -d '{
    "task_text": "优化钙钛矿钝化配方，提高 PCE",
    "recommendation_count": 3,
    "language": "zh",
    "use_llm": false
  }'
```

Fetch a run after replacing `RUN_ID` with the returned id:

```bash
curl http://localhost:8000/api/v1/agent-runs/RUN_ID
```

## Final Acceptance Checklist

最终验收前建议执行：

```bash
python -m pytest -q
```

```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run build
```

浏览器验收时，启动后端和前端，然后在浏览器中验证 MVP 流程：

```bash
python -m uvicorn api:app --reload --port 8000
```

```bash
cd frontend
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

浏览器流程：

1. 打开 Vite dev URL，通常是 `http://localhost:5173`。
2. 确认 UI 清楚表达三阶段：Initialization、Screening、Optimization。
3. 跑通 demo 路径，确认 recommendations、BO curve display 和 scientific-boundary copy 与本 README 一致。
4. 如果后端端口 `8000` 被占用，改用 `8010` 并设置 `VITE_API_BASE_URL=http://localhost:8010` 后重复浏览器流程。
