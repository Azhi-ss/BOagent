from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bo_core.benchmark.comparison import ComparisonRunner
from bo_core.benchmark.data_loader import DATA_LOADERS, DEFAULT_DATA_ROOT
from bo_core.benchmark.runner import BenchmarkRunner, run_multi_seed
from bo_core.optimization.knowledge import KnowledgeEngine
from bo_core.optimization.optimizer import BayesianOptimizer
from bo_core.optimization.space import ContinuousSearchSpace
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Load environment variables only from the project root.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")  # project root

app = FastAPI(
    title="BOagent API",
    version="0.2.0",
    description="PVK-BO Benchmark Agent — GP+LLM acquisition function evaluation.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://[::1]:3000",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://[::1]:4173",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://[::1]:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://[::1]:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://[::1]:5175",
        "http://0.0.0.0:5173",
        "http://0.0.0.0:5174",
        "http://0.0.0.0:5175",
    ],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1|\[::1\]|0\.0\.0\.0):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Backend event log (SSE-capable)
# ---------------------------------------------------------------------------

class BackendLogBuffer:
    def __init__(self, max_events: int = 500) -> None:
        self.max_events = max_events
        self._events: list[dict[str, Any]] = []
        self._next_id = 1

    def append(
        self,
        event: str,
        message: str,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        item = {
            "id": self._next_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "message": message,
            "detail": detail or {},
        }
        self._next_id += 1
        self._events.append(item)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]
        return item

    def recent(self, limit: int = 100, after: int = 0) -> list[dict[str, Any]]:
        events = [event for event in self._events if int(event["id"]) > after]
        return events[-limit:]

    def clear(self) -> None:
        self._events.clear()
        self._next_id = 1


backend_log = BackendLogBuffer()


def emit_backend_log(
    event: str,
    message: str,
    *,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return backend_log.append(event, message, detail)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class CreateBenchmarkBody(BaseModel):
    task_id: str = Field(default="band_alignment", pattern="^(band_alignment|defects_doping|buchwald_sub4|suzuki|battery_cathode)$")
    n_initial: int = Field(default=5, ge=1, le=50)
    n_trials: int = Field(default=20, ge=1, le=200)
    seed: int = Field(default=42, ge=0)
    seeds: list[int] | None = None
    sm_mode: str = Field(default="discriminative", pattern="^(discriminative|generative)$")
    n_candidates: int = Field(default=10, ge=1, le=50)
    n_templates: int = Field(default=2, ge=1, le=10)
    n_gens: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.1, ge=-1.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=100)
    output_dir: str = Field(default="results")


class TraditionalConfig(BaseModel):
    acquisition: str = Field(default="ei", pattern="^(ei|ucb|pi)$")
    xi: float = Field(default=0.01, ge=0.0, le=1.0)
    kappa: float = Field(default=2.576, ge=0.0, le=10.0)


class LLMBOConfig(BaseModel):
    acquisition: str = Field(default="ei", pattern="^(ei|ucb|pi)$")
    xi: float = Field(default=0.01, ge=0.0, le=1.0)
    kappa: float = Field(default=2.576, ge=0.0, le=10.0)
    n_candidates: int = Field(default=5, ge=1, le=50)
    n_templates: int = Field(default=2, ge=1, le=10)
    top_k: int = Field(default=20, ge=1, le=100)
    alpha: float = Field(default=0.1, ge=-1.0, le=1.0)
    chat_engine: str | None = None
    use_llm_heuristic: bool = False
    use_direct_full_pool: bool = False
    heuristic_weight: float = 0.3


class CompareBenchmarkBody(BaseModel):
    task_id: str = Field(default="band_alignment", pattern="^(band_alignment|defects_doping|buchwald_sub4|suzuki|battery_cathode)$")
    n_initial: int = Field(default=5, ge=1, le=50)
    n_trials: int = Field(default=20, ge=1, le=200)
    seeds: list[int] = Field(default=[42, 7, 100, 1, 21])
    traditional: TraditionalConfig = Field(default_factory=TraditionalConfig)
    llmbo: LLMBOConfig = Field(default_factory=LLMBOConfig)


class OperationalVariable(BaseModel):
    name: str
    min: float
    max: float
    unit: str = ""


class OperationalObservation(BaseModel):
    config: dict[str, float]
    score: float


class OperationalSuggestBody(BaseModel):
    target: str = "score"
    variables: list[OperationalVariable]
    history: list[OperationalObservation]
    llm_config: LLMBOConfig = Field(default_factory=LLMBOConfig)
    n_sample: int = Field(default=2000, ge=100, le=10000)
    seed: int = Field(default=42, ge=0)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def success(data: Any) -> dict[str, Any]:
    return {"data": data}


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    code = "not_found" if exc.status_code == status.HTTP_404_NOT_FOUND else "http_error"
    return error_response(exc.status_code, code, detail)


# ---------------------------------------------------------------------------
# Logs HTML + API
# ---------------------------------------------------------------------------

def _logs_page_html() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BOagent Live Logs</title>
  <style>
    body { margin: 0; background: #0f1115; color: #e5e7eb; font: 14px/1.55 ui-monospace, SFMono-Regular, Consolas, monospace; }
    header { position: sticky; top: 0; padding: 16px 20px; background: #171717; border-bottom: 1px solid #2f3542; }
    h1 { margin: 0; font-size: 16px; }
    main { padding: 16px 20px; }
    .event { border: 1px solid #2f3542; border-radius: 12px; padding: 12px; margin-bottom: 10px; background: #151922; }
    .meta { color: #9ca3af; font-size: 12px; margin-bottom: 6px; }
    .name { color: #d4af37; font-weight: 700; }
    pre { white-space: pre-wrap; word-break: break-word; margin: 8px 0 0; color: #cbd5e1; }
  </style>
</head>
<body>
  <header>
    <h1>BOagent Live Logs</h1>
    <div class="meta">Streaming from /api/v1/logs/stream</div>
  </header>
  <main id="events"></main>
  <script>
    const root = document.getElementById("events");
    function addEvent(item) {
      const node = document.createElement("section");
      node.className = "event";
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `#${item.id} · ${item.timestamp} · ${item.event}`;
      const message = document.createElement("div");
      message.textContent = item.message || "";
      const detail = document.createElement("pre");
      detail.textContent = JSON.stringify(item.detail || {}, null, 2);
      node.append(meta, message, detail);
      root.prepend(node);
    }
    fetch("/api/v1/logs?limit=50")
      .then((response) => response.json())
      .then((payload) => (payload.data.events || []).forEach(addEvent));
    const source = new EventSource("/api/v1/logs/stream");
    source.onmessage = (message) => addEvent(JSON.parse(message.data));
  </script>
</body>
</html>"""


@app.get("/logs", response_class=HTMLResponse)
def logs_page() -> HTMLResponse:
    return HTMLResponse(_logs_page_html())


@app.get("/architecture", response_class=HTMLResponse)
def architecture_page() -> HTMLResponse:
    """Serve the system architecture interactive documentation."""
    path = Path(__file__).parent.parent / "docs" / "architecture" / "architecture.html"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Architecture documentation file not found.",
        )
    return HTMLResponse(content=path.read_text(encoding="utf-8"))


@app.get("/api/v1/logs")
def list_backend_logs(limit: int = 100, after: int = 0) -> dict[str, Any]:
    return success({"events": backend_log.recent(limit=max(1, min(limit, 500)), after=after)})


@app.get("/api/v1/logs/stream")
async def stream_backend_logs(after: int = 0) -> StreamingResponse:
    async def event_stream():
        last_id = after
        while True:
            events = backend_log.recent(limit=100, after=last_id)
            for item in events:
                last_id = int(item["id"])
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Core API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health")
def health() -> dict[str, Any]:
    return success(
        {
            "status": "ok",
            "service": "boagent-api",
            "version": app.version,
        }
    )


@app.get("/api/v1/tasks")
def list_tasks() -> dict[str, Any]:
    tasks = []
    for task_id in DATA_LOADERS:
        name = task_id.replace("_", " ").title()
        tasks.append(
            {
                "task_id": task_id,
                "name": name,
                "objective": f"Maximize eta over {name.lower()} features.",
                "data_source": f"PVK-LLM:{task_id}",
                "data_available": True,
                "data_boundary": {
                    "notes": f"GP+LLM ACQ benchmark over the PVK-LLM {task_id} dataset.",
                    "constraints": [
                        "Benchmark evaluation only; uses custom_perovskite_dataset Excel workbooks.",
                    ],
                },
                "source_path": str(DEFAULT_DATA_ROOT),
            }
        )
    return success(tasks)


@app.post(
    "/api/v1/benchmark",
    status_code=status.HTTP_202_ACCEPTED,
)
def create_benchmark_run(body: CreateBenchmarkBody) -> dict[str, Any]:
    """Submit a benchmark run. Execution is synchronous; may take minutes to complete."""

    # Validate output_dir: reject path traversal attempts
    output_path = Path(body.output_dir)
    if output_path.is_absolute() or ".." in str(output_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="output_dir must be a relative path without '..' traversal.",
        )

    emit_backend_log(
        "benchmark.request",
        f"Benchmark request: {body.task_id}",
        detail={"task_id": body.task_id, "seed": body.seed, "seeds": body.seeds},
    )

    try:
        common_kwargs = {
            "task_id": body.task_id,
            "n_initial": body.n_initial,
            "n_trials": body.n_trials,
            "sm_mode": body.sm_mode,
            "chat_engine": os.environ.get("DEEPSEEK_FLASH_MODEL") or os.environ.get("DEEPSEEK_MODEL") or "deepseek-v4-flash",
            "n_candidates": body.n_candidates,
            "n_templates": body.n_templates,
            "n_gens": body.n_gens,
            "alpha": body.alpha,
            "top_k": body.top_k,
            "output_dir": body.output_dir,
        }

        if body.seeds:
            results = run_multi_seed(seeds=body.seeds, **common_kwargs)
        else:
            runner = BenchmarkRunner(seed=body.seed, **common_kwargs)
            result = runner.run()
            runner.save_results(result)
            results = [result]

        emit_backend_log(
            "benchmark.complete",
            f"Benchmark complete: {body.task_id}",
            detail={
                "task_id": body.task_id,
                "runs": len(results),
                "best_scores": [r["best_score"] for r in results],
            },
        )

        return success(
            {
                "task_id": body.task_id,
                "runs": len(results),
                "results": [
                    {
                        "seed": r["seed"],
                        "best_score": r["best_score"],
                        "best_generalization_score": r["best_generalization_score"],
                    }
                    for r in results
                ],
                "output_dir": str(
                    Path(body.output_dir)
                    / f"results_{body.sm_mode}"
                    / body.task_id
                ),
            }
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data file not found: {exc}",
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None
    except Exception:
        emit_backend_log("benchmark.error", "Benchmark failed: internal error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Benchmark run failed due to an internal error. Check backend logs for details.",
        ) from None


@app.post("/api/v1/benchmark/compare/stream")
async def compare_benchmark_stream(body: CompareBenchmarkBody) -> StreamingResponse:
    """Run traditional BO and LLMBO across multiple seeds over a shared
    train/test split, streaming aggregate convergence events as SSE.

    Each SSE ``data:`` line is one JSON event:
      - {"type": "meta", ...}        once at start
      - {"type": "seed_start", ...}  per seed
      - {"type": "step_start", ...}  per engine per step (drives busy indicator)
      - {"type": "aggregate", ...}   per seed completion (mean/std trajectories)
      - {"type": "done", ...}        once at end (final mean ± std summary)
      - {"type": "error", ...}       on failure
    """
    emit_backend_log(
        "compare.request",
        f"Comparison request: {body.task_id}",
        detail={"task_id": body.task_id, "seeds": body.seeds, "n_trials": body.n_trials},
    )

    runner = ComparisonRunner(
        task_id=body.task_id,
        n_initial=body.n_initial,
        n_trials=body.n_trials,
        seeds=body.seeds,
        traditional=body.traditional.model_dump(),
        llmbo=body.llmbo.model_dump(),
    )

    async def event_stream():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def produce():
            try:
                for event in runner.events():
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:  # surface engine errors to the client
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "error", "message": str(exc)}
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        task = loop.run_in_executor(None, produce)
        try:
            while True:
                event = await queue.get()
                if event is sentinel:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            await task

        emit_backend_log("compare.complete", f"Comparison complete: {body.task_id}")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/v1/operational/suggest")
def operational_suggest(body: OperationalSuggestBody) -> dict[str, Any]:
    """Provide next experiment suggestions for human-in-the-loop operational mode."""
    emit_backend_log(
        "operational.suggest",
        f"Operational suggest request for {body.target}",
        detail={"variables": len(body.variables), "history": len(body.history)},
    )

    try:
        # 1. Setup Space and Optimizer
        space = ContinuousSearchSpace(
            variables=[v.model_dump() for v in body.variables],
            n_samples=body.n_sample,
            seed=body.seed
        )
        
        chat_engine = body.llm_config.chat_engine or os.environ.get("DEEPSEEK_FLASH_MODEL") or "deepseek-v4-flash"
        knowledge = KnowledgeEngine(chat_engine=chat_engine)
        
        optimizer = BayesianOptimizer(
            space=space,
            target_name=body.target,
            knowledge_engine=knowledge,
            seed=body.seed
        )
        
        # 2. Reconstruct history
        for obs in body.history:
            optimizer.observe(obs.config, obs.score)
            
        # 3. Get suggestion
        result = optimizer.suggest(
            top_k=body.llm_config.top_k,
            n_candidates=body.llm_config.n_candidates,
            acquisition=body.llm_config.acquisition,
            kappa=body.llm_config.kappa,
            xi=body.llm_config.xi,
            use_llm=True,
            use_llm_heuristic=body.llm_config.use_llm_heuristic,
            use_direct_full_pool=body.llm_config.use_direct_full_pool,
            heuristic_weight=body.llm_config.heuristic_weight
        )

        return success({
            "suggestions": result.suggestions,
            "analysis": result.analysis,
            "prompt": result.prompt
        })

    except Exception as exc:
        emit_backend_log("operational.error", f"Operational suggest failed: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from None
