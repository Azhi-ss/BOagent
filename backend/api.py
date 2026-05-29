from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent_runtime import AgentRunRequest, AgentRunStore, serialize_run
from chat_agent import (
    ChatAgentStore,
    apply_allowed_state_update,
    apply_demo_run_consent,
    blocked_run_decision,
    can_accept_demo_run_request,
    can_request_bo_run,
    route_explicit_user_action,
)
from pvk_llm_bo_runtime import RealPvkBoUnavailableError
from pathlib import Path
import os
from benchmark.runner import BenchmarkRunner, run_multi_seed

try:
    from pvk_mvp import (
        build_passivation_target,
        compute_bo_curve,
        handle_mvp_chat_turn,
    )
except ModuleNotFoundError as exc:
    if exc.name != "pvk_mvp":
        raise

    def build_passivation_target(session: Any) -> dict[str, Any]:
        return {
            "session_id": _session_value(session, "session_id"),
            "task_id": _session_value(session, "task_id")
            or (_session_value(session, "task") or {}).get("task_id"),
            "metric": "PCE_percent",
            "passivators": ["3MTPAI", "PDAI2", "EDAI2", "PipDI"],
        }

    def compute_bo_curve(observed_fvals: list[float]) -> dict[str, Any]:
        best_so_far: float | None = None
        series = []
        for index, value in enumerate(observed_fvals):
            objective = float(value)
            best_so_far = objective if best_so_far is None else max(best_so_far, objective)
            series.append(
                {
                    "iteration": index,
                    "objective": objective,
                    "best": best_so_far,
                }
            )
        return {"metric": "PCE_percent", "series": series}

    def handle_mvp_chat_turn(
        message: str, session: Any, language: str = "zh"
    ) -> dict[str, Any]:
        del session
        assistant_message = (
            f"已收到问题：{message}"
            if language == "zh"
            else f"Received your question: {message}"
        )
        return {
            "assistant_message": assistant_message,
            "tool_calls": [
                {
                    "name": "handle_mvp_chat_turn",
                    "arguments": {"language": language},
                }
            ],
        }

try:
    from pvk_session_runtime import (
        OptimizationSessionRequest as PvkSessionRequest,
        OptimizationSessionRuntime,
    )
    from pvk_tools import list_tasks as list_pvk_tasks
except ModuleNotFoundError as exc:
    if exc.name not in {"pvk_session_runtime", "pvk_tools"}:
        raise

    @dataclass(frozen=True)
    class PvkSessionRequest:
        task_id: str
        n_initial: int = 3
        n_trials: int = 5
        seed: int = 0
        use_llm: bool = False
        language: str = "en"

    @dataclass
    class PvkSession:
        session_id: str
        status: str
        created_at: str
        task_id: str
        language: str
        config: dict[str, Any]
        step_count: int
        candidates: list[dict[str, Any]]
        artifacts: dict[str, Any]

    class PvkSessionStore:
        def __init__(self) -> None:
            self._tasks = {
                "pvk_passivation": {
                    "task_id": "pvk_passivation",
                    "title": "PVK passivation optimization",
                    "description": "Optimize perovskite passivation candidates with deterministic demo scoring.",
                }
            }
            self._sessions: dict[str, PvkSession] = {}

        def list_tasks(self) -> list[dict[str, Any]]:
            return [_normalize_task(task) for task in self._tasks.values()]

        def create_session(self, request: PvkSessionRequest) -> PvkSession:
            if request.task_id not in self._tasks:
                raise KeyError(request.task_id)
            candidates = [
                self._build_candidate(index=index, seed=request.seed, source="initial")
                for index in range(request.n_initial)
            ]
            session = PvkSession(
                session_id=f"session_{uuid4().hex[:12]}",
                status="created",
                created_at=datetime.now(UTC).isoformat(),
                task_id=request.task_id,
                language=request.language,
                config={
                    "n_initial": request.n_initial,
                    "n_trials": request.n_trials,
                    "seed": request.seed,
                    "use_llm": request.use_llm,
                },
                step_count=0,
                candidates=candidates,
                artifacts={
                    "initial-designs": {"candidates": candidates},
                    "optimization-log": [],
                },
            )
            self._sessions[session.session_id] = session
            return session

        def get_session(self, session_id: str) -> PvkSession | None:
            return self._sessions.get(session_id)

        def run_step(self, session_id: str) -> PvkSession | None:
            session = self.get_session(session_id)
            if session is None:
                return None
            if session.step_count >= session.config["n_trials"]:
                session.status = "completed"
                return session
            step_index = session.step_count + 1
            candidate = self._build_candidate(
                index=len(session.candidates),
                seed=session.config["seed"] + step_index,
                source="optimization",
            )
            session.candidates.append(candidate)
            session.step_count = step_index
            session.status = (
                "completed"
                if session.step_count >= session.config["n_trials"]
                else "running"
            )
            session.artifacts["optimization-log"].append(
                {
                    "step": step_index,
                    "selected_candidate": candidate,
                    "status": session.status,
                }
            )
            session.artifacts["candidates"] = {"candidates": session.candidates}
            return session

        def list_artifacts(self, session_id: str) -> dict[str, Any] | None:
            session = self.get_session(session_id)
            if session is None:
                return None
            return {
                "session_id": session_id,
                "artifacts": [
                    {"artifact_name": name, "content": content}
                    for name, content in session.artifacts.items()
                ],
            }

        def _build_candidate(
            self, index: int, seed: int, source: str
        ) -> dict[str, Any]:
            score = round(22.0 + ((seed + index) % 17) * 0.13, 3)
            return {
                "candidate_id": f"cand_{index + 1}",
                "source": source,
                "formulation": {
                    "passivator": ["PEAI", "FPEAI", "BAI"][(seed + index) % 3],
                    "concentration_mg_ml": 5 + ((seed + index) % 5),
                },
                "predicted_pce": score,
            }

    def serialize_session(session: PvkSession) -> dict[str, Any]:
        return _normalize_session(
            {
                "session_id": session.session_id,
                "status": session.status,
                "created_at": session.created_at,
                "task_id": session.task_id,
                "language": session.language,
                "config": session.config,
                "step_count": session.step_count,
                "candidates": session.candidates,
                "artifacts": list(session.artifacts.keys()),
            }
        )
else:

    class PvkSessionStore(OptimizationSessionRuntime):
        def list_tasks(self) -> list[dict[str, Any]]:
            tasks_by_id = {
                task["task_id"]: task
                for task in [
                    *list_pvk_tasks(),
                    *self.real_pvk_runtime.list_tasks(),
                ]
            }
            return [_normalize_task(task) for task in tasks_by_id.values()]

        def list_artifacts(self, session_id: str) -> dict[str, Any]:
            return _normalize_artifacts(self.get_artifacts(session_id))

    def serialize_session(session: dict[str, Any]) -> dict[str, Any]:
        return _normalize_session(session)


app = FastAPI(
    title="BOagent API",
    version="0.1.0",
    description="Claw-style API wrapper for the PVK-BO Agent demo pipeline.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):\d+$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
store = AgentRunStore()
session_store = PvkSessionStore()
chat_agent_store = ChatAgentStore()


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


class CreateAgentRunBody(BaseModel):
    task_text: str = Field(..., min_length=1)
    recommendation_count: int = Field(default=5, ge=3, le=5)
    language: str = Field(default="en", pattern="^(en|zh)$")
    use_llm: bool = False


class CreatePvkSessionBody(BaseModel):
    task_id: str = Field(..., min_length=1)
    n_initial: int = Field(default=3, ge=1, le=20)
    n_trials: int = Field(default=5, ge=1, le=100)
    seed: int = 0
    use_llm: bool = False
    language: str = Field(default="en", pattern="^(en|zh)$")


class ChatTurnBody(BaseModel):
    message: str = Field(..., min_length=1)
    language: str = Field(default="zh", pattern="^(zh|en)$")
    history: list[dict[str, Any]] | None = None


class AgentChatBody(BaseModel):
    conversation_id: str | None = None
    message: str = Field(..., min_length=1)
    language: str = Field(default="zh", pattern="^(zh|en)$")
    history: list[dict[str, Any]] = Field(default_factory=list)


class CreateBenchmarkBody(BaseModel):
    task_id: str = Field(default="band_alignment", pattern="^(band_alignment|defects_doping)$")
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


def _normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(task)
    task_id = normalized.get("task_id") or normalized.get("id")
    if task_id is not None:
        normalized.setdefault("id", task_id)
    if normalized.get("name") is not None:
        normalized.setdefault("title", normalized["name"])
    if normalized.get("objective") is not None:
        normalized.setdefault("description", normalized["objective"])
    normalized["data_boundary"] = _normalize_data_boundary(
        normalized.get("data_boundary"),
        data_source=normalized.get("data_source"),
        rows=normalized.get("record_count"),
        source_path=normalized.get("source_path"),
    )
    return normalized


def _normalize_session(session: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(session)
    task = normalized.get("task")
    if isinstance(task, dict):
        normalized.setdefault("task_id", task.get("task_id") or task.get("id"))
        normalized.setdefault("data_source", task.get("data_source"))
        normalized.setdefault("data_boundary", task.get("data_boundary"))
        normalized.setdefault("source_path", task.get("source_path"))
        normalized.setdefault("record_count", task.get("record_count"))
    normalized["data_boundary"] = _normalize_data_boundary(
        normalized.get("data_boundary"),
        data_source=normalized.get("data_source"),
        rows=normalized.get("record_count"),
        source_path=normalized.get("source_path"),
    )
    if normalized.get("session_id") is not None:
        normalized.setdefault("id", normalized["session_id"])
    if "current_step" in normalized:
        normalized.setdefault("step_count", normalized["current_step"])
        normalized.setdefault("iteration", normalized["current_step"])
    elif "step_count" in normalized:
        normalized.setdefault("current_step", normalized["step_count"])
        normalized.setdefault("iteration", normalized["step_count"])
    guardrails = normalized.get("guardrails") if isinstance(normalized.get("guardrails"), dict) else {}
    normalized.setdefault(
        "config",
        {
            "n_initial": max(
                0,
                len(normalized.get("observed_configs", []))
                - int(normalized.get("current_step", 0) or 0),
            ),
            "n_trials": guardrails.get("max_trials"),
            "use_llm": guardrails.get("llm_enabled"),
            "language": guardrails.get("language"),
        },
    )
    if isinstance(normalized.get("best_result"), dict):
        normalized["best_result"] = _normalize_best_result(normalized["best_result"])
    normalized["candidate_points"] = [
        _normalize_candidate(candidate)
        for candidate in normalized.get("candidate_points", [])
    ]
    normalized.setdefault(
        "observed_history",
        _build_observed_history(
            normalized.get("observed_configs", []),
            normalized.get("observed_fvals", []),
        ),
    )
    normalized["tool_trace"] = [
        _normalize_trace_event(event) for event in normalized.get("tool_trace", [])
    ]
    normalized.setdefault(
        "artifacts",
        ["summary", "observed-configs", "observed-fvals", "candidate-points", "tool-trace"],
    )
    return normalized


def _normalize_best_result(best_result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(best_result)
    if normalized.get("score") is not None:
        normalized.setdefault("objective", normalized["score"])
    config = normalized.get("config")
    if isinstance(config, dict):
        normalized.setdefault("parameters", config)
    normalized.setdefault("target", normalized.get("metric", "PCE_percent"))
    return normalized


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(candidate)
    candidate_id = normalized.get("candidate_id") or normalized.get("id")
    if candidate_id is not None:
        normalized.setdefault("id", candidate_id)
        normalized.setdefault("label", candidate_id)
    if normalized.get("mock_acquisition_score") is not None:
        normalized.setdefault("expected_improvement", normalized["mock_acquisition_score"])
    if normalized.get("score") is None and normalized.get("expected_improvement") is not None:
        normalized["score"] = normalized["expected_improvement"]
    normalized.setdefault("status", "pending")
    normalized.setdefault(
        "parameters",
        {
            key: value
            for key, value in normalized.items()
            if key
            not in {
                "id",
                "label",
                "status",
                "score",
                "expected_improvement",
                "uncertainty",
                "rationale",
                "candidate_id",
                "mock_acquisition_score",
                "acquisition_function",
            }
        },
    )
    return normalized


def _build_observed_history(
    observed_configs: list[dict[str, Any]], observed_fvals: list[float]
) -> list[dict[str, Any]]:
    history = []
    for index, config in enumerate(observed_configs):
        objective = observed_fvals[index] if index < len(observed_fvals) else None
        history.append(
            {
                "iteration": index,
                "candidate_id": config.get("experiment_id") or f"observation_{index + 1}",
                "objective": objective,
                "best": max(observed_fvals[: index + 1]) if objective is not None else None,
                "metrics": {"PCE_percent": objective} if objective is not None else {},
            }
        )
    return history


def _normalize_trace_event(event: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(event)
    step = normalized.get("step")
    detail = normalized.get("detail")
    if step is not None:
        normalized.setdefault("tool", step)
        normalized.setdefault("name", step)
    if detail is not None:
        normalized.setdefault("message", detail)
        normalized.setdefault("output", {"detail": detail})
    normalized.setdefault("status", "success")
    return normalized


def _normalize_data_boundary(
    boundary: Any,
    *,
    data_source: Any = None,
    rows: Any = None,
    source_path: Any = None,
) -> dict[str, Any]:
    if isinstance(boundary, dict):
        normalized = dict(boundary)
    else:
        normalized = {"notes": str(boundary or "Demo data boundary was not declared.")}

    normalized.setdefault("dataset", str(data_source or "unknown"))
    normalized.setdefault("source", str(source_path or data_source or "unknown"))
    normalized.setdefault("last_updated", "demo-runtime")
    normalized.setdefault("constraints", [])
    normalized.setdefault(
        "warnings",
        [
            "Demo-only optimization trace; recommendations are hypotheses, not experimentally validated results.",
        ],
    )
    if rows is not None:
        try:
            normalized.setdefault("rows", int(rows))
        except (TypeError, ValueError):
            pass
    return normalized


def _normalize_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(artifacts)
    if "data_boundary" in normalized:
        normalized["data_boundary"] = _normalize_data_boundary(
            normalized.get("data_boundary"),
            data_source=normalized.get("data_source"),
        )
    if "artifacts" not in normalized:
        artifact_keys = [
            key
            for key in normalized
            if key not in {"session_id", "data_source", "data_boundary"}
        ]
        normalized["artifacts"] = [
            {"name": key, "type": "json", "content": normalized[key]}
            for key in artifact_keys
        ]
    else:
        normalized["artifacts"] = [
            _normalize_artifact_item(item) for item in normalized["artifacts"]
        ]
    return normalized


def _normalize_artifact_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    normalized = dict(item)
    if normalized.get("artifact_name") is not None:
        normalized.setdefault("name", normalized["artifact_name"])
    return normalized


def _get_session_or_404(session_id: str) -> Any:
    try:
        session = session_store.get_session(session_id)
    except KeyError:
        session = None
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' was not found.",
        )
    return session


def _is_real_pvk_session(session: Any) -> bool:
    if not isinstance(session, dict):
        return False
    guardrails = session.get("guardrails")
    return isinstance(guardrails, dict) and guardrails.get("mode") == "real_pvk_llm_bo"


def _message_requests_bo_step(message: str) -> bool:
    normalized = message.lower()
    stripped = normalized.strip()
    if re.search(
        r"^(please\s+|can you\s+|let'?s\s+)?(run|start|execute|advance|perform)\b.{0,30}\b(bo|optimization|step)\b",
        stripped,
    ):
        return True
    if re.search(r"^(please\s+)?next step\s*(please|now)?[.!?]?$", stripped):
        return True

    original = message.strip()
    if re.match(r"^请?下一步(吧|。|！|!|$)", original):
        return True
    if re.match(
        r"^请?(运行|执行|推进|跑|调用|开始|做).{0,20}(bo|BO|优化|下一步|step|步骤)",
        original,
    ):
        return True
    return False


def _handle_real_pvkbo_chat_turn(
    session_id: str, message: str, session: dict[str, Any], language: str
) -> dict[str, Any]:
    ran_step = False
    active_session = session
    if _message_requests_bo_step(message) and session.get("status") != "completed":
        active_session = session_store.run_step(session_id)
        ran_step = True

    observed_fvals = active_session.get("observed_fvals", [])
    best_result = active_session.get("best_result") or {}
    best_score = best_result.get("score")
    phase = "Optimization" if ran_step or active_session.get("current_step", 0) else "Initialization"
    tool_calls = _real_pvk_tool_calls(active_session, include_step_only=ran_step)
    artifacts = {
        "bo_step": {
            "session_id": active_session.get("session_id"),
            "task_id": (active_session.get("task") or {}).get("task_id"),
            "current_step": active_session.get("current_step", 0),
            "best_score": best_score,
            "selected_candidate": (active_session.get("candidate_points") or [None])[0],
            "observed_fvals": observed_fvals,
        },
        "bo_curve": compute_bo_curve(observed_fvals),
        "data_boundary": (active_session.get("task") or {}).get("data_boundary"),
    }
    if language == "en":
        assistant_message = (
            "Real PVKBO agent state updated. "
            if ran_step
            else "Real PVKBO agent session is ready. "
        )
        assistant_message += (
            f"Best eta is {best_score:.4f}; tool trace comes from live LLM_ACQ, "
            "LLM surrogate selection, and workbook black-box lookup."
            if isinstance(best_score, (int, float))
            else "No best score is available yet."
        )
    else:
        assistant_message = (
            "真实 PVKBO Agent 已推进一轮优化："
            if ran_step
            else "真实 PVKBO Agent 会话已就绪："
        )
        assistant_message += (
            f"当前 best eta/PCE 为 {best_score:.4f}。这一步来自 LLM_ACQ 候选生成、"
            "LLM surrogate 选点和 Excel black-box lookup 评估；它是真实 BO 流程，"
            "但仍不是湿实验验证。"
            if isinstance(best_score, (int, float))
            else "当前还没有可用 best score。"
        )

    return {
        "assistant_message": assistant_message,
        "phase": phase,
        "tool_calls": tool_calls,
        "artifacts": artifacts,
    }


def _real_pvk_tool_calls(
    session: dict[str, Any], *, include_step_only: bool
) -> list[dict[str, Any]]:
    trace = session.get("tool_trace", [])
    step_names = {
        "LLM_ACQ.get_candidate_points",
        "LLM_ACQ.generate_candidate_points",
        "LLM_SURROGATE.select_query_point",
        "black_box.evaluate_candidate",
        "PVKBO.update_observations",
    }
    if include_step_only:
        trace = [
            event
            for event in trace
            if (event.get("step") or event.get("name")) in step_names
        ]
    return [
        {
            "name": _normalize_pvk_tool_name(
                event.get("step") or event.get("name") or "unknown_tool"
            ),
            "arguments": {
                "current_step": session.get("current_step", 0),
                "mode": "real_pvk_llm_bo",
            },
            "result": event.get("detail") or event.get("message"),
        }
        for event in trace
    ]


def _normalize_pvk_tool_name(name: str) -> str:
    if name == "LLM_ACQ.generate_candidate_points":
        return "LLM_ACQ.get_candidate_points"
    return name


def _session_value(session: Any, key: str, default: Any = None) -> Any:
    if isinstance(session, dict):
        return session.get(key, default)
    return getattr(session, key, default)


def _session_mapping(session: Any) -> dict[str, Any]:
    if isinstance(session, dict):
        return dict(session)
    if session is None:
        return {}
    try:
        serialized = serialize_session(session)
        if isinstance(serialized, dict):
            return serialized
    except Exception:
        pass
    return {
        "session_id": _session_value(session, "session_id"),
        "id": _session_value(session, "id"),
        "status": _session_value(session, "status"),
        "task_id": _session_value(session, "task_id"),
        "current_step": _session_value(session, "current_step"),
        "step_count": _session_value(session, "step_count"),
        "observed_fvals": _session_value(session, "observed_fvals", []),
        "best_result": _session_value(session, "best_result", {}),
        "candidate_points": _session_value(session, "candidate_points", []),
        "candidates": _session_value(session, "candidates", []),
        "task": _session_value(session, "task", {}),
        "data_boundary": _session_value(session, "data_boundary"),
        "tool_trace": _session_value(session, "tool_trace", []),
    }


def _serialize_chat_response(
    conversation: Any,
    decision: Any,
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "conversation_id": conversation.conversation_id,
        "state": conversation.state,
        "session_id": conversation.session_id,
        "assistant_message": decision.assistant_message,
        "message": {"role": "agent", "content": decision.assistant_message},
        "messages": [{"role": "agent", "content": decision.assistant_message}],
        "intent": decision.intent,
        "action": {"type": decision.action.type, "args": decision.action.args},
        "ui_hints": decision.ui_hints,
        "tool_calls": tool_calls or [],
        "artifacts": artifacts or {},
    }


def _run_agent_bo_step(
    conversation: Any,
    _action_args: dict[str, Any],
    language: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not conversation.session_id:
        # Chat-driven BO is intentionally fixed to the confirmed built-in demo task.
        # Free-form LLM action args must not enlarge cost or switch datasets.
        session = session_store.create_session(
            PvkSessionRequest(
                task_id="band_alignment",
                n_initial=3,
                n_trials=5,
                seed=0,
                use_llm=True,
                language=language,
            )
        )
        session_id = _session_value(session, "session_id") or _session_value(session, "id")
        if not session_id:
            raise ValueError("Created PVKBO session did not include a session_id.")
        conversation.session_id = str(session_id)

    try:
        active_session = session_store.run_step(conversation.session_id)
    except KeyError:
        active_session = None
    if active_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{conversation.session_id}' was not found.",
        )

    active = _session_mapping(active_session)
    session_id = active.get("session_id") or active.get("id") or conversation.session_id
    conversation.session_id = str(session_id)
    conversation.state = "reporting"

    task = active.get("task") if isinstance(active.get("task"), dict) else {}
    observed_fvals = active.get("observed_fvals") or []
    best_result = active.get("best_result") if isinstance(active.get("best_result"), dict) else {}
    candidate_points = active.get("candidate_points") or []
    current_step = active.get("current_step", active.get("step_count", 0))
    artifacts = {
        "bo_step": {
            "session_id": conversation.session_id,
            "task_id": task.get("task_id") or active.get("task_id"),
            "current_step": current_step,
            "best_score": best_result.get("score"),
            "best_result": best_result,
            "selected_candidate": candidate_points[0] if candidate_points else None,
            "observed_fvals": observed_fvals,
        },
        "bo_curve": compute_bo_curve(observed_fvals),
        "data_boundary": task.get("data_boundary") or active.get("data_boundary"),
    }
    return _real_pvk_tool_calls(active, include_step_only=True), artifacts


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
    return success(session_store.list_tasks())


@app.post("/api/v1/chat")
def create_agent_chat_turn(body: AgentChatBody) -> dict[str, Any]:
    conversation = chat_agent_store.get_or_create(body.conversation_id)
    emit_backend_log(
        "chat.request",
        "收到 Agent 对话请求",
        detail={
            "conversation_id": conversation.conversation_id,
            "message": body.message,
        },
    )
    decision = chat_agent_store.planner.plan(conversation, body.message, body.language)
    decision = route_explicit_user_action(conversation, decision, body.message)
    emit_backend_log(
        "chat.decision",
        "LLM 生成 Agent action",
        detail={
            "conversation_id": conversation.conversation_id,
            "intent": decision.intent,
            "action": decision.action.type,
            "state": conversation.state,
        },
    )
    if decision.action.type in {"run_bo_step", "run_demo_bo"} and not can_request_bo_run(conversation):
        if can_accept_demo_run_request(conversation, decision):
            apply_demo_run_consent(conversation)
            emit_backend_log(
                "gate.demo_consent",
                "隐藏 gate 接受内置 demo BO 直达请求",
                detail={"conversation_id": conversation.conversation_id},
            )
        else:
            decision = blocked_run_decision(body.language)
            emit_backend_log(
                "gate.blocked",
                "隐藏 gate 阻止 BO 工具调用",
                detail={
                    "conversation_id": conversation.conversation_id,
                    "action": "run_bo_step",
                },
            )
    previous_state = conversation.state
    previous_session_id = conversation.session_id
    previous_user_confirmed_run = conversation.user_confirmed_run
    apply_allowed_state_update(conversation, decision)
    tool_calls: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    if decision.action.type in {"run_bo_step", "run_demo_bo", "run_next_bo_step"} and conversation.user_confirmed_run:
        try:
            emit_backend_log(
                "bo.start",
                "开始执行 PVKBO step",
                detail={"conversation_id": conversation.conversation_id},
            )
            tool_calls, artifacts = _run_agent_bo_step(
                conversation, decision.action.args, body.language
            )
            decision.assistant_message = chat_agent_store.planner.summarize_bo_result(
                artifacts,
                tool_calls,
                body.language,
            )
            bo_step = artifacts.get("bo_step") if isinstance(artifacts.get("bo_step"), dict) else {}
            emit_backend_log(
                "bo.complete",
                "PVKBO step 完成",
                detail={
                    "conversation_id": conversation.conversation_id,
                    "session_id": conversation.session_id,
                    "best_score": bo_step.get("best_score"),
                    "tool_calls": [call.get("name") for call in tool_calls],
                },
            )
        except RealPvkBoUnavailableError as exc:
            conversation.state = previous_state
            conversation.session_id = previous_session_id
            conversation.user_confirmed_run = previous_user_confirmed_run
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from None
        except HTTPException:
            conversation.state = previous_state
            conversation.session_id = previous_session_id
            conversation.user_confirmed_run = previous_user_confirmed_run
            raise
        except (KeyError, ValueError):
            conversation.state = previous_state
            conversation.session_id = previous_session_id
            conversation.user_confirmed_run = previous_user_confirmed_run
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task '{decision.action.args.get('task_id') or 'band_alignment'}' was not found.",
            ) from None
        except Exception as exc:
            conversation.state = previous_state
            conversation.session_id = previous_session_id
            conversation.user_confirmed_run = previous_user_confirmed_run
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Real PVKBO agent step failed: {exc}",
            ) from None
    conversation.history.append({"role": "user", "content": body.message})
    conversation.history.append(
        {"role": "assistant", "content": decision.assistant_message}
    )
    return success(
        _serialize_chat_response(
            conversation,
            decision,
            tool_calls=tool_calls,
            artifacts=artifacts,
        )
    )


@app.post(
    "/api/v1/sessions",
    status_code=status.HTTP_201_CREATED,
)
def create_session(body: CreatePvkSessionBody) -> dict[str, Any]:
    try:
        session = session_store.create_session(
            PvkSessionRequest(
                task_id=body.task_id,
                n_initial=body.n_initial,
                n_trials=body.n_trials,
                seed=body.seed,
                use_llm=body.use_llm,
                language=body.language,
            )
        )
    except RealPvkBoUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from None
    except (KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{body.task_id}' was not found.",
        ) from None
    return success(serialize_session(session))


@app.get("/api/v1/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    session = _get_session_or_404(session_id)
    return success(serialize_session(session))


@app.post("/api/v1/sessions/{session_id}/steps")
def run_session_step(session_id: str) -> dict[str, Any]:
    existing_session = None
    try:
        existing_session = session_store.get_session(session_id)
    except (AttributeError, KeyError):
        existing_session = None
    try:
        session = session_store.run_step(session_id)
    except KeyError:
        session = None
    except RealPvkBoUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from None
    except Exception as exc:
        if _is_real_pvk_session(existing_session) or session_id.startswith("pvk_real_"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Real PVKBO step failed: {exc}",
            ) from None
        raise
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' was not found.",
        )
    return success(serialize_session(session))


@app.get("/api/v1/sessions/{session_id}/passivation-target")
def get_passivation_target(session_id: str) -> dict[str, Any]:
    session = _get_session_or_404(session_id)
    return success(build_passivation_target(session))


@app.get("/api/v1/sessions/{session_id}/bo-curve")
def get_bo_curve(session_id: str) -> dict[str, Any]:
    session = _get_session_or_404(session_id)
    observed_fvals = _session_value(session, "observed_fvals", [])
    return success(compute_bo_curve(observed_fvals))


@app.post("/api/v1/sessions/{session_id}/chat")
def create_chat_turn(session_id: str, body: ChatTurnBody) -> dict[str, Any]:
    session = _get_session_or_404(session_id)
    if _is_real_pvk_session(session):
        try:
            return success(
                _handle_real_pvkbo_chat_turn(
                    session_id, body.message, session, body.language
                )
            )
        except RealPvkBoUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from None
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Real PVKBO chat step failed: {exc}",
            ) from None
    return success(handle_mvp_chat_turn(body.message, session, body.language))


@app.get("/api/v1/sessions/{session_id}/artifacts")
def list_session_artifacts(session_id: str) -> dict[str, Any]:
    try:
        artifacts = session_store.list_artifacts(session_id)
    except KeyError:
        artifacts = None
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' was not found.",
        )
    return success(artifacts)


@app.post(
    "/api/v1/agent-runs",
    status_code=status.HTTP_201_CREATED,
)
def create_agent_run(body: CreateAgentRunBody) -> dict[str, Any]:
    run = store.create_run(
        AgentRunRequest(
            task_text=body.task_text,
            recommendation_count=body.recommendation_count,
            language=body.language,
            use_llm=body.use_llm,
        )
    )
    return success(serialize_run(run))


@app.get("/api/v1/agent-runs/{run_id}")
def get_agent_run(run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent run '{run_id}' was not found.",
        )
    return success(serialize_run(run))


@app.get("/api/v1/agent-runs/{run_id}/artifacts/{artifact_name}")
def get_agent_run_artifact(run_id: str, artifact_name: str) -> dict[str, Any]:
    artifact = store.get_artifact(run_id, artifact_name)
    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact '{artifact_name}' for run '{run_id}' was not found.",
        )
    return success(artifact)


@app.post(
    "/api/v1/benchmark",
    status_code=status.HTTP_202_ACCEPTED,
)
def create_benchmark_run(body: CreateBenchmarkBody) -> dict[str, Any]:
    """Submit a benchmark run. Returns immediately with run metadata;
    execution happens synchronously (for now)."""

    # Validate output_dir: reject path traversal attempts
    output_path = Path(body.output_dir)
    if output_path.is_absolute() or ".." in str(output_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="output_dir must be a relative path without '..' traversal.",
        )

    emit_backend_log(
        "benchmark.request",
        f"收到 benchmark 请求: {body.task_id}",
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
            f"Benchmark 完成: {body.task_id}",
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
        emit_backend_log(
            "benchmark.error",
            "Benchmark 失败: 内部错误",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Benchmark run failed due to an internal error. Check backend logs for details.",
        ) from None
