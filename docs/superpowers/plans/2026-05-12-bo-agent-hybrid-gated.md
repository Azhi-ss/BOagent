# BO Agent Hybrid Gated Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current template-based chat flow with a real LLM-driven Chat Planner that can only run PVKBO after explicit demo-data, goal, and run confirmations.

**Architecture:** Add an independent chat conversation layer before PVKBO sessions. The Chat Planner LLM emits structured decisions; a backend policy gate validates actions; the tool router updates conversation state or calls the existing PVKBO runtime only when allowed.

**Tech Stack:** FastAPI, Pydantic, DeepSeek OpenAI-compatible chat API via existing `llm_client.py`, React/Vite/TypeScript, pytest, Playwright.

---

## File Structure

- Create `chat_agent.py`
  - Owns chat states, structured action schema, LLM prompt construction, JSON parsing, policy gate, and `ChatAgentStore`.
- Modify `llm_client.py`
  - Add optional `extra_body` support for DeepSeek thinking-disabled chat responses.
- Modify `api.py`
  - Add `/api/v1/chat` conversation endpoint.
  - Keep existing session endpoints for compatibility.
  - Route allowed `run_bo_step` actions through existing `session_store`.
- Modify `tests/test_api.py`
  - Add API-level tests for no-auto-session greeting, demo selection, confirmations, gate rejection, and run flow.
- Modify `frontend/src/types.ts`
  - Add `ChatAgentState`, `ChatAgentResponse`, `ChatAgentAction`, and UI hint types.
- Modify `frontend/src/lib/api.ts`
  - Add `sendAgentChatMessage`.
- Modify `frontend/src/App.tsx`
  - Stop defaulting input to “run real PVKBO”.
  - Stop auto-creating PVKBO session in `handleSubmit`.
  - Use `/api/v1/chat` until a PVKBO session actually exists.
  - Render state-driven buttons/chips.
- Optional later: create permanent Playwright spec after the flow stabilizes. First implementation can keep E2E as a manual verification command.

Do not create a git commit during execution unless the user explicitly asks for one.

---

## Task 1: Add Chat Agent State, Schema, and Safe Planner

**Files:**
- Create: `chat_agent.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write failing tests for planner-safe greeting**

Add this test near the existing chat/session tests in `tests/test_api.py`:

```python
def test_agent_chat_greeting_uses_llm_without_creating_pvkbo_session(monkeypatch):
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            assert messages
            assert extra_body == {"thinking": {"type": "disabled"}}
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"greeting","next_state":"awaiting_data_choice","action":{"type":"none","args":{}},"assistant_message":"你好，我可以帮你做 PVK 贝叶斯优化。请选择内置 demo 数据或之后上传文件。","ui_hints":["show_demo_button","show_upload_disabled"]}',
                usage={"total_tokens": 42},
            )

    original_chat_store = api.chat_agent_store
    api.chat_agent_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "你好", "language": "zh"},
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["conversation_id"].startswith("chat_")
    assert payload["state"] == "awaiting_data_choice"
    assert payload["session_id"] is None
    assert payload["assistant_message"].startswith("你好")
    assert payload["ui_hints"] == ["show_demo_button", "show_upload_disabled"]
    assert payload["tool_calls"] == []
```

- [ ] **Step 2: Run test and confirm it fails**

Run:

```bash
python -m pytest -q tests/test_api.py::test_agent_chat_greeting_uses_llm_without_creating_pvkbo_session
```

Expected: FAIL because `/api/v1/chat`, `ChatAgentStore`, and `chat_agent.py` do not exist yet.

- [ ] **Step 3: Implement `chat_agent.py` models and planner**

Create `chat_agent.py` with:

```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from llm_client import DeepSeekClient, LlmCallResult

ChatState = Literal[
    "idle",
    "awaiting_data_choice",
    "awaiting_demo_confirm",
    "awaiting_goal_confirm",
    "ready_to_run",
    "running_bo",
    "reporting",
]

ActionType = Literal[
    "none",
    "select_demo_data",
    "confirm_demo_data",
    "confirm_goal",
    "create_bo_session",
    "run_bo_step",
    "explain_result",
]

KNOWN_STATES = {
    "idle",
    "awaiting_data_choice",
    "awaiting_demo_confirm",
    "awaiting_goal_confirm",
    "ready_to_run",
    "running_bo",
    "reporting",
}

KNOWN_ACTIONS = {
    "none",
    "select_demo_data",
    "confirm_demo_data",
    "confirm_goal",
    "create_bo_session",
    "run_bo_step",
    "explain_result",
}


@dataclass
class ChatAgentAction:
    type: ActionType
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatAgentDecision:
    intent: str
    next_state: ChatState
    action: ChatAgentAction
    assistant_message: str
    ui_hints: list[str] = field(default_factory=list)


@dataclass
class ChatAgentConversation:
    conversation_id: str
    state: ChatState = "idle"
    data_source: str | None = None
    data_source_confirmed: bool = False
    demo_disclaimer_confirmed: bool = False
    goal_confirmed: bool = False
    user_confirmed_run: bool = False
    session_id: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def state_payload(self) -> dict[str, Any]:
        allowed_actions = allowed_actions_for(self)
        return {
            "state": self.state,
            "data_source": self.data_source,
            "data_source_confirmed": self.data_source_confirmed,
            "demo_disclaimer_confirmed": self.demo_disclaimer_confirmed,
            "goal_confirmed": self.goal_confirmed,
            "user_confirmed_run": self.user_confirmed_run,
            "session_id": self.session_id,
            "allowed_actions": sorted(allowed_actions),
            "forbidden_actions": sorted(KNOWN_ACTIONS - allowed_actions),
        }


def allowed_actions_for(conversation: ChatAgentConversation) -> set[str]:
    if conversation.state in {"idle", "awaiting_data_choice"}:
        return {"none", "select_demo_data"}
    if conversation.state == "awaiting_demo_confirm":
        return {"none", "confirm_demo_data"}
    if conversation.state == "awaiting_goal_confirm":
        return {"none", "confirm_goal"}
    if conversation.state == "ready_to_run":
        return {"none", "run_bo_step", "explain_result"}
    if conversation.state in {"running_bo", "reporting"}:
        return {"none", "run_bo_step", "explain_result"}
    return {"none"}


def build_chat_planner_messages(
    conversation: ChatAgentConversation,
    message: str,
    language: str,
) -> list[dict[str, str]]:
    language_instruction = "Respond in Simplified Chinese." if language == "zh" else "Respond in English."
    system_prompt = (
        "You are a PVK Bayesian Optimization Agent. "
        "You help the user prepare and run a BO task through conversation. "
        "Every turn must return only valid JSON with intent, next_state, action, assistant_message, and ui_hints. "
        "Do not run BO unless backend state says it is allowed. "
        "If the user has not uploaded data, never imply user data exists. "
        "If demo data is used, explicitly call it built-in demo/reference data. "
        "Workbook lookup is reference evaluation, not wet-lab validation. "
        "Ask for missing prerequisites before proposing BO execution. "
        "Keep replies concise and action-oriented."
    )
    user_prompt = {
        "language_instruction": language_instruction,
        "backend_state": conversation.state_payload(),
        "user_message": message,
        "output_schema": {
            "intent": "string",
            "next_state": "known state string",
            "action": {"type": "known action string", "args": {}},
            "assistant_message": "string",
            "ui_hints": ["string"],
        },
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]


def parse_chat_decision(content: str) -> ChatAgentDecision:
    raw = json.loads(content)
    next_state = raw.get("next_state")
    action_raw = raw.get("action") or {}
    action_type = action_raw.get("type")
    if next_state not in KNOWN_STATES:
        raise ValueError(f"Unknown next_state: {next_state}")
    if action_type not in KNOWN_ACTIONS:
        raise ValueError(f"Unknown action type: {action_type}")
    assistant_message = str(raw.get("assistant_message") or "").strip()
    if not assistant_message:
        raise ValueError("assistant_message is required")
    ui_hints = raw.get("ui_hints") or []
    if not isinstance(ui_hints, list):
        ui_hints = []
    return ChatAgentDecision(
        intent=str(raw.get("intent") or "other"),
        next_state=next_state,
        action=ChatAgentAction(type=action_type, args=dict(action_raw.get("args") or {})),
        assistant_message=assistant_message,
        ui_hints=[str(item) for item in ui_hints],
    )


def fallback_decision(language: str, reason: str) -> ChatAgentDecision:
    message = (
        "我可以继续帮你规划 PVK 贝叶斯优化，但现在不会运行 BO。请先选择使用内置 demo 数据，或之后上传你的数据。"
        if language == "zh"
        else "I can help plan PVK Bayesian optimization, but I will not run BO yet. Please choose built-in demo data or upload data later."
    )
    return ChatAgentDecision(
        intent=f"fallback:{reason}",
        next_state="awaiting_data_choice",
        action=ChatAgentAction(type="none"),
        assistant_message=message,
        ui_hints=["show_demo_button", "show_upload_disabled"],
    )


class ChatAgentPlanner:
    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self.client = client or DeepSeekClient.from_env()

    def plan(
        self,
        conversation: ChatAgentConversation,
        message: str,
        language: str,
    ) -> ChatAgentDecision:
        result = self.client.chat(
            build_chat_planner_messages(conversation, message, language),
            max_tokens=900,
            extra_body={"thinking": {"type": "disabled"}},
        )
        if result.status != "success":
            return fallback_decision(language, result.error or result.status)
        try:
            return parse_chat_decision(result.content)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return fallback_decision(language, type(exc).__name__)


class ChatAgentStore:
    def __init__(self, planner_client: Any | None = None) -> None:
        self.planner = ChatAgentPlanner(client=planner_client)
        self._conversations: dict[str, ChatAgentConversation] = {}

    def get_or_create(self, conversation_id: str | None) -> ChatAgentConversation:
        if conversation_id and conversation_id in self._conversations:
            return self._conversations[conversation_id]
        conversation = ChatAgentConversation(conversation_id=f"chat_{uuid4().hex[:12]}")
        self._conversations[conversation.conversation_id] = conversation
        return conversation
```

- [ ] **Step 4: Extend `llm_client.py` for `extra_body`**

Change `DeepSeekClient.chat` signature and JSON payload:

```python
def chat(
    self,
    messages: list[dict[str, str]],
    max_tokens: int = 512,
    extra_body: dict[str, Any] | None = None,
) -> LlmCallResult:
```

Inside the request:

```python
payload = {
    "model": self.model,
    "messages": messages,
    "max_tokens": max_tokens,
    "stream": False,
}
if extra_body:
    payload.update(extra_body)
response = requests.post(
    f"{self.base_url}/chat/completions",
    headers={
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
    },
    json=payload,
    timeout=self.timeout_s,
)
```

- [ ] **Step 5: Add `/api/v1/chat` with no tool routing yet**

In `api.py`, import and instantiate:

```python
from chat_agent import ChatAgentStore

chat_agent_store = ChatAgentStore()
```

Add request model:

```python
class AgentChatBody(BaseModel):
    conversation_id: str | None = None
    message: str = Field(..., min_length=1)
    language: str = "zh"
    history: list[ChatMessage] = Field(default_factory=list)
```

Add endpoint:

```python
@app.post("/api/v1/chat")
def create_agent_chat_turn(body: AgentChatBody) -> dict[str, Any]:
    conversation = chat_agent_store.get_or_create(body.conversation_id)
    decision = chat_agent_store.planner.plan(conversation, body.message, body.language)
    conversation.state = decision.next_state
    conversation.history.append({"role": "user", "content": body.message})
    conversation.history.append({"role": "assistant", "content": decision.assistant_message})
    return success(
        {
            "conversation_id": conversation.conversation_id,
            "state": conversation.state,
            "session_id": conversation.session_id,
            "assistant_message": decision.assistant_message,
            "message": {"role": "agent", "content": decision.assistant_message},
            "messages": [{"role": "agent", "content": decision.assistant_message}],
            "intent": decision.intent,
            "action": {"type": decision.action.type, "args": decision.action.args},
            "ui_hints": decision.ui_hints,
            "tool_calls": [],
            "artifacts": {},
        }
    )
```

- [ ] **Step 6: Run test**

Run:

```bash
python -m pytest -q tests/test_api.py::test_agent_chat_greeting_uses_llm_without_creating_pvkbo_session
```

Expected: PASS.

- [ ] **Step 7: Checkpoint**

Run:

```bash
python -m pytest -q tests/test_api.py::test_agent_chat_greeting_uses_llm_without_creating_pvkbo_session
git diff -- chat_agent.py llm_client.py api.py tests/test_api.py
```

Do not commit unless the user explicitly asks.

---

## Task 2: Implement Policy Gate and State Transitions

**Files:**
- Modify: `chat_agent.py`
- Modify: `api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Add tests for demo selection and confirmations**

Add tests:

```python
def test_agent_chat_demo_selection_requires_explicit_demo_confirmation():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"select_demo_data","next_state":"awaiting_demo_confirm","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"我会使用内置 PVK demo 数据。它不是你的上传数据。请确认是否继续。","ui_hints":["show_confirm_demo_button"]}',
                usage={},
            )

    original_chat_store = api.chat_agent_store
    api.chat_agent_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    try:
        response = client.post("/api/v1/chat", json={"message": "用内置 demo", "language": "zh"})
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_demo_confirm"
    assert payload["session_id"] is None
```

Use this pattern for the next tests, but assert through response payload rather than store internals:

```python
def test_agent_chat_rejects_run_before_confirmations():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"request_run","next_state":"running_bo","action":{"type":"run_bo_step","args":{}},"assistant_message":"我将开始运行 BO。","ui_hints":[]}',
                usage={},
            )

    original_chat_store = api.chat_agent_store
    api.chat_agent_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    try:
        response = client.post("/api/v1/chat", json={"message": "开始优化", "language": "zh"})
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_data_choice"
    assert payload["session_id"] is None
    assert payload["action"]["type"] == "none"
    assert "不会运行 BO" in payload["assistant_message"]
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
python -m pytest -q tests/test_api.py::test_agent_chat_demo_selection_requires_explicit_demo_confirmation tests/test_api.py::test_agent_chat_rejects_run_before_confirmations
```

Expected: FAIL until policy gate is implemented.

- [ ] **Step 3: Add gate application helpers**

In `chat_agent.py`, add:

```python
def can_request_bo_run(conversation: ChatAgentConversation) -> bool:
    return (
        conversation.data_source == "demo_pvk"
        and conversation.data_source_confirmed
        and conversation.demo_disclaimer_confirmed
        and conversation.goal_confirmed
    )


def blocked_run_decision(language: str) -> ChatAgentDecision:
    message = (
        "现在不会运行 BO。请先明确选择内置 PVK demo 数据、确认它不是你的上传数据，并确认默认优化目标。"
        if language == "zh"
        else "I will not run BO yet. Please select built-in PVK demo data, confirm it is not uploaded user data, and confirm the default optimization goal first."
    )
    return ChatAgentDecision(
        intent="blocked:run_bo_step",
        next_state="awaiting_data_choice",
        action=ChatAgentAction(type="none"),
        assistant_message=message,
        ui_hints=["show_demo_button", "show_upload_disabled"],
    )


def apply_allowed_state_update(
    conversation: ChatAgentConversation,
    decision: ChatAgentDecision,
) -> ChatAgentDecision:
    action = decision.action.type
    if action == "select_demo_data":
        conversation.data_source = "demo_pvk"
        conversation.data_source_confirmed = True
        conversation.state = "awaiting_demo_confirm"
        decision.next_state = "awaiting_demo_confirm"
    elif action == "confirm_demo_data":
        conversation.demo_disclaimer_confirmed = True
        conversation.state = "awaiting_goal_confirm"
        decision.next_state = "awaiting_goal_confirm"
    elif action == "confirm_goal":
        conversation.goal_confirmed = True
        conversation.state = "ready_to_run"
        decision.next_state = "ready_to_run"
    elif action == "run_bo_step":
        conversation.user_confirmed_run = True
        conversation.state = "running_bo"
        decision.next_state = "running_bo"
    else:
        conversation.state = decision.next_state
    return decision
```

- [ ] **Step 4: Use gate in endpoint**

In `api.py`, after planning:

```python
from chat_agent import (
    ChatAgentStore,
    apply_allowed_state_update,
    blocked_run_decision,
    can_request_bo_run,
)
```

Then:

```python
decision = chat_agent_store.planner.plan(conversation, body.message, body.language)
if decision.action.type == "run_bo_step" and not can_request_bo_run(conversation):
    decision = blocked_run_decision(body.language)
else:
    decision = apply_allowed_state_update(conversation, decision)
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest -q tests/test_api.py::test_agent_chat_demo_selection_requires_explicit_demo_confirmation tests/test_api.py::test_agent_chat_rejects_run_before_confirmations
```

Expected: PASS.

---

## Task 3: Route Allowed BO Actions to Existing PVKBO Runtime

**Files:**
- Modify: `api.py`
- Modify: `chat_agent.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write test for confirmed run flow**

Add a fake planner that emits four decisions in sequence:

```python
def test_agent_chat_confirmed_demo_flow_runs_one_pvkbo_step():
    class FakePlannerClient:
        def __init__(self):
            self.contents = [
                '{"intent":"select_demo_data","next_state":"awaiting_demo_confirm","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"使用内置 demo。请确认。","ui_hints":["show_confirm_demo_button"]}',
                '{"intent":"confirm_demo_data","next_state":"awaiting_goal_confirm","action":{"type":"confirm_demo_data","args":{}},"assistant_message":"已确认 demo 数据。请确认目标。","ui_hints":["show_confirm_goal_button"]}',
                '{"intent":"confirm_goal","next_state":"ready_to_run","action":{"type":"confirm_goal","args":{}},"assistant_message":"默认目标是 band_alignment / eta maximize。可以开始第一轮。","ui_hints":["show_run_button"]}',
                '{"intent":"request_run","next_state":"running_bo","action":{"type":"run_bo_step","args":{}},"assistant_message":"开始运行第一轮 BO。","ui_hints":[]}',
            ]

        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content=self.contents.pop(0),
                usage={},
            )

    class FakeSessionStore:
        def create_session(self, request):
            assert request.task_id == "band_alignment"
            return {
                "session_id": "pvk_real_agent",
                "status": "running",
                "current_step": 0,
                "observed_fvals": [0.0, 0.002],
                "best_result": {"score": 0.002},
                "candidate_points": [],
                "task": {"task_id": "band_alignment", "data_boundary": {"notes": "demo"}},
                "tool_trace": [{"step": "PVKBO.initialize"}],
                "guardrails": {"mode": "real_pvk_llm_bo"},
            }

        def run_step(self, session_id):
            assert session_id == "pvk_real_agent"
            return {
                "session_id": session_id,
                "status": "completed",
                "current_step": 1,
                "observed_fvals": [0.0, 0.002, 0.003],
                "best_result": {"score": 0.003},
                "candidate_points": [{"candidate_id": "C1"}],
                "task": {"task_id": "band_alignment", "data_boundary": {"notes": "demo"}},
                "tool_trace": [
                    {"step": "LLM_ACQ.get_candidate_points"},
                    {"step": "LLM_SURROGATE.select_query_point"},
                    {"step": "black_box.evaluate_candidate"},
                    {"step": "PVKBO.update_observations"},
                ],
                "guardrails": {"mode": "real_pvk_llm_bo"},
            }

        def get_session(self, session_id):
            return None

    original_chat_store = api.chat_agent_store
    original_session_store = api.session_store
    api.chat_agent_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.session_store = FakeSessionStore()
    try:
        first = client.post("/api/v1/chat", json={"message": "用 demo", "language": "zh"}).json()["data"]
        second = client.post("/api/v1/chat", json={"conversation_id": first["conversation_id"], "message": "确认 demo", "language": "zh"}).json()["data"]
        third = client.post("/api/v1/chat", json={"conversation_id": first["conversation_id"], "message": "确认目标", "language": "zh"}).json()["data"]
        fourth = client.post("/api/v1/chat", json={"conversation_id": first["conversation_id"], "message": "开始运行", "language": "zh"}).json()["data"]
    finally:
        api.chat_agent_store = original_chat_store
        api.session_store = original_session_store

    assert second["state"] == "awaiting_goal_confirm"
    assert third["state"] == "ready_to_run"
    assert fourth["state"] == "reporting"
    assert fourth["session_id"] == "pvk_real_agent"
    assert [call["name"] for call in fourth["tool_calls"]] == [
        "LLM_ACQ.get_candidate_points",
        "LLM_SURROGATE.select_query_point",
        "black_box.evaluate_candidate",
        "PVKBO.update_observations",
    ]
    assert fourth["artifacts"]["bo_step"]["best_score"] == 0.003
```

- [ ] **Step 2: Run test and confirm it fails**

Run:

```bash
python -m pytest -q tests/test_api.py::test_agent_chat_confirmed_demo_flow_runs_one_pvkbo_step
```

Expected: FAIL because tool routing is not implemented yet.

- [ ] **Step 3: Add tool routing in `api.py`**

Create helper:

```python
def _serialize_chat_response(
    conversation,
    decision,
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
```

Create routing helper:

```python
def _run_agent_bo_step(conversation, language: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not conversation.session_id:
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
        conversation.session_id = session["session_id"] if isinstance(session, dict) else session.session_id
    active_session = session_store.run_step(conversation.session_id)
    if active_session is None:
        raise HTTPException(status_code=404, detail="PVKBO session not found")
    conversation.state = "reporting"
    observed_fvals = active_session.get("observed_fvals", [])
    best_result = active_session.get("best_result") or {}
    artifacts = {
        "bo_step": {
            "session_id": active_session.get("session_id"),
            "task_id": (active_session.get("task") or {}).get("task_id"),
            "current_step": active_session.get("current_step", 0),
            "best_score": best_result.get("score"),
            "selected_candidate": (active_session.get("candidate_points") or [None])[0],
            "observed_fvals": observed_fvals,
        },
        "bo_curve": compute_bo_curve(observed_fvals),
        "data_boundary": (active_session.get("task") or {}).get("data_boundary"),
    }
    return _real_pvk_tool_calls(active_session, include_step_only=True), artifacts
```

In endpoint:

```python
tool_calls: list[dict[str, Any]] = []
artifacts: dict[str, Any] = {}
if decision.action.type == "run_bo_step" and conversation.user_confirmed_run:
    tool_calls, artifacts = _run_agent_bo_step(conversation, body.language)
return success(_serialize_chat_response(conversation, decision, tool_calls=tool_calls, artifacts=artifacts))
```

- [ ] **Step 4: Run test**

Run:

```bash
python -m pytest -q tests/test_api.py::test_agent_chat_confirmed_demo_flow_runs_one_pvkbo_step
```

Expected: PASS.

---

## Task 4: Frontend Uses Chat Conversation Before PVKBO Session

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Add frontend types**

In `frontend/src/types.ts`, add:

```ts
export type ChatAgentState =
  | "idle"
  | "awaiting_data_choice"
  | "awaiting_demo_confirm"
  | "awaiting_goal_confirm"
  | "ready_to_run"
  | "running_bo"
  | "reporting";

export interface ChatAgentAction {
  type: string;
  args?: JsonMap;
}

export interface ChatAgentResponse extends ChatResponse {
  conversation_id: string;
  state: ChatAgentState;
  session_id?: string | null;
  intent?: string;
  action?: ChatAgentAction;
  ui_hints?: string[];
}
```

- [ ] **Step 2: Add API function**

In `frontend/src/lib/api.ts`, import `ChatAgentResponse`, then add:

```ts
export function sendAgentChatMessage(payload: ChatRequest & { conversation_id?: string | null }) {
  return request<ChatAgentResponse>("/api/v1/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
```

- [ ] **Step 3: Update initial frontend state**

In `frontend/src/App.tsx`:

```ts
const DEFAULT_PROMPT = "";
```

Change initial state:

```ts
const [conversationId, setConversationId] = useState<string | null>(null);
const [agentState, setAgentState] = useState<ChatAgentState>("idle");
const [messages, setMessages] = useState<ChatMessage[]>([
  {
    role: "agent",
    content:
      "你好，我可以帮你规划 PVK 贝叶斯优化。第一版支持内置 PVK demo 数据；上传你自己的文件会在下一版开放。你可以先选择是否用内置 demo 数据演示。",
  },
]);
const [input, setInput] = useState("");
```

- [ ] **Step 4: Replace `ensureSession` usage in normal submit**

In `handleSubmit`, remove:

```ts
const activeSessionId = await ensureSession();
const response = await sendChatMessage(activeSessionId, ...)
await syncAgentOutputs(activeSessionId);
```

Replace with:

```ts
const response = await sendAgentChatMessage({
  conversation_id: conversationId,
  message,
  history: normalizeChatHistory(nextHistory.slice(1)),
});
setConversationId(response.conversation_id);
setAgentState(response.state);
setMessages([...nextHistory, ...normalizeChatResponse(response)]);
setPhase(response.phase || response.state);
setToolCalls(response.tool_calls || []);
applyResponseArtifacts(response);
if (response.session_id) {
  setSession((current) => ({
    ...(current || {}),
    session_id: response.session_id || undefined,
    status: response.state === "reporting" ? "completed" : "running",
  }));
  await syncAgentOutputs(response.session_id);
}
```

- [ ] **Step 5: Render state chips**

Add a small chip area near the chat form:

```tsx
<div className="flex flex-wrap gap-2">
  {agentState === "awaiting_data_choice" && (
    <button type="button" onClick={() => onInputChange("使用内置 PVK demo 数据")}>
      使用内置 PVK demo 数据
    </button>
  )}
  {agentState === "awaiting_demo_confirm" && (
    <button type="button" onClick={() => onInputChange("我确认这是内置 demo 数据，不是我的上传数据")}>
      确认 demo 数据
    </button>
  )}
  {agentState === "awaiting_goal_confirm" && (
    <button type="button" onClick={() => onInputChange("确认默认目标：band_alignment，最大化 eta/PCE")}>
      确认目标
    </button>
  )}
  {agentState === "ready_to_run" && (
    <button type="button" onClick={() => onInputChange("开始运行第一轮 BO")}>
      开始第一轮 BO
    </button>
  )}
</div>
```

Use the existing button style classes from `ChatPanel` so the UI remains consistent.

- [ ] **Step 6: Disable old run button until session exists**

The “让 Agent 运行下一步 BO” button should be disabled unless `sessionId` exists and `agentState` is `ready_to_run` or `reporting`.

---

## Task 5: Verification and E2E

**Files:**
- Optional create: `frontend/tests/e2e/hybrid-gated-agent.spec.ts`
- Modify only if permanent E2E is desired: `frontend/package.json`

- [ ] **Step 1: Run backend tests**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run frontend build**

Run:

```bash
cd frontend && npm run build
```

Expected: TypeScript and Vite build pass.

- [ ] **Step 3: Run manual API smoke**

Run:

```bash
python - <<'PY'
from urllib.request import Request, urlopen
import json

base = "http://127.0.0.1:8010"
req = Request(
    base + "/api/v1/chat",
    data=json.dumps({"message": "你好", "language": "zh"}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(req, timeout=20) as response:
    payload = json.loads(response.read().decode())["data"]
print(response.status, payload["state"], payload["session_id"], payload["assistant_message"][:40])
PY
```

Expected:

```text
200 awaiting_data_choice None ...
```

- [ ] **Step 4: Run browser E2E path**

Use Playwright to verify:

```text
open page
send "你好"
assert no PVKBO tool trace appears
click/use "使用内置 PVK demo 数据"
confirm demo data
confirm default goal
start first BO step
assert LLM_ACQ / LLM_SURROGATE / black_box / PVKBO.update_observations appear
assert no API 4xx or 5xx responses
```

If converting this into permanent test, place it in:

```text
frontend/tests/e2e/hybrid-gated-agent.spec.ts
```

and run:

```bash
cd frontend && npx playwright test tests/e2e/hybrid-gated-agent.spec.ts --browser=chromium --reporter=list --timeout=300000
```

- [ ] **Step 5: Final code review**

Run a code-review subagent focused on:

```text
chat_agent.py
api.py
llm_client.py
frontend/src/App.tsx
frontend/src/lib/api.ts
frontend/src/types.ts
tests/test_api.py
```

Review goals:

- no BO session is created from greeting
- LLM action cannot bypass policy gate
- demo data is always labelled as built-in reference data
- errors do not fall back to fake BO results
- frontend cannot trigger run before `ready_to_run`

---

## Self-Review

Spec coverage:

- Real LLM chat layer: Task 1.
- Structured action JSON: Task 1.
- Backend policy gate: Task 2.
- Tool router and PVKBO execution: Task 3.
- Frontend no default run prompt and state chips: Task 4.
- Testing and E2E: Task 5.

Placeholder scan:

- No `TBD`, `TODO`, or unspecified implementation steps remain.

Type consistency:

- Backend state names match frontend `ChatAgentState`.
- Action names match the spec.
- Response fields match `ChatAgentResponse`.

Scope:

- Real upload support remains out of scope as agreed.
- Existing PVKBO runtime remains mostly unchanged.
