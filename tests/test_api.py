import pytest
from fastapi.testclient import TestClient

import api
from agent_runtime import AgentRunStore
from api import app
from llm_client import DeepSeekClient, LlmCallResult
from pvk_llm_bo_runtime import RealPvkBoRuntime


client = TestClient(app)


def test_health_endpoint_reports_api_status():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "ok"


def test_backend_logs_page_and_recent_events_are_available():
    api.backend_log.clear()
    api.emit_backend_log("test.event", "测试日志事件", detail={"step": 1})

    page_response = client.get("/logs")
    events_response = client.get("/api/v1/logs")

    assert page_response.status_code == 200
    assert "text/html" in page_response.headers["content-type"]
    assert "BOagent Live Logs" in page_response.text
    assert "/api/v1/logs/stream" in page_response.text
    assert events_response.status_code == 200
    events = events_response.json()["data"]["events"]
    assert events[-1]["event"] == "test.event"
    assert events[-1]["message"] == "测试日志事件"
    assert events[-1]["detail"] == {"step": 1}


def test_create_agent_run_returns_claw_style_run_envelope():
    response = client.post(
        "/api/v1/agent-runs",
        json={
            "task_text": "优化钙钛矿钝化配方，提高 PCE",
            "recommendation_count": 3,
            "language": "zh",
        },
    )

    assert response.status_code == 201
    payload = response.json()["data"]
    assert payload["run_id"].startswith("run_")
    assert payload["status"] == "completed"
    assert len(payload["stage_results"]) == 6
    assert payload["data_summary"]["total_records"] > 0
    assert len(payload["recommendations"]) == 3
    assert payload["guardrails"]["mode"] == "demo"
    assert payload["guardrails"]["llm_enabled"] is False
    assert payload["llm_notes"] is None


def test_get_agent_run_and_artifact_by_id():
    create_response = client.post(
        "/api/v1/agent-runs",
        json={
            "task_text": "Generate a next experiment batch",
            "recommendation_count": 5,
            "language": "en",
        },
    )
    run_id = create_response.json()["data"]["run_id"]

    run_response = client.get(f"/api/v1/agent-runs/{run_id}")
    artifact_response = client.get(
        f"/api/v1/agent-runs/{run_id}/artifacts/data-summary"
    )

    assert run_response.status_code == 200
    assert run_response.json()["data"]["run_id"] == run_id
    assert artifact_response.status_code == 200
    assert artifact_response.json()["data"]["artifact_name"] == "data-summary"
    assert artifact_response.json()["data"]["content"]["best_pce"] > 0


def test_missing_agent_run_returns_404_error_envelope():
    response = client.get("/api/v1/agent-runs/run_missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_create_agent_run_can_attach_mocked_deepseek_notes():
    class FakeDeepSeekClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            assert messages
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content="LLM note generated",
                usage={"total_tokens": 12},
            )

    original_store = api.store
    api.store = AgentRunStore(llm_client=FakeDeepSeekClient())
    try:
        response = client.post(
            "/api/v1/agent-runs",
            json={
                "task_text": "请生成带 DeepSeek 说明的推荐",
                "recommendation_count": 3,
                "language": "zh",
                "use_llm": True,
            },
        )
    finally:
        api.store = original_store

    assert response.status_code == 201
    payload = response.json()["data"]
    assert payload["guardrails"]["llm_enabled"] is True
    assert payload["llm_notes"]["provider"] == "deepseek"
    assert payload["llm_notes"]["content"] == "LLM note generated"
    assert "llm-notes" in payload["artifacts"]


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
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
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


def test_agent_chat_prompt_hides_internal_state_machine_names():
    captured_messages = []

    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            captured_messages.extend(messages)
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"greeting","action":{"type":"none","args":{}},"assistant_message":"可以，我先确认数据来源。","ui_hints":["show_demo_button"]}',
                usage={},
            )

    original_chat_store = api.chat_agent_store
    api.chat_agent_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    try:
        response = client.post("/api/v1/chat", json={"message": "你好", "language": "zh"})
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_data_choice"
    prompt_text = "\n".join(message["content"] for message in captured_messages)
    assert "awaiting_" not in prompt_text
    assert "ready_to_run" not in prompt_text
    assert "running_bo" not in prompt_text
    assert "forbidden_actions" not in prompt_text


def test_agent_chat_idle_greeting_is_guided_to_data_choice():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"greeting","next_state":"idle","action":{"type":"none","args":{}},"assistant_message":"你好！欢迎使用 PVK 贝叶斯优化助手。请告诉我你的优化目标是什么？","ui_hints":[]}',
                usage={},
            )

    original_chat_store = api.chat_agent_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "你好", "language": "zh"},
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_data_choice"
    assert payload["session_id"] is None
    assert payload["action"]["type"] == "none"
    assert payload["assistant_message"] == "你好！欢迎使用 PVK 贝叶斯优化助手。请告诉我你的优化目标是什么？"
    assert "show_demo_button" in payload["ui_hints"]


def test_agent_chat_rejects_unsafe_planner_state_from_idle():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            assert messages
            assert extra_body == {"thinking": {"type": "disabled"}}
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"request_run","next_state":"running_bo","action":{"type":"run_bo_step","args":{}},"assistant_message":"我将开始运行 BO。","ui_hints":[]}',
                usage={"total_tokens": 42},
            )

    original_chat_store = api.chat_agent_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "开始运行", "language": "zh"},
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_data_choice"
    assert payload["session_id"] is None
    assert payload["action"]["type"] == "none"
    assert payload["tool_calls"] == []


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
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
    try:
        response = client.post("/api/v1/chat", json={"message": "用内置 demo", "language": "zh"})
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_demo_confirm"
    assert payload["session_id"] is None
    assert payload["action"]["type"] == "select_demo_data"
    conversation = test_chat_store.get_or_create(payload["conversation_id"])
    assert conversation.data_source == "demo_pvk"
    assert conversation.data_source_confirmed is True
    assert conversation.demo_disclaimer_confirmed is False
    assert conversation.goal_confirmed is False


def test_agent_chat_explicit_demo_button_is_still_llm_authored():
    class FakePlannerClient:
        def __init__(self):
            self.calls = 0
            self.messages = []

        def chat(self, messages, max_tokens=512, extra_body=None):
            self.calls += 1
            self.messages.extend(messages)
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"select_demo_data","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"可以。我会用内置 PVK reference 数据做演示，并说明它不是上传数据或湿实验。","ui_hints":["show_run_button"]}',
                usage={},
            )

    original_chat_store = api.chat_agent_store
    fake_client = FakePlannerClient()
    test_chat_store = api.ChatAgentStore(planner_client=fake_client)
    api.chat_agent_store = test_chat_store
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "使用内置 PVK demo 数据", "language": "zh"},
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_demo_confirm"
    assert payload["action"]["type"] == "select_demo_data"
    assert payload["session_id"] is None
    assert payload["assistant_message"].startswith("可以。我会用内置 PVK reference 数据")
    assert fake_client.calls == 1
    assert fake_client.messages


def test_agent_chat_accepts_llm_action_as_string():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"select_demo_data","action":"select_demo_data","assistant_message":"可以，我会用内置 reference 数据演示。","ui_hints":["show_demo_button"]}',
                usage={},
            )

    original_chat_store = api.chat_agent_store
    api.chat_agent_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "使用内置 PVK demo 数据", "language": "zh"},
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_demo_confirm"
    assert payload["action"]["type"] == "select_demo_data"
    assert payload["assistant_message"] == "可以，我会用内置 reference 数据演示。"


def test_agent_chat_preserves_llm_language_when_no_tool_action():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"discuss_goal","action":{"type":"none","args":{}},"assistant_message":"可以，我们先把你想优化的目标说清楚：是 eta、稳定性，还是两者都看？","ui_hints":["show_demo_button"]}',
                usage={},
            )

    original_chat_store = api.chat_agent_store
    api.chat_agent_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "我想先聊一下目标", "language": "zh"},
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_data_choice"
    assert payload["action"]["type"] == "none"
    assert payload["assistant_message"] == "可以，我们先把你想优化的目标说清楚：是 eta、稳定性，还是两者都看？"


def test_agent_chat_rejects_non_demo_data_source_selection():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"select_demo_data","next_state":"awaiting_demo_confirm","action":{"type":"select_demo_data","args":{"source":"user_upload"}},"assistant_message":"我会使用你的上传数据。","ui_hints":[]}',
                usage={},
            )

    original_chat_store = api.chat_agent_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
    try:
        response = client.post(
            "/api/v1/chat", json={"message": "用我上传的数据", "language": "zh"}
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_data_choice"
    assert payload["session_id"] is None
    assert payload["action"]["type"] == "none"
    assert "第一版只支持内置 PVK demo 数据" in payload["assistant_message"]
    conversation = test_chat_store.get_or_create(payload["conversation_id"])
    assert conversation.data_source is None
    assert conversation.data_source_confirmed is False


def test_agent_chat_demo_selection_must_advance_to_demo_confirmation():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"select_demo_data","next_state":"awaiting_data_choice","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"我已选择 demo 数据。","ui_hints":[]}',
                usage={},
            )

    original_chat_store = api.chat_agent_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
    try:
        response = client.post(
            "/api/v1/chat", json={"message": "用内置 demo", "language": "zh"}
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "awaiting_data_choice"
    assert payload["session_id"] is None
    assert payload["action"]["type"] == "none"
    conversation = test_chat_store.get_or_create(payload["conversation_id"])
    assert conversation.data_source is None
    assert conversation.data_source_confirmed is False


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
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
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
    conversation = test_chat_store.get_or_create(payload["conversation_id"])
    assert conversation.session_id is None
    assert conversation.user_confirmed_run is False


def test_agent_chat_demo_confirmation_chain_reaches_ready_without_session():
    class FakePlannerClient:
        responses = [
            '{"intent":"select_demo_data","next_state":"awaiting_demo_confirm","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"请选择是否确认 demo 数据。","ui_hints":["show_confirm_demo_button"]}',
            '{"intent":"confirm_demo_data","next_state":"awaiting_goal_confirm","action":{"type":"confirm_demo_data","args":{}},"assistant_message":"已确认使用 demo 数据，请确认优化目标。","ui_hints":["show_confirm_goal_button"]}',
            '{"intent":"confirm_goal","next_state":"ready_to_run","action":{"type":"confirm_goal","args":{}},"assistant_message":"目标已确认，可以准备运行。","ui_hints":["show_run_button"]}',
        ]

        def __init__(self):
            self.index = 0

        def chat(self, messages, max_tokens=512, extra_body=None):
            content = self.responses[self.index]
            self.index += 1
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content=content,
                usage={},
            )

    original_chat_store = api.chat_agent_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
    try:
        select_response = client.post(
            "/api/v1/chat", json={"message": "用内置 demo", "language": "zh"}
        )
        conversation_id = select_response.json()["data"]["conversation_id"]
        demo_response = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "message": "确认 demo",
                "language": "zh",
            },
        )
        goal_response = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "message": "确认目标",
                "language": "zh",
            },
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert select_response.status_code == 200
    assert demo_response.status_code == 200
    assert goal_response.status_code == 200
    assert select_response.json()["data"]["state"] == "awaiting_demo_confirm"
    assert demo_response.json()["data"]["state"] == "awaiting_goal_confirm"
    payload = goal_response.json()["data"]
    assert payload["state"] == "ready_to_run"
    assert payload["session_id"] is None
    assert payload["action"]["type"] == "confirm_goal"
    conversation = test_chat_store.get_or_create(payload["conversation_id"])
    assert conversation.data_source == "demo_pvk"
    assert conversation.data_source_confirmed is True
    assert conversation.demo_disclaimer_confirmed is True
    assert conversation.goal_confirmed is True
    assert conversation.user_confirmed_run is False


def test_agent_chat_confirmation_chain_does_not_create_session_before_run():
    class FakePlannerClient:
        responses = [
            '{"intent":"select_demo_data","next_state":"awaiting_demo_confirm","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"请选择是否确认 demo 数据。","ui_hints":["show_confirm_demo_button"]}',
            '{"intent":"confirm_demo_data","next_state":"awaiting_goal_confirm","action":{"type":"confirm_demo_data","args":{}},"assistant_message":"已确认使用 demo 数据，请确认优化目标。","ui_hints":["show_confirm_goal_button"]}',
            '{"intent":"confirm_goal","next_state":"ready_to_run","action":{"type":"confirm_goal","args":{}},"assistant_message":"目标已确认，可以准备运行。","ui_hints":["show_run_button"]}',
        ]

        def __init__(self):
            self.index = 0

        def chat(self, messages, max_tokens=512, extra_body=None):
            content = self.responses[self.index]
            self.index += 1
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content=content,
                usage={},
            )

    class FakeSessionStore:
        def __init__(self):
            self.created = 0
            self.ran = 0

        def create_session(self, request):
            self.created += 1
            raise AssertionError("session should not be created before run_bo_step")

        def run_step(self, session_id):
            self.ran += 1
            raise AssertionError("BO should not run before run_bo_step")

    original_chat_store = api.chat_agent_store
    original_session_store = api.session_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    test_session_store = FakeSessionStore()
    api.chat_agent_store = test_chat_store
    api.session_store = test_session_store
    try:
        select_response = client.post(
            "/api/v1/chat", json={"message": "用内置 demo", "language": "zh"}
        )
        conversation_id = select_response.json()["data"]["conversation_id"]
        client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "message": "确认 demo",
                "language": "zh",
            },
        )
        goal_response = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "message": "确认目标",
                "language": "zh",
            },
        )
    finally:
        api.chat_agent_store = original_chat_store
        api.session_store = original_session_store

    assert goal_response.status_code == 200
    payload = goal_response.json()["data"]
    assert payload["state"] == "ready_to_run"
    assert payload["session_id"] is None
    assert payload["action"]["type"] == "confirm_goal"
    assert test_session_store.created == 0
    assert test_session_store.ran == 0
    conversation = test_chat_store.get_or_create(payload["conversation_id"])
    assert conversation.user_confirmed_run is False


def test_agent_chat_confirmed_demo_flow_runs_one_pvkbo_step():
    class FakePlannerClient:
        responses = [
            '{"intent":"select_demo_data","next_state":"awaiting_demo_confirm","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"使用内置 demo。请确认。","ui_hints":["show_confirm_demo_button"]}',
            '{"intent":"confirm_demo_data","next_state":"awaiting_goal_confirm","action":{"type":"confirm_demo_data","args":{}},"assistant_message":"已确认 demo 数据。请确认目标。","ui_hints":["show_confirm_goal_button"]}',
            '{"intent":"confirm_goal","next_state":"ready_to_run","action":{"type":"confirm_goal","args":{}},"assistant_message":"默认目标是 band_alignment / eta maximize。可以开始第一轮。","ui_hints":["show_run_button"]}',
            '{"intent":"request_run","next_state":"running_bo","action":{"type":"run_bo_step","args":{"task_id":"unexpected_task","n_trials":999}},"assistant_message":"开始运行第一轮 BO。","ui_hints":[]}',
        ]

        def __init__(self):
            self.index = 0

        def chat(self, messages, max_tokens=512, extra_body=None):
            content = self.responses[self.index]
            self.index += 1
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content=content,
                usage={},
            )

    class FakeSessionStore:
        def __init__(self):
            self.created = 0
            self.ran = 0

        def create_session(self, request):
            self.created += 1
            assert request.task_id == "band_alignment"
            assert request.n_trials == 5
            return {
                "session_id": "pvk_real_agent",
                "status": "running",
                "current_step": 0,
                "observed_fvals": [0.0, 0.002],
                "best_result": {"score": 0.002},
                "candidate_points": [],
                "task": {
                    "task_id": "band_alignment",
                    "data_boundary": {"notes": "demo"},
                },
                "tool_trace": [{"step": "PVKBO.initialize", "detail": "initialized"}],
                "guardrails": {"mode": "real_pvk_llm_bo"},
            }

        def run_step(self, session_id):
            self.ran += 1
            assert session_id == "pvk_real_agent"
            return {
                "session_id": session_id,
                "status": "completed",
                "current_step": 1,
                "observed_fvals": [0.0, 0.002, 0.003],
                "best_result": {"score": 0.003},
                "candidate_points": [{"candidate_id": "C1"}],
                "task": {
                    "task_id": "band_alignment",
                    "data_boundary": {"notes": "demo"},
                },
                "tool_trace": [
                    {"step": "LLM_ACQ.get_candidate_points", "detail": "generated"},
                    {"step": "LLM_SURROGATE.select_query_point", "detail": "selected"},
                    {"step": "black_box.evaluate_candidate", "detail": "eta=0.003"},
                    {"step": "PVKBO.update_observations", "detail": "stored"},
                ],
                "guardrails": {"mode": "real_pvk_llm_bo"},
            }

        def get_session(self, session_id):
            return None

    original_chat_store = api.chat_agent_store
    original_session_store = api.session_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    test_session_store = FakeSessionStore()
    api.chat_agent_store = test_chat_store
    api.session_store = test_session_store
    try:
        first = client.post(
            "/api/v1/chat", json={"message": "用 demo", "language": "zh"}
        ).json()["data"]
        second = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": first["conversation_id"],
                "message": "确认 demo",
                "language": "zh",
            },
        ).json()["data"]
        third = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": first["conversation_id"],
                "message": "确认目标",
                "language": "zh",
            },
        ).json()["data"]
        fourth = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": first["conversation_id"],
                "message": "开始运行",
                "language": "zh",
            },
        ).json()["data"]
    finally:
        api.chat_agent_store = original_chat_store
        api.session_store = original_session_store

    assert second["state"] == "awaiting_goal_confirm"
    assert third["state"] == "ready_to_run"
    assert fourth["state"] == "reporting"
    assert fourth["session_id"] == "pvk_real_agent"
    assert test_session_store.created == 1
    assert test_session_store.ran == 1
    assert [call["name"] for call in fourth["tool_calls"]] == [
        "LLM_ACQ.get_candidate_points",
        "LLM_SURROGATE.select_query_point",
        "black_box.evaluate_candidate",
        "PVKBO.update_observations",
    ]
    assert fourth["artifacts"]["bo_step"]["best_score"] == 0.003
    assert fourth["artifacts"]["bo_step"]["best_result"]["score"] == 0.003
    assert "0.003" in fourth["assistant_message"]
    assert "C1" in fourth["assistant_message"]


def test_agent_chat_llm_can_start_demo_bo_without_multi_step_confirmations():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"start_demo_bo","action":{"type":"run_bo_step","args":{"source":"demo_pvk","goal":"band_alignment_eta"}},"assistant_message":"可以，我会直接用内置 PVK reference 数据跑一轮 band_alignment BO；这不是上传数据或湿实验。","ui_hints":[]}',
                usage={},
            )

    class FakeSessionStore:
        def __init__(self):
            self.created = 0
            self.ran = 0

        def create_session(self, request):
            self.created += 1
            assert request.task_id == "band_alignment"
            return {"session_id": "pvk_real_agent"}

        def run_step(self, session_id):
            self.ran += 1
            return {
                "session_id": session_id,
                "status": "completed",
                "current_step": 1,
                "observed_fvals": [0.0, 0.002, 0.003],
                "best_result": {
                    "score": 0.003,
                    "config": {"CHI_PVK": 3.1},
                },
                "candidate_points": [{"candidate_id": "C1", "CHI_PVK": 3.1}],
                "task": {
                    "task_id": "band_alignment",
                    "data_boundary": {"notes": "reference lookup"},
                },
                "tool_trace": [
                    {"step": "LLM_ACQ.get_candidate_points", "detail": "generated"},
                    {"step": "LLM_SURROGATE.select_query_point", "detail": "selected"},
                    {"step": "black_box.evaluate_candidate", "detail": "eta=0.003"},
                    {"step": "PVKBO.update_observations", "detail": "stored"},
                ],
            }

    original_chat_store = api.chat_agent_store
    original_session_store = api.session_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    test_session_store = FakeSessionStore()
    api.chat_agent_store = test_chat_store
    api.session_store = test_session_store
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "使用内置 PVK demo 数据", "language": "zh"},
        )
    finally:
        api.chat_agent_store = original_chat_store
        api.session_store = original_session_store

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["state"] == "reporting"
    assert payload["session_id"] == "pvk_real_agent"
    assert payload["action"]["type"] == "run_bo_step"
    assert payload["artifacts"]["bo_step"]["best_score"] == 0.003
    assert test_session_store.created == 1
    assert test_session_store.ran == 1
    conversation = test_chat_store.get_or_create(payload["conversation_id"])
    assert conversation.data_source == "demo_pvk"
    assert conversation.demo_disclaimer_confirmed is True
    assert conversation.goal_confirmed is True


def test_agent_chat_routes_explicit_demo_start_even_when_llm_returns_none():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"discuss_demo","action":{"type":"none","args":{}},"assistant_message":"可以，我们用内置 reference 数据演示。","ui_hints":[]}',
                usage={},
            )

    class FakeSessionStore:
        def create_session(self, request):
            return {"session_id": "pvk_real_agent"}

        def run_step(self, session_id):
            return {
                "session_id": session_id,
                "status": "completed",
                "current_step": 1,
                "observed_fvals": [0.0, 0.005],
                "best_result": {"score": 0.005},
                "candidate_points": [{"candidate_id": "C4"}],
                "task": {"task_id": "band_alignment"},
                "tool_trace": [
                    {"step": "LLM_ACQ.get_candidate_points", "detail": "generated"},
                    {"step": "LLM_SURROGATE.select_query_point", "detail": "selected"},
                    {"step": "black_box.evaluate_candidate", "detail": "eta=0.005"},
                    {"step": "PVKBO.update_observations", "detail": "stored"},
                ],
            }

    original_chat_store = api.chat_agent_store
    original_session_store = api.session_store
    api.chat_agent_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.session_store = FakeSessionStore()
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "使用内置 PVK demo 数据直接开始", "language": "zh"},
        )
    finally:
        api.chat_agent_store = original_chat_store
        api.session_store = original_session_store

    payload = response.json()["data"]
    assert response.status_code == 200
    assert payload["state"] == "reporting"
    assert payload["action"]["type"] == "run_demo_bo"
    assert payload["artifacts"]["bo_step"]["best_score"] == 0.005


def test_agent_chat_accepts_run_demo_bo_action_name():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"start_demo_bo","action":{"type":"run_demo_bo","args":{"source":"demo_pvk","goal":"band_alignment_eta"}},"assistant_message":"我会直接用内置 reference 数据开始一轮 BO。","ui_hints":[]}',
                usage={},
            )

    class FakeSessionStore:
        def __init__(self):
            self.created = 0
            self.ran = 0

        def create_session(self, request):
            self.created += 1
            return {"session_id": "pvk_real_agent"}

        def run_step(self, session_id):
            self.ran += 1
            return {
                "session_id": session_id,
                "status": "completed",
                "current_step": 1,
                "observed_fvals": [0.0, 0.004],
                "best_result": {"score": 0.004, "config": {"CHI_PVK": 3.2}},
                "candidate_points": [{"candidate_id": "C2"}],
                "task": {"task_id": "band_alignment", "data_boundary": {"notes": "demo"}},
                "tool_trace": [
                    {"step": "LLM_ACQ.get_candidate_points", "detail": "generated"},
                    {"step": "LLM_SURROGATE.select_query_point", "detail": "selected"},
                    {"step": "black_box.evaluate_candidate", "detail": "eta=0.004"},
                    {"step": "PVKBO.update_observations", "detail": "stored"},
                ],
            }

    original_chat_store = api.chat_agent_store
    original_session_store = api.session_store
    api.chat_agent_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    fake_session_store = FakeSessionStore()
    api.session_store = fake_session_store
    try:
        response = client.post(
            "/api/v1/chat",
            json={"message": "用内置 demo 直接开始", "language": "zh"},
        )
    finally:
        api.chat_agent_store = original_chat_store
        api.session_store = original_session_store

    payload = response.json()["data"]
    assert response.status_code == 200
    assert payload["state"] == "reporting"
    assert payload["action"]["type"] == "run_demo_bo"
    assert payload["artifacts"]["bo_step"]["best_score"] == 0.004
    assert fake_session_store.created == 1
    assert fake_session_store.ran == 1


def test_agent_chat_accepts_run_next_bo_step_for_existing_session():
    class FakePlannerClient:
        def chat(self, messages, max_tokens=512, extra_body=None):
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content='{"intent":"run_next","action":{"type":"run_next_bo_step","args":{}},"assistant_message":"继续下一轮。","ui_hints":[]}',
                usage={},
            )

    class FakeSessionStore:
        def __init__(self):
            self.created = 0
            self.ran = 0

        def create_session(self, request):
            self.created += 1
            raise AssertionError("existing session should be reused")

        def run_step(self, session_id):
            self.ran += 1
            assert session_id == "pvk_existing"
            return {
                "session_id": session_id,
                "status": "completed",
                "current_step": 2,
                "observed_fvals": [0.0, 0.004, 0.006],
                "best_result": {"score": 0.006},
                "candidate_points": [{"candidate_id": "C3"}],
                "task": {"task_id": "band_alignment"},
                "tool_trace": [
                    {"step": "LLM_ACQ.get_candidate_points", "detail": "generated"},
                    {"step": "LLM_SURROGATE.select_query_point", "detail": "selected"},
                    {"step": "black_box.evaluate_candidate", "detail": "eta=0.006"},
                    {"step": "PVKBO.update_observations", "detail": "stored"},
                ],
            }

    original_chat_store = api.chat_agent_store
    original_session_store = api.session_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    conversation = test_chat_store.get_or_create(None)
    conversation.data_source = "demo_pvk"
    conversation.data_source_confirmed = True
    conversation.demo_disclaimer_confirmed = True
    conversation.goal_confirmed = True
    conversation.session_id = "pvk_existing"
    conversation.state = "reporting"
    api.chat_agent_store = test_chat_store
    fake_session_store = FakeSessionStore()
    api.session_store = fake_session_store
    try:
        response = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation.conversation_id,
                "message": "继续下一轮",
                "language": "zh",
            },
        )
    finally:
        api.chat_agent_store = original_chat_store
        api.session_store = original_session_store

    payload = response.json()["data"]
    assert response.status_code == 200
    assert payload["state"] == "reporting"
    assert payload["action"]["type"] == "run_next_bo_step"
    assert payload["artifacts"]["bo_step"]["best_score"] == 0.006
    assert fake_session_store.created == 0
    assert fake_session_store.ran == 1


def test_agent_chat_accepts_generate_candidate_points_trace_name():
    class FakePlannerClient:
        responses = [
            '{"intent":"select_demo_data","next_state":"awaiting_demo_confirm","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"使用内置 demo。请确认。","ui_hints":["show_confirm_demo_button"]}',
            '{"intent":"confirm_demo_data","next_state":"awaiting_goal_confirm","action":{"type":"confirm_demo_data","args":{}},"assistant_message":"已确认 demo 数据。请确认目标。","ui_hints":["show_confirm_goal_button"]}',
            '{"intent":"confirm_goal","next_state":"ready_to_run","action":{"type":"confirm_goal","args":{}},"assistant_message":"默认目标是 band_alignment / eta maximize。可以开始第一轮。","ui_hints":["show_run_button"]}',
            '{"intent":"request_run","next_state":"running_bo","action":{"type":"run_bo_step","args":{}},"assistant_message":"开始运行第一轮 BO。","ui_hints":[]}',
        ]

        def __init__(self):
            self.index = 0

        def chat(self, messages, max_tokens=512, extra_body=None):
            content = self.responses[self.index]
            self.index += 1
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content=content,
                usage={},
            )

    class FakeSessionStore:
        def create_session(self, request):
            return {"session_id": "pvk_real_agent"}

        def run_step(self, session_id):
            return {
                "session_id": session_id,
                "status": "completed",
                "current_step": 1,
                "observed_fvals": [0.0, 0.003],
                "best_result": {"score": 0.003},
                "candidate_points": [{"candidate_id": "C1"}],
                "task": {"task_id": "band_alignment"},
                "tool_trace": [
                    {"step": "LLM_ACQ.generate_candidate_points", "detail": "generated"},
                    {"step": "LLM_SURROGATE.select_query_point", "detail": "selected"},
                    {"step": "black_box.evaluate_candidate", "detail": "eta=0.003"},
                    {"step": "PVKBO.update_observations", "detail": "stored"},
                ],
            }

    original_chat_store = api.chat_agent_store
    original_session_store = api.session_store
    api.chat_agent_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.session_store = FakeSessionStore()
    try:
        first = client.post(
            "/api/v1/chat", json={"message": "用 demo", "language": "zh"}
        ).json()["data"]
        for message in ["确认 demo", "确认目标"]:
            client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": first["conversation_id"],
                    "message": message,
                    "language": "zh",
                },
            )
        response = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": first["conversation_id"],
                "message": "开始运行",
                "language": "zh",
            },
        )
    finally:
        api.chat_agent_store = original_chat_store
        api.session_store = original_session_store

    assert response.status_code == 200
    tool_names = [call["name"] for call in response.json()["data"]["tool_calls"]]
    assert tool_names
    assert "LLM_ACQ.get_candidate_points" in tool_names


def test_agent_chat_run_step_failure_returns_503_without_success_payload():
    class FakePlannerClient:
        responses = [
            '{"intent":"select_demo_data","next_state":"awaiting_demo_confirm","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"使用内置 demo。请确认。","ui_hints":["show_confirm_demo_button"]}',
            '{"intent":"confirm_demo_data","next_state":"awaiting_goal_confirm","action":{"type":"confirm_demo_data","args":{}},"assistant_message":"已确认 demo 数据。请确认目标。","ui_hints":["show_confirm_goal_button"]}',
            '{"intent":"confirm_goal","next_state":"ready_to_run","action":{"type":"confirm_goal","args":{}},"assistant_message":"默认目标是 band_alignment / eta maximize。可以开始第一轮。","ui_hints":["show_run_button"]}',
            '{"intent":"request_run","next_state":"running_bo","action":{"type":"run_bo_step","args":{}},"assistant_message":"开始运行第一轮 BO。","ui_hints":[]}',
        ]

        def __init__(self):
            self.index = 0

        def chat(self, messages, max_tokens=512, extra_body=None):
            content = self.responses[self.index]
            self.index += 1
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content=content,
                usage={},
            )

    class FakeSessionStore:
        def create_session(self, request):
            return {"session_id": "pvk_real_agent"}

        def run_step(self, session_id):
            raise RuntimeError("runtime exploded")

    original_chat_store = api.chat_agent_store
    original_session_store = api.session_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
    api.session_store = FakeSessionStore()
    try:
        first = client.post(
            "/api/v1/chat", json={"message": "用 demo", "language": "zh"}
        ).json()["data"]
        for message in ["确认 demo", "确认目标"]:
            client.post(
                "/api/v1/chat",
                json={
                    "conversation_id": first["conversation_id"],
                    "message": message,
                    "language": "zh",
                },
            )
        response = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": first["conversation_id"],
                "message": "开始运行",
                "language": "zh",
            },
        )
        conversation = test_chat_store.get_or_create(first["conversation_id"])
    finally:
        api.chat_agent_store = original_chat_store
        api.session_store = original_session_store

    assert response.status_code == 503
    payload = response.json()
    assert "data" not in payload
    assert "artifacts" not in payload
    assert "tool_calls" not in payload
    assert "runtime exploded" in payload["error"]["message"]
    assert conversation.state == "ready_to_run"
    assert conversation.session_id is None


def test_agent_chat_ready_none_cannot_transition_to_reporting():
    class FakePlannerClient:
        responses = [
            '{"intent":"select_demo_data","next_state":"awaiting_demo_confirm","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"请选择是否确认 demo 数据。","ui_hints":["show_confirm_demo_button"]}',
            '{"intent":"confirm_demo_data","next_state":"awaiting_goal_confirm","action":{"type":"confirm_demo_data","args":{}},"assistant_message":"已确认使用 demo 数据，请确认优化目标。","ui_hints":["show_confirm_goal_button"]}',
            '{"intent":"confirm_goal","next_state":"ready_to_run","action":{"type":"confirm_goal","args":{}},"assistant_message":"目标已确认，可以准备运行。","ui_hints":["show_run_button"]}',
            '{"intent":"idle","next_state":"reporting","action":{"type":"none","args":{}},"assistant_message":"我会进入报告页。","ui_hints":[]}',
        ]

        def __init__(self):
            self.index = 0

        def chat(self, messages, max_tokens=512, extra_body=None):
            content = self.responses[self.index]
            self.index += 1
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content=content,
                usage={},
            )

    original_chat_store = api.chat_agent_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
    try:
        select_response = client.post(
            "/api/v1/chat", json={"message": "用内置 demo", "language": "zh"}
        )
        conversation_id = select_response.json()["data"]["conversation_id"]
        client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "message": "确认 demo",
                "language": "zh",
            },
        )
        client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "message": "确认目标",
                "language": "zh",
            },
        )
        reporting_response = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "message": "继续",
                "language": "zh",
            },
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert reporting_response.status_code == 200
    payload = reporting_response.json()["data"]
    assert payload["state"] == "ready_to_run"
    assert payload["session_id"] is None
    assert payload["action"]["type"] == "none"
    conversation = test_chat_store.get_or_create(payload["conversation_id"])
    assert conversation.state == "ready_to_run"


def test_agent_chat_rejects_confirm_goal_before_demo_confirmation():
    class FakePlannerClient:
        responses = [
            '{"intent":"select_demo_data","next_state":"awaiting_demo_confirm","action":{"type":"select_demo_data","args":{"source":"demo_pvk"}},"assistant_message":"请选择是否确认 demo 数据。","ui_hints":["show_confirm_demo_button"]}',
            '{"intent":"confirm_goal","next_state":"ready_to_run","action":{"type":"confirm_goal","args":{}},"assistant_message":"目标已确认。","ui_hints":["show_run_button"]}',
        ]

        def __init__(self):
            self.index = 0

        def chat(self, messages, max_tokens=512, extra_body=None):
            content = self.responses[self.index]
            self.index += 1
            return LlmCallResult(
                status="success",
                provider="deepseek",
                model="deepseek-v4-flash",
                content=content,
                usage={},
            )

    original_chat_store = api.chat_agent_store
    test_chat_store = api.ChatAgentStore(planner_client=FakePlannerClient())
    api.chat_agent_store = test_chat_store
    try:
        select_response = client.post(
            "/api/v1/chat", json={"message": "用内置 demo", "language": "zh"}
        )
        conversation_id = select_response.json()["data"]["conversation_id"]
        forged_response = client.post(
            "/api/v1/chat",
            json={
                "conversation_id": conversation_id,
                "message": "直接确认目标",
                "language": "zh",
            },
        )
    finally:
        api.chat_agent_store = original_chat_store

    assert forged_response.status_code == 200
    payload = forged_response.json()["data"]
    assert payload["state"] == "awaiting_demo_confirm"
    assert payload["session_id"] is None
    assert payload["action"]["type"] == "none"
    conversation = test_chat_store.get_or_create(payload["conversation_id"])
    assert conversation.demo_disclaimer_confirmed is False
    assert conversation.goal_confirmed is False


def test_deepseek_chat_rejects_extra_body_protected_key_override():
    deepseek_client = DeepSeekClient(api_key="test-key")

    with pytest.raises(ValueError, match="protected"):
        deepseek_client.chat(
            [{"role": "user", "content": "hello"}],
            extra_body={"model": "override-model"},
        )


def test_list_tasks_returns_pvk_optimization_tasks():
    response = client.get("/api/v1/tasks")

    assert response.status_code == 200
    tasks = response.json()["data"]
    assert tasks
    assert tasks[0]["id"] == tasks[0]["task_id"]
    assert tasks[0]["task_id"]
    assert tasks[0]["title"]
    assert isinstance(tasks[0]["data_boundary"], dict)
    assert tasks[0]["data_boundary"]["notes"]
    assert tasks[0]["data_boundary"]["warnings"]
    assert {task["task_id"] for task in tasks}.issuperset(
        {"passivation_demo", "band_alignment", "defects_doping"}
    )


def test_create_real_pvkbo_session_reports_missing_workbook_as_dependency_error():
    original_session_store = api.session_store
    api.session_store = api.PvkSessionStore(
        real_pvk_runtime=RealPvkBoRuntime(data_root="/tmp/missing-pvk-dataset")
    )
    try:
        response = client.post(
            "/api/v1/sessions",
            json={
                "task_id": "band_alignment",
                "n_initial": 2,
                "n_trials": 1,
                "seed": 7,
                "use_llm": True,
                "language": "zh",
            },
        )
    finally:
        api.session_store = original_session_store

    assert response.status_code == 503
    assert "bandAlignment.xlsx" in response.json()["error"]["message"]


def test_create_session_run_step_and_list_artifacts():
    task_id = client.get("/api/v1/tasks").json()["data"][0]["task_id"]
    create_response = client.post(
        "/api/v1/sessions",
        json={
            "task_id": task_id,
            "n_initial": 2,
            "n_trials": 2,
            "seed": 7,
            "use_llm": False,
            "language": "zh",
        },
    )

    assert create_response.status_code == 201
    session = create_response.json()["data"]
    session_id = session["session_id"]
    assert session_id
    assert session["task_id"] == task_id
    assert session["status"] in {"created", "running", "completed"}
    assert session["config"]["n_initial"] == 2
    assert isinstance(session["data_boundary"], dict)
    assert session["data_boundary"]["notes"]

    read_response = client.get(f"/api/v1/sessions/{session_id}")
    step_response = client.post(f"/api/v1/sessions/{session_id}/steps")
    artifacts_response = client.get(f"/api/v1/sessions/{session_id}/artifacts")

    assert read_response.status_code == 200
    assert read_response.json()["data"]["session_id"] == session_id
    assert step_response.status_code == 200
    stepped_session = step_response.json()["data"]
    assert stepped_session["session_id"] == session_id
    assert stepped_session["step_count"] == 1
    assert artifacts_response.status_code == 200
    artifacts = artifacts_response.json()["data"]
    assert artifacts["session_id"] == session_id
    assert isinstance(artifacts["data_boundary"], dict)
    assert artifacts["data_boundary"]["notes"]
    assert artifacts["artifacts"]


def test_session_mvp_endpoints_return_target_curve_and_chat():
    task_id = client.get("/api/v1/tasks").json()["data"][0]["task_id"]
    create_response = client.post(
        "/api/v1/sessions",
        json={
            "task_id": task_id,
            "n_initial": 3,
            "n_trials": 2,
            "seed": 7,
            "use_llm": False,
            "language": "zh",
        },
    )
    session_id = create_response.json()["data"]["session_id"]

    target_response = client.get(
        f"/api/v1/sessions/{session_id}/passivation-target"
    )
    curve_response = client.get(f"/api/v1/sessions/{session_id}/bo-curve")
    chat_response = client.post(
        f"/api/v1/sessions/{session_id}/chat",
        json={
            "message": "请总结当前最好的钝化候选",
            "language": "zh",
            "history": [{"role": "user", "content": "上一轮问题"}],
        },
    )

    assert target_response.status_code == 200
    target = target_response.json()["data"]
    assert "PipDI" in target["passivators"]

    assert curve_response.status_code == 200
    curve = curve_response.json()["data"]
    assert curve["series"]

    assert chat_response.status_code == 200
    chat = chat_response.json()["data"]
    assert chat["assistant_message"]
    assert chat["tool_calls"]


def test_real_pvkbo_chat_can_trigger_next_bo_step():
    class FakeRealSessionStore:
        def __init__(self):
            self.session = {
                "session_id": "pvk_real_fake",
                "status": "running",
                "current_step": 0,
                "observed_fvals": [22.4, 23.1],
                "candidate_points": [],
                "best_result": {"score": 23.1, "metric": "eta", "config": {}},
                "task": {"task_id": "band_alignment", "data_source": "PVK-LLM:bandAlignment.xlsx"},
                "tool_trace": [{"step": "PVKBO.initialize", "detail": "initialized"}],
                "guardrails": {"mode": "real_pvk_llm_bo", "language": "zh"},
            }

        def get_session(self, session_id):
            return self.session if session_id == "pvk_real_fake" else None

        def run_step(self, session_id):
            assert session_id == "pvk_real_fake"
            self.session = {
                **self.session,
                "current_step": 1,
                "observed_fvals": [22.4, 23.1, 24.2],
                "candidate_points": [
                    {"candidate_id": "PVK-CAND-01-01", "CHI_PVK": 4.0}
                ],
                "best_result": {
                    "score": 24.2,
                    "metric": "eta",
                    "config": {"CHI_PVK": 4.0},
                },
                "tool_trace": [
                    {"step": "PVKBO.initialize", "detail": "initialized"},
                    {"step": "LLM_ACQ.get_candidate_points", "detail": "generated"},
                    {"step": "LLM_SURROGATE.select_query_point", "detail": "selected"},
                    {"step": "black_box.evaluate_candidate", "detail": "eta=24.2"},
                    {"step": "PVKBO.update_observations", "detail": "stored"},
                ],
            }
            return self.session

    original_session_store = api.session_store
    api.session_store = FakeRealSessionStore()
    try:
        response = client.post(
            "/api/v1/sessions/pvk_real_fake/chat",
            json={"message": "运行下一步真实 BO", "language": "zh"},
        )
    finally:
        api.session_store = original_session_store

    assert response.status_code == 200
    chat = response.json()["data"]
    assert chat["phase"] == "Optimization"
    assert "真实 PVKBO" in chat["assistant_message"]
    assert chat["artifacts"]["bo_step"]["best_score"] == 24.2
    assert [call["name"] for call in chat["tool_calls"]] == [
        "LLM_ACQ.get_candidate_points",
        "LLM_SURROGATE.select_query_point",
        "black_box.evaluate_candidate",
        "PVKBO.update_observations",
    ]


def test_real_pvkbo_step_intent_does_not_trigger_on_bo_substrings():
    assert api._message_requests_bo_step("run BO step now") is True
    assert api._message_requests_bo_step("please execute the next optimization step") is True
    assert api._message_requests_bo_step("next step") is True
    assert api._message_requests_bo_step("运行下一步真实 BO") is True
    assert api._message_requests_bo_step("下一步吧") is True
    assert api._message_requests_bo_step("tell me about robot behavior") is False
    assert api._message_requests_bo_step("就上面的结果解释一下") is False
    assert api._message_requests_bo_step("解释一下真实数据的边界") is False
    assert api._message_requests_bo_step("优化算法是怎么工作的？") is False
    assert api._message_requests_bo_step("解释下一步会发生什么") is False
    assert api._message_requests_bo_step("explain how to start the optimization") is False
    assert api._message_requests_bo_step("next step is to analyze data") is False


def test_missing_session_returns_404_error_envelope():
    response = client.get("/api/v1/sessions/session_missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize(
    ("method", "path", "json"),
    [
        ("get", "/api/v1/sessions/session_missing/passivation-target", None),
        ("get", "/api/v1/sessions/session_missing/bo-curve", None),
        (
            "post",
            "/api/v1/sessions/session_missing/chat",
            {"message": "hello", "language": "en"},
        ),
    ],
)
def test_missing_session_mvp_endpoints_return_404_error_envelope(method, path, json):
    request = getattr(client, method)
    response = request(path, json=json) if json is not None else request(path)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
