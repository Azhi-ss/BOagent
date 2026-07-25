import api
import pytest
from api import app
from bo_core.llm_client import DeepSeekClient
from fastapi.testclient import TestClient

client = TestClient(app)


# ---------------------------------------------------------------------------
# Health & Logs
# ---------------------------------------------------------------------------

def test_health_endpoint_reports_api_status():
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["data"]["status"] == "ok"


def test_backend_logs_page_and_recent_events_are_available():
    api.backend_log.clear()
    api.emit_backend_log("test.event", "test log event", detail={"step": 1})

    page_response = client.get("/logs")
    events_response = client.get("/api/v1/logs")

    assert page_response.status_code == 200
    assert "text/html" in page_response.headers["content-type"]
    assert "BOagent Live Logs" in page_response.text

    assert events_response.status_code == 200
    payload = events_response.json()
    events = payload["data"]["events"]
    assert len(events) >= 1
    assert events[-1]["event"] == "test.event"


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def test_list_tasks_returns_benchmark_tasks():
    response = client.get("/api/v1/tasks")

    assert response.status_code == 200
    tasks = response.json()["data"]

    assert isinstance(tasks, list)
    assert len(tasks) == 5
    task_ids = {t["task_id"] for t in tasks}
    assert task_ids == {"band_alignment", "defects_doping", "buchwald_sub4", "suzuki", "battery_cathode"}


    for task in tasks:
        assert "name" in task
        assert "objective" in task
        assert "data_available" in task
        assert task["data_available"] is True


# ---------------------------------------------------------------------------
# Benchmark endpoint
# ---------------------------------------------------------------------------

def test_create_benchmark_rejects_invalid_task_id():
    response = client.post(
        "/api/v1/benchmark",
        json={"task_id": "invalid_task"},
    )
    assert response.status_code == 422  # pydantic validation


def test_create_benchmark_rejects_path_traversal_output_dir():
    response = client.post(
        "/api/v1/benchmark",
        json={
            "task_id": "band_alignment",
            "n_trials": 1,
            "n_initial": 2,
            "output_dir": "../../etc",
        },
    )
    assert response.status_code == 400
    assert "output_dir" in response.json()["error"]["message"]


def test_create_benchmark_rejects_absolute_output_dir():
    response = client.post(
        "/api/v1/benchmark",
        json={
            "task_id": "band_alignment",
            "n_trials": 1,
            "n_initial": 2,
            "output_dir": "/tmp/results",
        },
    )
    assert response.status_code == 400
    assert "output_dir" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# DeepSeek client (unit test — no API call)
# ---------------------------------------------------------------------------

def test_deepseek_chat_rejects_extra_body_protected_key_override():
    ds = DeepSeekClient(api_key="sk-test")
    with pytest.raises(ValueError, match="extra_body cannot override"):
        ds.chat(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"model": "gpt-5"},
        )
