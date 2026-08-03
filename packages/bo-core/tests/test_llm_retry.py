from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from bo_core import llm_client
from bo_core.llm_client import DeepSeekClient, LlmCallResult


def _ok_response() -> dict:
    return {
        "choices": [
            {"message": {"content": "Yes"}, "logprobs": None}
        ],
        "usage": {},
    }


def _resp(
    status_code: int,
    retry_after: str | None = None,
    body: str = "{}",
    payload: dict | None = None,
):
    class _R:
        def __init__(self) -> None:
            self.ok = 200 <= status_code < 300
            self.status_code = status_code
            self.text = body
            self.headers = {"Retry-After": retry_after} if retry_after else {}

        def json(self) -> dict:
            return payload if payload is not None else _ok_response()

    return _R()


def _client() -> DeepSeekClient:
    c = DeepSeekClient(api_key="sk-test", base_url="https://relay.test/v1", model="m")
    c.timeout_s = 5
    return c


def test_retries_on_429_then_succeeds():
    client = _client()
    side = [_resp(429, retry_after="0"), _resp(200)]
    with patch("bo_core.llm_client.requests.post", side_effect=side) as mock_post, \
         patch("bo_core.llm_client.time.sleep") as _sleep:
        result = client.chat([{"role": "user", "content": "hi"}])
    assert result.status == "success"
    assert result.content == "Yes"
    assert mock_post.call_count == 2


def test_retries_on_503_then_succeeds():
    client = _client()
    side = [_resp(503), _resp(503), _resp(200)]
    with patch("bo_core.llm_client.requests.post", side_effect=side) as mock_post, \
         patch("bo_core.llm_client.time.sleep"):
        result = client.chat([{"role": "user", "content": "hi"}])
    assert result.status == "success"
    assert mock_post.call_count == 3


def test_no_retry_on_non_retryable_400():
    client = _client()
    with patch("bo_core.llm_client.requests.post", side_effect=[_resp(400)]) as mock_post, \
         patch("bo_core.llm_client.time.sleep") as sleep:
        result = client.chat([{"role": "user", "content": "hi"}])
    assert result.status == "error"
    assert "HTTP 400" in result.error
    assert mock_post.call_count == 1
    assert sleep.call_count == 0


def test_gives_up_after_max_retries():
    client = _client()
    with patch("bo_core.llm_client.requests.post", side_effect=[_resp(429)] * 10) as mock_post, \
         patch("bo_core.llm_client.time.sleep"):
        result = client.chat([{"role": "user", "content": "hi"}])
    assert result.status == "error"
    assert "429" in result.error
    # initial attempt + MAX_RETRIES
    from bo_core.llm_client import MAX_RETRIES
    assert mock_post.call_count == MAX_RETRIES + 1

def test_tool_call_succeeds_with_empty_content() -> None:
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "propose_sparse_subspace", "arguments": "{}"},
        }
    ]
    response = _resp(
        200,
        payload={
            "choices": [{"message": {"content": "", "tool_calls": tool_calls}}],
            "usage": {},
        },
    )
    with patch("bo_core.llm_client.requests.post", return_value=response):
        result = _client().chat([{"role": "user", "content": "hi"}])

    assert result.status == "success"
    assert result.content == ""
    assert result.tool_calls == tool_calls


def test_chat_temperature_default_override_and_positional_extra_body() -> None:
    with patch("bo_core.llm_client.requests.post", return_value=_resp(200)) as post:
        _client().chat([{"role": "user", "content": "hi"}], 100, {"seed": 7})
        default_payload = post.call_args.kwargs["json"]

        _client().chat(
            [{"role": "user", "content": "hi"}],
            temperature=0.2,
        )
        override_payload = post.call_args.kwargs["json"]

    assert default_payload["temperature"] == 0.0
    assert default_payload["max_tokens"] == 100
    assert default_payload["seed"] == 7
    assert override_payload["temperature"] == 0.2


def test_extra_body_cannot_override_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        _client().chat(
            [{"role": "user", "content": "hi"}],
            extra_body={"temperature": 0.9},
        )


def test_llm_call_result_tool_calls_default_to_none() -> None:
    result = LlmCallResult("success", "deepseek", "m", "text", {})

    assert result.tool_calls is None


def test_project_env_is_the_only_file_source_and_process_env_wins(
    monkeypatch, tmp_path: Path
) -> None:
    assert llm_client.PROJECT_ENV_PATH == Path(__file__).resolve().parents[3] / ".env"
    project_env = tmp_path / ".env"
    project_env.write_text(
        "DEEPSEEK_API_KEY=file-key\nDEEPSEEK_MODEL=file-model\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(llm_client, "PROJECT_ENV_PATH", project_env)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-key")
    monkeypatch.delenv("DEEPSEEK_FLASH_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)

    client = DeepSeekClient.from_env()

    assert client.api_key == "process-key"
    assert client.model == "file-model"
    assert llm_client.PROJECT_ENV_PATH == project_env
