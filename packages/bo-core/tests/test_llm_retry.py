from __future__ import annotations

from unittest.mock import patch

from bo_core.llm_client import DeepSeekClient


def _ok_response() -> dict:
    return {
        "choices": [
            {"message": {"content": "Yes"}, "logprobs": None}
        ],
        "usage": {},
    }


def _resp(status_code: int, retry_after: str | None = None, body: str = "{}"):
    class _R:
        def __init__(self) -> None:
            self.ok = 200 <= status_code < 300
            self.status_code = status_code
            self.text = body
            self.headers = {"Retry-After": retry_after} if retry_after else {}

        def json(self) -> dict:
            return _ok_response()

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
