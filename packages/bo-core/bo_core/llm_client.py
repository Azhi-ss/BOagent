from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

PROJECT_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
PROTECTED_CHAT_PAYLOAD_KEYS = {"model", "messages", "max_tokens", "stream"}

# Retry transient upstream rate-limiting / gateway errors so concurrent workers
# stay under a provider's per-minute quota instead of failing the whole BO step.
# Total backoff is bounded so a single chat() returns well within the ~30s
# caller-side timeout used by the viability concurrent evaluator.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 3
RETRY_BASE_SLEEP = 1.0
RETRY_MAX_TOTAL_SLEEP = 15.0


@dataclass(frozen=True)
class LlmCallResult:
    status: str
    provider: str
    model: str
    content: str
    usage: dict[str, Any]
    error: str | None = None
    logprobs: dict[str, Any] | None = None


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        timeout_s: int = 30,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> DeepSeekClient:
        load_env_file()
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        model = (
            os.environ.get("DEEPSEEK_FLASH_MODEL")
            or os.environ.get("DEEPSEEK_MODEL")
            or "deepseek-v4-flash"
        )
        return cls(api_key=api_key, base_url=base_url, model=model)

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_deepseek_api_key_here")

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 2048,
        extra_body: dict[str, Any] | None = None,
    ) -> LlmCallResult:
        protected_overrides = PROTECTED_CHAT_PAYLOAD_KEYS & set(extra_body or {})
        if protected_overrides:
            keys = ", ".join(sorted(protected_overrides))
            raise ValueError(f"extra_body cannot override protected chat payload keys: {keys}")

        if not self.is_configured():
            return LlmCallResult(
                status="skipped",
                provider="deepseek",
                model=self.model,
                content="",
                usage={},
                error="DEEPSEEK_API_KEY is missing or still uses the placeholder.",
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        if extra_body:
            payload.update(extra_body)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = None
        last_error = ""
        slept = 0.0
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout_s
                )
            except requests.RequestException as exc:
                response = None
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= MAX_RETRIES or slept >= RETRY_MAX_TOTAL_SLEEP:
                    break
                delay = min(RETRY_BASE_SLEEP * (2 ** attempt), RETRY_MAX_TOTAL_SLEEP - slept)
                time.sleep(delay)
                slept += delay
                continue

            if response.ok:
                break

            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in RETRYABLE_STATUS:
                break
            if attempt >= MAX_RETRIES or slept >= RETRY_MAX_TOTAL_SLEEP:
                break
            # Honor server Retry-After when present; else exponential backoff,
            # capped so the whole call stays within caller timeouts.
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else RETRY_BASE_SLEEP * (2 ** attempt)
            except (TypeError, ValueError):
                delay = RETRY_BASE_SLEEP * (2 ** attempt)
            delay = min(delay, RETRY_MAX_TOTAL_SLEEP - slept)
            time.sleep(delay)
            slept += delay

        if response is None or not response.ok:
            return LlmCallResult(
                status="error",
                provider="deepseek",
                model=self.model,
                content="",
                usage={},
                error=last_error or "request failed after retries",
            )

        payload = response.json()
        choice = payload.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        logprobs = choice.get("logprobs", None)
        return LlmCallResult(
            status="success",
            provider="deepseek",
            model=self.model,
            content=content.strip(),
            usage=payload.get("usage", {}),
            logprobs=logprobs,
        )


def load_env_file() -> None:
    if not PROJECT_ENV_PATH.exists():
        return

    # Default high-performance CPU threading for AMD 20-thread CPU if not set
    for thread_var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(thread_var, "20")

    for raw_line in PROJECT_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
