from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"
PROTECTED_CHAT_PAYLOAD_KEYS = {"model", "messages", "max_tokens", "stream"}


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
    def from_env(cls, env_path: Path = DEFAULT_ENV_PATH) -> "DeepSeekClient":
        load_env_file(env_path)
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
        max_tokens: int = 512,
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

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            return LlmCallResult(
                status="error",
                provider="deepseek",
                model=self.model,
                content="",
                usage={},
                error=f"{type(exc).__name__}: {exc}",
            )

        if not response.ok:
            return LlmCallResult(
                status="error",
                provider="deepseek",
                model=self.model,
                content="",
                usage={},
                error=f"HTTP {response.status_code}: {response.text[:300]}",
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


def load_env_file(path: Path = DEFAULT_ENV_PATH) -> None:
    # Try current path first, then parent
    if not path.exists():
        parent_env = path.parent.parent / ".env"
        if parent_env.exists():
            path = parent_env
        else:
            return
            
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # Use direct assignment so .env overrides existing env vars (e.g. broken ones)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")
