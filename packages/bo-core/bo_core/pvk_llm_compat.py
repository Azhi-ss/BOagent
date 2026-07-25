"""Compatibility patches required by PVK-LLM before importing PVKBO.

These monkey-patches fix API incompatibilities between the legacy PVK-LLM
code and modern versions of pandas, langchain, and the OpenAI SDK.
Call `install_all_compat_patches()` before any `from pvk_bo.pvk_bo import PVKBO`.
"""

from __future__ import annotations

from typing import Any


def install_all_compat_patches() -> None:
    """Apply all PVK-LLM compatibility monkey-patches."""
    _install_langchain_prompt_compat()
    _install_pandas_series_int_position_compat()
    _install_openai_single_completion_compat()


def _install_langchain_prompt_compat() -> None:
    """Expose langchain_core prompt classes on the langchain top-level module.

    PVK-LLM references ``langchain.FewShotPromptTemplate`` and
    ``langchain.PromptTemplate``, which were moved to langchain_core
    in newer versions.
    """
    try:
        import langchain
        from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
    except ImportError:
        return
    if not hasattr(langchain, "FewShotPromptTemplate"):
        langchain.FewShotPromptTemplate = FewShotPromptTemplate
    if not hasattr(langchain, "PromptTemplate"):
        langchain.PromptTemplate = PromptTemplate


def _install_pandas_series_int_position_compat() -> None:
    """Restore legacy integer-positional indexing on pd.Series.

    Older PVK-LLM code uses ``series[0]`` to access the first element
    by position.  Modern pandas raises ``KeyError`` when the integer
    is not in the index.  This wrapper falls back to ``iloc``.
    """
    import pandas as pd

    if getattr(pd.Series, "_boagent_legacy_int_position_compat", False):
        return

    original_getitem = pd.Series.__getitem__

    def getitem_with_legacy_int_position(self: pd.Series, key: Any) -> Any:
        try:
            return original_getitem(self, key)
        except KeyError:
            if isinstance(key, int) and key not in self.index:
                return self.iloc[key]
            raise

    pd.Series.__getitem__ = getitem_with_legacy_int_position
    pd.Series._boagent_legacy_int_position_compat = True


def _install_openai_single_completion_compat() -> None:
    """Force sane defaults on DeepSeek-bound OpenAI chat-completion calls.

    DeepSeek does not support ``n > 1``, requires ``max_tokens >= 512``,
    and emits a warning when the ``thinking`` extra-body is absent.
    This wrapper silently adjusts those parameters for any model whose
    name starts with ``deepseek``.
    """
    try:
        from openai.resources.chat.completions import AsyncCompletions
    except ImportError:
        return
    if getattr(AsyncCompletions, "_boagent_force_single_completion", False):
        return

    original_create = AsyncCompletions.create

    async def create_with_single_completion(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model") or "").lower()
        if model.startswith("deepseek"):
            if kwargs.get("n", 1) != 1:
                kwargs["n"] = 1
            if int(kwargs.get("max_tokens") or 0) < 512:
                kwargs["max_tokens"] = 512
            extra_body = dict(kwargs.get("extra_body") or {})
            extra_body.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = extra_body
        return await original_create(self, *args, **kwargs)

    AsyncCompletions.create = create_with_single_completion
    AsyncCompletions._boagent_force_single_completion = True
