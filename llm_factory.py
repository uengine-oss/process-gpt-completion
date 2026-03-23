"""Local LLM helper (no external llm_factory dependency)."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Union

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

TimeoutType = Union[float, Tuple[float, float]]


def get_llm_model(default: str = "gpt-4o") -> str:
    """Resolve chat model alias/name from environment."""
    model = (os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "").strip()
    return model or default


def _proxy_base_url() -> str:
    return (
        os.getenv("LLM_PROXY_URL")
        or "http://litellm-proxy:4000"
    )


def _proxy_api_key() -> str:
    api_key = (
        os.getenv("LLM_PROXY_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Missing LiteLLM proxy API key. Set `LLM_PROXY_API_KEY` "
            "(virtual key, sk-...) or fallback to `OPENAI_API_KEY`."
        )
    return api_key


def create_llm(
    model: Optional[str] = None,
    streaming: bool = False,
    temperature: float = 0.0,
    timeout: Optional[TimeoutType] = (10.0, 120.0),
    max_retries: int = 6,
    **kwargs: Any,
):
    """Standard ChatOpenAI constructor wrapper used across the project."""
    resolved_model = (model or "").strip() or get_llm_model()

    base_url = kwargs.pop("base_url", None) or _proxy_base_url()
    api_key = kwargs.pop("api_key", None) or _proxy_api_key()

    return ChatOpenAI(
        model=resolved_model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        streaming=streaming,
        disable_streaming=not streaming,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )


def create_openai_llm(
    model: Optional[str] = None,
    streaming: bool = False,
    **kwargs: Any,
):
    return create_llm(model=model, streaming=streaming, **kwargs)


def create_embedding(
    model: Optional[str] = None,
    timeout: Optional[TimeoutType] = (10.0, 120.0),
    max_retries: int = 6,
    **kwargs: Any,
):
    if model is None:
        model = os.getenv("LLM_EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL")
    if not model:
        raise RuntimeError(
            "Missing embedding model alias. Set `LLM_EMBEDDING_MODEL` "
            "or at least `LLM_MODEL`."
        )

    base_url = kwargs.pop("base_url", None) or _proxy_base_url()
    api_key = kwargs.pop("api_key", None) or _proxy_api_key()

    # Some setups require `deployment`; for LiteLLM OpenAI-compatible API,
    # passing the same model value is typically sufficient.
    deployment = kwargs.pop("deployment", None) or model

    return OpenAIEmbeddings(
        model=model,
        deployment=deployment,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )

