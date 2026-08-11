"""Local LLM helper for polling_service runtime."""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Union

TimeoutType = Union[float, Tuple[float, float]]


def _env_int(name: str, default: int) -> int:
    try:
        return int((os.getenv(name) or "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float((os.getenv(name) or "").strip() or default)
    except ValueError:
        return default


def _default_timeout() -> TimeoutType:
    # (connect, read). read 는 스트리밍에서 "청크 간 간격" 이므로 120초를 유지한다.
    # (비스트리밍 대형 생성이 60초를 넘는 경우가 있어 축소하지 않는다.)
    return (_env_float("LLM_CONNECT_TIMEOUT", 5.0), _env_float("LLM_READ_TIMEOUT", 120.0))


def _default_max_retries() -> int:
    # 기존 기본값 6 은 최악의 경우 7회 시도 × 120초 ≈ 14분 동안 워크아이템이 컨슈머 락을 점유했다.
    # LangChain 기본값(2)에 맞춰 꼬리 지연을 줄인다.
    return _env_int("LLM_MAX_RETRIES", 2)


def get_llm_model(default: str = "gpt-4o") -> str:
    model = (os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "").strip()
    return model or default


def _proxy_base_url() -> str:
    return (
        os.getenv("LLM_PROXY_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or "http://litellm-proxy:4000"
    )


def _is_openrouter_url(url: str) -> bool:
    return "openrouter.ai" in (url or "").lower()


def _proxy_api_key() -> str:
    base_url = _proxy_base_url()
    if _is_openrouter_url(base_url):
        candidates = [
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("LLM_PROXY_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ]
        api_key = next((k for k in candidates if k and k.startswith("sk-or-v1-")), None)
        if not api_key:
            api_key = next((k for k in candidates if k), None)
    else:
        api_key = (
            os.getenv("LLM_PROXY_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

    if not api_key:
        raise RuntimeError(
            "Missing API key. Set `OPENROUTER_API_KEY` when using OpenRouter, "
            "or set `LLM_PROXY_API_KEY` / `OPENAI_API_KEY`."
        )
    return api_key


def create_llm(
    model: Optional[str] = None,
    streaming: bool = False,
    temperature: float = 0.0,
    timeout: Optional[TimeoutType] = None,
    max_retries: Optional[int] = None,
    **kwargs: Any,
):
    # Import lazily to keep module importable in test/CI environments
    # where optional LLM dependencies may not be installed.
    from langchain_openai import ChatOpenAI

    if timeout is None:
        timeout = _default_timeout()
    if max_retries is None:
        max_retries = _default_max_retries()

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
    timeout: Optional[TimeoutType] = None,
    max_retries: Optional[int] = None,
    **kwargs: Any,
):
    # Import lazily to keep module importable in test/CI environments
    # where optional LLM dependencies may not be installed.
    from langchain_openai import OpenAIEmbeddings

    if timeout is None:
        timeout = _default_timeout()
    if max_retries is None:
        max_retries = _default_max_retries()

    if model is None:
        model = os.getenv("LLM_EMBEDDING_MODEL") or os.getenv("OPENAI_EMBEDDING_MODEL")
    if not model:
        raise RuntimeError(
            "Missing embedding model alias. Set `LLM_EMBEDDING_MODEL` "
            "or at least `LLM_MODEL`."
        )

    base_url = kwargs.pop("base_url", None) or _proxy_base_url()
    api_key = kwargs.pop("api_key", None) or _proxy_api_key()

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

