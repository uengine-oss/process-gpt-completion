"""LLM 호출에 상관키를 붙이는 httpx 클라이언트.

ChatOpenAI 의 default_headers 는 생성 시점에 고정된다. LLM 객체가 여러
워크아이템에 걸쳐 재사용되면 헤더가 낡은 trace 를 가리키게 되므로,
httpx event hook 으로 매 요청마다 지금의 상관키를 붙인다.

LiteLLM 프록시의 processgpt_callback 이 이 헤더를 읽어 llm_call_log 에
남기고, 그래서 LLM 스팬이 MCP 스팬과 같은 trace_id 로 묶인다.
"""
from __future__ import annotations

import logging

from .mcp_trace import correlation_headers

logger = logging.getLogger(__name__)

_HEADER = "x-processgpt-trace"


def _attach(request) -> None:
    try:
        for key, value in correlation_headers().items():
            request.headers[key] = value
    except Exception as exc:  # noqa: BLE001
        logger.debug("상관키 헤더 주입 실패: %s", exc)


async def _attach_async(request) -> None:
    """AsyncClient 의 훅은 **awaitable 이어야 한다.**

    동기 함수를 걸면 httpx 가 그 반환값 None 을 await 하고, 오류는
    `TypeError: object NoneType can't be used in 'await' expression` 으로 났다가
    OpenAI SDK 안에서 `APIConnectionError: Connection error.` 로 덮인다 —
    네트워크 문제처럼 보이지만 실제로는 계측 훅이 원인이다.
    동기 클라이언트만 쓰는 경로는 멀쩡해서 오래 드러나지 않았다.
    """
    _attach(request)


def tracing_http_clients() -> dict:
    """ChatOpenAI 에 넘길 http_client / http_async_client.

    httpx 가 없거나 클라이언트를 못 만들면 빈 dict 를 돌려준다 —
    계측 때문에 LLM 호출이 막히면 안 된다.
    """
    try:
        import httpx

        return {
            "http_client": httpx.Client(
                event_hooks={"request": [_attach]}, timeout=120.0),
            "http_async_client": httpx.AsyncClient(
                event_hooks={"request": [_attach_async]}, timeout=120.0),
        }
    except Exception as exc:  # noqa: BLE001
        logger.debug("트레이싱 HTTP 클라이언트 생성 실패: %s", exc)
        return {}
