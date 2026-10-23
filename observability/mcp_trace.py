"""MCP 호출 계측 — 공용 래퍼.

엔진마다 이벤트 발행 코드를 흩뿌리는 대신, LangChain BaseTool 을 감싸는
ObservedTool 하나가 유일한 발행처가 된다. 엔진은 툴을 로드하는 지점에서
observe_tools() 로 감싸기만 하면 된다.

원칙:
  - 관측 실패는 절대 툴 호출 실패로 번지지 않는다 (전부 삼킨다).
  - 전송은 fire-and-forget 배치 — 툴 지연에 영향을 주지 않는다.
  - 상관키(trace_id/proc_inst_id/todo_id/job_id/agent_id)를 함께 실어보내
    LiteLLM 콜백이 남기는 LLM 스팬과 같은 trace 로 묶이게 한다.

이 파일은 completion · deepagents · cli-agent 세 곳에 같은 내용으로 놓인다
(세 서비스가 공유하는 패키지가 없기 때문). 수정할 때는 세 곳을 함께 고칠 것.
공통 의존성은 httpx 뿐이며, DB 쓰기는 mcp-hub 가 맡는다.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)

MCP_HUB_URL = os.getenv("MCP_HUB_URL", "http://mcp-hub:8500").rstrip("/")
TRACE_ENABLED = os.getenv("MCP_TRACE_ENABLED", "true").lower() != "false"
FLUSH_INTERVAL = float(os.getenv("MCP_TRACE_FLUSH_SEC", "0.5"))
FLUSH_SIZE = int(os.getenv("MCP_TRACE_FLUSH_SIZE", "25"))
PREVIEW_CHARS = int(os.getenv("MCP_TRACE_PREVIEW_CHARS", "500"))
QUEUE_MAX = int(os.getenv("MCP_TRACE_QUEUE_MAX", "2000"))

# 페이로드(전체 인자·결과) 캡처. 핫 로그의 500자 preview 와 별개로,
# 프로파일러가 볼 원문을 사이드카 테이블에 따로 보낸다.
#   off      : 보내지 않음
#   sampled  : 오류는 항상, 나머지는 SAMPLE_RATE 확률 (기본)
#   on       : 전부
PAYLOAD_CAPTURE = os.getenv("MCP_TRACE_PAYLOAD_CAPTURE", "sampled").lower()
PAYLOAD_SAMPLE_RATE = float(os.getenv("MCP_TRACE_PAYLOAD_SAMPLE_RATE", "0.1"))
PAYLOAD_MAX_BYTES = int(os.getenv("MCP_TRACE_PAYLOAD_MAX_BYTES", str(256 * 1024)))

_MASK_KEYS = ("password", "token", "secret", "authorization", "api_key", "apikey",
              "credential", "passwd", "private_key")

# LangChain 이 툴 호출에 끼워 넣는 실행 배관. 툴의 "입력"이 아니고
# 직렬화도 되지 않으므로 미리보기에서 제외한다.
_PLUMBING_KEYS = ("run_manager", "config", "callbacks", "callback_manager",
                  "tool_call_id", "run_id", "parent_run_id")


# ---------------------------------------------------------------------------
# 컨텍스트
# ---------------------------------------------------------------------------
@dataclass
class MCPContext:
    """한 실행 단위의 상관키."""
    tenant_id: str
    engine: str = "unknown"
    proc_inst_id: str | None = None
    todo_id: str | None = None
    job_id: str | None = None
    agent_id: str | None = None
    user_id: str | None = None
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_span_id: str | None = None
    # 화면에서 "이 실행은 상세 추적" 을 켰을 때 True. 콜백의 샘플링보다 우선한다.
    capture: bool = False

    def child(self, **overrides: Any) -> "MCPContext":
        """서브에이전트용 — trace_id 는 물려받고 parent_span_id 만 바뀐다."""
        return replace(self, **overrides)


_current: contextvars.ContextVar[MCPContext | None] = contextvars.ContextVar(
    "mcp_trace_context", default=None
)

# 지금 진행 중인 LLM 턴의 스팬 id.
# correlation_headers() 가 LLM 요청마다 새로 발급하고, 그 사이에 일어난 툴
# 호출이 이 값을 부모로 삼는다 — "몇 번째 턴이 이 툴을 불렀나" 가 남는다.
_current_llm_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mcp_llm_span", default=None
)


def begin_llm_span() -> str:
    """새 LLM 스팬을 발급하고 현재 스팬으로 세운다."""
    span_id = uuid.uuid4().hex
    _current_llm_span.set(span_id)
    return span_id


def current_llm_span() -> str | None:
    return _current_llm_span.get()


def set_context(ctx: MCPContext | None) -> contextvars.Token:
    return _current.set(ctx)


def get_context() -> MCPContext | None:
    return _current.get()


def correlation_headers() -> dict[str, str]:
    """LLM 호출에 실어보낼 상관키 헤더.

    LiteLLM 프록시 콜백이 이 헤더를 읽어 llm_call_log 에 남긴다.

    여기서 LLM 스팬 id 를 새로 발급한다 — httpx event hook 이 요청마다
    불리므로 "HTTP 요청 1건 = LLM 스팬 1개" 가 되고, 그 뒤에 일어난 툴
    호출은 이 스팬을 부모로 삼는다. 재시도도 별도 스팬이 된다.
    """
    ctx = get_context()
    if not ctx:
        return {}
    payload = {
        "tenant_id": ctx.tenant_id, "engine": ctx.engine,
        "proc_inst_id": ctx.proc_inst_id, "todo_id": ctx.todo_id,
        "job_id": ctx.job_id, "agent_id": ctx.agent_id,
        "user_id": ctx.user_id, "trace_id": ctx.trace_id,
        "span_id": begin_llm_span(),
        # LLM 스팬의 부모는 job 루트다. (직전 턴이 아니다 — 턴끼리는 형제)
        "parent_span_id": ctx.parent_span_id,
        "capture": ctx.capture or None,
    }
    return {"x-processgpt-trace": json.dumps(
        {k: v for k, v in payload.items() if v is not None}, ensure_ascii=False)}


# ---------------------------------------------------------------------------
# 전송
# ---------------------------------------------------------------------------
class _Sink:
    """배치 큐. 큐가 차면 조용히 버리고 카운터만 올린다."""

    def __init__(self, path: str = "/ingest/mcp-calls", label: str = "MCP 트레이스") -> None:
        self.path = path
        self.label = label
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self.dropped = 0
        self.failures = 0

    def _ensure(self) -> asyncio.Queue | None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return None
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=QUEUE_MAX)
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())
        return self._queue

    def submit(self, record: dict) -> None:
        if not TRACE_ENABLED:
            return
        q = self._ensure()
        if q is None:
            return
        try:
            q.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped += 1
            if self.dropped % 100 == 1:
                logger.warning("%s 큐 포화 — %d건 버림", self.label, self.dropped)

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            batch: list[dict] = []
            try:
                batch.append(await self._queue.get())
                deadline = time.monotonic() + FLUSH_INTERVAL
                while len(batch) < FLUSH_SIZE and time.monotonic() < deadline:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._queue.get(), timeout))
                    except asyncio.TimeoutError:
                        break
                await self._post(batch)
            except asyncio.CancelledError:
                if batch:
                    await self._post(batch)
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug("트레이스 전송 루프 오류: %s", exc)

    async def _post(self, batch: list[dict]) -> None:
        if not batch:
            return
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5) as client:
                await client.post(f"{MCP_HUB_URL}{self.path}",
                                  json={"records": batch})
            self.failures = 0
        except Exception as exc:  # noqa: BLE001
            # 관측이 죽어도 업무는 계속되어야 한다. 다만 조용히 묻히면
            # 안 되므로 첫 실패는 눈에 띄게 남긴다.
            self.failures += 1
            log = logger.warning if self.failures == 1 else logger.debug
            log("%s 전송 실패(%d건, 연속 %d회): %s",
                self.label, len(batch), self.failures, exc)

    async def flush(self) -> None:
        q = self._queue
        if q is None:
            return
        pending: list[dict] = []
        while not q.empty():
            try:
                pending.append(q.get_nowait())
            except asyncio.QueueEmpty:
                break
        await self._post(pending)


_sink = _Sink()
# 전체 페이로드는 별도 경로로 보낸다 — 캡처를 꺼도 핫 로그는 계속 흘러야 하고,
# 배치 크기가 20배 달라 플러시 튜닝이 다르다.
_payload_sink = _Sink("/ingest/mcp-payloads", "MCP 페이로드")


async def flush() -> None:
    """실행 종료 직전에 남은 스팬을 밀어낸다."""
    await _sink.flush()
    await _payload_sink.flush()


# ---------------------------------------------------------------------------
# 미리보기 마스킹
# ---------------------------------------------------------------------------
def _mask(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if any(m in str(k).lower() for m in _MASK_KEYS):
                out[k] = "***"
            else:
                out[k] = _mask(v)
        return out
    if isinstance(value, list):
        return [_mask(v) for v in value[:10]]
    if isinstance(value, str) and len(value) > PREVIEW_CHARS:
        return value[:PREVIEW_CHARS] + f"…(+{len(value) - PREVIEW_CHARS}자)"
    return value


def _preview(value: Any) -> dict | None:
    """JSON 으로 안전하게 실려 나갈 수 있는 미리보기를 만든다.

    json.dumps(default=str) 로 한 번 왕복시켜, 직렬화 불가능한 객체가
    끼어 있어도 전송 단계에서 터지지 않게 한다.
    """
    try:
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in _PLUMBING_KEYS}
        masked = _mask(value)
        blob = json.dumps(masked, ensure_ascii=False, default=str)
        if len(blob) > PREVIEW_CHARS * 4:
            return {"_truncated": True, "preview": blob[: PREVIEW_CHARS * 4]}
        safe = json.loads(blob)
        return safe if isinstance(safe, dict) else {"value": safe}
    except Exception:  # noqa: BLE001
        return None


def _full(value: Any) -> tuple[Any, bool, int | None]:
    """프로파일러용 전체 페이로드.

    마스킹과 plumbing 제거는 _preview 와 같지만, 자르는 기준이
    500자가 아니라 PAYLOAD_MAX_BYTES 다. 반환값은 (값, 잘림여부, 원본바이트).
    """
    try:
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in _PLUMBING_KEYS}
        masked = _mask_full(value)
        blob = json.dumps(masked, ensure_ascii=False, default=str)
        orig = len(blob.encode())
        if orig > PAYLOAD_MAX_BYTES:
            return ({"_truncated": True,
                     "preview": blob[:PAYLOAD_MAX_BYTES]}, True, orig)
        safe = json.loads(blob)
        return (safe if isinstance(safe, dict) else {"value": safe}, False, orig)
    except Exception:  # noqa: BLE001
        return (None, False, None)


def _mask_full(value: Any) -> Any:
    """_mask 와 같되 문자열을 자르지 않는다 (자르기는 _full 이 총량으로 한다)."""
    if isinstance(value, dict):
        return {
            k: ("***" if any(m in str(k).lower() for m in _MASK_KEYS) else _mask_full(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_mask_full(v) for v in value]
    return value


def _should_capture(status: str) -> bool:
    if PAYLOAD_CAPTURE == "off":
        return False
    if PAYLOAD_CAPTURE == "on":
        return True
    if status != "success":
        return True          # 실패는 항상 남긴다 — 원인 추적이 목적이므로
    ctx = get_context()
    if ctx is not None and ctx.capture:
        return True          # 화면에서 켠 실행은 샘플링을 건너뛴다
    import random

    return random.random() < PAYLOAD_SAMPLE_RATE


def _size(value: Any) -> int | None:
    try:
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in _PLUMBING_KEYS}
        return len(json.dumps(value, ensure_ascii=False, default=str).encode())
    except Exception:  # noqa: BLE001
        return None


def _classify(exc: BaseException) -> str:
    text = str(exc).lower()
    if isinstance(exc, asyncio.TimeoutError) or "timeout" in text or "timed out" in text:
        return "TIMEOUT"
    if "unauthor" in text or "forbidden" in text or "401" in text or "403" in text:
        return "AUTH"
    if "validation" in text or "invalid" in text or "schema" in text:
        return "VALIDATION"
    return "UPSTREAM"


# ---------------------------------------------------------------------------
# 래퍼
# ---------------------------------------------------------------------------
def _emit(ctx: MCPContext, server_name: str, server_id: str | None, tool_name: str,
          span_id: str, started: datetime, duration_ms: int, status: str,
          error_type: str | None, error_message: str | None,
          input_preview: dict | None, input_bytes: int | None,
          output_preview: dict | None, output_bytes: int | None,
          raw_input: Any = None, raw_output: Any = None) -> None:
    parent = current_llm_span() or ctx.parent_span_id
    _sink.submit({
        "tenant_id": ctx.tenant_id,
        "server_id": server_id,
        "server_name": server_name,
        "tool_name": tool_name,
        "trace_id": ctx.trace_id,
        "span_id": span_id,
        # 직전 LLM 턴이 이 툴을 불렀다. 그 턴이 없으면 job 루트에 붙인다.
        "parent_span_id": current_llm_span() or ctx.parent_span_id,
        "proc_inst_id": ctx.proc_inst_id,
        "todo_id": ctx.todo_id,
        "job_id": ctx.job_id,
        "agent_id": ctx.agent_id,
        "user_id": ctx.user_id,
        "engine": ctx.engine,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "status": status,
        "error_type": error_type,
        "error_message": (error_message or "")[:1000] or None,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "input_preview": input_preview,
        "output_preview": output_preview,
    })

    # 프로파일러용 전체 페이로드는 사이드카로 따로 보낸다.
    if not _should_capture(status):
        return
    tool_input, in_trunc, in_orig = _full(raw_input)
    tool_output, out_trunc, out_orig = _full(raw_output)
    _payload_sink.submit({
        "span_kind": "mcp",
        "span_id": span_id,
        "tenant_id": ctx.tenant_id,
        "trace_id": ctx.trace_id,
        "proc_inst_id": ctx.proc_inst_id,
        "todo_id": ctx.todo_id,
        "job_id": ctx.job_id,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "truncated": in_trunc or out_trunc,
        "orig_bytes": (in_orig or 0) + (out_orig or 0),
    })


def observe_tool(tool: Any, server_name: str, ctx: MCPContext,
                 server_id: str | None = None) -> Any:
    """BaseTool 하나를 감싼다. 실패하면 원본을 그대로 돌려준다."""
    if tool is None or getattr(tool, "_mcp_observed", False):
        return tool

    tool_name = getattr(tool, "name", "unknown")
    original_arun = getattr(tool, "_arun", None)
    original_run = getattr(tool, "_run", None)

    async def _traced_arun(*args: Any, **kwargs: Any) -> Any:
        # 래핑 당시의 ctx 를 클로저에 가두면, 툴 객체가 여러 실행에 재사용될 때
        # 전부 첫 실행의 워크아이템으로 귀속된다. 호출 시점 컨텍스트를 우선한다.
        run_ctx = get_context() or ctx
        span_id = uuid.uuid4().hex
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        payload = kwargs if kwargs else ({"args": list(args)} if args else {})
        in_prev, in_size = _preview(payload), _size(payload)
        try:
            result = await original_arun(*args, **kwargs)
        except BaseException as exc:
            _emit(run_ctx, server_name, server_id, tool_name, span_id, started,
                  int((time.perf_counter() - t0) * 1000),
                  "timeout" if isinstance(exc, asyncio.TimeoutError) else "error",
                  _classify(exc), str(exc), in_prev, in_size, None, None,
                  raw_input=payload)
            raise
        _emit(run_ctx, server_name, server_id, tool_name, span_id, started,
              int((time.perf_counter() - t0) * 1000), "success", None, None,
              in_prev, in_size, _preview(result), _size(result),
              raw_input=payload, raw_output=result)
        return result

    def _traced_run(*args: Any, **kwargs: Any) -> Any:
        run_ctx = get_context() or ctx
        span_id = uuid.uuid4().hex
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        payload = kwargs if kwargs else ({"args": list(args)} if args else {})
        in_prev, in_size = _preview(payload), _size(payload)
        try:
            result = original_run(*args, **kwargs)
        except BaseException as exc:
            _emit(run_ctx, server_name, server_id, tool_name, span_id, started,
                  int((time.perf_counter() - t0) * 1000), "error",
                  _classify(exc), str(exc), in_prev, in_size, None, None,
                  raw_input=payload)
            raise
        _emit(run_ctx, server_name, server_id, tool_name, span_id, started,
              int((time.perf_counter() - t0) * 1000), "success", None, None,
              in_prev, in_size, _preview(result), _size(result),
              raw_input=payload, raw_output=result)
        return result

    # LangChain 은 inspect.signature 로 _arun 이 config / run_manager 를 받는지
    # 판단해 인자를 넘긴다. 래퍼가 (*args, **kwargs) 로 시그니처를 가리면
    # config 가 전달되지 않아 원본 호출이 TypeError 로 깨진다.
    # functools.wraps 가 붙여주는 __wrapped__ 를 inspect 가 따라가므로
    # 원본 시그니처가 그대로 보이게 된다.
    try:
        if original_arun is not None:
            functools.update_wrapper(_traced_arun, original_arun)
            object.__setattr__(tool, "_arun", _traced_arun)
        if original_run is not None:
            functools.update_wrapper(_traced_run, original_run)
            object.__setattr__(tool, "_run", _traced_run)
        object.__setattr__(tool, "_mcp_observed", True)
        object.__setattr__(tool, "_mcp_server_name", server_name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("툴 계측 실패 (%s): %s — 원본을 그대로 사용", tool_name, exc)
        return tool
    return tool


def observe_tools(tools: Iterable[Any], server_name: str, ctx: MCPContext | None,
                  server_id: str | None = None) -> list:
    """서버 단위로 툴 목록을 감싼다.

    ctx 가 없으면(상관키를 모르면) 아무것도 하지 않는다 — 귀속되지 않는
    스팬을 쌓아봐야 분석에 쓸 수 없다.
    """
    tools = list(tools or [])
    if not TRACE_ENABLED or ctx is None:
        return tools
    return [observe_tool(t, server_name, ctx, server_id) for t in tools]
