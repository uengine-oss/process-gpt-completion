"""보상(되돌리기) 코드 생성이 작업 이력 전체를 근거로 삼는지 고정한다.

되돌릴 대상은 MCP 호출만이 아니다. 스킬 절차를 셸 스크립트로 수행한 활동은 셸과
파일 조작으로 세상을 바꾼다. 그것들이 생성 프롬프트에 실리지 않으면 되돌릴 방법이
없다. 그리고 생성이 실패해도 워크아이템은 멈추지 않아야 한다.
"""

import ast
import sys
import types

import pytest


@pytest.fixture
def handler(monkeypatch):
    """DB/LLM 의존을 스텁으로 대체한 compensation_handler."""
    captured: dict = {}

    db = types.ModuleType("database")
    db.fetch_tenant_mcp_config = lambda tenant_id: {"mcpServers": {"pg": {"url": "x"}}}
    db.fetch_mcp_python_code = lambda *a, **k: None
    db.upsert_mcp_python_code = lambda record: captured.setdefault("saved", record)
    db.fetch_events_by_proc_inst_id_until_activity = lambda *a, **k: captured.get("events", [])
    db.upsert_workitem = lambda record: captured.setdefault("workitem", record)
    db.fetch_user_info_by_uid = lambda uid: None
    monkeypatch.setitem(sys.modules, "database", db)

    llm = types.ModuleType("llm_factory")

    class _Resp:
        def __init__(self, content):
            self.content = content

    class _LLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return _Resp(captured.get("reply", ""))

    llm.create_llm = lambda **kwargs: _LLM()
    monkeypatch.setitem(sys.modules, "llm_factory", llm)

    sys.modules.pop("compensation_handler", None)
    import compensation_handler as module

    module._captured = captured
    # 실제 MCP 서버 조회는 하지 않는다.
    module.build_tool_index_from_tenant = lambda tenant_id: {"db_exec": "pg"}
    return module


def _entries(handler):
    import work_history as wh

    return wh.normalize_events([
        {"event_type": "tool_usage_finished", "timestamp": "t1",
         "data": {"tool_name": "read_file", "args": {"file_path": "/skills/inventory/SKILL.md"}}},
        {"event_type": "tool_usage_finished", "timestamp": "t2",
         "data": {"tool_name": "execute", "args": {"command": "python3 scripts/load.py --qty 20"}}},
        {"event_type": "tool_usage_finished", "timestamp": "t3",
         "data": {"tool_name": "write_file", "args": {"file_path": "/workspace/out.txt", "content": "x"}}},
        {"event_type": "tool_usage_finished", "timestamp": "t4",
         "data": {"tool_name": "db_exec", "args": {"sql": "UPDATE product SET stock = stock + 20"}}},
    ])


def _generate(handler, reply):
    import work_history as wh

    handler._captured["reply"] = reply
    trace = _entries(handler)
    return handler.generate_deterministic_compensation_code(
        "tn", "재고 20 입고", wh.to_log_entries(trace), wh.summarize(trace)
    )


_VALID_REPLY = '''# -*- coding: utf-8 -*-
import asyncio
from typing import Dict, Any, List


async def run(inputs: Dict[str, Any], timeout_s: int = 60) -> List[Dict[str, Any]]:
    results = []
    for log in reversed(inputs["event_logs"]):
        if log["kind"] == "mcp_call":
            results.append(await call_tool("pg", "db_exec", {"sql": "UPDATE product SET stock = stock - 20"}))
        elif log["kind"] == "shell":
            results.append(await run_shell("python3 scripts/unload.py --qty 20", ""))
    return results
'''


def test_prompt_carries_every_kind_of_work(handler):
    """셸·파일·스킬 읽기가 모두 프롬프트에 실린다. MCP 호출만 보내면 되돌릴 수 없다."""
    _generate(handler, _VALID_REPLY)
    prompt = handler._captured["prompt"]
    assert '"kind": "shell"' in prompt
    assert '"kind": "file_write"' in prompt
    assert '"kind": "skill_read"' in prompt
    assert "/skills/inventory/SKILL.md" in prompt      # 어떤 절차를 따랐는지
    assert "python3 scripts/load.py --qty 20" in prompt


def test_prompt_offers_a_primitive_for_each_kind(handler):
    """골격이 셸·파일 되돌리기 원시 동작을 제공한다는 사실이 지시에 포함된다."""
    _generate(handler, _VALID_REPLY)
    prompt = handler._captured["prompt"]
    for primitive in ("call_tool(", "run_shell(", "write_file(", "read_file("):
        assert primitive in prompt
    assert "CONTEXT ONLY" in prompt  # 읽기는 되돌리지 않는다


def test_valid_generation_is_returned(handler):
    code = _generate(handler, _VALID_REPLY)
    ast.parse(code)
    assert "run_shell(" in code


def test_markdown_fence_is_stripped(handler):
    code = _generate(handler, "```python\n" + _VALID_REPLY + "\n```")
    ast.parse(code)  # 펜스가 남으면 첫 줄에서 문법 오류가 난다


@pytest.mark.parametrize("reply", [
    "죄송합니다. 되돌릴 방법을 찾지 못했습니다.",       # 코드가 아님
    "async def run(inputs):\n    return []",           # 되돌리는 일을 하지 않음
    "async def run(inputs:\n    call_tool(",           # 문법 오류
])
def test_unusable_generation_falls_back_to_a_noop_script(handler, reply):
    """생성이 실패해도 예외를 던지지 않는다 — 워크아이템은 멈추면 안 된다."""
    code = _generate(handler, reply)
    ast.parse(code)
    assert "async def run(" in code
    body = code.split("results = []", 1)[1].split("return results", 1)[0]
    assert body.strip() == "pass"   # 부수효과 없음


def test_llm_failure_falls_back_instead_of_raising(handler, monkeypatch):
    import llm_factory

    def _boom(**kwargs):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(llm_factory, "create_llm", _boom)
    monkeypatch.setattr(handler, "create_llm", _boom)
    code = _generate(handler, _VALID_REPLY)
    ast.parse(code)


@pytest.mark.asyncio
async def test_activity_without_effects_generates_nothing(handler):
    """조회만 한 활동은 되돌릴 것이 없다. 빈 보상 코드를 저장하지 않는다."""
    handler._captured["events"] = [
        {"event_type": "tool_usage_finished", "timestamp": "t1",
         "data": {"tool_name": "db_exec", "args": {"sql": "SELECT 1"}}},
        {"event_type": "tool_usage_finished", "timestamp": "t2",
         "data": {"tool_name": "read_file", "args": {"file_path": "/skills/x/SKILL.md"}}},
    ]

    class _Workitem:
        id = "w1"
        proc_def_id = "pd"
        activity_id = "act"
        tenant_id = "tn"
        proc_inst_id = "pi"
        query = "조회"
        assignees = []
        agent_orch = "deepagents"
        user_id = "u"
        username = "n"

    await handler.generate_compensation(_Workitem(), {"id": "w2"})
    assert "saved" not in handler._captured
