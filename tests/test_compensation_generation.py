"""보상(되돌리기) 저장 경로가 관측 위에 서는지 고정한다.

예전에는 이력을 LLM에게 보여 주고 되돌리는 코드를 지어내게 했다. 생성이 실패하면 아무
일도 하지 않는 골격을 저장했는데, 그러면 재작업 때 되돌리기가 조용히 아무것도 하지 않고
순방향 코드가 그 위에 덧씌워진다 — 같은 내용이 두 번 반영된다.

지금은 되돌릴 수 있는지를 관측이 정하고, 하나라도 되돌릴 수 없으면 **아무것도 저장하지
않는다**. 저장된 보상이 없으면 실행 런타임이 재작업 전체를 에이전트에게 넘긴다.
"""

import ast
import asyncio
import json
import sys
import types

import pytest


@pytest.fixture
def handler(monkeypatch):
    """DB 의존을 스텁으로 대체한 compensation_handler."""
    captured: dict = {}

    db = types.ModuleType("database")
    db.fetch_tenant_mcp_config = lambda tenant_id: {"mcpServers": {"pg": {"url": "x"}}}
    db.fetch_mcp_python_code = lambda *a, **k: None
    db.upsert_mcp_python_code = lambda record: captured.setdefault("saved", record)
    db.fetch_events_by_todo_id = lambda todo_id: captured.get("events", [])
    db.upsert_workitem = lambda record: captured.setdefault("workitem", record)
    db.fetch_user_info_by_uid = lambda uid: None
    monkeypatch.setitem(sys.modules, "database", db)

    sys.modules.pop("compensation_handler", None)
    import compensation_handler as module

    module._captured = captured
    # 실제 MCP 서버 조회는 하지 않는다.
    module.build_tool_index_from_tenant = lambda tenant_id: {"db_exec": "pg"}
    return module


class _Workitem:
    id = "todo-1"
    proc_def_id = "pd"
    activity_id = "act"
    tenant_id = "tn"
    proc_inst_id = "pi"
    query = "재고 20 입고"
    user_id = "u1"
    username = "홍길동"
    assignees = None
    agent_orch = "deepagents"


def _event(tool, args, timestamp):
    return {"event_type": "tool_usage_finished", "timestamp": timestamp,
            "data": {"tool_name": tool, "args": args}}


def _run(handler, events):
    handler._captured["events"] = events
    asyncio.run(handler.generate_compensation(_Workitem(), {"id": "next"}))
    return handler._captured


def test_a_reversible_history_is_stored(handler):
    captured = _run(handler, [
        _event("read_file", {"file_path": "/skills/inventory/SKILL.md"}, "t1"),
        _event("db_exec", {"sql": "UPDATE product SET stock = stock + 20 WHERE name = 'x'"}, "t2"),
    ])
    saved = captured.get("saved")
    assert saved is not None
    ast.parse(saved["compensation"])
    # 되돌리는 일을 실제로 하는 코드여야 한다.
    body = saved["compensation"].split("results = []", 1)[1].split("return results", 1)[0]
    assert "await call_tool(" in body


def test_the_reworked_workitem_keeps_its_runtime(handler):
    """실행 런타임을 바꾸면 어떤 폴링 워커도 그 워크아이템을 가져가지 않는다."""
    captured = _run(handler, [
        _event("db_exec", {"sql": "INSERT INTO t (a) VALUES (1)"}, "t1"),
    ])
    assert captured["workitem"]["agent_orch"] == "deepagents"
    assert captured["workitem"]["status"] == "IN_PROGRESS"


def test_the_rework_scope_decides_which_activity_starts(handler):
    """재작업 범위가 뒤 단계를 TODO로 만들었으면 그대로 둬야 한다.

    여기서 IN_PROGRESS로 못박으면 앞 단계가 다시 끝나기도 전에 뒤 단계가 옛 입력으로
    돌아간다. 되돌릴 수 있는 활동만 그렇게 되므로 순서가 보상 유무에 따라 갈린다.
    """
    handler._captured["events"] = [_event("db_exec", {"sql": "INSERT INTO t (a) VALUES (1)"}, "t1")]
    asyncio.run(handler.generate_compensation(_Workitem(), {"id": "next", "status": "TODO"}))
    assert handler._captured["workitem"]["status"] == "TODO"


@pytest.mark.parametrize("events,why", [
    ([_event("execute", {"command": "python3 scripts/load.py --qty 20"}, "t1")],
     "셸 실행은 되돌릴 방법이 이력에 없다"),
    ([_event("send_email", {"to": "a@b.c", "body": "본문"}, "t1")],
     "나간 메일은 되돌릴 수 없다"),
    ([_event("db_exec", {"sql": "DELETE FROM t WHERE id = 1"}, "t1")],
     "지워진 행을 모른다"),
    ([_event("db_exec", {"sql": "INSERT INTO t (a) VALUES (1)"}, "t1"),
      _event("execute", {"command": "python3 load.py"}, "t2")],
     "하나라도 되돌릴 수 없으면 전체를 포기한다"),
])
def test_an_irreversible_history_is_not_stored(handler, events, why):
    captured = _run(handler, events)
    assert "saved" not in captured, why
    assert "workitem" not in captured, "되돌리지도 못하면서 재작업을 시작시키면 안 된다"


def _frozen_run_event(results, timestamp="t1"):
    """고착화 실행이 남기는 완료 이벤트. 도구 이벤트가 아니라 결과 봉투 하나다."""
    return {
        "event_type": "task_completed",
        "timestamp": timestamp,
        "data": json.dumps(
            {"ok": True, "execution_mode": "deterministic", "llm_calls": 0,
             "undo_results": [], "results": results},
            ensure_ascii=False,
        ),
    }


def test_a_frozen_activity_can_still_get_a_compensation(handler):
    """한 번 굳으면 이후 모든 실행이 결과 봉투만 남긴다.

    그 봉투를 못 읽으면 "되돌릴 부수효과가 없는 활동"으로 보여 보상이 영영 만들어지지
    않는다. 그러면 재작업 때 되돌리기가 조용히 건너뛰어지고 같은 행이 두 번 들어간다.
    """
    captured = _run(handler, [_frozen_run_event([
        {"kind": "mcp_call", "tool": "db_exec", "server": "pg",
         "args": {"sql": "INSERT INTO ledger (applicant) VALUES ('김철수')"}, "data": None},
    ])])
    assert captured.get("saved") is not None
    ast.parse(captured["saved"]["compensation"])


def test_a_frozen_run_without_recorded_arguments_is_not_read_as_harmless(handler):
    """무엇을 넣었는지 모르는 호출은 되돌릴 수 없다 — 없었던 일로 읽으면 안 된다."""
    captured = _run(handler, [_frozen_run_event([
        {"kind": "mcp_call", "tool": "db_exec", "data": None},
    ])])
    assert "saved" not in captured
    assert "workitem" not in captured


def test_an_activity_without_effects_is_skipped(handler):
    captured = _run(handler, [_event("db_exec", {"sql": "SELECT * FROM t"}, "t1")])
    assert "saved" not in captured


def test_no_events_means_nothing_to_do(handler):
    assert _run(handler, []) .get("saved") is None


def test_only_this_workitems_history_is_considered(handler):
    """앞 액티비티의 부수효과는 이 재작업의 되돌리기 대상이 아니다.

    인스턴스 전체를 가져오면 두 방향으로 틀린다 — 남의 셸이 되돌릴 수 없으면 멀쩡한
    활동의 보상이 막히고, 되돌릴 수 있으면 재작업하지도 않을 앞 액티비티의 INSERT까지
    지운다. 실행 런타임도 같은 액티비티의 이전 회차만 보고 되돌린다.
    """
    asked: dict = {}

    def _events(todo_id):
        asked["todo"] = todo_id
        return []

    handler.fetch_events_by_todo_id = _events
    asyncio.run(handler.generate_compensation(_Workitem(), {"id": "next"}))
    assert asked["todo"] == "todo-1"
