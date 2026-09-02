# -*- coding: utf-8 -*-
"""관측 기반 보상(되돌리기) 생성.

순방향은 관측으로 코드를 만드는데 되돌리기만 LLM이 지어내면, 재작업 경로 전체가 추측
위에 선다. 여기서 고정하는 계약은 셋이다.

1. 되돌릴 수 있는 구문은 관측에서 **정확히** 뒤집는다.
2. 되돌릴 수 없는 것은 지어내지 않고 **되돌릴 수 없다고 말한다**.
3. 하나라도 되돌릴 수 없으면 계획을 만들지 않는다 — 절반만 되돌린 세상은 되돌리지
   않은 세상보다 나쁘다.
"""

import ast

import pytest

from compensation import UNDO_DELETE_FILE, UNDO_RESTORE_FILE, UNDO_TOOL, invert, undo_plan
from deterministic_signature import invert_sql
from deterministic_template import undo_script
from work_history import to_action


def _trace(*calls):
    return [to_action(tool, args, result=result) for tool, args, result in calls]


# --------------------------------------------------------------------------
# SQL 역연산
# --------------------------------------------------------------------------

def test_insert_is_reversed_by_deleting_exactly_what_was_inserted():
    """일부 컬럼만 조건에 넣으면 남의 행까지 지운다. 넣은 값 전부를 조건으로 쓴다."""
    reversed_sql = invert_sql(
        "insert into expense_ledger (tenant_id, applicant, amount) "
        "values ('localhost', '한지민', 29000);"
    )
    assert reversed_sql == (
        "DELETE FROM expense_ledger WHERE tenant_id = 'localhost' "
        "AND applicant = '한지민' AND amount = 29000"
    )


def test_in_place_increment_is_reversed_by_flipping_the_sign():
    assert invert_sql("UPDATE product SET stock = stock + 20 WHERE name = 'x'") == (
        "UPDATE product SET stock = stock - 20 WHERE name = 'x'"
    )
    assert invert_sql("UPDATE product SET stock = stock - 3") == (
        "UPDATE product SET stock = stock + 3"
    )


@pytest.mark.parametrize("sql", [
    "UPDATE product SET stock = 5 WHERE id = 1",   # 이전 값을 모른다
    "DELETE FROM t WHERE id = 1",                  # 지워진 행을 모른다
    "INSERT INTO t (a, b) VALUES (now(), 'x')",    # 값과 컬럼의 짝이 어긋난다
    "DROP TABLE t",
    "SELECT * FROM t",
])
def test_statements_without_a_known_previous_state_are_not_reversed(sql):
    assert invert_sql(sql) is None


# --------------------------------------------------------------------------
# 이력 → 되돌리기 단계
# --------------------------------------------------------------------------

def test_undo_runs_in_reverse_order():
    """마지막에 한 일을 먼저 되돌려야 중간 상태가 앞뒤로 어긋나지 않는다."""
    steps, reasons = invert(_trace(
        ("db_exec", {"sql": "INSERT INTO a (x) VALUES (1)"}, "ok"),
        ("db_exec", {"sql": "INSERT INTO b (y) VALUES (2)"}, "ok"),
    ))
    assert reasons == []
    assert [step.args["sql"] for step in steps] == [
        "DELETE FROM b WHERE y = 2",
        "DELETE FROM a WHERE x = 1",
    ]


def test_a_file_this_run_created_is_removed():
    steps, reasons = invert(_trace(
        ("write_file", {"file_path": "/out/a.md", "content": "문서"}, "ok"),
    ))
    assert reasons == []
    assert steps[0].kind == UNDO_DELETE_FILE and steps[0].path == "/out/a.md"


def test_a_file_read_before_writing_is_restored_not_removed():
    """덮어쓰기 전 내용을 읽어 두었다면 되돌리기는 삭제가 아니라 복원이다."""
    steps, reasons = invert(_trace(
        ("read_file", {"file_path": "/out/a.md"}, {"content": "예전 내용"}),
        ("write_file", {"file_path": "/out/a.md", "content": "새 내용"}, "ok"),
    ))
    assert reasons == []
    assert steps[0].kind == UNDO_RESTORE_FILE
    assert steps[0].content == "예전 내용"


def test_reads_are_never_undone():
    steps, reasons = invert(_trace(
        ("read_file", {"file_path": "/skills/x/SKILL.md"}, "내용"),
        ("db_exec", {"sql": "SELECT 1"}, "[]"),
        ("db_exec", {"sql": "INSERT INTO a (x) VALUES (1)"}, "ok"),
    ))
    assert reasons == []
    assert len(steps) == 1


@pytest.mark.parametrize("call,fragment", [
    (("send_email", {"to": "a@b.c", "body": "본문"}, "ok"), "send_email"),
    (("execute", {"command": "python3 load.py"}, "ok"), "셸 실행"),
    (("db_exec", {"sql": "DELETE FROM t WHERE id = 1"}, "ok"), "되돌릴 수 없는 SQL"),
    (("delete_file", {"file_path": "/out/a.md"}, "ok"), "파일 조작"),
])
def test_irreversible_actions_are_named_not_guessed(call, fragment):
    """지어낸 되돌리기보다 '못 되돌린다'가 낫다 — 사유가 남아야 사람이 판단한다."""
    steps, reasons = invert(_trace(call))
    assert steps == []
    assert any(fragment in reason for reason in reasons)


def test_one_irreversible_action_cancels_the_whole_plan():
    """절반만 되돌린 세상은 되돌리지 않은 세상보다 나쁘다 — 무엇이 남았는지 모른다."""
    assert undo_plan(_trace(
        ("db_exec", {"sql": "INSERT INTO a (x) VALUES (1)"}, "ok"),
        ("send_email", {"to": "a@b.c"}, "ok"),
    )) is None


def test_an_activity_with_nothing_to_undo_has_no_plan():
    assert undo_plan(_trace(("db_exec", {"sql": "SELECT 1"}, "[]"))) is None


# --------------------------------------------------------------------------
# 저장되는 코드
# --------------------------------------------------------------------------

def test_the_stored_undo_code_actually_undoes():
    """되돌리는 일을 실제로 하지 않는 코드는 저장되지 않아야 한다.

    예전에는 생성이 실패하면 아무 일도 하지 않는 골격을 저장했다. 그러면 재작업 때
    되돌리기가 조용히 아무것도 하지 않고, 순방향 코드가 그 위에 덧씌워진다.
    """
    code = undo_script({"db_exec": "pg"})
    ast.parse(code)
    body = code.split("results = []", 1)[1].split("return results", 1)[0]
    assert "await call_tool(" in body
    assert "await remove_file(" in body
    assert "await write_file(" in body


def test_the_stored_undo_code_records_what_it_undid():
    """무엇을 되돌렸는지가 결과에 남아야 사람이 재작업을 안심하고 요청할 수 있다."""
    import asyncio

    code = undo_script({"db_exec": "pg"})
    namespace: dict = {}
    exec(compile(code, "compensation.py", "exec"), namespace)

    calls: list = []

    async def fake_call_tool(server, tool, args, timeout_s=60):
        calls.append((server, tool, args))
        return {"kind": "mcp_call", "tool": tool}

    namespace["call_tool"] = fake_call_tool
    plan = undo_plan(_trace(("db_exec", {"sql": "INSERT INTO a (x) VALUES (1)"}, "ok")))
    results = asyncio.run(namespace["run"](plan))

    assert calls == [("pg", "db_exec", {"sql": "DELETE FROM a WHERE x = 1"})]
    assert results[0]["undone"].startswith("db_exec: DELETE FROM a")


def test_an_unknown_tool_fails_instead_of_skipping():
    """서버를 못 찾으면 조용히 건너뛰지 않는다 — 되돌리지 못한 채 되돌렸다고 하면 안 된다."""
    import asyncio

    code = undo_script({})
    namespace: dict = {}
    exec(compile(code, "compensation.py", "exec"), namespace)
    plan = undo_plan(_trace(("db_exec", {"sql": "INSERT INTO a (x) VALUES (1)"}, "ok")))
    with pytest.raises(RuntimeError):
        asyncio.run(namespace["run"](plan))
