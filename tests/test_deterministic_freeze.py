"""고착화(결정론적 코드 생성) 트리거와 실행 지문 검증.

핵심 계약 세 가지를 고정한다.
1. SQL 판별과 읽기 전용 제외가 **도구 이름에 의존하지 않는다**. MCP 구성마다 SQL
   실행 도구의 이름이 다르므로 `execute_sql` 같은 이름으로 분기하면 안 된다.
2. 실행 지문이 구조를 구분한다. 리터럴만 다른 실행은 같고, 대상 테이블이 다르면 다르다.
3. 파라미터 식별이 LLM 추측이 아니라 표본 대조(관측)로 이루어진다.
"""

import ast
import asyncio
import sys
import types

import pytest

from deterministic_signature import (
    execution_fingerprint,
    identify_parameters,
    is_readonly_sql,
    is_write_call,
    looks_like_sql,
    normalize_sql,
)


# --------------------------------------------------------------------------
# SQL 판별 — 도구 이름 비의존
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("UPDATE product SET a = 1", True),
    ("  -- 주석\n  SELECT 1", True),
    ("WITH c AS (SELECT 1) INSERT INTO t SELECT * FROM c", True),
    ("안녕하세요", False),
    (42, False),
    (None, False),
])
def test_sql_detection_uses_content(value, expected):
    assert looks_like_sql(value) is expected


def test_write_cte_is_not_readonly():
    """`WITH ... INSERT`는 SELECT로 시작하지만 부수효과가 있다."""
    assert is_readonly_sql("SELECT * FROM t") is True
    assert is_readonly_sql("WITH c AS (SELECT 1) INSERT INTO t SELECT * FROM c") is False


def test_write_call_ignores_tool_name():
    """도구 이름이 execute_sql이 아니어도 인자 내용으로 판정한다."""
    excluded = {"mem0"}
    assert is_write_call("db_exec", {"sql": "UPDATE t SET a = 1"}, excluded) is True
    assert is_write_call("db_exec", {"sql": "SELECT * FROM t"}, excluded) is False
    assert is_write_call("postgres_query", {"query": "DELETE FROM t"}, excluded) is True
    assert is_write_call("mem0", {"sql": "UPDATE t SET a = 1"}, excluded) is False
    # SQL 인자가 없는 도구는 부수효과가 있다고 보수적으로 가정한다.
    assert is_write_call("send_email", {"to": "a@b.com"}, excluded) is True


# --------------------------------------------------------------------------
# 실행 지문
# --------------------------------------------------------------------------

def test_literals_are_normalized_but_identifiers_are_not():
    a = normalize_sql("UPDATE product SET s = s + 20 WHERE name = '노트북'")
    b = normalize_sql("update  product  set s = s + 50 where name = '마우스'")
    assert a.signature == b.signature
    assert [lit.value for lit in a.literals] == [20, "노트북"]


def test_double_quoted_identifier_is_preserved():
    """PostgreSQL에서 큰따옴표는 식별자다. 지우면 다른 테이블이 같은 지문이 된다."""
    a = normalize_sql('UPDATE "Order" SET a = 1')
    b = normalize_sql('UPDATE "Item" SET a = 1')
    assert a.signature != b.signature
    assert '"Order"' in a.signature


def test_different_tables_produce_different_fingerprints():
    x = execution_fingerprint([("db_exec", {"sql": "UPDATE product SET s = 1 WHERE n = 'a'"})])
    y = execution_fingerprint([("db_exec", {"sql": "UPDATE orders SET s = 1 WHERE n = 'a'"})])
    assert x != y


def test_in_list_length_is_absorbed():
    a = normalize_sql("DELETE FROM t WHERE id IN (1, 2, 3)").signature
    b = normalize_sql("DELETE FROM t WHERE id IN (7, 8)").signature
    assert a == b


def test_call_order_matters():
    up = ("db_exec", {"sql": "UPDATE t SET a = 1"})
    ins = ("db_exec", {"sql": "INSERT INTO log VALUES (1)"})
    assert execution_fingerprint([up, ins]) != execution_fingerprint([ins, up])


# --------------------------------------------------------------------------
# 파라미터 식별 — 관측 기반
# --------------------------------------------------------------------------

def _sample(qty, name):
    return [
        ("db_exec", {"sql": f"UPDATE product SET stock = stock + {qty} "
                            f"WHERE product_name = '{name}' AND region = 'KR'"}),
        ("db_exec", {"sql": f"INSERT INTO inventory_log (product_name, reason) "
                            f"VALUES ('{name}', '입고')"}),
    ]


def test_varying_positions_become_parameters_constants_stay():
    samples = [_sample(20, "노트북"), _sample(50, "마우스"), _sample(35, "키보드")]
    plan = identify_parameters(samples)
    names = {p["name"] for p in plan.parameters}
    assert names == {"stock", "product_name"}
    # region='KR'과 '입고'는 모든 표본에서 같으므로 상수로 남는다.
    assert not any(p["example"] in ("KR", "입고") for p in plan.parameters)


def test_positions_sharing_a_value_collapse_into_one_parameter():
    """상품명이 두 호출에 나타나도 값이 늘 같으면 파라미터 하나로 묶인다."""
    samples = [_sample(20, "노트북"), _sample(50, "마우스"), _sample(35, "키보드")]
    plan = identify_parameters(samples)
    product_slots = [s for s in plan.slots if s.name == "product_name"]
    assert {s.call_index for s in product_slots} == {0, 1}


def test_parameter_types_are_inferred_from_literals():
    samples = [_sample(20, "노트북"), _sample(50, "마우스"), _sample(35, "키보드")]
    plan = identify_parameters(samples)
    by_name = {p["name"]: p["type"] for p in plan.parameters}
    assert by_name["stock"] == "integer"
    assert by_name["product_name"] == "string"


# --------------------------------------------------------------------------
# 코드 생성 + 트리거
# --------------------------------------------------------------------------

@pytest.fixture
def generator(monkeypatch):
    """DB/MCP 의존을 스텁으로 대체한 deterministic_generator."""
    events: dict = {}
    workitems: list = []
    saved: list = []

    db = types.ModuleType("database")
    db.fetch_mcp_python_code = lambda *a, **k: None
    db.upsert_mcp_python_code = saved.append
    db.fetch_last_deactivated_at = lambda *a, **k: None
    db.fetch_workitems_by_activity = lambda *a, **k: workitems
    db.fetch_events_by_todo_id = lambda todo_id: events.get(todo_id, [])
    monkeypatch.setitem(sys.modules, "database", db)

    index = types.ModuleType("mcp_tool_index")
    index.build_tool_index_from_tenant = lambda tenant_id, timeout_s=None: {"db_exec": "pg"}
    monkeypatch.setitem(sys.modules, "mcp_tool_index", index)

    for name in list(sys.modules):
        if name == "deterministic_generator":
            del sys.modules[name]
    import deterministic_generator as module

    module._test_events = events
    module._test_workitems = workitems
    module._test_saved = saved
    return module


class _Workitem:
    id = "todo-x"
    proc_def_id = "pd"
    activity_id = "act"
    tenant_id = "tn"


def _add_run(module, todo_id, sql, timestamp, rework=0, instance=None):
    module._test_events[todo_id] = [
        {"event_type": "tool_usage_finished", "timestamp": timestamp,
         "data": {"tool_name": "db_exec", "args": {"sql": sql}}},
    ]
    module._test_workitems.append({
        "id": todo_id,
        "proc_inst_id": instance or f"pi-{todo_id}",
        "rework_count": rework,
        "updated_at": timestamp,
    })


def test_freeze_requires_three_samples(generator):
    _add_run(generator, "a", "UPDATE t SET s = s + 1 WHERE n = 'x'", "2026-08-01")
    _add_run(generator, "b", "UPDATE t SET s = s + 2 WHERE n = 'y'", "2026-08-02")
    assert generator.try_freeze(_Workitem()) is None


def test_freeze_requires_matching_fingerprints(generator):
    _add_run(generator, "a", "UPDATE t SET s = s + 1 WHERE n = 'x'", "2026-08-01")
    _add_run(generator, "b", "UPDATE t SET s = s + 2 WHERE n = 'y'", "2026-08-02")
    _add_run(generator, "c", "UPDATE orders SET s = s + 3 WHERE n = 'z'", "2026-08-03")
    assert generator.try_freeze(_Workitem()) is None


def test_freeze_succeeds_on_three_matching_samples(generator):
    for i, (todo, name) in enumerate([("a", "x"), ("b", "y"), ("c", "z")], start=1):
        _add_run(generator, todo, f"UPDATE t SET s = s + {i} WHERE n = '{name}'", f"2026-08-0{i}")
    record = generator.try_freeze(_Workitem())
    assert record is not None
    ast.parse(record["code"])
    assert generator._test_saved == [record]


def test_reworked_runs_are_excluded_from_samples(generator):
    # pi1은 재작업이 이어졌으므로 그 첫 시도는 표본에서 빠져 유효 표본이 2건뿐이다.
    _add_run(generator, "a", "UPDATE t SET s = s + 1 WHERE n = 'x'", "2026-08-01", 0, "pi1")
    _add_run(generator, "b", "UPDATE t SET s = s + 2 WHERE n = 'y'", "2026-08-02", 1, "pi1")
    _add_run(generator, "c", "UPDATE t SET s = s + 3 WHERE n = 'z'", "2026-08-03", 0, "pi2")
    assert generator.try_freeze(_Workitem()) is None


def test_readonly_calls_are_not_frozen(generator):
    for i, todo in enumerate(["a", "b", "c"], start=1):
        generator._test_events[todo] = [
            {"event_type": "tool_usage_finished", "timestamp": f"2026-08-0{i}",
             "data": {"tool_name": "db_exec", "args": {"sql": "SELECT * FROM t"}}},
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{i}",
        })
    assert generator.try_freeze(_Workitem()) is None


def test_generated_code_substitutes_parameters_and_keeps_constants(generator):
    for i, name in enumerate(["노트북", "마우스", "키보드"], start=1):
        generator._test_events[f"t{i}"] = [
            {"event_type": "tool_usage_finished", "timestamp": f"2026-08-0{i}T00:01",
             "data": {"tool_name": "db_exec", "args": {
                 "sql": f"UPDATE product SET stock = stock + {i * 10} "
                        f"WHERE product_name = '{name}' AND region = 'KR'"}}},
        ]
        generator._test_workitems.append({
            "id": f"t{i}", "proc_inst_id": f"pi{i}", "rework_count": 0,
            "updated_at": f"2026-08-0{i}",
        })

    record = generator.try_freeze(_Workitem())
    assert record is not None

    namespace: dict = {}
    exec(compile(record["code"], "generated.py", "exec"), namespace)
    captured = []

    async def fake_call_tool(server, tool, args, timeout_s=60):
        captured.append(args)
        return {"tool": tool}

    namespace["call_tool"] = fake_call_tool
    asyncio.run(namespace["run"]({"stock": 99, "product_name": "모니터"}))

    sql = captured[0]["sql"]
    assert "stock + 99" in sql          # 파라미터가 새 값으로 치환된다
    assert "'모니터'" in sql             # 문자열 파라미터도 치환된다
    assert "region = 'KR'" in sql       # 상수는 코드에 박힌 채 남는다


def test_event_order_follows_timestamp_not_query_order(generator):
    """이벤트 조회는 최신순이지만 지문은 시간 오름차순이어야 한다."""
    for i in range(1, 4):
        generator._test_events[f"t{i}"] = [
            {"event_type": "tool_usage_finished", "timestamp": f"2026-08-0{i}T00:02",
             "data": {"tool_name": "db_exec", "args": {"sql": f"INSERT INTO log VALUES ({i})"}}},
            {"event_type": "tool_usage_finished", "timestamp": f"2026-08-0{i}T00:01",
             "data": {"tool_name": "db_exec", "args": {"sql": f"UPDATE t SET s = {i}"}}},
        ]
        generator._test_workitems.append({
            "id": f"t{i}", "proc_inst_id": f"pi{i}", "rework_count": 0,
            "updated_at": f"2026-08-0{i}",
        })

    record = generator.try_freeze(_Workitem())
    assert record is not None
    assert record["code"].index("UPDATE t SET") < record["code"].index("INSERT INTO log")


# --------------------------------------------------------------------------
# 작업 이력 전체를 읽는다 — MCP 호출만이 아니라 셸·파일·스킬까지
# --------------------------------------------------------------------------

def _event(tool, args, timestamp):
    return {"event_type": "tool_usage_finished", "timestamp": timestamp,
            "data": {"tool_name": tool, "args": args}}


def _mixed_run(generator, todo, index, qty, name, extra_reads=0):
    """스킬을 읽고 → 셸 스크립트를 돌리고 → 파일을 쓰고 → DB를 갱신한 실행 한 건."""
    events = [
        _event("read_file", {"file_path": "/skills/inventory/SKILL.md"}, f"{todo}-01"),
    ]
    # 맥락 행위(조회)의 횟수는 실행마다 달라진다. 그래도 고착화는 성립해야 한다.
    for k in range(extra_reads):
        events.append(_event("ls", {"path": f"/workspace/dir{k}"}, f"{todo}-02-{k}"))
    events += [
        _event("execute", {"command": f"python3 scripts/load.py --qty {qty} --name {name}"}, f"{todo}-03"),
        _event("write_file", {"file_path": "/workspace/report.txt", "content": f"입고 {qty}건"}, f"{todo}-04"),
        _event("db_exec", {"sql": f"UPDATE product SET stock = stock + {qty} "
                                  f"WHERE product_name = '{name}' AND region = 'KR'"}, f"{todo}-05"),
    ]
    generator._test_events[todo] = events
    generator._test_workitems.append({
        "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
        "updated_at": f"2026-08-0{index}",
    })


def _freeze_mixed(generator):
    for index, (todo, qty, name) in enumerate(
        [("m1", 10, "노트북"), ("m2", 20, "마우스"), ("m3", 30, "키보드")], start=1
    ):
        _mixed_run(generator, todo, index, qty, name, extra_reads=index)
    return generator.try_freeze(_Workitem())


def test_shell_and_file_work_is_frozen_not_only_mcp_calls(generator):
    record = _freeze_mixed(generator)
    assert record is not None
    code = record["code"]
    ast.parse(code)
    assert "await run_shell(" in code       # 셸 스크립트 실행이 코드에 남는다
    assert "await write_file(" in code      # 파일 생성도 남는다
    assert "await call_tool(" in code       # MCP 호출도 남는다


def test_context_only_actions_do_not_become_steps(generator):
    """스킬·파일 읽기는 되돌릴 것도 재현할 것도 아니다. 단계로 굳지 않는다."""
    record = _freeze_mixed(generator)
    steps = record["code"].split("results = []", 1)[1].split("return results", 1)[0]
    assert "SKILL.md" not in steps
    assert "ls" not in steps.replace("results", "")


def test_context_actions_are_recorded_as_provenance(generator):
    """읽은 스킬은 단계가 아니라 출처로 남는다 — 나중에 코드를 의심할 근거가 된다."""
    record = _freeze_mixed(generator)
    assert record["work_history"]["skills_read"] == ["/skills/inventory/SKILL.md"]
    assert record["work_history"]["by_kind"]["skill_read"] == 1
    assert record["work_history"]["sample_count"] == 3
    assert "/skills/inventory/SKILL.md" in record["code"]  # 헤더 주석


def test_varying_context_counts_do_not_block_freezing(generator):
    """조회 횟수는 실행마다 다르다. 그것까지 일치를 요구하면 아무것도 고착화되지 않는다."""
    assert _freeze_mixed(generator) is not None


def test_shell_command_is_parameterized_by_token(generator):
    """명령 전체가 아니라 값이 변한 토큰만 파라미터가 된다."""
    record = _freeze_mixed(generator)
    names = {p["name"] for p in record["parameters"]["parameters"]}
    assert {"qty", "name"} <= names  # `--qty`/`--name` 옵션 이름에서 딴다
    steps = record["code"].split("results = []", 1)[1].split("return results", 1)[0]
    assert steps.count("python3 scripts/load.py") == 1  # 명령 본체는 상수로 굳는다
    assert "--qty ${qty} --name ${name}" in steps


def test_generated_mixed_code_runs_with_new_inputs(generator):
    """생성된 코드가 입력값을 받는 일반화된 형태로 실제 실행된다."""
    record = _freeze_mixed(generator)
    namespace: dict = {}
    exec(compile(record["code"], "generated.py", "exec"), namespace)

    calls: list = []

    async def fake_call_tool(server, tool, args, timeout_s=60):
        calls.append(("mcp", args))
        return {"tool": tool}

    async def fake_run_shell(command, cwd="", timeout_s=300):
        calls.append(("shell", command))
        return {"command": command}

    async def fake_write_file(path, content):
        calls.append(("file", path, content))
        return {"path": path}

    namespace["call_tool"] = fake_call_tool
    namespace["run_shell"] = fake_run_shell
    namespace["write_file"] = fake_write_file

    inputs = {p["name"]: p["example"] for p in record["parameters"]["parameters"]}
    inputs.update({"qty": 77, "name": "모니터", "stock": 77, "product_name": "모니터"})
    asyncio.run(namespace["run"](inputs))

    kinds = [c[0] for c in calls]
    assert kinds == ["shell", "file", "mcp"]          # 관측된 순서 그대로 재현된다
    assert "--qty 77 --name 모니터" in calls[0][1]      # 셸 파라미터가 치환된다
    assert "stock + 77" in calls[2][1]["sql"]         # SQL 파라미터도 치환된다
    assert "region = 'KR'" in calls[2][1]["sql"]      # 상수는 그대로 굳는다


def test_freeze_is_skipped_when_shell_commands_differ_structurally(generator):
    """구조가 다른 실행은 고착화하지 않는다 — 재현 대상이 하나로 정해지지 않는다."""
    for index, (todo, command) in enumerate(
        [("s1", "python3 a.py"), ("s2", "python3 b.py"), ("s3", "bash c.sh --force")], start=1
    ):
        generator._test_events[todo] = [_event("execute", {"command": command}, f"{todo}-01")]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}",
        })
    assert generator.try_freeze(_Workitem()) is None


def test_dollar_in_observed_content_is_escaped(generator):
    """관측된 원문의 `$`가 파라미터 치환 지시로 오해되면 실행이 깨진다."""
    for index, (todo, qty) in enumerate([("d1", 1), ("d2", 2), ("d3", 3)], start=1):
        generator._test_events[todo] = [
            _event("execute", {"command": f"echo $HOME && python3 load.py --qty {qty}"}, f"{todo}-01"),
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}",
        })
    record = generator.try_freeze(_Workitem())
    assert record is not None

    namespace: dict = {}
    exec(compile(record["code"], "generated.py", "exec"), namespace)
    seen: list = []

    async def fake_run_shell(command, cwd="", timeout_s=300):
        seen.append(command)
        return {}

    namespace["run_shell"] = fake_run_shell
    asyncio.run(namespace["run"]({"qty": 9}))
    assert seen == ["echo $HOME && python3 load.py --qty 9"]


def test_one_value_appearing_across_kinds_becomes_one_parameter(generator):
    """같은 값이 셸 인자·파일 내용·SQL 리터럴에 나타나면 파라미터 하나로 묶인다.

    묶지 않으면 호출자가 같은 값을 이름만 다르게 서너 번 넘겨야 한다. 자리마다
    표현이 다를 뿐(셸에서는 `"20"`, SQL에서는 `20`) 하나의 입력이다.
    """
    for index, (todo, qty, name) in enumerate(
        [("u1", 10, "노트북"), ("u2", 20, "마우스"), ("u3", 30, "키보드")], start=1
    ):
        generator._test_events[todo] = [
            _event("execute", {"command": f"python3 load.py --qty {qty} --name {name}"}, f"{todo}-1"),
            _event("write_file", {"file_path": "/workspace/report.md",
                                  "content": f"# 보고\n{name}: {qty}"}, f"{todo}-2"),
            _event("db_exec", {"sql": f"UPDATE product SET stock = stock + {qty} "
                                      f"WHERE product_name = '{name}'"}, f"{todo}-3"),
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}",
        })

    record = generator.try_freeze(_Workitem())
    assert record is not None
    by_name = {p["name"]: p for p in record["parameters"]["parameters"]}
    assert set(by_name) == {"qty", "name"}
    assert by_name["qty"]["type"] == "integer"

    namespace: dict = {}
    exec(compile(record["code"], "generated.py", "exec"), namespace)
    seen: list = []

    async def _record(*args, **kwargs):
        seen.append(args)
        return {}

    namespace["run_shell"] = _record
    namespace["write_file"] = _record
    namespace["call_tool"] = _record
    asyncio.run(namespace["run"]({"qty": 7, "name": "모니터"}))

    assert seen[0][0] == "python3 load.py --qty 7 --name 모니터"
    assert seen[1][1] == "# 보고\n모니터: 7"
    assert seen[2][2]["sql"] == "UPDATE product SET stock = stock + 7 WHERE product_name = '모니터'"


# --------------------------------------------------------------------------
# 파라미터 이름표 — 다음 실행에서 값을 되찾는 열쇠
# --------------------------------------------------------------------------

def _labelled_runs(generator, queries):
    """지시문만 다른 같은 구조의 실행 3건."""
    for index, (todo, qty, name, query) in enumerate(queries, start=1):
        generator._test_events[todo] = [
            _event("execute", {"command": f"printf '%s' {qty} > loaded_{name}.txt"}, f"{todo}-1"),
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}", "query": query,
        })
    return generator.try_freeze(_Workitem())


def test_parameter_label_is_observed_from_the_workitem_instruction(generator):
    """자리 번호에서 나온 이름만으로는 다음 실행에서 값을 찾을 수 없다.

    값이 관측될 때 지시문에서 그 앞에 무엇이 적혀 있었는지를 함께 남긴다.
    """
    record = _labelled_runs(generator, [
        ("l1", 10, "노트북", "상품명은 노트북, 입고 수량은 10"),
        ("l2", 20, "마우스", "상품명은 마우스, 입고 수량은 20"),
        ("l3", 30, "키보드", "상품명은 키보드, 입고 수량은 30"),
    ])
    assert record is not None
    by_label = {p.get("label"): p for p in record["parameters"]["parameters"]}
    assert "수량" in by_label and by_label["수량"]["type"] == "integer"
    assert "상품명" in by_label
    assert all(p["label_position"] == "before" for p in by_label.values())


def test_label_after_the_value_is_also_observed(generator):
    """`Galaxy 상품의` 처럼 이름표가 값 뒤에 붙는 표현도 잡는다."""
    record = _labelled_runs(generator, [
        ("a1", 10, "노트북", "노트북 상품의 입고 수량은 10"),
        ("a2", 20, "마우스", "마우스 상품의 입고 수량은 20"),
        ("a3", 30, "키보드", "키보드 상품의 입고 수량은 30"),
    ])
    assert record is not None
    labels = {p.get("label"): p.get("label_position") for p in record["parameters"]["parameters"]}
    assert labels.get("상품") == "after"     # 값 뒤에 붙은 이름표
    assert labels.get("수량") == "before"    # 값 앞에 붙은 이름표


def test_label_is_dropped_when_it_does_not_hold_across_samples(generator):
    """한 표본에서만 맞아떨어진 이름표를 믿으면 다음 실행에서 엉뚱한 값을 넣는다."""
    record = _labelled_runs(generator, [
        ("v1", 10, "노트북", "상품명은 노트북, 수량은 10"),
        ("v2", 20, "마우스", "제품은 마우스, 개수는 20"),
        ("v3", 30, "키보드", "품목은 키보드, 총량은 30"),
    ])
    assert record is not None
    assert all("label" not in p for p in record["parameters"]["parameters"])


def test_colliding_labels_are_dropped(generator):
    """이름표가 겹치면 어느 파라미터인지 정해지지 않는다. 확신 있게 틀리는 것보다 낫다."""
    record = _labelled_runs(generator, [
        ("c1", 10, "10", "수량은 10, 수량은 10"),
        ("c2", 20, "20", "수량은 20, 수량은 20"),
        ("c3", 30, "30", "수량은 30, 수량은 30"),
    ])
    if record is None:
        pytest.skip("표본 구조가 달라 고착화되지 않음")
    labels = [p.get("label") for p in record["parameters"]["parameters"] if p.get("label")]
    assert len(labels) == len(set(labels))


# --------------------------------------------------------------------------
# 파일 조작 — 조작 종류마다 대응하는 동작이 있어야 한다
# --------------------------------------------------------------------------

def _file_runs(generator, steps_per_run):
    for index, steps in enumerate(steps_per_run, start=1):
        todo = f"f{index}"
        generator._test_events[todo] = [
            _event(tool, args, f"{todo}-{k}") for k, (tool, args) in enumerate(steps)
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}", "query": "",
        })
    return generator.try_freeze(_Workitem())


def test_delete_is_not_compiled_as_a_write(generator):
    """삭제를 쓰기로 뭉뚱그리면 지우는 대신 빈 파일을 만든다 — 조용히 틀린다."""
    record = _file_runs(generator, [
        [("delete_file", {"file_path": f"/tmp/report_{n}.csv"})] for n in ("a", "b", "c")
    ])
    assert record is not None
    steps = record["code"].split("results = []", 1)[1].split("return results", 1)[0]
    assert "remove_file(" in steps
    assert "write_file(" not in steps


def test_mkdir_and_move_use_their_own_primitives(generator):
    record = _file_runs(generator, [
        [("mkdir", {"path": f"/tmp/out_{n}"})] for n in ("a", "b", "c")
    ])
    assert record is not None and "make_dir(" in record["code"]

    generator._test_events.clear()
    generator._test_workitems.clear()
    record = _file_runs(generator, [
        [("move_file", {"source": f"/tmp/in_{n}.csv", "destination": f"/tmp/done_{n}.csv"})]
        for n in ("a", "b", "c")
    ])
    assert record is not None
    assert "move_file(" in record["code"].split("results = []", 1)[1]


def test_file_write_without_a_target_is_refused_not_mislabelled(generator):
    """경로를 못 찾은 파일 쓰기는 재현할 수 없다.

    예전에는 이런 호출이 `mcp_call`로 새어 "MCP 서버를 못 찾았다"는 엉뚱한 사유로
    막혔다. 되돌릴 대상에서도 파일이 빠졌다. 지금은 파일 조작으로 인식하되, 대상을
    모르므로 고착화를 거부한다 — 틀린 코드보다 없는 편이 낫다.
    """
    import work_history as wh

    action = wh.to_action("write_file", {"content": "보고", "session_id": "s1"})
    assert action.kind == wh.FILE_WRITE  # 되돌릴 대상으로는 보인다

    record = _file_runs(generator, [
        [("execute", {"command": f"echo {n} > out.txt"}),
         ("write_file", {"content": f"보고 {n}", "session_id": "s1"})]
        for n in ("a", "b", "c")
    ])
    assert record is None  # 재현할 수 없으므로 고착화하지 않는다


# --------------------------------------------------------------------------
# 실패한 실행은 흉내 낼 대상이 아니다
# --------------------------------------------------------------------------

def _event_with_result(tool, args, result, timestamp):
    return {"event_type": "tool_usage_finished", "timestamp": timestamp,
            "data": {"tool_name": tool, "args": args, "result": result}}


@pytest.mark.parametrize("failure", [
    # 어댑터가 덧붙이는 문구
    "Tool 'request_approval' failed after 2 attempts with ToolException: bad uuid",
    # 도구가 본문으로 돌려주는 구조화된 오류(MCP 콘텐츠 블록)
    [{"type": "text", "text": '{"result":"error","error_kind":"INVALID","message":"unknown po"}'}],
    # MCP 표준 오류 플래그
    {"isError": True, "content": []},
])
def test_runs_with_a_failed_side_effect_are_not_frozen(generator, failure):
    """실패한 호출을 굳히면 고착화 코드가 **실패를 충실히 재현한다**.

    그것도 매번 LLM 없이, 사람이 알아채기 어려운 채로. 실제 운영 이력에서 세 표본이
    모두 실패한 `request_approval` 이었던 적이 있어 이 계약을 못박는다.
    """
    for index, todo in enumerate(["x1", "x2", "x3"], start=1):
        generator._test_events[todo] = [
            _event_with_result("db_exec", {"sql": f"UPDATE t SET s = {index}"}, failure, f"{todo}-1"),
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}", "query": "",
        })
    assert generator.try_freeze(_Workitem()) is None


def test_successful_runs_are_still_frozen(generator):
    """성공 판정이 지나치게 넓어 정상 실행까지 걸러내면 아무것도 고착화되지 않는다."""
    for index, todo in enumerate(["ok1", "ok2", "ok3"], start=1):
        generator._test_events[todo] = [
            _event_with_result(
                "db_exec", {"sql": f"UPDATE t SET s = {index}"},
                [{"type": "text", "text": '{"result":"ok","rows":1}'}], f"{todo}-1",
            ),
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}", "query": "",
        })
    assert generator.try_freeze(_Workitem()) is not None


def test_failure_in_a_context_action_does_not_disqualify_the_run(generator):
    """조회가 한 번 실패한 것까지 실행 전체를 부정하면 표본이 남지 않는다.

    자격을 좌우하는 것은 **부수효과**의 성패다.
    """
    for index, todo in enumerate(["m1", "m2", "m3"], start=1):
        generator._test_events[todo] = [
            _event_with_result("ls", {"path": "/tmp"}, "Tool 'ls' failed after 2 attempts", f"{todo}-1"),
            _event_with_result("db_exec", {"sql": f"UPDATE t SET s = {index}"}, "ok", f"{todo}-2"),
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}", "query": "",
        })
    assert generator.try_freeze(_Workitem()) is not None
