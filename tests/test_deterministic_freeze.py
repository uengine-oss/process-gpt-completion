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
    build_output_template,
    identify_parameters,
    unrecoverable_parameters,
    is_readonly_sql,
    is_write_call,
    looks_like_sql,
    normalize_sql,
    procedure_pins,
    same_value,
    structured_fields,
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
    upstream: dict = {}

    db = types.ModuleType("database")
    db.fetch_mcp_python_code = lambda *a, **k: None
    db.upsert_mcp_python_code = saved.append
    db.fetch_last_deactivated_at = lambda *a, **k: None
    db.fetch_workitems_by_activity = lambda *a, **k: workitems
    db.fetch_events_by_todo_id = lambda todo_id: events.get(todo_id, [])
    db.fetch_related_workitem_outputs = lambda tenant, root, inst, **k: upstream.get(
        str(k.get("exclude_id") or ""), []
    )
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
    module._test_upstream = upstream
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


# --------------------------------------------------------------------------
# 실행 식별자 — 지시문에서 되찾을 수 없는 자리
# --------------------------------------------------------------------------

def _ledger_sample(applicant, amount, proc_inst_id, todo_id):
    return [
        ("db_exec", {"sql": (
            "INSERT INTO expense_ledger "
            "(tenant_id, applicant, amount, proc_inst_id, todo_id) "
            f"VALUES ('localhost', '{applicant}', {amount}, "
            f"'{proc_inst_id}', '{todo_id}')"
        )}),
    ]


_LEDGER_IDENTITIES = [
    {"id": "todo-1", "proc_inst_id": "pi-1", "tenant_id": "localhost"},
    {"id": "todo-2", "proc_inst_id": "pi-2", "tenant_id": "localhost"},
    {"id": "todo-3", "proc_inst_id": "pi-3", "tenant_id": "localhost"},
]


def test_insert_values_take_their_column_names():
    """VALUES 목록의 값은 앞에 이름이 없어 자리 번호로만 불렸다.

    `query_7` 이라는 이름으로는 그 자리가 무엇인지 알 수 없고, 다음 실행에서 값을
    되찾을 수도 없다. 컬럼 목록과 순서로 대응시켜 이름을 준다.
    """
    samples = [
        _ledger_sample("홍길동", 50000, "pi-1", "todo-1"),
        _ledger_sample("김철수", 30000, "pi-2", "todo-2"),
        _ledger_sample("이영희", 45000, "pi-3", "todo-3"),
    ]
    plan = identify_parameters(samples, identities=_LEDGER_IDENTITIES)
    assert {p["name"] for p in plan.parameters} == {
        "applicant", "amount", "proc_inst_id", "todo_id",
    }


def test_runtime_identifiers_are_bound_to_the_workitem_row():
    """실행 때 정해지는 값은 지시문이 아니라 워크아이템 행에서 읽는다."""
    samples = [
        _ledger_sample("홍길동", 50000, "pi-1", "todo-1"),
        _ledger_sample("김철수", 30000, "pi-2", "todo-2"),
        _ledger_sample("이영희", 45000, "pi-3", "todo-3"),
    ]
    plan = identify_parameters(samples, identities=_LEDGER_IDENTITIES)
    by_name = {p["name"]: p for p in plan.parameters}
    assert by_name["proc_inst_id"]["runtime"] == "proc_inst_id"
    assert by_name["todo_id"]["runtime"] == "id"
    # 업무 입력은 그대로 지시문에서 되찾는다.
    assert "runtime" not in by_name["applicant"]
    assert "runtime" not in by_name["amount"]


def test_identifier_column_binds_even_when_the_agent_filled_it_wrong():
    """에이전트가 그 자리를 잘못 채운 표본이 섞여도 자리의 뜻은 컬럼 이름이 정한다.

    실제 이력에서 todo_id 자리에는 빈 문자열과 **다른 워크아이템의** id 가 섞여 있었다.
    관측만 보면 식별자로 인정되지 않아, 위치 폴백이 엉뚱한 값을 채웠다.
    """
    samples = [
        _ledger_sample("홍길동", 50000, "pi-1", "다른-워크아이템-id"),
        _ledger_sample("김철수", 30000, "pi-2", ""),
        _ledger_sample("이영희", 45000, "pi-3", "또-다른-id"),
    ]
    plan = identify_parameters(samples, identities=_LEDGER_IDENTITIES)
    by_name = {p["name"]: p for p in plan.parameters}
    assert by_name["todo_id"]["runtime"] == "id"


def test_business_values_are_not_mistaken_for_identifiers():
    """식별자와 무관한 컬럼은 그대로 지시문에서 되찾는다."""
    samples = [
        _ledger_sample("홍길동", 50000, "pi-1", "todo-1"),
        _ledger_sample("김철수", 30000, "pi-2", "todo-2"),
        _ledger_sample("이영희", 45000, "pi-3", "todo-3"),
    ]
    plan = identify_parameters(samples, identities=None)
    by_name = {p["name"]: p for p in plan.parameters}
    # identities 가 없어도 컬럼 이름만으로 식별자 자리는 가려낸다.
    assert by_name["proc_inst_id"]["runtime"] == "proc_inst_id"
    assert "runtime" not in by_name["applicant"]


# --------------------------------------------------------------------------
# 폼 산출물 템플릿
# --------------------------------------------------------------------------

_OBSERVATIONS = {"applicant": ("김철수", "이영희", "박지은"), "amount": (30000, 45000, 27000)}


def _form(text):
    return {"expense_form": {"ledger_result": text}}


def test_constant_output_is_frozen_as_is():
    outputs = [_form("등록 완료")] * 3
    built = build_output_template(outputs, _OBSERVATIONS)
    assert built == {
        "form_id": "expense_form",
        "fields": {"ledger_result": {"const": "등록 완료"}},
    }


def test_varying_output_folds_into_a_parameter_template():
    """값의 차이가 파라미터로 전부 설명되면 템플릿으로 접힌다."""
    outputs = [
        _form("신청자 김철수, 금액 30000 등록 완료"),
        _form("신청자 이영희, 금액 45000 등록 완료"),
        _form("신청자 박지은, 금액 27000 등록 완료"),
    ]
    built = build_output_template(outputs, _OBSERVATIONS)
    assert built["fields"]["ledger_result"] == {
        "template": "신청자 ${applicant}, 금액 ${amount} 등록 완료"
    }


def test_prose_that_varies_in_wording_is_not_frozen():
    """에이전트가 매번 다르게 쓴 문장은 굳히지 않는다.

    표본 하나의 문장을 굳히면 그 표본의 사실이 다음 실행에 그대로 남는다. 실행기가
    이번 실행의 사실로 채우도록 `None` 을 남긴다.
    """
    outputs = [
        _form("INSERT 1건 완료"),
        _form("지출 내역 1건이 등록되었습니다"),
        _form("1건 INSERT를 실행했습니다"),
    ]
    assert build_output_template(outputs, _OBSERVATIONS)["fields"]["ledger_result"] is None


def test_literal_dollar_survives_rendering():
    """원문의 `$` 가 치환 기호로 읽히면 렌더가 통째로 깨진다."""
    from string import Template

    outputs = [
        _form("비용 $30000 처리"),
        _form("비용 $45000 처리"),
        _form("비용 $27000 처리"),
    ]
    template = build_output_template(outputs, _OBSERVATIONS)["fields"]["ledger_result"]
    assert Template(template["template"]).substitute({"amount": 51000}) == "비용 $51000 처리"


def test_mismatched_forms_are_not_folded():
    """폼 아이디나 필드 구성이 다르면 같은 산출물이 아니다."""
    assert build_output_template(
        [_form("a"), {"other_form": {"ledger_result": "a"}}, _form("a")], _OBSERVATIONS
    ) is None
    assert build_output_template(
        [_form("a"), {"expense_form": {"different_field": "a"}}, _form("a")], _OBSERVATIONS
    ) is None
    # 산출물을 남기지 않은 표본이 섞여도 접지 않는다.
    assert build_output_template([_form("a"), None, _form("a")], _OBSERVATIONS) is None


# --------------------------------------------------------------------------
# 되찾을 수 없는 파라미터 — 굳히기 전에 거른다
# --------------------------------------------------------------------------

def _write_sample(applicant, amount, body):
    return [
        ("db_exec", {"sql": f"INSERT INTO ledger (applicant, amount) "
                            f"VALUES ('{applicant}', {amount})"}),
        ("write_file", {"path": f"/out/{applicant}.md", "content": body}),
    ]


_QUERIES = [
    '[InputData]\n{"form": {"applicant": "김철수", "amount": 30000}}',
    '[InputData]\n{"form": {"applicant": "이영희", "amount": 45000}}',
    '[InputData]\n{"form": {"applicant": "박지은", "amount": 27000}}',
]


def test_parameters_found_in_the_instruction_are_recoverable():
    samples = [
        _write_sample("김철수", 30000, "김철수 문서"),
        _write_sample("이영희", 45000, "이영희 문서"),
        _write_sample("박지은", 27000, "박지은 문서"),
    ]
    plan = identify_parameters(samples, _QUERIES)
    # content 는 `{신청자} 문서` 라 신청자 파라미터로 접히고, 남는 자리는 지시문에 있다.
    assert unrecoverable_parameters(plan, _QUERIES) == []


def test_a_value_absent_from_every_instruction_blocks_the_freeze():
    """문서 본문처럼 지시문에 없는 값은 다음 실행에서 되찾을 수 없다.

    실행기의 위치 폴백은 지시문에서 순서대로 값을 집으므로 **언제나 성공**하고,
    그 값으로 실제 도구를 부른다. 굳히기 전에 막아야 한다.
    """
    samples = [
        _write_sample("김철수", 30000, "# 보고서\n\n지난 분기 매출은 상승했다."),
        _write_sample("이영희", 45000, "# 보고서\n\n신규 고객이 늘었다."),
        _write_sample("박지은", 27000, "# 보고서\n\n재고 회전율이 개선됐다."),
    ]
    plan = identify_parameters(samples, _QUERIES)
    assert "content" in unrecoverable_parameters(plan, _QUERIES)


def test_runtime_identifiers_are_exempt():
    """실행 식별자는 지시문이 아니라 워크아이템 행에서 읽으므로 검사 대상이 아니다."""
    samples = [
        [("db_exec", {"sql": f"INSERT INTO t (applicant, proc_inst_id) "
                             f"VALUES ('{who}', '{pi}')"})]
        for who, pi in (("김철수", "pi-1"), ("이영희", "pi-2"), ("박지은", "pi-3"))
    ]
    identities = [{"proc_inst_id": f"pi-{i}"} for i in (1, 2, 3)]
    plan = identify_parameters(samples, _QUERIES, identities)
    by_name = {p["name"]: p for p in plan.parameters}
    assert by_name["proc_inst_id"]["runtime"] == "proc_inst_id"
    assert unrecoverable_parameters(plan, _QUERIES) == []


def test_number_formatting_does_not_count_as_unrecoverable():
    """`27,000` 과 `27000` 은 같은 값이다. 서식 차이로 막으면 굳을 수 있는 활동이 안 굳는다."""
    samples = [
        [("db_exec", {"sql": f"UPDATE t SET amount = {amount}"})]
        for amount in (30000, 45000, 27000)
    ]
    formatted = [
        '[InputData]\n{"form": {"amount": "30,000"}}',
        '[InputData]\n{"form": {"amount": "45,000"}}',
        '[InputData]\n{"form": {"amount": "27,000"}}',
    ]
    plan = identify_parameters(samples, formatted)
    assert unrecoverable_parameters(plan, formatted) == []


def test_no_instructions_means_no_verdict():
    """지시문이 없으면 판정할 근거가 없다.

    비어 있는 지시문에서는 실행기의 폴백이 집을 것도 없어 그냥 실패하고 에이전트로
    넘어간다 — 안전한 쪽이다. 이 게이트는 쓸모없는 코드가 아니라 확신 있게 틀리는
    코드를 막는다.
    """
    samples = [_write_sample("김철수", 30000, "a"), _write_sample("이영희", 45000, "b"),
               _write_sample("박지은", 27000, "c")]
    plan = identify_parameters(samples)
    assert unrecoverable_parameters(plan, None) == []
    assert unrecoverable_parameters(plan, ["", "  ", ""]) == []


def test_freeze_is_withheld_when_a_parameter_cannot_be_recovered(generator):
    """지시문에 없는 값이 파라미터로 승격되면 활동 전체를 굳히지 않는다.

    이 게이트가 없으면 코드는 만들어지고, 실행기는 이름표 없는 자리를 위치 폴백으로
    채운다. 폴백은 지시문에서 순서대로 값을 집으므로 언제나 성공하고, 그 값으로 실제
    도구를 부른다. 실패가 아니라 확신 있는 오답이라 사람이 알아채기 어렵다.
    """
    for index, (todo, applicant, body) in enumerate(
        [("u1", "김철수", "지난 분기 매출이 올랐다"),
         ("u2", "이영희", "신규 고객이 늘었다"),
         ("u3", "박지은", "재고 회전율이 개선됐다")], start=1
    ):
        generator._test_events[todo] = [
            _event("db_exec", {"sql": f"INSERT INTO t (applicant) VALUES ('{applicant}')"},
                   f"{todo}-1"),
            _event("write_file", {"file_path": "/out/report.md", "content": body}, f"{todo}-2"),
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}",
            "query": f'[InputData]\n{{"form": {{"applicant": "{applicant}"}}}}',
        })
    assert generator.try_freeze(_Workitem()) is None
    assert generator._test_saved == []


def test_freeze_proceeds_when_every_parameter_is_recoverable(generator):
    """같은 구조라도 변한 값이 전부 지시문에 있으면 굳는다.

    위 테스트의 대조군이다 — 게이트가 지나치게 넓으면 아무것도 고착화되지 않는다.
    """
    for index, (todo, applicant) in enumerate(
        [("r1", "김철수"), ("r2", "이영희"), ("r3", "박지은")], start=1
    ):
        generator._test_events[todo] = [
            _event("db_exec", {"sql": f"INSERT INTO t (applicant) VALUES ('{applicant}')"},
                   f"{todo}-1"),
            _event("write_file", {"file_path": "/out/report.md", "content": f"{applicant} 보고서"},
                   f"{todo}-2"),
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}",
            "query": f'[InputData]\n{{"form": {{"applicant": "{applicant}"}}}}',
        })
    record = generator.try_freeze(_Workitem())
    assert record is not None
    assert [p["name"] for p in record["parameters"]["parameters"]] == ["applicant"]


# --------------------------------------------------------------------------
# 단계 간 데이터 흐름 — 앞 단계의 결과가 뒤 단계의 인자로 흐른다
# --------------------------------------------------------------------------
#
# 지금까지 인자는 둘 중 하나였다. 표본 전부에서 같았던 상수이거나, 다음 실행의
# 지시문에서 되찾는 파라미터이거나. 그래서 "시각을 찍어 문서에 적는" 활동은 굳지
# 않았다 — 그 시각은 지시문 어디에도 없고 표본마다 다르니 상수도 아니다.

def _event_with_result(tool, args, result, timestamp):
    return {"event_type": "tool_usage_finished", "timestamp": timestamp,
            "data": {"tool_name": tool, "args": args, "result": result}}


def _shell_result(stamp):
    """실제 러너가 남기는 셸 결과 모양. 표준출력 뒤에 어댑터의 문구가 붙는다."""
    return f"{stamp}\n\n[Command succeeded with exit code 0]"


_RESOLUTION_SAMPLES = [
    ("s1", "김철수", 30000, "2026-09-01T02:46:44Z"),
    ("s2", "이영희", 45000, "2026-09-01T03:02:24Z"),
    ("s3", "박지은", 27000, "2026-09-01T03:17:21Z"),
]


def _document_run(generator, todo, index, applicant, amount, stamp, written_stamp=None):
    """시각을 찍고(셸) → 그 시각을 담은 문서를 쓴 실행 한 건."""
    generator._test_events[todo] = [
        _event("ls", {"path": "/workspace/out"}, f"{todo}-1"),
        _event_with_result(
            "execute", {"command": "mkdir -p /out && date -Iseconds"},
            _shell_result(stamp), f"{todo}-2",
        ),
        _event("write_file", {
            "file_path": f"/out/{applicant}.md",
            "content": f"# 지출결의서\n| 신청자 | {applicant} |\n| 금액 | {amount} |\n"
                       f"| 등록 시각 | {written_stamp or stamp} |\n",
        }, f"{todo}-3"),
    ]
    generator._test_workitems.append({
        "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
        "updated_at": f"2026-08-0{index}",
        "query": f'[InputData]\n{{"form": {{"applicant": "{applicant}", "amount": {amount}}}}}',
    })


def _freeze_document(generator, samples=None):
    for index, (todo, applicant, amount, stamp) in enumerate(samples or _RESOLUTION_SAMPLES, start=1):
        _document_run(generator, todo, index, applicant, amount, stamp)
    return generator.try_freeze(_Workitem())


def test_a_value_produced_by_an_earlier_step_is_not_a_parameter(generator):
    """앞 단계가 만들어 낸 값은 밖에서 받을 것이 아니라 이어받을 것이다."""
    record = _freeze_document(generator)
    assert record is not None
    names = {p["name"] for p in record["parameters"]["parameters"]}
    assert names == {"applicant", "amount"}  # 시각은 파라미터가 아니다
    assert "from_step(results, 0, \"output\", [], 0)" in record["code"]


def test_step_binding_lets_the_document_activity_freeze(generator):
    """이어받기가 없으면 이 활동은 "되찾을 수 없는 입력"으로 영영 보류된다.

    시각은 지시문에 없다. 파라미터로 승격되는 한 게이트에 걸리고, 게이트를 열면
    실행기의 위치 폴백이 엉뚱한 값을 시각 자리에 넣는다. 세 번째 갈래가 필요한 이유다.
    """
    assert _freeze_document(generator) is not None
    assert generator._test_saved != []


def test_step_binding_is_recorded_as_provenance(generator):
    """무엇을 어디서 이어받았는지 남는다 — 앞 단계가 달라졌을 때 의심할 근거다."""
    record = _freeze_document(generator)
    bindings = record["work_history"]["step_bindings"]
    assert len(bindings) == 1
    assert bindings[0]["from"] == 0
    assert bindings[0]["path"] == ["output", "line 0"]


def test_bound_value_comes_from_this_run_not_from_the_sample(generator):
    """재실행하면 문서의 시각이 **이번 실행**의 시각으로 채워진다.

    표본 하나의 시각을 굳히면 그 표본의 사실이 다음 실행에 그대로 남아 조용히 틀린다.
    """
    record = _freeze_document(generator)
    namespace: dict = {}
    exec(compile(record["code"], "generated.py", "exec"), namespace)

    written: list = []

    async def fake_run_shell(command, cwd="", timeout_s=300):
        return {"kind": "shell", "command": command, "output": "2026-12-25T09:00:00Z\n"}

    async def fake_write_file(path, content):
        written.append((path, content))
        return {"path": path}

    namespace["run_shell"] = fake_run_shell
    namespace["write_file"] = fake_write_file
    asyncio.run(namespace["run"]({"applicant": "최수정", "amount": 12000}))

    path, content = written[0]
    assert path == "/out/최수정.md"
    assert "| 등록 시각 | 2026-12-25T09:00:00Z |" in content   # 이번 실행의 시각
    assert "2026-09-01" not in content                          # 표본의 시각이 아니다
    assert "| 신청자 | 최수정 |" in content


def test_a_broken_step_binding_fails_instead_of_guessing(generator):
    """이어받을 자리가 비면 값을 지어내지 않고 실패한다 — 실행기가 에이전트로 넘긴다."""
    record = _freeze_document(generator)
    namespace: dict = {}
    exec(compile(record["code"], "generated.py", "exec"), namespace)

    async def empty_shell(command, cwd="", timeout_s=300):
        return {"kind": "shell", "command": command, "output": ""}

    async def fake_write_file(path, content):
        raise AssertionError("이어받을 값이 없는데 문서를 썼습니다")

    namespace["run_shell"] = empty_shell
    namespace["write_file"] = fake_write_file
    with pytest.raises(RuntimeError):
        asyncio.run(namespace["run"]({"applicant": "최수정", "amount": 12000}))


def test_binding_requires_every_sample_to_agree(generator):
    """한 표본에서만 맞아떨어진 자리를 믿으면 다음 실행에서 엉뚱한 값을 이어받는다.

    한 실행에서 에이전트가 찍은 시각 대신 제 손으로 지어낸 시각을 적었다면, 그 자리는
    앞 단계의 결과가 아니다. 이어받지 않고 파라미터로 두면 되찾을 수 없어 보류된다.
    """
    samples = list(_RESOLUTION_SAMPLES)
    for index, (todo, applicant, amount, stamp) in enumerate(samples, start=1):
        written = "2026-01-01T00:00:00Z" if todo == "s2" else stamp
        _document_run(generator, todo, index, applicant, amount, stamp, written_stamp=written)
    assert generator.try_freeze(_Workitem()) is None


def test_number_formatting_does_not_split_a_bound_value(generator):
    """금액 표기가 실행마다 흔들려도 자리 대조는 성립한다."""
    for index, (todo, applicant, amount, stamp) in enumerate(_RESOLUTION_SAMPLES, start=1):
        formatted = f"{amount:,}원" if index % 2 else str(amount)
        _document_run(generator, todo, index, applicant, formatted, stamp)
    record = generator.try_freeze(_Workitem())
    assert record is not None
    assert {p["name"] for p in record["parameters"]["parameters"]} == {"applicant", "amount"}


# --------------------------------------------------------------------------
# 액티비티를 건너뛰는 데이터 흐름 — 앞 워크아이템의 산출물
# --------------------------------------------------------------------------

_LEDGER_FORM = "expense_resolution_process_register_expense_ledger_form"


def _upstream_run(generator, todo, index, applicant, stamp):
    """앞 액티비티의 산출물(완료 시각)을 문서에 옮겨 적은 실행 한 건."""
    generator._test_events[todo] = [
        _event("write_file", {
            "file_path": f"/out/{applicant}.md",
            "content": f"| 신청자 | {applicant} |\n| 대장 등록 시각 | {stamp} |\n",
        }, f"{todo}-1"),
    ]
    generator._test_workitems.append({
        "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
        "updated_at": f"2026-08-0{index}",
        "query": f'[InputData]\n{{"form": {{"applicant": "{applicant}"}}}}',
    })
    generator._test_upstream[todo] = [{
        "workitemId": f"w-{todo}",
        "procInstId": f"pi-{todo}",
        "activityId": "register_expense_ledger",
        "activityName": "경비 대장 등록",
        "endDate": stamp,
        "output": {_LEDGER_FORM: {"ledger_result": f"{applicant} 1건 등록"}},
    }]


def test_a_value_from_the_previous_activity_is_bound_to_its_output(generator):
    """지시문에 없어도 앞 워크아이템의 산출물에 있으면 되찾을 수 있다."""
    for index, (todo, applicant, stamp) in enumerate(
        [("u1", "김철수", "2026-09-01T11:55:47.027549"),
         ("u2", "이영희", "2026-09-01T12:02:18.949265"),
         ("u3", "박지은", "2026-09-01T12:20:55.413880")], start=1
    ):
        _upstream_run(generator, todo, index, applicant, stamp)
    record = generator.try_freeze(_Workitem())
    assert record is not None
    bound = [p for p in record["parameters"]["parameters"] if p.get("upstream")]
    assert len(bound) == 1
    assert bound[0]["upstream"] == {
        "activity_id": "register_expense_ledger", "path": ["endDate"],
    }
    # 이름표를 함께 남기면 실행기가 지시문을 먼저 뒤져 엉뚱한 값을 집는다.
    assert "label" not in bound[0]


def test_upstream_binding_needs_every_sample_to_agree(generator):
    """한 표본에서만 앞 산출물과 맞았다면 근거가 아니다 — 굳히지 않는다."""
    for index, (todo, applicant, stamp) in enumerate(
        [("u1", "김철수", "2026-09-01T11:55:47.027549"),
         ("u2", "이영희", "2026-09-01T12:02:18.949265"),
         ("u3", "박지은", "2026-09-01T12:20:55.413880")], start=1
    ):
        _upstream_run(generator, todo, index, applicant, stamp)
    # 한 실행만 앞 산출물에 없는 시각을 적었다.
    generator._test_upstream["u2"][0]["endDate"] = "2026-09-01T00:00:00.000000"
    assert generator.try_freeze(_Workitem()) is None


# --------------------------------------------------------------------------
# 값 표기 정규화
# --------------------------------------------------------------------------

@pytest.mark.parametrize("left,right,expected", [
    ("29,000원", "29000", True),
    ("₩27000", "27000.0", True),
    ("2026-09-01T02:46:44Z", "2026-09-01 02:46:44", True),
    ("2026-09-09", "2026년 9월 9일", True),
    ("2026-09-01T02:46:44Z", "2026-09-01T02:46:45Z", False),
    ("1건", "1개", False),          # 단위가 다르면 다른 값이다
    ("29000", "29001", False),
    ("노트북", "마우스", False),
])
def test_formatting_differences_do_not_make_different_values(left, right, expected):
    assert same_value(left, right) is expected


# --------------------------------------------------------------------------
# 합성 자리 — 이미 아는 값들의 조합으로 다시 적는다
# --------------------------------------------------------------------------

_DOC_QUERY = (
    "[Instruction]\n출력 파일 경로를 expense/{{used_at}}_{{applicant}}.md 로 고정한다.\n\n"
    '[InputData]\n{{"form": {{"applicant": "{applicant}", "used_at": "{used_at}", '
    '"purpose": "{purpose}"}}}}'
)

_COMPOSED = [
    ("c1", "김철수", "2026-09-02", "협력사 미팅 택시비", "pi-aaaa1111"),
    ("c2", "이영희", "2026-09-03", "세미나 참석 택시비", "pi-bbbb2222"),
    ("c3", "박지은", "2026-09-04", "지사 출장 택시비", "pi-cccc3333"),
]


def _composed_run(generator, todo, index, applicant, used_at, purpose, instance):
    """워크스페이스 경로에 실행 식별자가, 문서 본문에 여러 낱말짜리 입력이 들어간 실행."""
    generator._test_events[todo] = [
        _event("write_file", {
            "file_path": f"/workspace/.bpmn/{instance}/expense/{used_at}_{applicant}.md",
            "content": f"| 신청자 | {applicant} |\n| 사유 | {purpose} |\n",
        }, f"{todo}-1"),
    ]
    generator._test_workitems.append({
        "id": todo, "proc_inst_id": instance, "rework_count": 0,
        "updated_at": f"2026-08-0{index}",
        "query": _DOC_QUERY.format(applicant=applicant, used_at=used_at, purpose=purpose),
    })


def _freeze_composed(generator):
    for index, args in enumerate(_COMPOSED, start=1):
        _composed_run(generator, args[0], index, *args[1:])
    return generator.try_freeze(_Workitem())


def test_a_path_holding_the_run_identity_is_composed_not_guessed(generator):
    """`/workspace/.bpmn/{proc_inst_id}/…` 는 한 낱말이라 조각으로 쪼개면 토막이 남는다.

    토막은 지시문에 없어 되찾을 수 없다. 그런데 그 토막을 이루는 값은 하나하나 알고
    있으므로, 자리 대조 대신 아는 값들의 조합으로 다시 적는다.
    """
    record = _freeze_composed(generator)
    assert record is not None
    assert "/workspace/.bpmn/${proc_inst_id}/expense/${used_at}_${applicant}.md" in record["code"]
    identity = [p for p in record["parameters"]["parameters"] if p.get("runtime")]
    assert [p["name"] for p in identity] == ["proc_inst_id"]


def test_a_fragment_of_one_input_field_is_replaced_by_the_whole_field(generator):
    """`사유: 지사 출장 택시비` 는 실행기가 칸 하나로 읽는다. 낱말로 쪼개면 못 집는다.

    쪼갠 토막은 지시문 어딘가에 있지만, 이름표도 없어 위치 폴백으로 떨어진다 — 사유
    칸에 신청자 이름이 들어간 문서가 만들어진다. 칸 값 전체로 다시 적어야 한다.
    """
    record = _freeze_composed(generator)
    assert "| 사유 | ${purpose} |" in record["code"]
    purpose = next(p for p in record["parameters"]["parameters"] if p["name"] == "purpose")
    assert purpose["example"] == "협력사 미팅 택시비"   # 토막이 아니라 칸 값 전체


def test_composition_is_recorded_as_provenance(generator):
    record = _freeze_composed(generator)
    composed = {c["into"]: set(c["from"]) for c in record["work_history"]["composed_args"]}
    assert composed["0:path"] == {"proc_inst_id", "used_at", "applicant"}
    assert composed["0:content"] == {"applicant", "purpose"}


def test_composed_code_runs_with_a_new_workitems_values(generator):
    record = _freeze_composed(generator)
    namespace: dict = {}
    exec(compile(record["code"], "generated.py", "exec"), namespace)
    written: list = []

    async def fake_write_file(path, content):
        written.append((path, content))
        return {"path": path}

    namespace["write_file"] = fake_write_file
    asyncio.run(namespace["run"]({
        "applicant": "최수정", "used_at": "2026-10-11", "purpose": "본사 회의 참석 택시비",
        "proc_inst_id": "pi-dddd4444",
    }))
    path, content = written[0]
    assert path == "/workspace/.bpmn/pi-dddd4444/expense/2026-10-11_최수정.md"
    assert "| 사유 | 본사 회의 참석 택시비 |" in content


def test_composition_is_refused_when_leftovers_disagree(generator):
    """아는 값으로 설명되지 않는 글자가 남으면 표본마다 템플릿이 갈린다 — 굳히지 않는다."""
    for index, (todo, applicant, used_at, purpose, instance) in enumerate(_COMPOSED, start=1):
        _composed_run(generator, todo, index, applicant, used_at, purpose, instance)
        # 파일 이름에 지시문 어디에도 없는 일련번호가 붙는다.
        events = generator._test_events[todo]
        args = events[0]["data"]["args"]
        args["file_path"] = args["file_path"].replace(".md", f"-{index * 37}.md")
    assert generator.try_freeze(_Workitem()) is None


def test_instruction_placeholders_do_not_hide_the_structured_input():
    """`expense/{used_at}_{applicant}.md` 같은 자리표시자가 지시문에 흔히 들어 있다.

    첫 `{` 부터 마지막 `}` 까지를 한 덩어리로 집으면 JSON 파싱이 통째로 실패해 구조화
    입력이 빈 것으로 보인다. 그러면 실행기는 지시문의 따옴표를 순서대로 긁어 **칸
    이름**을 값으로 집는다.
    """
    query = _DOC_QUERY.format(applicant="김철수", used_at="2026-09-02", purpose="협력사 미팅 택시비")
    assert "expense/{used_at}_{applicant}.md" in query   # 자리표시자는 그대로 남는다
    assert structured_fields(query) == {
        "applicant": "김철수", "used_at": "2026-09-02", "purpose": "협력사 미팅 택시비",
    }


# --------------------------------------------------------------------------
# 절차와 내용을 가른다 — 되찾을 수 없는 자리 중 무엇을 굳혀도 되는가
# --------------------------------------------------------------------------
#
# 되찾을 수 없다는 것은 그 값이 이 워크아이템이 준 것이 아니라는 뜻이다. 에이전트가
# 스스로 정한 값인데, 거기에는 두 갈래가 있다 — "어떻게 할지"를 정하는 절차(시각 서식)와
# 에이전트가 지어낸 내용(메일 본문)이다. 앞의 것은 굳혀도 되고, 뒤의 것은 굳히면 그
# 표본의 사실이 모든 실행에 남는다.

def _plan_with(observations, contexts):
    """관측값만 있는 최소 계획. 절차/내용 판정만 시험한다."""
    from deterministic_signature import ParameterPlan
    return ParameterPlan(
        parameters=tuple({"name": name} for name in observations),
        slots=(),
        observations={name: tuple(values) for name, values in observations.items()},
    )


_MAIL_QUERIES = [
    '[InputData]\n{"form": {"recipient": "김철수", "product": "노트북"}}',
    '[InputData]\n{"form": {"recipient": "이영희", "product": "마우스"}}',
    '[InputData]\n{"form": {"recipient": "박지은", "product": "키보드"}}',
]


def test_a_value_the_agent_reuses_is_procedure_not_input():
    """시각 서식처럼 에이전트가 즐겨 쓰는 방식은 과반으로 굳힌다.

    어느 서식을 골라도 다음 실행이 제대로 돌아간다. 이 자리 하나 때문에 나머지가 전부
    멀쩡한 활동을 영영 보류하면, 반복되는 작업이 매번 에이전트를 거친다.
    """
    plan = _plan_with(
        {"command_7": ["'+%Y-%m-%d %H:%M:%S'", "'+%Y-%m-%d %H:%M:%S'", "-Iseconds"]},
        _MAIL_QUERIES,
    )
    assert procedure_pins(plan, ["command_7"], _MAIL_QUERIES) == {
        "command_7": "'+%Y-%m-%d %H:%M:%S'"
    }


def test_agent_written_content_is_never_pinned():
    """에이전트가 쓴 메일 본문은 굳히면 안 된다.

    표본마다 다르고, 그 워크아이템의 사실을 담고 있다. 하나를 굳히면 김철수에게 보낼
    문장이 모든 수신자에게 나간다 — 실제로 그렇게 굳은 코드를 지운 적이 있다.
    """
    plan = _plan_with(
        {"body": ["김철수님, 노트북 발송했습니다.",
                  "이영희님, 마우스 발송했습니다.",
                  "박지은님, 키보드 발송했습니다."]},
        _MAIL_QUERIES,
    )
    assert procedure_pins(plan, ["body"], _MAIL_QUERIES) == {}


def test_a_repeated_value_carrying_workitem_data_is_not_pinned():
    """과반으로 반복돼도 이 활동의 값을 품고 있으면 내용이다.

    반복은 우연일 수 있다. 값이 워크아이템의 입력을 담고 있다면 그것은 절차가 아니다.
    """
    plan = _plan_with(
        {"body": ["김철수님께 발송", "김철수님께 발송", "박지은님께 발송"]},
        _MAIL_QUERIES,
    )
    assert procedure_pins(plan, ["body"], _MAIL_QUERIES) == {}


def test_all_distinct_values_are_never_pinned():
    """표본마다 다르면 '에이전트가 즐겨 쓰는 방식'이라고 볼 근거가 없다."""
    plan = _plan_with(
        {"stamp": ["2026-09-01 00:00", "2026-09-02 11:00", "2026-09-03 09:30"]},
        _MAIL_QUERIES,
    )
    assert procedure_pins(plan, ["stamp"], _MAIL_QUERIES) == {}


def test_recoverable_parameters_are_never_pinned():
    """되찾을 수 있는 자리는 애초에 판정 대상이 아니다 — 굳히지 말고 되찾아야 한다."""
    plan = _plan_with({"recipient": ["김철수", "김철수", "박지은"]}, _MAIL_QUERIES)
    assert procedure_pins(plan, [], _MAIL_QUERIES) == {}


def test_a_procedure_option_does_not_block_the_whole_activity(generator):
    """서식 하나가 다르다고 활동 전체를 보류하지 않는다 — 나머지는 전부 되찾을 수 있다."""
    formats = ["'+%Y-%m-%d'", "'+%Y-%m-%d'", "-Iseconds"]
    for index, (todo, applicant, stamp) in enumerate(
        [("f1", "김철수", "2026-09-02"), ("f2", "이영희", "2026-09-03"),
         ("f3", "박지은", "2026-09-04")], start=1
    ):
        generator._test_events[todo] = [
            _event_with_result(
                "execute", {"command": f"mkdir -p /out && date {formats[index - 1]}"},
                _shell_result(stamp), f"{todo}-1",
            ),
            _event("write_file", {
                "file_path": f"/out/{applicant}.md",
                "content": f"| 신청자 | {applicant} |\n| 시각 | {stamp} |\n",
            }, f"{todo}-2"),
        ]
        generator._test_workitems.append({
            "id": todo, "proc_inst_id": f"pi-{todo}", "rework_count": 0,
            "updated_at": f"2026-08-0{index}",
            "query": f'[InputData]\n{{"form": {{"applicant": "{applicant}"}}}}',
        })
    record = generator.try_freeze(_Workitem())
    assert record is not None
    # 과반 서식이 상수로 굳는다.
    assert "date '+%Y-%m-%d'" in record["code"]
    assert "-Iseconds" not in record["code"]
