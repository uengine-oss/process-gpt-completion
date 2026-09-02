"""작업 이력 정규화가 런타임에 매이지 않는지, 그리고 무엇까지 읽어내는지 고정한다.

고착화·보상은 "그 활동이 실제로 무엇을 했는가"를 근거로 삼는다. 그 근거를 MCP 도구
호출로만 좁히면 스킬의 셸 스크립트로 일하는 활동은 영영 고착화되지 않고 되돌릴
대상도 놓친다. 여기서 고정하는 계약은 두 가지다.

1. 이벤트 봉투의 모양(런타임)이 달라도 같은 행위 목록으로 읽힌다.
2. 도구 호출뿐 아니라 셸 실행·파일 쓰기·스킬 읽기까지 종류를 나눠 읽는다.
"""

import pytest

import work_history as wh


# --------------------------------------------------------------------------
# 런타임 비의존 — 봉투 모양이 달라도 같은 결론
# --------------------------------------------------------------------------

def _event(event_type, data, timestamp="2026-08-01T00:00:01"):
    return {"event_type": event_type, "timestamp": timestamp, "data": data}


def test_deepagents_and_crewai_envelopes_read_identically():
    """DeepAgents는 tool/args/result, CrewAI는 tool_name/query/args로 남긴다."""
    deepagents = _event("tool_usage_finished", {
        "tool": "db_exec", "tool_name": "db_exec",
        "args": {"sql": "UPDATE t SET a = 1"}, "result": "ok",
    })
    crewai = _event("tool_usage_finished", {
        "tool_name": "db_exec", "query": "UPDATE t SET a = 1",
        "args": {"sql": "UPDATE t SET a = 1"},
    })
    a, = wh.normalize_events([deepagents])
    b, = wh.normalize_events([crewai])
    assert (a.kind, a.tool, a.args) == (b.kind, b.tool, b.args) == (wh.MCP_CALL, "db_exec", {"sql": "UPDATE t SET a = 1"})


def test_json_string_payload_is_parsed():
    """data 컬럼이 JSON 문자열로 저장된 러너도 있다."""
    actions = wh.normalize_events([
        _event("tool_usage_finished", '{"tool_name": "db_exec", "args": {"sql": "DELETE FROM t"}}')
    ])
    assert [a.kind for a in actions] == [wh.MCP_CALL]


def test_nested_payload_is_unwrapped():
    """페이로드를 한 겹 더 감싸는 러너를 흡수한다."""
    actions = wh.normalize_events([
        _event("tool_end", {"payload": {"name": "execute", "input": {"command": "rm -rf /tmp/x"}}})
    ])
    assert [(a.kind, a.args["command"]) for a in actions] == [(wh.SHELL, "rm -rf /tmp/x")]


def test_unknown_finished_event_type_is_still_read():
    """이벤트 타입 이름을 하나로 못박지 않는다."""
    actions = wh.normalize_events([
        _event("agent_tool_completed", {"tool_name": "db_exec", "args": {"sql": "INSERT INTO t VALUES (1)"}})
    ])
    assert len(actions) == 1


def test_started_events_are_a_fallback_not_a_duplicate():
    """끝난 이벤트가 있으면 시작 이벤트는 세지 않는다. 없으면 시작 이벤트로 읽는다."""
    started = _event("tool_usage_started", {"tool_name": "db_exec", "args": {"sql": "UPDATE t SET a = 1"}})
    finished = _event("tool_usage_finished", {"tool_name": "db_exec", "args": {"sql": "UPDATE t SET a = 1"}})
    assert len(wh.normalize_events([started, finished])) == 1
    assert len(wh.normalize_events([started])) == 1


def test_actions_are_ordered_by_timestamp():
    late = _event("tool_usage_finished", {"tool_name": "b", "args": {"sql": "UPDATE t SET a = 2"}}, "2026-08-01T00:02")
    early = _event("tool_usage_finished", {"tool_name": "a", "args": {"sql": "UPDATE t SET a = 1"}}, "2026-08-01T00:01")
    assert [a.tool for a in wh.normalize_events([late, early])] == ["a", "b"]


# --------------------------------------------------------------------------
# 무엇을 읽어내는가 — 도구 호출만이 아니다
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tool,args,expected", [
    ("execute", {"command": "python3 import.py"}, wh.SHELL),
    ("bash", {"command": "rm -rf /tmp/x"}, wh.SHELL),
    # 세상을 바꾸지 않는 셸은 맥락이다 — SQL의 SELECT와 같은 기준.
    ("bash", {"command": "ls"}, wh.INSPECT),
    # 이름을 몰라도 "명령 문자열 하나"만 받으면 셸로 본다.
    ("run_it", {"command": "make build", "cwd": "/workspace"}, wh.SHELL),
    ("write_file", {"file_path": "/workspace/out.csv", "content": "a,b"}, wh.FILE_WRITE),
    ("edit_file", {"file_path": "/workspace/a.py", "old_string": "x", "new_string": "y"}, wh.FILE_WRITE),
    ("read_file", {"file_path": "/skills/inventory/SKILL.md"}, wh.SKILL_READ),
    ("read_file", {"file_path": "/workspace/data.json"}, wh.FILE_READ),
    ("ls", {"path": "/workspace"}, wh.INSPECT),
    ("task", {"subagent_type": "researcher", "description": "조사"}, wh.DELEGATE),
    ("write_todos", {"todos": []}, wh.PLAN),
    ("mem0", {"query": "x"}, wh.INTERNAL),
    ("db_exec", {"sql": "SELECT 1"}, wh.INSPECT),
    ("db_exec", {"sql": "UPDATE t SET a = 1"}, wh.MCP_CALL),
    ("send_email_tool", {"to": "a@b.com"}, wh.MCP_CALL),
])
def test_action_kinds(tool, args, expected):
    assert wh.classify(tool, args) == expected


def test_mcp_prefixed_tool_names_are_recognized():
    """`mcp__pg__execute_sql` 처럼 접두어가 붙어도 끝 마디로 판정한다."""
    assert wh.classify("mcp__fs__read_file", {"path": "/skills/x/SKILL.md"}) == wh.SKILL_READ
    assert wh.classify("sandbox.execute", {"command": "rm -rf /tmp/x"}) == wh.SHELL


def test_command_named_arg_on_a_real_tool_is_not_shell():
    """`command` 인자를 쓰는 다른 도구를 셸로 오인하지 않는다."""
    assert wh.classify("printer", {"command": "print", "copies": 3, "tray": "A4"}) == wh.MCP_CALL


def test_effect_actions_exclude_context_only_work():
    trace = wh.normalize_events([
        _event("tool_usage_finished", {"tool_name": "read_file", "args": {"file_path": "/skills/inv/SKILL.md"}}, "t1"),
        _event("tool_usage_finished", {"tool_name": "ls", "args": {"path": "/workspace"}}, "t2"),
        _event("tool_usage_finished", {"tool_name": "execute", "args": {"command": "python3 load.py"}}, "t3"),
        _event("tool_usage_finished", {"tool_name": "db_exec", "args": {"sql": "SELECT 1"}}, "t4"),
        _event("tool_usage_finished", {"tool_name": "db_exec", "args": {"sql": "UPDATE t SET a = 1"}}, "t5"),
    ])
    assert [a.kind for a in wh.effect_actions(trace)] == [wh.SHELL, wh.MCP_CALL]


def test_summary_records_skills_shell_and_files():
    """생성 근거를 사람이 읽을 수 있게 남긴다 — 어떤 절차를 따랐는지 포함."""
    trace = wh.normalize_events([
        _event("tool_usage_finished", {"tool_name": "read_file", "args": {"file_path": "/skills/inv/SKILL.md"}}, "t1"),
        _event("tool_usage_finished", {"tool_name": "read_file", "args": {"file_path": "/workspace/in.csv"}}, "t2"),
        _event("tool_usage_finished", {"tool_name": "execute", "args": {"command": "python3 load.py --file in.csv"}}, "t3"),
        _event("tool_usage_finished", {"tool_name": "write_file", "args": {"file_path": "/workspace/out.json", "content": "{}"}}, "t4"),
        _event("tool_usage_finished", {"tool_name": "task", "args": {"subagent_type": "checker"}}, "t5"),
    ])
    summary = wh.summarize(trace)
    assert summary["skills_read"] == ["/skills/inv/SKILL.md"]
    assert summary["files_read"] == ["/workspace/in.csv"]
    assert summary["files_written"] == ["/workspace/out.json"]
    assert summary["shell_commands"] == ["python3 load.py --file in.csv"]
    assert summary["subagents"] == ["checker"]