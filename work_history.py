"""워크아이템 작업 이력의 런타임 비의존 정규화.

고착화(결정론적 코드 생성)와 보상(undo) 생성은 "그 활동이 실제로 무엇을 했는가"를
읽어야 성립한다. 그런데 그 이력은 실행 런타임마다 다른 모양으로 남는다.

- CrewAI 러너: ``{"tool_name": ..., "query": ..., "args": {...}}``
- DeepAgents 러너: ``{"tool": ..., "tool_name": ..., "args": {...}, "result": ...}``
- 앞으로 붙을 러너: 아직 모른다.

그래서 특정 런타임의 봉투 모양이나 도구 이름에 기대지 않는다. 이 모듈은 events
테이블의 원시 행을 받아 **정규화된 행위(Action) 목록**으로 바꾼다. 정규화된 행위는
"어떤 종류의 일을 했는가"(kind)로 구분되며, 생성기와 보상기는 그 종류만 본다.

읽어내는 종류:
- ``mcp_call``   외부 도구(MCP) 호출 — 부수효과 있음
- ``shell``      셸/스크립트 실행 — 부수효과 있음
- ``file_write`` 파일 생성·수정 — 부수효과 있음
- ``file_read``  파일 읽기 — 부수효과 없음(맥락)
- ``skill_read`` 스킬 파일(SKILL.md 등) 읽기 — 부수효과 없음(절차 출처)
- ``delegate``   서브에이전트 위임 — 부수효과는 위임 안쪽 행위로 따로 남는다
- ``inspect``    조회(ls/glob/grep/읽기 전용 SQL) — 부수효과 없음
- ``plan``       계획·메모(write_todos 등) — 부수효과 없음
- ``internal``   추적 대상이 아닌 내부 도구(mem0 등)

부수효과 판정은 **도구 이름이 아니라 인자 내용**으로 한다. SQL 실행 도구의 이름은
테넌트 MCP 구성마다 다르고, 셸 도구의 이름도 런타임마다 다르기 때문이다. 이름은
후보를 좁히는 힌트로만 쓰고, 최종 판정은 인자의 모양이 결정한다.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from deterministic_signature import (
    EXCLUDED_TOOLS,
    FILE_WRITE_KIND,
    MCP_CALL_KIND,
    SHELL_KIND,
    is_readonly_shell,
    is_readonly_sql,
    looks_like_sql,
    shell_made_dirs,
)

# ---------------------------------------------------------------------------
# 행위 종류
# ---------------------------------------------------------------------------

# 종류 이름의 정의는 지문 모듈에 둔다. 실행 지문이 셸 명령을 특별 취급하고, 단계 간
# 이어받기가 종류별로 결과의 어느 키를 볼지 정하기 때문이다.
MCP_CALL = MCP_CALL_KIND
SHELL = SHELL_KIND
FILE_WRITE = FILE_WRITE_KIND
FILE_READ = "file_read"
SKILL_READ = "skill_read"
DELEGATE = "delegate"
INSPECT = "inspect"
PLAN = "plan"
INTERNAL = "internal"

# 세상을 바꾸는 종류. 고착화 대상이자 보상 대상이다.
EFFECT_KINDS = (MCP_CALL, SHELL, FILE_WRITE)


@dataclass(frozen=True)
class Action:
    """정규화된 작업 이력 한 건.

    ``args``는 종류별 표준 모양으로 맞춘 값이다. 생성기는 이 모양만 보고 코드를
    만들므로 런타임 고유의 인자 이름(``file_path`` vs ``path`` 등)을 몰라도 된다.
    ``raw_args``에는 원본을 남겨 추적에 쓴다.
    """

    kind: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    raw_args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    timestamp: str = ""
    failed: bool = False

    @property
    def has_effect(self) -> bool:
        return self.kind in EFFECT_KINDS


# ---------------------------------------------------------------------------
# 봉투 해체 — 런타임마다 다른 이벤트 모양을 흡수한다
# ---------------------------------------------------------------------------

# "도구 사용이 끝났다"를 뜻하는 이벤트 타입들. 끝난 이벤트에는 인자와 결과가 함께
# 실려 있어 재현·되돌리기의 근거가 된다.
_FINISHED_TYPES = {
    "tool_usage_finished", "tool_finished", "tool_end", "tool_call_finished",
    "tool_result", "action_finished", "tool_completed",
}
# 시작 이벤트. 끝난 이벤트가 하나도 없는 런타임을 위한 폴백으로만 쓴다.
_STARTED_TYPES = {
    "tool_usage_started", "tool_started", "tool_start", "tool_call",
    "action_started", "tool_use",
}

_TOOL_KEYS = ("tool_name", "tool", "toolName", "name", "function_name", "function", "action")
_ARG_KEYS = ("args", "tool_args", "arguments", "input", "tool_input", "parameters", "params", "kwargs")
_RESULT_KEYS = ("result", "output", "response", "content", "observation")
_NESTED_KEYS = ("data", "payload", "event", "body")

# 도구 이름 정규화: `mcp__postgres__execute_sql`, `postgres.execute_sql` 등에서 끝
# 마디만 남긴다. 호출에는 원래 이름을 쓰고, 종류 판정에만 이 값을 쓴다.
_NAME_TAIL = re.compile(r"[^A-Za-z0-9]+")


def _loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return value


def _first(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _payload(event: dict[str, Any]) -> dict[str, Any] | None:
    """이벤트 행에서 도구 호출 페이로드를 꺼낸다.

    ``data``는 JSON 문자열일 수도, 딕셔너리일 수도, 한 겹 더 감싸여 있을 수도 있다.
    """
    data = _loads(event.get("data"))
    for _ in range(3):  # 중첩은 실무상 한두 겹이다. 상한을 두어 순환을 막는다.
        if not isinstance(data, dict):
            return None
        if _first(data, _TOOL_KEYS) is not None:
            return data
        nested = _loads(_first(data, _NESTED_KEYS))
        if not isinstance(nested, dict):
            return None
        data = nested
    return None


def _tool_name(payload: dict[str, Any]) -> str:
    raw = _first(payload, _TOOL_KEYS)
    if isinstance(raw, dict):  # OpenAI 형식의 {"function": {"name": ...}}
        raw = _first(raw, ("name", "tool_name"))
    return str(raw or "").strip()


def _arguments(payload: dict[str, Any]) -> dict[str, Any]:
    raw = _loads(_first(payload, _ARG_KEYS))
    if isinstance(raw, dict):
        return raw
    if raw is None:
        # 인자를 별도 키로 감싸지 않고 페이로드에 펼쳐 놓는 런타임을 위한 폴백.
        reserved = set(_TOOL_KEYS) | set(_ARG_KEYS) | set(_RESULT_KEYS) | set(_NESTED_KEYS)
        spread = {k: v for k, v in payload.items() if k not in reserved}
        return spread
    return {"input": raw}


# ---------------------------------------------------------------------------
# 종류 판정 — 이름은 힌트, 인자가 결정한다
# ---------------------------------------------------------------------------

_SHELL_TOOLS = {
    "execute", "bash", "sh", "shell", "run_shell", "run_shell_command", "run_command",
    "execute_command", "execute_shell", "terminal", "command", "subprocess",
    "code_execution", "run_script", "code_interpreter",
}
_SHELL_ARG_KEYS = ("command", "cmd", "script", "shell_command", "bash_command", "code")

_READ_TOOLS = {
    "read_file", "read", "view", "view_file", "cat", "open_file", "read_text_file",
    "get_file", "fetch_file", "load_file",
}
# 파일을 바꾸는 도구. 조작 종류별로 나눈다 — 삭제·이동을 쓰기로 뭉뚱그리면 생성 코드가
# 지우는 대신 빈 파일을 만든다(조용히 틀리는 종류의 오류다).
_FILE_OPS = {
    "write": (
        "write_file", "create_file", "update_file", "append_file", "write", "save_file",
        "write_process_definition", "update_process_definition",
    ),
    "edit": (
        "edit_file", "apply_patch", "patch_file", "str_replace", "str_replace_editor",
        "multi_edit", "notebook_edit",
    ),
    "delete": ("delete_file", "remove_file", "rm", "unlink"),
    "move": ("move_file", "rename_file", "mv", "rename"),
    "copy": ("copy_file", "cp"),
    "mkdir": ("mkdir", "make_directory", "create_directory"),
}
_WRITE_TOOLS = {name for names in _FILE_OPS.values() for name in names}
_PATH_ARG_KEYS = ("file_path", "path", "filepath", "filename", "file", "target_file", "notebook_path")
_CONTENT_ARG_KEYS = ("content", "text", "contents", "data", "body", "definition")
_SOURCE_ARG_KEYS = ("source", "src", "from", "old_path", "source_path")
_DEST_ARG_KEYS = ("destination", "dst", "to", "new_path", "target_path")

_INSPECT_TOOLS = {
    "ls", "list", "list_files", "glob", "grep", "find", "search", "search_files",
    "search_documents", "web_search", "fetch_webpage", "tree", "list_directory",
}
_DELEGATE_TOOLS = {"task", "subagent", "delegate", "spawn_agent", "run_subagent", "agent"}
_PLAN_TOOLS = {
    "write_todos", "todo_write", "update_plan", "plan", "think", "thinking",
    "sequentialthinking", "note",
}

# 스킬 문서로 보이는 경로. 어떤 절차를 따랐는지를 남기기 위해 따로 센다.
_SKILL_PATH = re.compile(r"(^|/)(skills?)/|SKILL\.md$|\.claude/skills/", re.IGNORECASE)


def _short_name(tool: str) -> str:
    """`mcp__pg__execute_sql` → `execute_sql`. 종류 판정에만 쓴다."""
    parts = [p for p in _NAME_TAIL.split(tool or "") if p]
    return (parts[-1] if parts else "").lower()


def _tail_names(tool: str) -> set[str]:
    """이름 판정 후보. 끝 마디와 마지막 두 마디를 함께 본다(`run_shell` 등)."""
    parts = [p for p in _NAME_TAIL.split(tool or "") if p]
    lowered = [p.lower() for p in parts]
    candidates = {"_".join(lowered)}
    if lowered:
        candidates.add(lowered[-1])
    if len(lowered) >= 2:
        candidates.add("_".join(lowered[-2:]))
    return candidates


def _text_arg(args: dict[str, Any], keys: Iterable[str]) -> tuple[str, str] | None:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return key, value
    return None


def looks_like_skill_path(path: str) -> bool:
    return bool(_SKILL_PATH.search(path or ""))


def _sql_values(args: dict[str, Any]) -> list[str]:
    return [v for v in (args or {}).values() if looks_like_sql(v)]


# 셸 도구는 "명령 문자열 하나"를 받는다. 실행 방식을 조절하는 곁가지 인자
# (작업 디렉터리, 상한, 스트리밍 여부)는 있어도 셸이지만, 그 밖의 실질 인자가 붙으면
# `command`라는 이름을 쓰는 다른 도구일 뿐이다.
_SHELL_SIDE_ARGS = {
    "cwd", "working_dir", "workdir", "directory", "timeout", "timeout_s", "timeout_ms",
    "shell", "env", "stream", "background", "restart", "description",
}


def _looks_like_shell_args(args: dict[str, Any]) -> bool:
    """이름을 모르는 도구가 셸 실행처럼 보이는지 인자 모양으로 본다."""
    found = _text_arg(args, _SHELL_ARG_KEYS)
    if found is None or _sql_values(args):
        return False
    command_key = found[0]
    rest = {k for k in args if k != command_key and args[k] not in (None, "")}
    return not (rest - _SHELL_SIDE_ARGS)


# 도구 이름의 동사로 부수효과 유무를 가린다. MCP 도구 이름은 `list_orders`,
# `create_rfq` 처럼 동사로 시작하는 관례가 강해서, 이름만으로도 조회와 변경이 꽤
# 정확히 갈린다. 이 판정이 없으면 `list_warehouse_stock` 같은 조회까지 "되돌릴
# 대상"이 되어, 보상 코드가 조회를 되돌리려 들고 실행 지문도 조회 횟수만큼 흔들린다.
#
# 쓰기 동사가 읽기 동사보다 우선한다(`get_or_create_order` 는 변경이다). 어느 쪽도
# 아니면 변경으로 본다 — 놓치는 쪽이 더 위험하기 때문이다.
_READ_VERBS = {
    "get", "list", "search", "find", "read", "fetch", "query", "describe",
    "count", "lookup", "show", "retrieve", "view", "inspect", "browse", "preview",
}
_WRITE_VERBS = {
    "create", "insert", "add", "update", "modify", "edit", "patch", "set", "put",
    "delete", "remove", "drop", "clear", "reset", "send", "post", "write", "save",
    "store", "upload", "publish", "submit", "confirm", "approve", "reject",
    "request", "cancel", "apply", "register", "execute", "run", "start", "stop",
    "keep", "record", "assign", "move", "copy", "rename", "generate", "issue",
}


def _classify_by_verb(tool: str) -> str:
    words = {p.lower() for p in _NAME_TAIL.split(tool or "") if p}
    if words & _WRITE_VERBS:
        return MCP_CALL
    if words & _READ_VERBS:
        return INSPECT
    return MCP_CALL


def classify(tool: str, args: dict[str, Any]) -> str:
    """도구 호출 하나의 종류를 판정한다.

    이름이 알려진 것이면 그 힌트를 먼저 쓰고, 모르는 이름이면 인자 모양으로 민다.
    끝까지 못 가리면 ``mcp_call``(부수효과 있음)로 보수적으로 본다 — 되돌릴 수
    있어야 할 것을 놓치는 쪽보다, 되돌릴 필요 없는 것을 되돌리려다 실패하는 쪽이
    드러나기 쉽기 때문이다.
    """
    args = args or {}
    names = _tail_names(tool)

    if not tool or names & EXCLUDED_TOOLS or (tool in EXCLUDED_TOOLS):
        return INTERNAL
    if names & _PLAN_TOOLS:
        return PLAN
    if names & _DELEGATE_TOOLS:
        return DELEGATE
    if names & _SHELL_TOOLS or _looks_like_shell_args(args):
        # 셸도 SQL과 같은 기준으로 본다 — 이름이 아니라 하는 일로. `date`·`ls`·`printf`
        # 처럼 세상을 바꾸지 않는 명령은 맥락이지 재현 대상이 아니다. 이걸 부수효과로
        # 세면 결과에 아무 기여도 없는 행위가 실행 지문에 들어가, 같은 결과를 낸 실행이
        # 서로 다른 방식으로 갈린다.
        found = _text_arg(args, _SHELL_ARG_KEYS)
        if found and is_readonly_shell(found[1]):
            return INSPECT
        return SHELL
    if names & _WRITE_TOOLS and (
        _text_arg(args, _PATH_ARG_KEYS)
        or _text_arg(args, _CONTENT_ARG_KEYS)
        or _text_arg(args, _SOURCE_ARG_KEYS)
    ):
        # 경로 인자를 못 찾아도 파일 조작으로 본다. mcp_call 로 새면 되돌릴 대상에서
        # 파일이 빠지고, 생성 단계에서는 "MCP 서버를 못 찾았다"는 엉뚱한 사유로 막힌다.
        return FILE_WRITE
    if names & _READ_TOOLS:
        # 인자가 없는 이벤트도 있다(결과만 기록한 러너). 읽기 도구는 이름만으로도
        # 부수효과가 없다고 볼 수 있다 — 되돌릴 것도, 재현할 것도 없다.
        path = _text_arg(args, _PATH_ARG_KEYS)
        if path and looks_like_skill_path(path[1]):
            return SKILL_READ
        return FILE_READ
    if names & _INSPECT_TOOLS:
        return INSPECT

    sql = _sql_values(args)
    if sql:
        return INSPECT if all(is_readonly_sql(v) for v in sql) else MCP_CALL
    return _classify_by_verb(tool)


def _canonical_args(kind: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """종류별 표준 인자 모양. 생성기는 이 모양만 안다."""
    args = args or {}
    if kind == SHELL:
        found = _text_arg(args, _SHELL_ARG_KEYS)
        command = found[1] if found else ""
        cwd = args.get("cwd") or args.get("working_dir") or args.get("workdir") or ""
        return {"command": command, "cwd": str(cwd or "")}
    if kind in (FILE_WRITE, FILE_READ, SKILL_READ):
        path = _text_arg(args, _PATH_ARG_KEYS)
        canonical: dict[str, Any] = {"path": path[1] if path else ""}
        if kind == FILE_WRITE:
            content = _text_arg(args, _CONTENT_ARG_KEYS)
            if content is not None:
                canonical["content"] = content[1]
            else:
                raw = args.get("definition")
                canonical["content"] = (
                    json.dumps(raw, ensure_ascii=False, indent=2) if raw is not None else ""
                )
            for old_key in ("old_string", "old_str", "old"):
                if isinstance(args.get(old_key), str):
                    canonical["old_string"] = args[old_key]
                    break
            for new_key in ("new_string", "new_str", "new"):
                if isinstance(args.get(new_key), str):
                    canonical["new_string"] = args[new_key]
                    break
            canonical["op"] = _file_op(tool, canonical)
            source = _text_arg(args, _SOURCE_ARG_KEYS)
            if source:
                canonical["source"] = source[1]
            destination = _text_arg(args, _DEST_ARG_KEYS)
            if destination:
                canonical["destination"] = destination[1]
        return canonical
    if kind == DELEGATE:
        return {
            "subagent": str(args.get("subagent_type") or args.get("agent") or tool),
            "description": str(args.get("description") or args.get("prompt") or ""),
        }
    return dict(args)


def _file_op(tool: str, canonical: dict[str, Any]) -> str:
    """파일 조작의 종류. 이름으로 못 가리면 인자 모양으로 민다."""
    names = _tail_names(tool)
    for op, tools in _FILE_OPS.items():
        if names & set(tools):
            return op
    return "edit" if "old_string" in canonical else "write"


# 도구가 실패했음을 알리는 표식. 런타임(어댑터)이 붙이는 문구와 도구 자신이 돌려주는
# 구조화된 오류 본문 두 갈래를 모두 본다.
_FAILURE_MARKERS = (
    "ToolException",
    "Error calling tool",
    "failed after",
    "Traceback (most recent call last)",
)
_ERROR_FIELDS = ("error", "err", "exception")


def _payload_says_error(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("isError") is True or node.get("is_error") is True:
        return True
    if node.get("ok") is False or node.get("success") is False:
        return True
    for key in ("result", "status", "state", "outcome"):
        if str(node.get(key) or "").strip().lower() in ("error", "failed", "failure"):
            return True
    if any(node.get(key) for key in _ERROR_FIELDS):
        return True
    return bool(node.get("error_kind"))


def looks_failed(result: Any) -> bool:
    """도구 호출 결과가 실패를 말하고 있는가.

    실패한 호출을 성공한 실행으로 착각해 굳히면, 고착화된 코드가 **실패를 충실히
    재현한다**. 그것도 매번 LLM 없이, 사람이 못 알아채는 채로. 표본 자격을 성공한
    실행으로 좁히는 이유가 여기 있다.

    판정은 특정 런타임의 오류 형식에 기대지 않는다. 어댑터가 덧붙이는 문구와 도구가
    본문으로 돌려주는 구조화된 오류를 모두 본다.
    """
    if result is None:
        return False
    if isinstance(result, dict):
        if _payload_says_error(result):
            return True
        # MCP 콘텐츠 블록은 본문을 text/content 에 문자열로 담는다. 도구가 돌려준
        # 구조화된 오류는 그 안에 들어 있으므로 한 겹 더 열어 본다.
        return any(
            looks_failed(result[key])
            for key in ("text", "content", "data", "body", "output")
            if isinstance(result.get(key), (str, dict, list))
        )
    if isinstance(result, (list, tuple)):
        return any(looks_failed(item) for item in result)
    text = str(result)
    if any(marker in text for marker in _FAILURE_MARKERS):
        return True
    parsed = _loads(text)
    if isinstance(parsed, (dict, list)):
        return looks_failed(parsed)
    return False


def to_action(tool: str, args: dict[str, Any], *, result: Any = None, timestamp: str = "") -> Action:
    kind = classify(tool, args)
    return Action(
        kind=kind,
        tool=tool,
        args=_canonical_args(kind, tool, args),
        raw_args=dict(args or {}),
        result=result,
        timestamp=timestamp or "",
        failed=looks_failed(result),
    )


# ---------------------------------------------------------------------------
# 고착화 실행이 남긴 이력
# ---------------------------------------------------------------------------
# 고착화된 코드로 돈 실행은 도구 이벤트(`tool_usage_finished`)를 남기지 않는다. 실행
# 골격이 단계마다 돌려준 결과가 완료 이벤트 하나에 통째로 실린다. 그 이벤트를 읽지
# 못하면 **한 번 굳은 활동은 이후 모든 실행이 "아무 일도 하지 않은 것"으로 읽힌다** —
# 재작업 때 되돌릴 것이 없다고 판정되고, 순방향 코드가 그 위에 덧씌워져 같은 값이 두
# 번 반영된다. 보상이 만들어지기 전에 굳은 활동은 영영 보상을 갖지 못한다.

# 실행 결과의 `kind` → 그 모양을 남긴 골격 원시 동작의 이름. 되읽을 때 이름으로 종류를
# 다시 판정하므로(`to_action`), 에이전트가 같은 일을 했을 때와 같은 결론이 나온다.
_RESULT_FILE_OP_TOOLS = {
    "write": "write_file",
    "edit": "edit_file",
    "delete": "delete_file",
    "move": "move_file",
    "copy": "copy_file",
    "mkdir": "mkdir",
}


def execution_results(event: dict[str, Any]) -> list[Any] | None:
    """이 이벤트가 고착화 실행의 결과 봉투인가. 아니면 None.

    타입 이름으로 고르지 않는다 — 결과 봉투는 `task_completed` 로 남고, 그 이름은
    에이전트 경로의 다른 카드도 함께 쓴다. 대신 **모양**으로 고른다: 실행 방식과
    단계별 결과 목록을 함께 싣는 것은 이 봉투뿐이다.
    """
    data = _loads(event.get("data"))
    if not isinstance(data, dict):
        return None
    results = data.get("results")
    if not isinstance(results, list):
        return None
    if "execution_mode" not in data and "llm_calls" not in data:
        return None
    return results


def action_from_execution_result(entry: Any, timestamp: str = "") -> Action | None:
    """실행 골격이 돌려준 결과 한 건을 행위로 되읽는다.

    골격의 원시 동작이 남기는 모양을 그대로 뒤집어 **도구 호출이었던 모습**으로
    되돌린 뒤, 에이전트 이력과 같은 분류기에 넣는다. 판정을 따로 짜면 같은 일을
    에이전트가 했을 때와 고착화 코드가 했을 때의 결론이 갈린다.

    되읽을 수 없는 것은 버리지 않는다. 버리면 "되돌릴 것이 없다"로 읽혀 조용히
    덧씌워지지만, 인자를 모르는 도구 호출로 남기면 "되돌릴 수 없다"가 되어 재작업
    전체가 에이전트에게 넘어간다 — 실패 방향을 안전한 쪽에 둔다.
    """
    if not isinstance(entry, dict):
        return None
    kind = str(entry.get("kind") or "").strip()
    if not kind:
        return None
    if kind == SHELL:
        return to_action(
            "run_shell",
            {"command": str(entry.get("command") or ""), "cwd": str(entry.get("cwd") or "")},
            result=entry.get("output"),
            timestamp=timestamp,
        )
    if kind == FILE_WRITE:
        # 조작 종류를 이름으로 옮긴다. 삭제·이동을 "쓰기"로 뭉뚱그리면 되돌리기가
        # 지워진 파일을 **또 지우려** 든다.
        operation = str(entry.get("op") or entry.get("mode") or "write")
        args: dict[str, Any] = {
            "file_path": str(entry.get("path") or entry.get("source") or ""),
        }
        if isinstance(entry.get("content"), str):
            args["content"] = entry["content"]
        return to_action(
            _RESULT_FILE_OP_TOOLS.get(operation, "write_file"),
            args,
            timestamp=timestamp,
        )
    if kind == FILE_READ:
        return to_action(
            "read_file",
            {"file_path": str(entry.get("path") or "")},
            result={"content": entry.get("content")},
            timestamp=timestamp,
        )
    arguments = entry.get("args")
    if isinstance(arguments, dict) and arguments:
        return to_action(
            str(entry.get("tool") or kind),
            arguments,
            result=entry.get("data"),
            timestamp=timestamp,
        )
    # 인자를 남기지 않은 옛 골격의 결과. 무엇을 했는지 모르므로 되돌릴 수도 없다.
    return Action(
        kind=MCP_CALL,
        tool=str(entry.get("tool") or kind),
        raw_args=dict(entry),
        result=entry.get("data"),
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# 이벤트 → 행위 목록
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 되돌리기로 넘긴 행위 — 그 회차의 "작업"이 아니다
# ---------------------------------------------------------------------------
# 재작업 회차의 이력에는 두 가지가 섞여 남는다. 직전 회차를 **되돌린 일**과 이번 회차가
# **새로 한 일**이다. 되돌리기까지 그 회차의 작업으로 읽으면, 다음 재작업은 그 되돌리기
# 마저 되돌리려 든다. 그런데 INSERT 를 되돌린 것은 DELETE 이고 DELETE 는 되돌릴 수 없다
# — 그 순간부터 이 활동은 영영 "되돌리기 불가"가 되어 재작업이 통째로 에이전트에게
# 넘어가고, 다시는 결정론적 경로로 돌아오지 못한다.
#
# 무엇을 되돌리라고 넘겼는지는 그 회차의 실행 카드에 단계 그대로 실려 있다. 짐작하지
# 않고 그것과 대조해서 일치하는 행위만 뺀다.

UNDO_STEP_TOOL = "mcp_call"
UNDO_STEP_DELETE_FILE = "file_delete"
UNDO_STEP_RESTORE_FILE = "file_restore"


def _sql_signature(value: Any) -> str:
    """같은 SQL인지 비교할 때 쓰는 모양. 공백·끝 세미콜론·대소문자는 무시한다."""
    return " ".join(str(value or "").split()).rstrip(";").casefold()


def _content_signature(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _args_signature(args: Any) -> str:
    return json.dumps(args or {}, ensure_ascii=False, sort_keys=True, default=str)


def handed_undo_steps(events: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """이 회차에 "먼저 되돌려라"라고 넘긴 단계들. 실행 카드에 남아 있다."""
    steps: list[dict[str, Any]] = []
    for event in events or []:
        data = _loads((event or {}).get("data"))
        if not isinstance(data, dict):
            continue
        found = data.get("undo_steps")
        if isinstance(found, list):
            steps.extend(step for step in found if isinstance(step, dict))
    return steps


def _undo_signatures(steps: Iterable[dict[str, Any]] | None) -> set[tuple]:
    signatures: set[tuple] = set()
    for step in steps or []:
        kind = str(step.get("kind") or "")
        if kind == UNDO_STEP_TOOL:
            args = step.get("args") if isinstance(step.get("args"), dict) else {}
            for value in args.values():
                if looks_like_sql(value):
                    signatures.add(("sql", _sql_signature(value)))
            signatures.add(("tool", str(step.get("tool") or ""), _args_signature(args)))
        elif kind == UNDO_STEP_DELETE_FILE:
            signatures.add(("file", "delete", str(step.get("path") or "")))
        elif kind == UNDO_STEP_RESTORE_FILE:
            # 되돌려 쓴 **내용**까지 같아야 한다. 경로만 보고 빼면 같은 파일에 대한 이번
            # 회차의 새 작업까지 함께 사라져, 다음 재작업이 그것을 되돌리지 못한다.
            signatures.add(
                ("file", "write", str(step.get("path") or ""),
                 _content_signature(step.get("content")))
            )
    return signatures


def _action_signatures(action: Action) -> set[tuple]:
    signatures: set[tuple] = set()
    if action.kind == MCP_CALL:
        args = action.raw_args or {}
        for value in args.values():
            if looks_like_sql(value):
                signatures.add(("sql", _sql_signature(value)))
        signatures.add(("tool", action.tool, _args_signature(args)))
    elif action.kind == FILE_WRITE:
        canonical = action.args or {}
        path = str(canonical.get("path") or "")
        operation = str(canonical.get("op") or "write")
        if operation == "delete":
            signatures.add(("file", "delete", path))
        elif operation == "write":
            signatures.add(
                ("file", "write", path, _content_signature(canonical.get("content")))
            )
    return signatures


def strip_undo_actions(
    actions: Iterable[Action], steps: Iterable[dict[str, Any]] | None
) -> list[Action]:
    """넘겨받은 되돌리기 단계와 일치하는 행위를 뺀다. 나머지가 그 회차의 작업이다."""
    undo = _undo_signatures(steps)
    if not undo:
        return list(actions)
    return [action for action in actions if not (_action_signatures(action) & undo)]


def normalize_events(events: Iterable[dict[str, Any]] | None) -> list[Action]:
    """events 행 목록을 시간 오름차순 행위 목록으로 바꾼다.

    끝난 이벤트를 우선한다(인자와 결과가 함께 있다). 끝난 이벤트가 **하나도** 없는
    런타임에 한해 시작 이벤트로 폴백한다 — 둘을 섞으면 같은 호출이 두 번 세어져
    실행 지문이 어긋난다.
    """
    rows = sorted(events or [], key=lambda e: str(e.get("timestamp") or ""))
    finished: list[Action] = []
    started: list[Action] = []

    for event in rows:
        results = execution_results(event)
        if results is not None:
            # 고착화 실행. 결과 봉투 하나에 그 실행이 한 일이 전부 들어 있다.
            for entry in results:
                action = action_from_execution_result(
                    entry, str(event.get("timestamp") or "")
                )
                if action is not None:
                    finished.append(action)
            continue

        event_type = str(event.get("event_type") or "").strip().lower()
        is_finished = event_type in _FINISHED_TYPES or (
            "tool" in event_type and event_type.endswith(("finished", "end", "completed"))
        )
        is_started = event_type in _STARTED_TYPES or (
            "tool" in event_type and event_type.endswith(("started", "start"))
        )
        if not (is_finished or is_started):
            continue

        payload = _payload(event)
        if payload is None:
            continue
        tool = _tool_name(payload)
        if not tool:
            continue

        action = to_action(
            tool,
            _arguments(payload),
            result=_first(payload, _RESULT_KEYS) if is_finished else None,
            timestamp=str(event.get("timestamp") or ""),
        )
        (finished if is_finished else started).append(action)

    # 되돌리기로 넘긴 단계는 이 회차의 작업이 아니다. 함께 읽으면 다음 재작업이
    # 되돌리기마저 되돌리려 들고, 그 시점부터 이 활동은 영영 "되돌리기 불가"가 된다.
    return strip_undo_actions(finished or started, handed_undo_steps(rows))


def effect_actions(actions: Iterable[Action]) -> list[Action]:
    """세상을 바꾼 행위만. 고착화 대상이자 보상 대상이다."""
    return [a for a in actions if a.has_effect]


def canonicalize(actions: Iterable[Action]) -> list[Action]:
    """결과에 남는 것만 남긴 최소 행위 목록.

    고착화의 자격은 "같은 발자국을 밟았는가"가 아니라 **"같은 것을 남겼는가"** 여야 한다.
    에이전트는 같은 산출물을 두고도 매번 조금씩 다르게 움직인다 — 폴더를 미리 만들기도
    하고, 내용을 한 번 찍어 보기도 하고, 곧바로 파일을 쓰기도 한다. 그 곁가지까지 방식의
    일부로 세면 같은 결과를 낸 실행이 서로 다른 방식으로 갈려, 반복되는 작업인데도 영영
    굳지 않는다.

    그래서 지문을 재기 전에 이력을 결과 기준으로 접는다.

    - 세상을 바꾸지 않는 행위는 이미 맥락으로 빠진다(`classify`).
    - 뒤따르는 파일 쓰기가 어차피 만들 디렉터리를 미리 만드는 행위는 뺀다. 골격의
      `write_file` 이 부모 디렉터리를 만들므로 재현해도 남는 것이 같다.
    - 같은 경로에 여러 번 덮어썼으면 마지막 것만 남긴다. 최종 상태가 곧 결과다.

    부분 수정(`edit`)은 접지 않는다. 앞의 내용에 기대어 고치는 것이라 마지막 하나만
    재현하면 다른 결과가 나온다.

    남긴 것이 없어도 **결과가 뒤 단계로 흘러갔으면** 접지 않는다. `mkdir -p x && date`
    는 디렉터리 하나 만드는 것이 전부지만, 찍은 시각이 다음 단계의 문서에 들어갔다면
    그 행위는 결과의 일부다. 접어 버리면 다음 실행에서 그 값을 만들 방법이 사라진다.
    """
    effects = [action for action in actions if action.has_effect]

    def _result_feeds_later(action: Action, rest: list[Action]) -> bool:
        """이 행위가 낸 값이 뒤 단계의 인자로 흘러갔는가."""
        if action.result is None:
            return False
        text = (
            action.result if isinstance(action.result, str)
            else json.dumps(action.result, ensure_ascii=False, default=str)
        )
        # 짧은 토막은 아무 데나 우연히 걸린다(`ok` 가 `okay` 에 걸리는 식).
        produced = {line.strip() for line in str(text).splitlines() if len(line.strip()) >= 4}
        if not produced:
            return False
        for later in rest:
            blob = json.dumps(later.args, ensure_ascii=False, default=str)
            if any(value in blob for value in produced):
                return True
        return False

    def _dirs_made(action: Action) -> tuple[str, ...]:
        if action.kind == FILE_WRITE and str(action.args.get("op") or "") == "mkdir":
            path = str(action.args.get("path") or "")
            return (path,) if path else ()
        if action.kind == SHELL:
            return shell_made_dirs(str(action.args.get("command") or "")) or ()
        return ()

    def _written_paths(rest: list[Action]) -> list[str]:
        return [
            str(a.args.get("path") or "")
            for a in rest
            if a.kind == FILE_WRITE and str(a.args.get("op") or "write") in ("write", "edit")
        ]

    kept: list[Action] = []
    for index, action in enumerate(effects):
        rest = effects[index + 1:]

        made = _dirs_made(action)
        if made and not _result_feeds_later(action, rest):
            covered = _written_paths(rest)
            if covered and all(
                any(written.startswith(directory.rstrip("/") + "/") for written in covered)
                for directory in made
            ):
                continue

        if action.kind == FILE_WRITE and str(action.args.get("op") or "write") == "write":
            path = str(action.args.get("path") or "")
            if path and any(
                a.kind == FILE_WRITE
                and str(a.args.get("op") or "write") == "write"
                and str(a.args.get("path") or "") == path
                for a in rest
            ):
                continue

        kept.append(action)
    return kept


def summarize(actions: Iterable[Action]) -> dict[str, Any]:
    """작업 이력 요약. 생성된 코드의 출처를 사람이 읽을 수 있게 남긴다.

    코드가 왜 이렇게 생겼는지를 되짚을 때 필요한 것은 부수효과 목록만이 아니다.
    어떤 스킬 절차를 따랐고 어떤 파일을 참고했는지가 함께 있어야 나중에 그 절차가
    바뀌었을 때 코드를 의심할 수 있다.
    """
    actions = list(actions)
    by_kind: dict[str, int] = {}
    for action in actions:
        by_kind[action.kind] = by_kind.get(action.kind, 0) + 1

    def _paths(kind: str) -> list[str]:
        seen: list[str] = []
        for a in actions:
            if a.kind == kind:
                path = str(a.args.get("path") or "")
                if path and path not in seen:
                    seen.append(path)
        return seen

    return {
        "action_count": len(actions),
        "by_kind": by_kind,
        "tools_used": list(dict.fromkeys(a.tool for a in actions if a.tool)),
        "mcp_calls": list(dict.fromkeys(a.tool for a in actions if a.kind == MCP_CALL)),
        "skills_read": _paths(SKILL_READ),
        "files_read": _paths(FILE_READ),
        "files_written": _paths(FILE_WRITE),
        "shell_commands": list(dict.fromkeys(
            str(a.args.get("command") or "") for a in actions if a.kind == SHELL
        )),
        "subagents": list(dict.fromkeys(
            str(a.args.get("subagent") or "") for a in actions if a.kind == DELEGATE
        )),
    }


