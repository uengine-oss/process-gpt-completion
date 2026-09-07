"""실행 지문(execution fingerprint) 생성과 파라미터 식별.

고착화 트리거는 "같은 활동이 같은 구조로 N번 실행되었는가"로 판정한다. 도구
이름만으로는 판정할 수 없다 — SQL 실행 도구의 이름은 테넌트 MCP 구성마다 다르고,
같은 이름이어도 전혀 다른 작업일 수 있다. 그래서 인자 내용에서 리터럴만 지운
구조 지문으로 비교한다(`pg_stat_statements`의 쿼리 정규화와 같은 발상).

지문이 같은 표본들을 리터럴 위치별로 대조하면 무엇이 파라미터이고 무엇이 상수인지
관측으로 드러난다. LLM 추측이 필요 없다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any

# 인자 값이 SQL인지 판별하는 선두 키워드. 도구 이름이나 인자 키 이름에 의존하지 않는다.
_SQL_HEADS = (
    "SELECT", "INSERT", "UPDATE", "DELETE", "MERGE",
    "WITH", "CREATE", "ALTER", "DROP", "TRUNCATE", "EXPLAIN",
)
# 부수효과가 없는 읽기 전용 구문. 지문과 보상 대상에서 제외한다.
_READONLY_HEADS = ("SELECT", "WITH", "EXPLAIN", "SHOW")

# 부수효과 추적 대상이 아닌 내부 도구.
EXCLUDED_TOOLS = {"mem0", "memento", "human_asked", "dmn_rule"}

_IDENT_BEFORE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|<>|!=|>=|<=|>|<|\+|-|,|\()?\s*$")


@dataclass(frozen=True)
class SqlLiteral:
    """정규화 과정에서 `?`로 치환된 리터럴 하나."""

    value: Any
    kind: str  # "string" | "number"
    start: int  # 원문에서의 시작 오프셋
    end: int  # 원문에서의 끝 오프셋(배타)
    name_hint: str | None = None


@dataclass(frozen=True)
class SqlNormalization:
    signature: str
    literals: tuple[SqlLiteral, ...] = field(default=())


def looks_like_sql(value: Any) -> bool:
    """문자열 값이 SQL 구문으로 시작하는지 본다."""
    if not isinstance(value, str):
        return False
    head = _strip_leading_noise(value)
    if not head:
        return False
    return head.split(None, 1)[0].upper() in _SQL_HEADS


def is_readonly_sql(value: str) -> bool:
    """읽기 전용 SQL인지 본다. 쓰기 구문이 하나라도 섞이면 False."""
    head = _strip_leading_noise(value)
    if not head:
        return False
    if head.split(None, 1)[0].upper() not in _READONLY_HEADS:
        return False
    # `WITH ... INSERT/UPDATE/DELETE` 형태의 쓰기 CTE를 걸러낸다.
    stripped = _blank_out_literals(value).upper()
    return not re.search(r"\b(INSERT|UPDATE|DELETE|MERGE|CREATE|ALTER|DROP|TRUNCATE)\b", stripped)


def _strip_leading_noise(value: str) -> str:
    """선두 공백과 주석을 걷어낸 나머지를 돌려준다."""
    i, n = 0, len(value)
    while i < n:
        if value[i].isspace():
            i += 1
        elif value.startswith("--", i):
            j = value.find("\n", i)
            i = n if j < 0 else j + 1
        elif value.startswith("/*", i):
            j = value.find("*/", i + 2)
            i = n if j < 0 else j + 2
        else:
            break
    return value[i:]


def _scan(sql: str) -> list[tuple[str, int, int]]:
    """SQL을 훑어 (종류, 시작, 끝) 구간 목록을 만든다.

    종류: literal-string, literal-number, ident-quoted, comment, plain.
    큰따옴표는 PostgreSQL에서 **식별자**이므로 리터럴로 취급하지 않는다. 이를 지우면
    서로 다른 테이블이 같은 지문을 갖게 되어 잘못된 고착화로 이어진다.
    """
    spans: list[tuple[str, int, int]] = []
    i, n = 0, len(sql)
    plain_start = 0

    def flush(upto: int) -> None:
        if upto > plain_start:
            spans.append(("plain", plain_start, upto))

    while i < n:
        ch = sql[i]
        if sql.startswith("--", i):
            flush(i)
            j = sql.find("\n", i)
            j = n if j < 0 else j
            spans.append(("comment", i, j))
            i = plain_start = j
        elif sql.startswith("/*", i):
            flush(i)
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            spans.append(("comment", i, j))
            i = plain_start = j
        elif ch == "'" or (ch in "eE" and sql.startswith("'", i + 1)):
            flush(i)
            start = i
            escaped = ch in "eE"
            i += 2 if escaped else 1
            while i < n:
                if escaped and sql[i] == "\\":
                    i += 2
                    continue
                if sql[i] == "'":
                    if sql.startswith("''", i):  # 표준 SQL의 '' 이스케이프
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            spans.append(("literal-string", start, i))
            plain_start = i
        elif ch == '"':
            flush(i)
            start = i
            i += 1
            while i < n:
                if sql[i] == '"':
                    if sql.startswith('""', i):
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            spans.append(("ident-quoted", start, i))
            plain_start = i
        elif ch.isdigit() and (i == 0 or not (sql[i - 1].isalnum() or sql[i - 1] == "_")):
            flush(i)
            start = i
            while i < n and (sql[i].isdigit() or sql[i] == "."):
                i += 1
            spans.append(("literal-number", start, i))
            plain_start = i
        else:
            i += 1
    flush(n)
    return spans


def _blank_out_literals(sql: str) -> str:
    """리터럴과 주석을 공백으로 지운 문자열(키워드 탐색용)."""
    out = list(sql)
    for kind, start, end in _scan(sql):
        if kind in ("literal-string", "literal-number", "comment", "ident-quoted"):
            for k in range(start, end):
                out[k] = " "
    return "".join(out)


def _literal_value(kind: str, raw: str) -> Any:
    if kind == "literal-number":
        return float(raw) if "." in raw else int(raw)
    body = raw
    if body[:1] in ("e", "E"):
        body = body[1:]
    body = body[1:-1] if len(body) >= 2 else body
    return body.replace("''", "'")


# `INSERT INTO t (컬럼들) VALUES (값들)` 을 찾는다. 값 목록 안에 함수 호출 같은
# 괄호가 끼면 대응이 어긋나므로, 괄호가 없는 단순한 형태만 본다.
_INSERT_COLUMNS = re.compile(
    r"\bINSERT\s+INTO\s+[^\s(]+\s*\(([^()]*)\)\s*VALUES\s*\(", re.IGNORECASE
)


def _insert_column_hints(sql: str, literals: list[SqlLiteral]) -> list[SqlLiteral]:
    """`INSERT ... VALUES` 의 값 자리에 대응하는 컬럼 이름을 붙인다.

    `_IDENT_BEFORE` 는 값 **바로 앞** 의 식별자를 본다. VALUES 목록에서는 값 앞이
    쉼표뿐이라 어떤 이름도 붙지 않고, 파라미터 이름이 `query_7` 같은 자리 번호가 된다.
    그러면 그 자리가 무엇인지 알 길이 없어 다음 실행에서 값을 되찾지 못하고, 이름표도
    없으니 위치 폴백으로 떨어져 **엉뚱한 값을 확신 있게** 넣는다(실제로 proc_inst_id
    자리에 신청자 이름이 들어간 적이 있다). 컬럼 목록과 값 목록은 순서로 대응하므로
    여기서 이어 준다.

    개수가 맞지 않으면 아무것도 하지 않는다. 어긋난 대응은 없는 것만 못하다.
    """
    blanked = _blank_out_literals(sql)
    match = _INSERT_COLUMNS.search(blanked)
    if not match:
        return literals
    columns = [c.strip().strip('"').lower() for c in match.group(1).split(",")]
    if not all(columns):
        return literals
    close = blanked.find(")", match.end())
    if close < 0:
        return literals
    inside = [i for i, lit in enumerate(literals) if match.end() <= lit.start < close]
    if len(inside) != len(columns):
        return literals
    hinted = list(literals)
    for column, index in zip(columns, inside):
        hinted[index] = replace(hinted[index], name_hint=column)
    return hinted


def normalize_sql(sql: str) -> SqlNormalization:
    """리터럴을 `?`로 치환한 지문과 치환된 리터럴 목록을 만든다.

    비리터럴 구간은 대문자로 접고 공백을 단일화한다. 큰따옴표 식별자는 대소문자가
    의미를 가지므로 원문 그대로 둔다.
    """
    parts: list[str] = []
    literals: list[SqlLiteral] = []
    for kind, start, end in _scan(sql):
        chunk = sql[start:end]
        if kind == "comment":
            parts.append(" ")
        elif kind == "ident-quoted":
            parts.append(chunk)
        elif kind in ("literal-string", "literal-number"):
            prefix = "".join(parts)
            match = _IDENT_BEFORE.search(prefix.rstrip())
            literals.append(
                SqlLiteral(
                    value=_literal_value(kind, chunk),
                    kind="number" if kind == "literal-number" else "string",
                    start=start,
                    end=end,
                    name_hint=match.group(1).lower() if match else None,
                )
            )
            parts.append("?")
        else:
            parts.append(chunk.upper())

    signature = re.sub(r"\s+", " ", "".join(parts)).strip()
    # IN (?, ?, ?) 처럼 길이만 다른 목록은 같은 구조로 본다.
    signature = re.sub(r"\(\s*\?(?:\s*,\s*\?)+\s*\)", "(?)", signature)
    return SqlNormalization(
        signature=signature, literals=tuple(_insert_column_hints(sql, literals))
    )


# ---------------------------------------------------------------------------
# 셸 명령의 부수효과 판정
# ---------------------------------------------------------------------------
#
# SQL은 오래전부터 읽기 전용을 가려냈다(`is_readonly_sql`). 셸은 그러지 않아서
# `date` 로 시각을 찍는 것, `printf` 로 내용을 미리 보는 것, `ls` 로 확인하는 것이 전부
# "세상을 바꾼 행위"로 세어졌다. 그 결과 산출물에 아무 기여도 없는 행위가 실행 지문에
# 들어가, 같은 결과를 낸 실행이 서로 다른 방식으로 갈렸다 — 문서 하나를 만드는 활동이
# 21번을 돌고도 굳지 않은 주된 이유다.
#
# 판정 기준은 SQL과 같다. 이름이 아니라 **하는 일**로 본다.

# 세상을 바꾸지 않는 프로그램. 여기 없는 이름은 바꾸는 것으로 본다 — 놓치는 쪽이 더 위험하다.
_READONLY_PROGRAMS = {
    "date", "echo", "printf", "ls", "ll", "cat", "pwd", "whoami", "hostname",
    "wc", "head", "tail", "grep", "egrep", "fgrep", "find", "which", "type",
    "stat", "du", "df", "env", "printenv", "basename", "dirname", "realpath",
    "readlink", "sort", "uniq", "cut", "tr", "diff", "file", "test", "true",
    "false", "sleep", "id", "uname", "seq", "expr", "jq", "awk", "sed", "python",
    "python3", "node",
}
# 자리에서 고쳐 쓰는 옵션. `sed -i` 는 읽기 도구처럼 보이지만 파일을 바꾼다.
_IN_PLACE = re.compile(r"^-i(\.|$)|^--in-place")
# 표준 스트림을 옮기는 표기(`2>&1`, `>&2`). 파일로 쓰는 리다이렉션과 구분한다.
_STREAM_REDIRECT = re.compile(r"\d*>&\d*|&>")
# 절 구분자. 이 중 하나라도 쓰는 절이 있으면 명령 전체가 부수효과를 낸다.
_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|\n]")
# 인터프리터에 스크립트를 먹이는 형태(`python - <<PY`, `node -e "…"`). 안에서 무엇을
# 하는지 알 수 없으므로 읽기 전용으로 보지 않는다.
_SCRIPT_INTERPRETERS = {"python", "python3", "node"}


def _mask_quoted(text: str) -> str:
    """따옴표 안을 공백으로 덮는다. 구분자·리다이렉션 탐색용."""
    out = list(text)
    quote = ""
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = ""
            out[index] = " "
        elif char in "'\"":
            quote = char
            out[index] = " "
    return "".join(out)


def _shell_segments(command: str) -> list[str]:
    """명령을 절 단위로 나눈다. 따옴표 안의 구분자는 무시한다."""
    masked = _mask_quoted(command)
    segments: list[str] = []
    cursor = 0
    for match in _SEGMENT_SPLIT.finditer(masked):
        segments.append(command[cursor:match.start()])
        cursor = match.end()
    segments.append(command[cursor:])
    return [segment for segment in segments if segment.strip()]


def _program_and_args(segment: str) -> tuple[str, list[str]]:
    """절의 실행 프로그램과 인자. 앞에 붙는 환경변수 대입은 건너뛴다."""
    tokens = [token for token in segment.split() if token]
    while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
        tokens = tokens[1:]
    if not tokens:
        return "", []
    program = tokens[0].strip("\"'").rsplit("/", 1)[-1]
    return program, tokens[1:]


def is_readonly_shell(command: str) -> bool:
    """셸 명령이 세상을 바꾸지 않는가.

    절 하나라도 바꾸면 명령 전체가 바꾸는 것이다 — `mkdir -p x && date` 는 읽기 전용이
    아니다. 파일로 내보내는 리다이렉션(`>`, `>>`)이 있어도 마찬가지다. 스트림을 옮기는
    `2>&1` 은 파일을 만들지 않으므로 리다이렉션으로 세지 않는다.
    """
    text = str(command or "")
    if not text.strip():
        return False
    masked = _STREAM_REDIRECT.sub(" ", _mask_quoted(text))
    if ">" in masked or "<<" in masked:
        return False
    for segment in _shell_segments(text):
        program, arguments = _program_and_args(segment)
        if program not in _READONLY_PROGRAMS:
            return False
        if program in _SCRIPT_INTERPRETERS:
            # 스크립트를 먹이는 호출은 안에서 무엇을 하는지 알 수 없다.
            return False
        if any(_IN_PLACE.match(argument) for argument in arguments):
            return False
    return True


def shell_made_dirs(command: str) -> tuple[str, ...] | None:
    """이 명령이 디렉터리 만들기 말고는 세상을 바꾸지 않는가. 맞으면 만든 경로들.

    `mkdir -p "…/expense" && date -Iseconds` 처럼, 뒤에 오는 파일 쓰기가 어차피 만들
    디렉터리를 미리 만들어 두는 절이 흔하다. 그 자체로는 결과에 아무 흔적도 남기지
    않으므로(파일 쓰기가 부모 디렉터리를 만든다) 재현 대상에서 뺄 수 있는지 가린다.
    """
    text = str(command or "")
    if not text.strip():
        return None
    masked = _STREAM_REDIRECT.sub(" ", _mask_quoted(text))
    if ">" in masked or "<<" in masked:
        return None
    made: list[str] = []
    for segment in _shell_segments(text):
        program, arguments = _program_and_args(segment)
        if program == "mkdir":
            paths = [a.strip("\"'") for a in arguments if not a.startswith("-")]
            if not paths:
                return None
            made.extend(paths)
            continue
        if program not in _READONLY_PROGRAMS or program in _SCRIPT_INTERPRETERS:
            return None
        if any(_IN_PLACE.match(argument) for argument in arguments):
            return None
    return tuple(made) or None


def _value_shape(value: Any) -> Any:
    """SQL이 아닌 값의 타입 골격. 값 자체는 버리고 구조만 남긴다."""
    if isinstance(value, dict):
        return {k: _value_shape(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        shapes: list[Any] = []
        for item in value:
            shape = _value_shape(item)
            if shape not in shapes:
                shapes.append(shape)
        return shapes  # 길이는 무시하고 원소 타입 집합만 본다
    if isinstance(value, bool):
        return "<bool>"
    if isinstance(value, int):
        return "<int>"
    if isinstance(value, float):
        return "<float>"
    if value is None:
        return "<null>"
    return "<str>"


def as_call(item: Any) -> tuple[str, str, dict[str, Any]]:
    """지문·파라미터 식별의 입력을 `(kind, tool, args)`로 통일한다.

    입력은 `work_history.Action`(kind가 있는 정규화된 행위)일 수도, 예전처럼
    `(tool, args)` 튜플일 수도 있다. 이 모듈은 work_history를 import하지 않는다 —
    반대 방향 의존이 이미 있어 순환이 되기 때문이다. 그래서 속성 유무로 본다.
    """
    kind = getattr(item, "kind", None)
    if kind is not None:
        return str(kind), str(getattr(item, "tool", "")), dict(getattr(item, "args", {}) or {})
    tool, args = item
    return "", str(tool or ""), dict(args or {})


# 행위 종류 이름. `work_history`가 이 값을 그대로 쓴다(정의는 한 곳에 둔다).
# 지문과 단계 간 바인딩이 종류별로 다르게 움직이므로 이 모듈이 이름을 안다.
SHELL_KIND = "shell"
MCP_CALL_KIND = "mcp_call"
FILE_WRITE_KIND = "file_write"

# 골격이 각 행위의 결과를 담는 키. 관측된 결과와 생성 코드의 `results[N]` 를 잇는
# 다리다 — 셸은 표준출력을 `output` 에, MCP 도구는 반환값을 `data` 에 싣는다.
# 파일 조작 결과에는 경로와 크기밖에 없어 이어받을 값이 없으므로 넣지 않는다.
RESULT_BODY_KEY: dict[str, str] = {SHELL_KIND: "output", MCP_CALL_KIND: "data"}


def _command_shape(command: str) -> str:
    """셸 명령의 구조. 실행 프로그램과 인자 개수만 남긴다.

    일반 문자열 인자는 값이 달라도 같은 구조로 본다(`<str>`). 셸 명령은 다르다 —
    문자열 자체가 곧 수행할 행위이므로, `python3 load.py` 와 `bash clean.sh --all` 을
    같은 실행으로 보면 서로 다른 일을 하나로 굳혀 버린다.
    """
    tokens = text_segments(command)
    program = str(tokens[0].value) if tokens else ""
    return f"{program}#{len(tokens)}"


# 파일 조작의 경로 인자. 지문에서 구조를 따로 본다.
_PATH_ARGS = ("path", "source", "destination")


def _path_shape(value: str) -> str:
    """파일 경로의 구조. 깊이와 확장자만 남긴다.

    일반 문자열처럼 `<str>` 로 뭉뚱그리면 안 된다 — `/workspace/expense/a.md` 와
    `/workspace/.bpmn/{인스턴스}/expense/a.md` 가 같은 실행으로 묶인다. 둘은 **다른 자리**에
    쓴 것이고, 그런 표본으로 굳히면 어느 한쪽은 엉뚱한 곳에 파일을 만든다. 실제로 같은
    활동에서 두 경로가 번갈아 나왔다.

    반대로 값까지 보면 안 된다. 경로에는 실행마다 달라지는 조각(인스턴스 식별자, 신청자
    이름)이 들어 있어, 값을 보면 어떤 두 실행도 같은 지문을 갖지 못한다. 셸 명령을
    `프로그램#토큰수` 로 보는 것과 같은 발상이다.
    """
    text = str(value)
    parts = [p for p in text.split("/") if p]
    tail = parts[-1] if parts else ""
    suffix = tail[tail.rfind("."):] if "." in tail[1:] else ""
    return f"{'/' if text.startswith('/') else ''}#{len(parts)}{suffix}"


def call_fingerprint(item: Any, args: dict[str, Any] | None = None) -> str:
    """행위 하나의 구조 지문.

    `call_fingerprint(action)` 과 `call_fingerprint(tool, args)` 를 모두 받는다.
    """
    kind, tool, call_args = as_call(item if args is None else (item, args))
    rendered: dict[str, Any] = {}
    for key in sorted(call_args):
        value = call_args[key]
        if looks_like_sql(value):
            rendered[key] = normalize_sql(value).signature
        elif kind == SHELL_KIND and key == "command" and isinstance(value, str):
            rendered[key] = _command_shape(value)
        elif kind == FILE_WRITE_KIND and key in _PATH_ARGS and isinstance(value, str):
            rendered[key] = _path_shape(value)
        else:
            rendered[key] = _value_shape(value)
    prefix = f"{kind}:" if kind else ""
    return f"{prefix}{tool}{json.dumps(rendered, ensure_ascii=False, sort_keys=True)}"


def execution_fingerprint(calls: list[Any]) -> str:
    """실행 하나(행위 목록)의 지문. 행위 종류·순서·횟수가 반영된다."""
    joined = "\n".join(call_fingerprint(call) for call in calls)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def is_write_call(tool_name: str, args: dict[str, Any], excluded: set[str]) -> bool:
    """지문·보상 대상이 되는 쓰기 호출인지 판정한다.

    도구 이름이 아니라 인자 내용으로 SQL 여부와 읽기 전용 여부를 본다.
    """
    if not tool_name or tool_name in excluded:
        return False
    sql_values = [v for v in (args or {}).values() if looks_like_sql(v)]
    if sql_values:
        return any(not is_readonly_sql(v) for v in sql_values)
    return True


# ---------------------------------------------------------------------------
# SQL 역연산 — 관측된 구문에서 되돌리는 구문을 만든다
# ---------------------------------------------------------------------------
#
# 순방향은 관측으로 만드는데 되돌리기는 LLM이 추측으로 지어 왔다. 되돌리기가 추측이면
# 재작업 경로 전체가 추측 위에 선다 — 무엇이 지워졌는지 아무도 모른 채 다시 실행된다.
#
# 되돌릴 수 있는 구문은 몇 안 된다. 그 몇 개만 관측에서 정확히 만들고, 나머지는
# **되돌릴 수 없다고 말한다**. 지어낸 되돌리기보다 "못 되돌린다"가 낫다.

_INSERT_STMT = re.compile(
    r"\bINSERT\s+INTO\s+([^\s(]+)\s*\(([^()]*)\)\s*VALUES\s*\(", re.IGNORECASE
)
# `UPDATE t SET c = c + 3 WHERE …` 처럼 제자리 증감만 되돌린다. 값을 덮어쓴 UPDATE는
# 이전 값을 모르므로 되돌릴 수 없다.
_UPDATE_DELTA = re.compile(
    r"^\s*UPDATE\s+(?P<table>[^\s]+)\s+SET\s+(?P<column>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?P=column)\s*(?P<sign>[+-])\s*(?P<amount>\d+(?:\.\d+)?)\s*(?P<rest>WHERE\b.*)?$",
    re.IGNORECASE | re.DOTALL,
)


def invert_sql(sql: str) -> str | None:
    """이 SQL을 되돌리는 SQL. 되돌릴 수 없으면 None.

    - `INSERT INTO t (컬럼들) VALUES (값들)` → 넣은 값 전부를 조건으로 하는 `DELETE`.
      일부 컬럼만 조건에 넣으면 남의 행까지 지운다. 값은 원문 그대로 옮겨 따옴표·형변환을
      보존한다.
    - `UPDATE t SET c = c + n …` → 부호를 뒤집은 같은 구문. `WHERE` 절은 그대로 둔다.
    - `DELETE`, 값을 덮어쓴 `UPDATE`, `CREATE`/`DROP` → 이전 상태를 모르므로 되돌릴 수 없다.
    """
    text = str(sql or "")
    if not looks_like_sql(text):
        return None
    blanked = _blank_out_literals(text)

    match = _INSERT_STMT.search(blanked)
    if match:
        table = text[match.start(1):match.end(1)]
        columns = [c.strip() for c in text[match.start(2):match.end(2)].split(",")]
        close = blanked.find(")", match.end())
        if close < 0 or not all(columns):
            return None
        literals = [
            lit for lit in normalize_sql(text).literals if match.end() <= lit.start < close
        ]
        if len(literals) != len(columns):
            # 값 자리에 함수 호출 같은 것이 섞였다. 짝이 어긋난 조건으로 지우면
            # 엉뚱한 행이 사라진다.
            return None
        pairs = [
            f"{column} = {text[lit.start:lit.end]}"
            for column, lit in zip(columns, literals)
        ]
        return f"DELETE FROM {table} WHERE " + " AND ".join(pairs)

    delta = _UPDATE_DELTA.match(text.strip())
    if delta:
        flipped = "-" if delta.group("sign") == "+" else "+"
        column, amount = delta.group("column"), delta.group("amount")
        rest = (delta.group("rest") or "").strip()
        statement = f"UPDATE {delta.group('table')} SET {column} = {column} {flipped} {amount}"
        return f"{statement} {rest}".strip()

    return None


# ---------------------------------------------------------------------------
# 값 표기 정규화 — 같은 값을 같은 값으로 본다
# ---------------------------------------------------------------------------

# 금액 표기. 자릿수 쉼표와 통화 기호·단위만 걷어낸다. 걷어낼 것을 좁게 잡아야
# `1건` 과 `1개` 가 같은 값으로 접히지 않는다 — 단위가 다르면 다른 값이다.
_NUMBER_TEXT = re.compile(
    r"^[\u20a9$\u20ac\u00a5\s]*(-?\d[\d,]*(?:\.\d+)?)\s*(?:\uc6d0|KRW|USD|won)?$",
    re.IGNORECASE,
)
# 날짜·시각 표기. `2026-09-01` / `2026/9/1` / `2026\ub1449\uc6d41\uc77c` /
# `2026-09-01T03:17:21+00:00` / `2026-09-01 03:02:24` / `2026-09-01T02:46:44Z` 를 모두 읽는다.
_TEMPORAL_TEXT = re.compile(
    r"^\s*(?P<y>\d{4})\s*(?:[-/.]|\ub144)\s*(?P<m>\d{1,2})\s*(?:[-/.]|\uc6d4)\s*(?P<d>\d{1,2})\s*\uc77c?"
    r"(?:[T\s]+(?P<hh>\d{1,2}):(?P<mi>\d{2})(?::(?P<ss>\d{2}))?(?:\.\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?\s*$"
)
# 지시문 안에서 값 후보를 찾을 때 쓰는 토막.
_NUMBER_TOKEN = re.compile(r"-?[\d,]*\d(?:\.\d+)?")
_DATE_TOKEN = re.compile(
    r"\d{4}\s*(?:[-/.]|\ub144)\s*\d{1,2}\s*(?:[-/.]|\uc6d4)\s*\d{1,2}\s*\uc77c?"
    r"(?:[T\s]+\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
)


def _as_number(text: Any) -> float | None:
    """금액·수량 표기를 하나의 수로 접는다. `27,000\uc6d0` \u00b7 `\u20a927000` \u00b7 `27000.0` \u2192 27000.0"""
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return float(text)
    match = _NUMBER_TEXT.match(str(text).strip())
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _as_temporal(text: Any) -> tuple[int, ...] | None:
    """날짜·시각 표기를 (연, 월, 일[, 시, 분, 초])로 접는다.

    시간대 표기(`Z` \u00b7 `+00:00`)와 소수 이하 초는 버리고 벽시계 값만 본다. 같은 실행이
    남긴 같은 시각을 다른 서식으로 적은 것을 잇는 것이 목적이라, 서로 다른 지역시를
    같은 순간으로 환산할 일은 없다.
    """
    match = _TEMPORAL_TEXT.match(str(text))
    if not match:
        return None
    parts = [int(match.group("y")), int(match.group("m")), int(match.group("d"))]
    if not (1 <= parts[1] <= 12 and 1 <= parts[2] <= 31):
        return None
    if match.group("hh") is not None:
        parts += [int(match.group("hh")), int(match.group("mi")), int(match.group("ss") or 0)]
    return tuple(parts)


def same_value(left: Any, right: Any) -> bool:
    """두 관측값이 같은 값을 가리키는가. 표기 차이는 무시한다.

    실행마다 에이전트가 금액을 `29,000\uc6d0` 으로도 `29000` 으로도 적고, 시각을
    `2026-09-01T02:46:44Z` 로도 `2026-09-01 02:46:44` 로도 적는다. 표기가 다르다는
    이유로 다른 값이라고 보면, 자리 대조가 어긋나 굳을 수 있는 활동이 영영 안 굳는다.

    문자열이 다르고 수로도 시각으로도 읽히지 않으면 다른 값이다. 접는 범위를 넓히면
    반대쪽 위험이 온다 \u2014 다른 값을 같다고 보면 변하는 자리를 상수로 굳혀 버린다.
    """
    if left is None or right is None:
        return False
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    text_left, text_right = str(left).strip(), str(right).strip()
    if text_left == text_right:
        return bool(text_left)
    if not text_left or not text_right:
        return False
    number_left, number_right = _as_number(text_left), _as_number(text_right)
    if number_left is not None and number_right is not None:
        return number_left == number_right
    temporal_left, temporal_right = _as_temporal(text_left), _as_temporal(text_right)
    if temporal_left is not None and temporal_right is not None:
        return temporal_left == temporal_right
    return False



# ---------------------------------------------------------------------------
# 표본 대조 → 파라미터 식별
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """치환 가능한 값 조각 하나. SQL 리터럴이거나 일반 문자열의 토큰이다."""

    value: Any
    start: int
    end: int
    quoted: bool = False  # 치환할 때 따옴표를 다시 씌워야 하는가(SQL 문자열 리터럴)
    name_hint: str | None = None


# 일반 문자열의 토큰. 네 갈래로 나눈다.
# 1) 따옴표로 묶인 덩어리는 통째로 한 토큰 — 셸의 `-c "UPDATE ..."` 를 쪼개면 어긋난다.
# 2) 날짜·시각과 금액은 통째로 한 토큰. 구두점에서 끊으면 `2026-09-01T02:46:44Z` 가
#    `2026-09-01T02` / `46` / `44Z` 로, `30,000원` 이 `30` / `,` / `000원` 으로 갈라져
#    값이 아니라 토막이 된다. 토막은 지시문에도 앞 단계 결과에도 없어 되찾을 길이 없고,
#    무엇보다 표기가 흔들리면(`30,000원` 대 `45000`) 표본마다 조각 수가 달라져 자리
#    대조 자체가 무산된다 — 그러면 값 전체가 통짜 파라미터가 된다.
# 3) 값처럼 보이는 낱말(경로·옵션·수량·식별자). 경로 구분자와 하이픈은 낱말에 포함한다.
# 4) 그 밖의 구두점 한 글자.
# 2)의 두 갈래는 낱말 한가운데를 자르지 않도록 뒤가 낱말 문자가 아닐 때만 인정한다.
# 그러지 않으면 `2026-09-09_홍길동.md` 나 `1.2.3` 이 조각으로 쪼개진다.
# 구두점을 낱말에서 떼어내야 `노트북:` 과 `노트북` 이 같은 값으로 묶인다. 붙여 두면
# 같은 값이 자리마다 다른 파라미터로 갈라져, 호출자가 같은 값을 여러 번 넘겨야 한다.
_WORD_CHAR = r"(?![\w./+@-])"
_TOKEN = re.compile(
    r"""\"[^\"]*\"|'[^']*'"""
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?" + _WORD_CHAR
    + r"|-?\d[\d,]*(?:\.\d+)?(?:원|KRW|USD)?" + _WORD_CHAR
    + r"|[\w./+@-]+|[^\s\w]"
)
# 값 앞에 붙는 옵션 이름. `--file report.csv` 의 `file` 을 파라미터 이름으로 쓴다.
_FLAG = re.compile(r"^-{1,2}([A-Za-z][A-Za-z0-9_-]*)$")


def text_segments(value: str) -> tuple[Segment, ...]:
    """SQL이 아닌 문자열을 토큰 단위로 쪼갠다.

    셸 명령이나 파일 경로처럼 "대부분 같고 일부만 다른" 문자열에서 다른 자리만
    파라미터로 뽑기 위한 것이다. 값 전체를 파라미터로 잡으면 명령 전체를 호출자가
    다시 조립해야 해서 고착화의 의미가 없어진다.
    """
    segments: list[Segment] = []
    previous: str | None = None
    for match in _TOKEN.finditer(value):
        token = match.group()
        flag = _FLAG.match(previous or "")
        segments.append(
            Segment(
                value=token,
                start=match.start(),
                end=match.end(),
                name_hint=flag.group(1).replace("-", "_").lower() if flag else None,
            )
        )
        previous = token
    return tuple(segments)


def segments_of(value: Any) -> tuple[str, tuple[Segment, ...]]:
    """값에서 치환 가능한 조각과 그 종류를 뽑는다.

    종류: ``"sql"``(리터럴), ``"text"``(토큰), ``""``(쪼갤 수 없음 — 값 전체가 단위).
    """
    if looks_like_sql(value):
        literals = normalize_sql(value).literals
        return "sql", tuple(
            Segment(
                value=lit.value,
                start=lit.start,
                end=lit.end,
                quoted=lit.kind == "string",
                name_hint=lit.name_hint,
            )
            for lit in literals
        )
    if isinstance(value, str):
        return "text", text_segments(value)
    return "", ()


def render_template(value: str, replacements: list[tuple[int, int, str, bool]]) -> str:
    """값에서 파라미터 구간만 `${name}`으로 바꾼 `string.Template` 템플릿을 만든다.

    구간은 호출자가 정해서 넘긴다. 여기서 조각을 다시 계산하지 않는 이유는, 파라미터
    구간이 조각 전체가 아닐 수 있기 때문이다 — `loaded_노트북.txt` 에서 실제로 변하는
    것은 가운데 `노트북` 뿐이다.

    상수 구간은 원문 그대로 남되, 그 안의 `$`는 `$$`로 escape한다. escape를 먼저
    하고 치환하면 오프셋이 밀려 엉뚱한 자리를 자르므로 한 번의 훑기로 함께 처리한다.
    """
    parts: list[str] = []
    cursor = 0
    for start, end, name, quoted in sorted(replacements):
        if start < cursor:
            continue
        parts.append(value[cursor:start].replace("$", "$$"))
        rendered = "${%s}" % name
        parts.append("'%s'" % rendered if quoted else rendered)
        cursor = end
    parts.append(value[cursor:].replace("$", "$$"))
    return "".join(parts)


@dataclass(frozen=True)
class ParameterSlot:
    """여러 표본에서 값이 달랐던 자리 하나."""

    call_index: int
    arg_key: str
    segment_index: int | None  # 값 안의 조각 위치. None이면 인자 값 전체
    name: str
    type: str
    example: Any
    segment_kind: str = ""  # "sql" | "text" | "" (값 전체)
    # 대표 표본의 인자 값 안에서 치환할 구간. segment_index가 None이면 None.
    span: tuple[int, int] | None = None
    quoted: bool = False  # 치환할 때 따옴표를 다시 씌우는가(SQL 문자열 리터럴)


@dataclass(frozen=True)
class ParameterPlan:
    parameters: tuple[dict[str, Any], ...]
    slots: tuple[ParameterSlot, ...]
    # 파라미터 이름 -> 표본별 관측값. 저장하지 않고 생성 단계에서만 쓴다. 산출물
    # 템플릿을 접을 때 "이 표본에서 이 파라미터가 어떤 값이었는가"가 필요하다.
    observations: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    # 앞 단계의 결과를 이어받는 자리들. 파라미터가 아니므로 명세에 실리지 않고,
    # 생성 코드 안에 `from_step(...)` 호출로 굳는다.
    bindings: tuple["StepBinding", ...] = ()
    # 값 전체를 이미 아는 값들의 조합으로 다시 적은 인자들. 자리 대조로는 쪼개지지 않는
    # 합성 문자열(워크스페이스 경로 등)을 위한 것이다.
    composites: tuple["CompositeArg", ...] = ()

    def as_specification(self) -> dict[str, Any]:
        return {"parameters": [dict(p) for p in self.parameters]}


def _arg_segmentation(values: list[Any]) -> tuple[str, int]:
    """한 인자 자리에 대해 표본들을 어떻게 쪼갤지 정한다.

    표본마다 조각 수가 다르면 자리 대조가 성립하지 않는다. 그때는 값 전체를 하나의
    단위로 본다 — 잘못 짝지어 엉뚱한 자리를 파라미터로 만드는 것보다 낫다.
    """
    kinds = {segments_of(v)[0] for v in values}
    if len(kinds) != 1:
        return "", 0
    kind = kinds.pop()
    if not kind:
        return "", 0
    counts = {len(segments_of(v)[1]) for v in values}
    if len(counts) != 1:
        return "", 0
    count = counts.pop()
    if count == 0:
        return "", 0
    if kind == "sql" and len({normalize_sql(v).signature for v in values}) != 1:
        return "", 0
    return kind, count


def _slot_values(
    samples: list[list[Any]],
) -> tuple[
    dict[tuple[int, str, int | None], list[Any]],
    dict[tuple[int, str], str],
    dict[tuple[int, str, int | None], tuple[int, int]],
    dict[tuple[int, str, int | None], bool],
]:
    """자리별로 표본들의 값을 모은다. 지문이 같으므로 인자 구성은 동일하다."""
    per_arg: dict[tuple[int, str], list[Any]] = {}
    for calls in samples:
        for call_index, call in enumerate(calls):
            _kind, _tool, args = as_call(call)
            for arg_key in sorted(args):
                per_arg.setdefault((call_index, arg_key), []).append(args[arg_key])

    collected: dict[tuple[int, str, int | None], list[Any]] = {}
    segment_kinds: dict[tuple[int, str], str] = {}
    # 대표 표본(첫 번째)의 인자 값 안에서 치환할 구간과 따옴표 여부.
    spans: dict[tuple[int, str, int | None], tuple[int, int]] = {}
    quoted: dict[tuple[int, str, int | None], bool] = {}

    for (call_index, arg_key), values in per_arg.items():
        kind, count = _arg_segmentation(values)
        segment_kinds[(call_index, arg_key)] = kind
        if not kind:
            collected[(call_index, arg_key, None)] = list(values)
            continue
        for index in range(count):
            head = segments_of(values[0])[1][index]
            slot_values = [segments_of(value)[1][index].value for value in values]
            start, end = head.start, head.end
            if kind == "text":
                trimmed = _trim_common_affixes(slot_values)
                if trimmed is not slot_values:
                    prefix, suffix = _common_affixes([str(v) for v in slot_values])
                    start, end = start + len(prefix), end - len(suffix)
                    slot_values = trimmed
            key = (call_index, arg_key, index)
            collected[key] = slot_values
            spans[key] = (start, end)
            quoted[key] = head.quoted
    return collected, segment_kinds, spans, quoted


def _common_affixes(values: list[str]) -> tuple[str, str]:
    """표본 전체가 공유하는 접두사와 접미사. **구분자 경계까지만** 인정한다.

    경계를 두지 않으면 `10`/`20`/`30` 의 공통 접미 `0` 까지 깎아, 파라미터가 `1`이고
    상수가 `0`인 엉터리 템플릿(`--qty ${qty}0`)이 나온다. 사람이 한 덩어리로 읽는
    낱말은 구분자로 갈리므로, 깎기도 거기서 멈춘다.
    """
    prefix = os.path.commonprefix(values)
    suffix = os.path.commonprefix([v[::-1] for v in values])[::-1]

    # 접두는 마지막 구분자까지만, 접미는 첫 구분자부터만.
    cut = max((i for i, ch in enumerate(prefix) if not ch.isalnum()), default=None)
    prefix = prefix[: cut + 1] if cut is not None else ""
    cut = next((i for i, ch in enumerate(suffix) if not ch.isalnum()), None)
    suffix = suffix[cut:] if cut is not None else ""

    # 접두와 접미가 겹치면(값 하나가 다른 값의 부분) 깎지 않는다.
    shortest = min(len(v) for v in values)
    while suffix and len(prefix) + len(suffix) > shortest:
        suffix = suffix[1:]
    return prefix, suffix


def _trim_common_affixes(values: list[Any]) -> list[Any]:
    """토큰에서 표본마다 실제로 달라진 가운데 토막만 남긴다.

    ``loaded_노트북.txt`` / ``loaded_마우스.txt`` 에서 변한 것은 상품명뿐이다. 토큰
    전체를 파라미터로 잡으면 (1) 호출자가 파일명 규칙까지 알아야 하고, (2) 그 값이
    워크아이템 지시문에 그대로 나오지 않아 다음 실행에서 되찾을 수 없다. 실제로 변한
    토막만 남겨야 지시문의 ``상품명은 노트북`` 과 이어진다.
    """
    if len(values) < 2 or not all(isinstance(v, str) for v in values):
        return values
    if len({*values}) < 2:
        return values
    # 날짜와 금액은 통째로 하나의 값이다. 공통 앞뒤를 깎으면 `01T02:46:44Z` 같은 토막이
    # 남는데, 그 토막은 지시문에도 앞 단계 결과에도 없어 다음 실행에서 되찾을 수 없다.
    if all(_as_temporal(v) is not None for v in values) or all(_as_number(v) is not None for v in values):
        return values
    prefix, suffix = _common_affixes(values)
    if not prefix and not suffix:
        return values
    trimmed = [v[len(prefix): len(v) - len(suffix) if suffix else None] for v in values]
    # 깎고 나서 빈 값이 생기거나 구분이 사라지면 원래대로 둔다.
    if any(not t for t in trimmed) or len({*trimmed}) < len({*values}):
        return values
    return trimmed


def _name_for(samples: list[list[Any]], key: tuple[int, str, int | None]) -> str:
    call_index, arg_key, segment_index = key
    if segment_index is None:
        return arg_key
    _kind, _tool, args = as_call(samples[0][call_index])
    _seg_kind, segments = segments_of(args[arg_key])
    hint = segments[segment_index].name_hint if segment_index < len(segments) else None
    return hint or f"{arg_key}_{segment_index + 1}"


# 지시문에서 값 앞에 붙는 이름표를 딸 때 걷어낼 조사·구분자.
_LABEL_TAIL = re.compile(r"[\s:=,\.\-\(\[\{'\"]*(?:은|는|이|가|을|를|로|으로)?[\s:=]*$")
_LABEL_WORD = re.compile(r"[^\s,\.:;=\(\)\[\]\{\}'\"]+$")
# 값 바로 뒤에 붙는 이름표(`Galaxy 상품의`). 조사는 이름표에서 떼어 낸다.
_LABEL_LEAD = re.compile(r"\s*([^\s,\.:;=\(\)\[\]\{\}'\"]{2,}?)(?:은|는|이|가|을|를|의|에|으로|로)?(?=[\s,\.:;]|$)")


def _label_from_context(value: Any, context: str) -> tuple[str, str] | None:
    """워크아이템 지시문에서 값에 붙어 있는 이름표와 그 위치를 딴다.

    ``"상품명은 노트북, 입고 수량은 10"`` → ``노트북``의 이름표는 앞의 ``상품명``.
    ``"Galaxy 상품의 발주 입고"``      → ``Galaxy``의 이름표는 뒤의 ``상품``.

    이름표가 필요한 이유는 다음 실행 때문이다. 고착화된 코드는 새 워크아이템의
    지시문에서 입력값을 뽑아 실행된다. 그런데 파라미터 이름이 자리 번호에서 나온
    ``command_3`` 이면 지시문의 어느 값이 그 자리인지 알 길이 없다. 값이 관측될 때
    그 곁에 무엇이 적혀 있었는지를 함께 적어 두어야 다음 실행에서 되찾을 수 있다.

    앞을 먼저 본다. 한국어 지시문에서 ``이름표는 값`` 이 값을 특정하는 가장 흔한
    표현이고, 뒤에 오는 말은 ``값 상품의`` 처럼 값을 수식하는 경우라 덜 확정적이다.

    글자 그대로 못 찾으면 표기를 접어서 다시 찾는다. 관측된 값이 ``30,000원`` 인데
    지시문에는 ``"amount": 30000`` 으로 적혀 있는 일이 흔하다. 여기서 포기하면 이름표가
    없는 자리가 되고, 이름표가 없으면 실행기가 위치 폴백으로 엉뚱한 값을 집는다.
    """
    text = str(value)
    if not text or not context:
        return None
    index, text = _locate(text, context)
    if index < 0:
        return None

    if index > 0:
        prefix = _LABEL_TAIL.sub("", context[:index])
        match = _LABEL_WORD.search(prefix)
        if match:
            label = match.group().strip()
            if len(label) >= 2 and label != text:
                return label, "before"

    suffix = context[index + len(text):]
    match = _LABEL_LEAD.match(suffix)
    if match:
        label = match.group(1).strip()
        if len(label) >= 2 and label != text:
            return label, "after"
    return None


def _locate(text: str, context: str) -> tuple[int, str]:
    """지시문에서 값이 적힌 자리와 **그 지시문에 적힌 모양**을 찾는다.

    이름표를 딸 자리를 정하려면 값이 실제로 놓인 구간을 알아야 한다. 표기가 다르면
    구간의 길이도 다르므로, 찾아낸 쪽의 글자를 함께 돌려준다.
    """
    index = context.find(text)
    if index >= 0:
        return index, text
    pattern = None
    if _as_temporal(text) is not None:
        pattern = _DATE_TOKEN
    elif _as_number(text) is not None:
        pattern = _NUMBER_TOKEN
    if pattern is None:
        return -1, text
    for match in pattern.finditer(context):
        if same_value(text, match.group()):
            return match.start(), match.group()
    return -1, text


def _shared_label(values: list[Any], contexts: list[str] | None) -> tuple[str, str] | None:
    """모든 표본에서 같은 이름표가 나올 때만 그 이름표를 인정한다.

    한 표본에서만 우연히 맞아떨어진 이름표를 믿으면, 다음 실행에서 엉뚱한 값을
    뽑아 그대로 실행해 버린다. 우연이 아님을 표본 수만큼 확인한다.
    """
    if not contexts or len(contexts) != len(values):
        return None
    labels = {_label_from_context(value, context) for value, context in zip(values, contexts)}
    if len(labels) != 1:
        return None
    return labels.pop()


# 실행 때 정해지는 식별자. 키는 도구 인자·SQL 컬럼에 쓰이는 이름, 값은 그 값을 읽어
# 올 워크아이템 행의 키다. 이 자리들은 워크아이템마다 달라지므로 파라미터로 승격되지만,
# 지시문에는 적혀 있지 않아 텍스트로는 되찾을 수 없다.
RUNTIME_IDENTITY_FIELDS: dict[str, str] = {
    "todo_id": "id",
    "workitem_id": "id",
    "proc_inst_id": "proc_inst_id",
    "root_proc_inst_id": "root_proc_inst_id",
    "proc_def_id": "proc_def_id",
    "activity_id": "activity_id",
    "tenant_id": "tenant_id",
}


def _runtime_binding(
    name: str, values: list[Any], identities: list[dict[str, Any]] | None
) -> tuple[str, str] | None:
    """이 자리가 실행 식별자인지 보고, 맞으면 (이름, 워크아이템 행의 키)를 돌려준다.

    근거는 둘이다.

    하나는 관측이다. 표본마다 그 자리의 값이 **그 워크아이템 자신의** 식별자와 같았다면,
    그 자리는 사람이 적어 준 입력이 아니라 실행 때 정해지는 값이다.

    다른 하나는 자리 이름이다. `INSERT ... (proc_inst_id, todo_id) VALUES (...)` 처럼
    컬럼 이름이 식별자 그대로면 그 자리의 뜻은 분명하다. 이름까지 보는 이유는, 에이전트가
    그 값을 잘못 채운 표본이 섞이기 때문이다 — 실제로 같은 활동의 표본 4건 중 둘은 빈
    문자열을, 둘은 **다른 워크아이템의** id 를 넣었다. 관측만 보면 어느 것도 식별자로
    인정되지 않고, 그러면 다음 실행에서 위치 폴백이 엉뚱한 값을 채운다.

    어느 쪽으로든 식별자로 판정되면 지시문을 뒤지지 않는다. 이 자리들은 이름표가 붙지
    않아, 되찾기에 실패하면 곧바로 위치·타입 폴백으로 떨어지는 자리다.
    """
    if identities and len(identities) == len(values):
        for field_name, row_key in RUNTIME_IDENTITY_FIELDS.items():
            expected = [str(identity.get(row_key) or "") for identity in identities]
            if all(expected) and [str(value) for value in values] == expected:
                return field_name, row_key
    row_key = RUNTIME_IDENTITY_FIELDS.get(str(name or "").lower())
    return (str(name).lower(), row_key) if row_key else None



# ---------------------------------------------------------------------------
# 단계 간 데이터 흐름 — 앞 단계의 결과를 뒤 단계의 인자로 잇는다
# ---------------------------------------------------------------------------
#
# 지금까지 생성 코드의 인자는 둘 중 하나였다. 표본 전부에서 같았던 상수이거나,
# 다음 실행의 지시문·워크아이템 행에서 되찾는 파라미터이거나. 그래서 **앞 단계가
# 만들어 낸 값**이 뒤 단계로 흐르는 활동은 굳지 않았다 — `date` 로 찍은 시각을 문서에
# 적는 활동이 그렇다. 그 시각은 지시문 어디에도 없고, 표본마다 다르니 상수도 아니다.
#
# 여기서 그 자리를 세 번째 갈래로 만든다. 값의 출처가 **이번 실행의 앞 단계 결과**인
# 자리다. 판정 근거는 관측이다 — 표본 **전부**에서 그 값이 앞 단계 결과의 **같은
# 자리**에 있었다면, 다음 실행에서도 거기 있다.


def _maybe_json(value: Any) -> Any:
    """JSON 문자열이면 풀어서 돌려준다. 아니면 그대로.

    도구 결과는 러너에 따라 구조 그대로 오기도, JSON 문자열로 오기도 한다. 생성 코드의
    `from_step` 도 같은 방식으로 푼다 — 양쪽이 같은 자리를 가리켜야 한다.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "{[":
        return value
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return value


def _result_body(action: Any) -> tuple[str, Any] | None:
    """행위의 관측 결과에서 (골격 결과의 본문 키, 본문)을 딴다.

    이어받을 수 있는 종류만 돌려준다. 파일 조작 결과에는 경로와 크기밖에 없다.
    """
    root = RESULT_BODY_KEY.get(str(getattr(action, "kind", "") or ""))
    if not root:
        return None
    result = getattr(action, "result", None)
    if result is None or (isinstance(result, str) and not result.strip()):
        return None
    return root, result


# 결과 안을 뒤지는 깊이 상한. 이보다 깊은 자리는 다음 실행에서도 같은 자리일 것이라고
# 믿기 어렵고, 값 하나를 찾자고 큰 결과를 통째로 훑을 이유도 없다.
_MAX_RESULT_DEPTH = 6


def _extraction_paths(body: Any, value: Any, depth: int = 0) -> set[tuple[tuple[Any, ...], int | None]]:
    """본문 안에서 값을 꺼낼 수 있는 자리들. `(경로, 줄번호)` 의 집합.

    줄번호는 잎이 여러 줄짜리 문자열일 때만 붙는다. 셸 결과가 그렇다 — 표준출력 첫 줄이
    값이고 뒤에 다른 줄이 붙는다. 값이 여러 줄에 걸쳐 나타나면 어느 줄인지 정해지지
    않으므로 아무것도 돌려주지 않는다.
    """
    if depth > _MAX_RESULT_DEPTH or body is None:
        return set()
    node = _maybe_json(body) if isinstance(body, str) else body

    if isinstance(node, dict):
        found: set[tuple[tuple[Any, ...], int | None]] = set()
        for key, child in node.items():
            found |= {((key,) + p, line) for p, line in _extraction_paths(child, value, depth + 1)}
        return found
    if isinstance(node, (list, tuple)):
        found = set()
        for index, child in enumerate(node):
            found |= {((index,) + p, line) for p, line in _extraction_paths(child, value, depth + 1)}
        return found

    ways: set[tuple[tuple[Any, ...], int | None]] = set()
    if same_value(node, value):
        ways.add(((), None))
    if isinstance(node, str):
        lines = node.splitlines()
        if len(lines) > 1:
            hits = [i for i, line in enumerate(lines) if same_value(line, value)]
            if len(hits) == 1:
                ways.add(((), hits[0]))
    return ways


def _path_order(way: tuple[tuple[Any, ...], int | None]) -> tuple[Any, ...]:
    """자리 후보의 정렬 기준. 얕은 자리, 그다음 이름순."""
    path, line = way
    return (len(path), tuple(str(p) for p in path), -1 if line is None else line)


# 이어받기로 인정할 최소 길이. 한 글자 값은 아무 결과에나 우연히 들어 있다.
_MIN_BINDING_LENGTH = 2


def _bindable(values: list[Any]) -> bool:
    return all(len(str(value).strip()) >= _MIN_BINDING_LENGTH for value in values)


def step_binding(
    samples: list[list[Any]], call_index: int, values: list[Any]
) -> tuple[int, str, tuple[Any, ...], int | None] | None:
    """이 자리가 앞 단계의 결과를 이어받은 자리인지 본다.

    가까운 단계부터 거슬러 올라가며, 표본 **전부**에서 같은 자리에 값이 있었던 단계를
    찾는다. 한 표본에서만 맞아떨어진 자리를 믿으면 다음 실행에서 엉뚱한 값을 이어받는다.

    후보가 여럿이면(같은 값이 결과 안에 두 번 나오면) 표본 전부에서 공통인 자리만 남기고
    그중 가장 얕은 것을 고른다. 어느 것을 골라도 표본 전부에서 그 값이었던 자리다.
    """
    if not _bindable(values):
        return None
    for source_index in range(call_index - 1, -1, -1):
        bodies = [
            _result_body(sample[source_index]) if source_index < len(sample) else None
            for sample in samples
        ]
        if any(body is None for body in bodies):
            continue
        roots = {body[0] for body in bodies}
        if len(roots) != 1:
            continue
        common: set[tuple[tuple[Any, ...], int | None]] | None = None
        for (_root, content), value in zip(bodies, values):
            ways = _extraction_paths(content, value)
            common = ways if common is None else (common & ways)
            if not common:
                break
        if common:
            path, line = min(common, key=_path_order)
            return source_index, roots.pop(), path, line
    return None


@dataclass(frozen=True)
class StepBinding:
    """앞 단계의 결과를 이어받는 자리 하나.

    파라미터와 달리 다음 실행의 지시문에서 값을 찾지 않는다. 이번 실행에서 앞 단계가
    실제로 낸 값을 그대로 쓴다 — 그래서 재실행하면 문서의 시각이 그 실행 기준으로 채워진다.
    """

    call_index: int
    arg_key: str
    segment_index: int | None
    name: str
    source_index: int  # 이어받을 앞 단계의 번호(`results[N]`)
    root: str  # 골격 결과에서 본문이 실린 키
    path: tuple[Any, ...]  # 본문 안에서의 자리
    line: int | None  # 잎이 여러 줄일 때 고를 줄. 아니면 None
    example: Any
    span: tuple[int, int] | None = None
    quoted: bool = False
    # 표본별 관측값. 저장하지 않고 생성 단계에서만 쓴다 — 합성 자리를 접을 때 "이 표본에서
    # 이 자리가 어떤 값이었는가"가 필요하다.
    values: tuple[Any, ...] = ()

    def as_record(self) -> dict[str, Any]:
        """출처 기록용 표현. 무엇을 어디서 이어받았는지 사람이 읽을 수 있게 남긴다."""
        return {
            "name": self.name,
            "into": f"{self.call_index}:{self.arg_key}",
            "from": self.source_index,
            "path": [self.root, *self.path] + ([f"line {self.line}"] if self.line is not None else []),
            "example": self.example,
        }


# ---------------------------------------------------------------------------
# 합성 자리 — 이미 아는 값들의 조합으로 다시 적는다
# ---------------------------------------------------------------------------
#
# 자리 대조는 값 하나가 조각 하나에 담겨 있을 때 성립한다. 그런데 한 조각에 여러 값이
# 붙어 있는 자리가 있다. 워크스페이스 경로가 그렇다 —
# `/workspace/.bpmn/{proc_inst_id}/expense/{used_at}_{applicant}_TRV.md` 는 통째로 한
# 낱말이라, 쪼개면 `5ee72ce5-…/expense/2026-09-04_박지은` 같은 토막이 남는다. 그 토막은
# 지시문에도 앞 단계 결과에도 없으므로 되찾을 수 없고, 활동 전체가 보류된다.
#
# 그런데 그 토막을 이루는 값들은 하나하나 이미 알고 있다. 신청자도 사용일도 파라미터로
# 잡혀 있고, 프로세스 인스턴스 식별자는 워크아이템 행에서 읽는다. 그러면 그 자리는
# 되찾을 수 없는 것이 아니라 **다시 적을 수 있는** 자리다.
#
# 근거는 여전히 관측이다. 표본 전부에서 같은 템플릿으로 접혀야 인정한다 — 한 표본에서만
# 맞아떨어진 조합은 다음 실행에서 다른 문자열을 만든다.


@dataclass(frozen=True)
class CompositeArg:
    """값 전체를 템플릿으로 다시 적은 인자 하나."""

    call_index: int
    arg_key: str
    template: str
    names: tuple[str, ...]


def _compose_template(value: str, known: list[tuple[str, str]]) -> str:
    """관측된 문자열을 이미 아는 값들의 조합으로 다시 적는다.

    긴 값부터 바꾼다. 짧은 값이 긴 값의 일부일 때 짧은 쪽이 먼저 먹으면 남은 자리가
    어긋난다. 글자 그대로 없으면 표기를 접어서 한 번 더 찾는다 — 문서에 `27,000원` 으로
    적힌 값이 지시문에는 `"amount": 27000` 으로 있는 일이 흔하다.
    """
    marked = value
    for name, observed in sorted(known, key=lambda item: -len(str(item[1]))):
        text = str(observed).strip()
        if len(text) < _MIN_BINDING_LENGTH:
            continue
        if text in marked:
            marked = marked.replace(text, f"\x00{name}\x00")
            continue
        index, found = _locate(text, marked)
        if index >= 0 and found:
            marked = marked.replace(found, f"\x00{name}\x00")
    parts = marked.split("\x00")
    return "".join(
        _escape_literal(part) if index % 2 == 0 else "${" + part + "}"
        for index, part in enumerate(parts)
    )


def _field_candidates(
    contexts: list[str] | None, count: int
) -> list[tuple[str, list[str]]]:
    """표본 전부의 지시문에 있는 구조화 입력 칸들. `(칸 이름, 표본별 값)`.

    실행기가 이름으로 곧장 집을 수 있는 값들이다. 이름이 `string.Template` 식별자가
    될 수 없으면(한글 칸 이름) 템플릿에 쓸 수 없으므로 뺀다.
    """
    if not contexts or len(contexts) != count:
        return []
    per_sample = [structured_fields(context) for context in contexts]
    if not all(per_sample):
        return []
    shared = set(per_sample[0])
    for fields in per_sample[1:]:
        shared &= set(fields)
    candidates: list[tuple[str, list[str]]] = []
    for key in sorted(shared):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        values = [fields[key] for fields in per_sample]
        if any(not isinstance(v, (str, int, float)) or isinstance(v, bool) for v in values):
            continue
        texts = [str(v).strip() for v in values]
        if all(len(text) >= _MIN_BINDING_LENGTH for text in texts):
            candidates.append((key, texts))
    return candidates


def _identity_candidates(
    identities: list[dict[str, Any]] | None,
) -> list[tuple[str, str, list[str]]]:
    """표본 전부에서 값이 있는 실행 식별자들. `(이름, 행의 키, 표본별 값)`."""
    if not identities:
        return []
    seen: dict[str, str] = {}
    for field_name, row_key in RUNTIME_IDENTITY_FIELDS.items():
        seen.setdefault(row_key, field_name)
    candidates: list[tuple[str, str, list[str]]] = []
    for row_key, field_name in seen.items():
        values = [str((identity or {}).get(row_key) or "") for identity in identities]
        if all(len(value) >= _MIN_BINDING_LENGTH for value in values):
            candidates.append((field_name, row_key, values))
    return candidates


def procedure_pins(
    plan: ParameterPlan, blocked: list[str], contexts: list[str] | None
) -> dict[str, str]:
    """되찾을 수 없는 자리 중 **입력이 아니라 절차**인 것들과, 굳힐 값.

    되찾을 수 없다는 것은 그 값이 이 워크아이템이 준 것이 아니라는 뜻이다 — 지시문에도,
    앞 액티비티의 산출물에도, 워크아이템 행에도 없다. 에이전트가 스스로 정한 값이다.
    그런 값에는 두 갈래가 있다.

    - **절차** — 시각 서식(`-Iseconds` 대 `'+%Y-%m-%d %H:%M:%S'`), 옵션처럼 "어떻게 할지"를
      정하는 자리. 어느 것을 골라도 다음 실행이 제대로 돌아간다.
    - **지어낸 내용** — 에이전트가 쓴 메일 본문, 없는 시각. 하나를 굳히면 그 표본의 사실이
      모든 실행에 남는다. 실제로 그렇게 굳은 코드 여섯 건을 지운 적이 있다.

    둘을 가르는 관측은 **반복**이다. 절차는 서로 다른 워크아이템에서 같은 값이 다시
    나온다(에이전트가 즐겨 쓰는 방식이 있다). 지어낸 내용은 워크아이템마다 다르다 —
    그 워크아이템의 사실을 담고 있기 때문이다.

    그래서 조건은 둘이다. 표본 과반에서 **같은 값**이 나왔고, 그 값이 이 활동이 다루는
    **어떤 값도 품고 있지 않아야** 한다. 둘 중 하나라도 어긋나면 굳히지 않는다.
    """
    if not blocked:
        return {}

    # 이 활동이 다루는 값들. 절차 값이 이것들을 품고 있으면 내용이지 절차가 아니다.
    carried: set[str] = set()
    for name, values in plan.observations.items():
        if name in blocked:
            continue
        carried.update(str(value).strip() for value in values)
    for context in contexts or []:
        carried.update(
            str(value).strip() for value in structured_fields(context).values()
            if isinstance(value, (str, int, float))
        )
    carried = {value for value in carried if len(value) >= _MIN_BINDING_LENGTH}

    pins: dict[str, str] = {}
    for name in blocked:
        values = [str(value) for value in plan.observations.get(name, ())]
        if len(values) < 3:
            continue
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        winner, hits = max(counts.items(), key=lambda item: (item[1], -len(item[0])))
        if hits * 2 <= len(values):
            # 과반이 아니면 "에이전트가 즐겨 쓰는 방식"이라고 볼 근거가 없다.
            continue
        if any(value in winner for value in carried):
            # 이 활동의 값을 품고 있다 — 절차가 아니라 내용이다.
            continue
        pins[name] = winner
    return pins


def compose_from_known(
    plan: ParameterPlan,
    samples: list[list[Any]],
    identities: list[dict[str, Any]] | None,
    unrecoverable: list[str],
    contexts: list[str] | None = None,
    pins: dict[str, str] | None = None,
) -> ParameterPlan:
    """되찾을 수 없는 자리를 품은 인자를, 이미 아는 값들의 조합으로 다시 적는다.

    되찾을 수 없다고 판정된 자리만 손댄다. 멀쩡히 굳던 인자를 템플릿으로 바꾸면 얻는
    것 없이 깨질 여지만 는다.

    쓰는 재료는 넷뿐이다 — 되찾을 수 있는 파라미터, 앞 단계에서 이어받는 값, 지시문의
    구조화 입력 칸, 그리고 워크아이템 행에서 읽는 실행 식별자. 다음 실행에서 값을 구할
    수 있는 것만 재료가 된다는 뜻이다. 남은 글자가 표본마다 다르면 템플릿이 갈리므로
    그때는 포기한다.

    ``pins`` 는 절차로 판정된 자리와 굳힐 값이다(`procedure_pins`). 표본마다 다르게 적힌
    그 자리를 과반 값으로 맞춘 뒤 템플릿을 만든다 — 그러지 않으면 시각 서식 하나가
    다르다는 이유로 나머지가 전부 멀쩡한 활동이 영영 굳지 않는다.
    """
    if not unrecoverable:
        return plan
    blocked = set(unrecoverable)
    targets = {
        (slot.call_index, slot.arg_key) for slot in plan.slots if slot.name in blocked
    }
    if not targets:
        return plan

    known: list[tuple[str, list[str]]] = [
        (str(spec["name"]), [str(v) for v in plan.observations.get(str(spec["name"]), ())])
        for spec in plan.parameters
        if str(spec.get("name")) not in blocked
    ]
    known += [
        (binding.name, [str(v) for v in binding.values])
        for binding in plan.bindings if binding.values
    ]
    identity_pool = _identity_candidates(identities)
    known += [(name, values) for name, _row_key, values in identity_pool]
    # 되찾을 수 없다고 판정된 자리와 이름이 겹치는 칸은 재료로 쓴다. 그 자리가 바로
    # 칸 값의 토막이기 때문이다 — `사유: 지사 출장 택시비` 의 `지사` 가 그렇다. 토막 대신
    # 칸 값 전체를 넣어야 문장이 온전해진다.
    declared_names = {
        str(spec.get("name")) for spec in plan.parameters
        if str(spec.get("name")) not in blocked
    }
    field_pool = [
        (name, values)
        for name, values in _field_candidates(contexts, len(samples))
        if name not in declared_names
    ]
    known += field_pool
    known = [(name, values) for name, values in known if len(values) == len(samples)]

    composites: list[CompositeArg] = []
    for call_index, arg_key in sorted(targets):
        observed = [
            str(as_call(sample[call_index])[2].get(arg_key, "")) for sample in samples
        ]
        if not all(observed):
            continue
        # 절차로 판정된 자리는 과반 값으로 맞춘다. 값 자체가 코드에 상수로 굳는다.
        aligned = list(observed)
        pinned_here = False
        for name, winner in (pins or {}).items():
            for index, value in enumerate(plan.observations.get(name, ())):
                text = str(value)
                if text and text != winner and index < len(aligned) and text in aligned[index]:
                    aligned[index] = aligned[index].replace(text, winner)
                    pinned_here = True

        templates = {
            _compose_template(value, [(name, values[index]) for name, values in known])
            for index, value in enumerate(aligned)
        }
        if len(templates) != 1:
            continue
        template = templates.pop()
        names = tuple(sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", template))))
        if not names and not pinned_here:
            # 치환할 것도 없고 맞춘 것도 없으면 손댈 이유가 없다. 반대로 절차 자리를
            # 맞춰 순수 상수가 된 인자는 그대로 굳혀야 한다 — 그게 목적이다.
            continue
        composites.append(CompositeArg(call_index, arg_key, template, names))

    if not composites:
        return plan

    # 합성으로 다시 적은 인자의 자리들은 더 이상 필요 없다. 자리가 하나도 남지 않은
    # 파라미터는 명세에서도 뺀다 — 실행기가 쓰지도 않을 값을 지시문에서 찾다 실패한다.
    replaced = {(composite.call_index, composite.arg_key) for composite in composites}
    slots = tuple(
        slot for slot in plan.slots if (slot.call_index, slot.arg_key) not in replaced
    )
    # 합성 템플릿이 이름으로 참조하는 이어받기는 그대로 둔다. 자리(span)는 더 이상
    # 쓰이지 않지만, 값은 실행 시점에 앞 단계에서 구해 템플릿에 넣어야 한다.
    referenced = {name for composite in composites for name in composite.names}
    bindings = tuple(
        binding for binding in plan.bindings
        if (binding.call_index, binding.arg_key) not in replaced or binding.name in referenced
    )
    used_names = {slot.name for slot in slots}
    for composite in composites:
        used_names.update(composite.names)
    field_names = {name for name, _values in field_pool}
    parameters = [
        spec for spec in plan.parameters
        if str(spec.get("name")) in used_names
        # 칸 값으로 갈아 끼울 토막 자리는 뺀다. 아래에서 칸 이름으로 다시 적는다.
        and not (str(spec.get("name")) in field_names and str(spec.get("name")) in blocked)
    ]

    # 합성에 쓴 실행 식별자는 명세에 없을 수 있다. 실행기가 워크아이템 행에서 읽도록
    # 그 자리를 새로 적어 둔다.
    declared = {str(spec.get("name")) for spec in parameters}
    for name, row_key, values in identity_pool:
        if name in used_names and name not in declared:
            parameters.append(
                {"name": name, "type": "string", "example": values[0], "runtime": row_key}
            )
            declared.add(name)
    # 지시문의 칸에서 곧장 읽는 자리도 마찬가지다. 이름이 칸 이름과 같으므로 실행기가
    # 이름표 없이도 그 칸을 찾는다.
    observations = dict(plan.observations)
    for name, values in field_pool:
        if name in used_names and name not in declared:
            parameters.append({"name": name, "type": "string", "example": values[0]})
            declared.add(name)
            # 관측값도 칸 값으로 바꿔 둔다. 토막이 남아 있으면 되찾기 판정이 그 토막을
            # 보고 또 막고, 산출물 템플릿도 토막으로 접힌다.
            observations[name] = tuple(values)
    for name in list(observations):
        if name not in declared and name not in {b.name for b in bindings}:
            observations.pop(name)
    return replace(
        plan,
        parameters=tuple(parameters),
        slots=slots,
        bindings=bindings,
        composites=tuple(composites),
        observations=observations,
    )


# ---------------------------------------------------------------------------
# 앞 단계 워크아이템의 산출물 — 액티비티를 건너뛰는 데이터 흐름
# ---------------------------------------------------------------------------
#
# `results[N]` 은 **같은 워크아이템 안**의 앞 단계다. 그런데 값은 앞 **액티비티**에서도
# 온다. 에이전트는 그것을 지시문의 `[InputData]` 로 받거나 `get_related_workitem_outputs`
# 로 직접 읽는다. 앞의 것은 지시문에 적혀 있어 이미 되찾을 수 있지만, 뒤의 것은 —
# 앞 워크아이템의 완료 시각 같은 — 지시문에 없어서 지금까지 고착화를 막아 왔다.
#
# 그런 자리는 이름표 대신 "앞 액티비티의 산출물 어디" 를 적어 둔다. 실행기가 그 자리를
# 워크아이템 행이 아니라 앞 워크아이템의 산출물에서 읽는다(`runtime` 과 같은 얼개다).


def _upstream_binding(
    values: list[Any], related: list[list[dict[str, Any]]]
) -> dict[str, Any] | None:
    """표본 전부에서 같은 앞 액티비티의 같은 자리에 있었던 값인지 본다."""
    if not _bindable(values) or len(values) != len(related):
        return None
    common: set[tuple[str, tuple[Any, ...]]] | None = None
    for value, records in zip(values, related):
        ways: set[tuple[str, tuple[Any, ...]]] = set()
        for record in records or []:
            activity = str((record or {}).get("activityId") or "")
            if not activity:
                continue
            for path, line in _extraction_paths(record, value):
                # 산출물은 줄 단위로 자르지 않는다. 여러 줄짜리 산출물의 몇 번째 줄인지는
                # 다음 실행에서 같으리라고 믿을 근거가 없다.
                if line is not None or not path or path[0] == "activityId":
                    continue
                ways.add((activity, path))
        common = ways if common is None else (common & ways)
        if not common:
            return None
    if not common:
        return None
    activity, path = min(common, key=lambda way: (len(way[1]), tuple(str(p) for p in way[1])))
    return {"activity_id": activity, "path": list(path)}


def bind_upstream(
    plan: "ParameterPlan", names: list[str], related: list[list[dict[str, Any]]]
) -> "ParameterPlan":
    """되찾을 수 없는 파라미터를 앞 액티비티의 산출물에 잇는다.

    되찾을 수 없다고 판정된 자리만 본다. 지시문에서 이미 되찾을 수 있는 값을 굳이 앞
    단계에서 끌어오면, 이어받기가 끊겼을 때(앞 워크아이템이 아직 없을 때) 멀쩡하던
    활동까지 폴백으로 떨어진다.
    """
    if not names or not related:
        return plan
    parameters = [dict(spec) for spec in plan.parameters]
    for spec in parameters:
        name = str(spec.get("name") or "")
        if name not in names:
            continue
        values = plan.observations.get(name)
        if not values:
            continue
        binding = _upstream_binding(list(values), related)
        if not binding:
            continue
        spec["upstream"] = binding
        # 이름표를 함께 남기면 실행기가 지시문을 먼저 뒤져 엉뚱한 값을 집을 여지가 생긴다.
        spec.pop("label", None)
        spec.pop("label_position", None)
    return replace(plan, parameters=tuple(parameters))



def _escape_literal(text: str) -> str:
    """`string.Template` 이 치환 기호로 읽지 않도록 문자 그대로의 `$` 를 접는다."""
    return text.replace("$", "$$")


def _as_template(value: str, sample_values: list[tuple[str, str]]) -> str:
    """관측된 문자열에서 파라미터 값이 있던 자리를 `${이름}` 으로 바꾼다.

    긴 값부터 바꾼다. 짧은 값이 긴 값의 일부일 때(`3` 과 `30000`) 짧은 쪽이 먼저
    먹으면 남은 자리가 어긋난다. 치환 자리는 먼저 표시만 해 두고, 나머지 글자를
    이스케이프한 뒤에 기호로 되돌린다 — 그래야 원문의 `$` 와 우리가 넣은 `${...}` 가
    섞이지 않는다.
    """
    marked = value
    for name, observed in sorted(sample_values, key=lambda item: -len(item[1])):
        if len(observed) < 2:
            # 한 글자 값은 아무 데나 걸린다(`1` 이 `1건` 의 1 을 먹는다). 바꾸지 않으면
            # 표본끼리 템플릿이 어긋나 산출물 고착화를 포기하게 된다 — 안전한 쪽이다.
            continue
        marked = marked.replace(observed, f"\x00{name}\x00")
    parts = marked.split("\x00")
    # 짝수 자리는 원문, 홀수 자리는 파라미터 이름이다.
    return "".join(
        _escape_literal(part) if index % 2 == 0 else "${" + part + "}"
        for index, part in enumerate(parts)
    )


def build_output_template(
    outputs: list[Any], observations: dict[str, tuple[Any, ...]]
) -> dict[str, Any] | None:
    """표본들의 폼 산출물을 파라미터 템플릿 하나로 접는다.

    에이전트는 도구를 부르고 끝나지 않는다. 마지막에 워크아이템의 폼 필드 값을 내놓고,
    다음 활동은 그 값을 입력으로 받는다. 고착화 코드가 도구 호출만 재현하면 폼이 빈 채로
    워크아이템이 완료되어 다음 단계가 굶는다.

    필드마다 셋 중 하나로 정한다.

    - 표본 전체에서 값이 같았다 → 상수. 그대로 굳힌다.
    - 값이 달랐고 그 차이가 파라미터로 **전부** 설명된다 → 템플릿. 다음 실행의 입력으로
      렌더한다.
    - 설명되지 않는 차이가 남는다 → 굳히지 않는다(`None`). 남은 조각은 표본 하나의
      사실이라 그대로 굳으면 다음 실행에서 조용히 틀린다. 실행기가 이번 실행에서 실제로
      일어난 일로 채운다.

    표본들의 폼 아이디나 필드 구성이 다르면 접지 않는다 — 같은 활동의 같은 산출물이
    아니라는 뜻이다.
    """
    forms = [output for output in outputs if isinstance(output, dict) and len(output) == 1]
    if len(forms) != len(outputs) or not forms:
        return None
    form_ids = {next(iter(form)) for form in forms}
    if len(form_ids) != 1:
        return None
    form_id = form_ids.pop()

    bodies = [form[form_id] for form in forms]
    if not all(isinstance(body, dict) for body in bodies):
        return None
    keys = {tuple(sorted(body)) for body in bodies}
    if len(keys) != 1:
        return None

    per_sample = [
        [(name, str(values[index])) for name, values in observations.items()
         if index < len(values) and str(values[index]).strip()]
        for index in range(len(bodies))
    ]

    fields: dict[str, Any] = {}
    for key in keys.pop():
        values = [body[key] for body in bodies]
        if len({_hashable(value) for value in values}) == 1:
            fields[key] = {"const": values[0]}
            continue
        if not all(isinstance(value, str) for value in values):
            fields[key] = None
            continue
        templates = {
            _as_template(value, per_sample[index]) for index, value in enumerate(values)
        }
        fields[key] = {"template": templates.pop()} if len(templates) == 1 else None

    return {"form_id": form_id, "fields": fields}


# ---------------------------------------------------------------------------
# 지시문의 구조화 입력 — 실행기가 값을 집는 자리
# ---------------------------------------------------------------------------
#
# 이 시스템의 지시문에는 앞 단계의 폼 값이 `[InputData]` JSON 으로 붙는다. 실행기는
# 파라미터 이름·이름표를 그 칸 이름과 맞춰 값을 집는다(`replay._parameter_values`).
# 여기서 같은 방식으로 읽어야, 생성 단계가 "실행기가 이 값을 집을 수 있는가"를 미리
# 판정할 수 있다. 읽는 방식이 갈리면 굳힐 때는 되찾을 수 있다고 보고 실행할 때는 못
# 찾는다.


def _json_objects(text: str):
    """텍스트에 박힌 JSON 객체들을 찾는다.

    첫 `{` 부터 마지막 `}` 까지를 한 덩어리로 집으면 안 된다. 지시문의 [Instruction]
    에는 `expense/{used_at}_{applicant}.md` 같은 자리표시자가 흔히 들어 있어, 그것까지
    한 덩어리에 들어가면 JSON 파싱이 통째로 실패한다. 그러면 구조화 입력이 빈 것으로
    보이고, 실행기는 지시문의 따옴표를 순서대로 긁어 **칸 이름**을 값으로 집는다.

    중괄호 균형을 세어 객체 하나씩 자른다. 파싱되지 않는 덩어리는 건너뛰고 다음 `{`
    부터 다시 본다.
    """
    index, length = 0, len(text or "")
    while index < length:
        if text[index] != "{":
            index += 1
            continue
        depth, cursor, in_string, escaped = 0, index, False, False
        while cursor < length:
            char = text[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    cursor += 1
                    break
            cursor += 1
        try:
            parsed = json.loads(text[index:cursor])
        except (ValueError, TypeError):
            index += 1
            continue
        yield parsed
        index = cursor


def structured_fields(context: str) -> dict[str, Any]:
    """지시문에 실린 구조화 입력을 키→값으로 편다. 중첩은 잎만, 겹치면 먼저 나온 것."""
    fields: dict[str, Any] = {}

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    _walk(value)
                elif key not in fields:
                    fields[key] = value
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    for parsed in _json_objects(context or ""):
        _walk(parsed)
    return fields


def _is_input_fragment(value: Any, context: str) -> bool:
    """이 값이 구조화 입력 칸 하나의 **일부**인가.

    칸 값과 같으면 토막이 아니다 — 실행기가 그 칸을 읽어 그대로 넣으면 된다. 칸 값 안에
    들어 있기만 하면 토막이다.
    """
    text = str(value).strip()
    if len(text) < _MIN_BINDING_LENGTH:
        return False
    fields = structured_fields(context)
    if not fields:
        return False
    inside = False
    for raw in fields.values():
        if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
            continue
        field_text = str(raw).strip()
        if same_value(text, field_text):
            return False
        if text in field_text:
            inside = True
    return inside


def _appears_in(value: Any, context: str) -> bool:
    """값이 그 워크아이템의 지시문에 실제로 나타나는가.

    글자 그대로 찾지 않는다. 금액과 날짜는 표기가 흔들리기 때문이다(`27000` /
    `27,000원`, `2026-09-01` / `2026년9월1일`). 서식 차이만으로 "되찾을 수 없다"고
    판정하면, 굳을 수 있는 활동이 영영 안 굳는다. 지시문에서 같은 종류의 토막만 뽑아
    정규화해 대조한다.
    """
    raw = str(value).strip()
    if not raw or not context:
        return False
    if raw in context:
        return True
    if _as_temporal(raw) is not None:
        return any(same_value(raw, match.group()) for match in _DATE_TOKEN.finditer(context))
    if _as_number(raw) is not None:
        return any(same_value(raw, match.group()) for match in _NUMBER_TOKEN.finditer(context))
    return False


def unrecoverable_parameters(
    plan: ParameterPlan, contexts: list[str] | None
) -> list[str]:
    """다음 실행에서 값을 되찾을 수 없는 파라미터 이름들.

    고착화된 코드는 새 워크아이템의 **지시문에서 값을 뽑아** 실행된다. 그런데 그 값이
    표본의 지시문에도 없었다면, 다음 지시문에도 없을 것이다. 그때 실행기는 실패하지
    않는다 — 이름표가 없는 자리는 위치·타입 폴백으로 떨어지고, 폴백은 지시문에서
    순서대로 값을 집으므로 **언제나 무언가를 채우는 데 성공**한다. 그리고 그 값으로
    실제 도구를 부른다. 경비 대장의 `proc_inst_id` 자리에 신청자 이름이, `todo_id`
    자리에 계정과목이 들어간 적이 있다. 조용히 틀리는 게 아니라 확신 있게 틀린다.

    그래서 지금까지 "재현할 수 없는 **행위**"만 거르던 것을(대상 경로 없는 파일 조작,
    서버를 못 찾은 MCP 도구) "되찾을 수 없는 **입력**"까지 넓힌다. 값이 어디서 오는지
    모르는 채로 굳히느니 에이전트에게 맡긴다.

    막는 것은 "지시문에 내용이 있는데 그 값이 없는" 경우다. 지시문이 비어 있으면 폴백이
    집을 것도 없어 실행기가 그냥 실패하고 에이전트로 넘어간다 — 안전한 쪽이라 막지 않는다.
    이 게이트는 쓸모없는 코드가 아니라 **확신 있게 틀리는 코드**를 막는다.

    값이 지시문에 있다는 것만으로는 부족하다. 실행기가 집을 수 있는 단위여야 한다.
    구조화 입력의 값이 여러 낱말이면(`사유: 지사 출장 택시비`) 자리 대조는 그것을 낱말
    단위로 쪼갠다. 그 토막은 지시문 어딘가에 분명히 있지만, 실행기는 칸 하나를 통째로
    읽으므로 토막만 따로 집을 방법이 없다 — 이름표도 붙지 않아 위치 폴백으로 떨어지고,
    사유 칸에 신청자 이름이 들어간 채로 문서가 만들어진다. 그런 자리도 되찾을 수 없는
    자리로 본다. 합성으로 다시 적을 수 있으면 그때 살아난다.

    내용이 있는 표본 **전부**에서 나타나야 인정한다. 한 표본에서만 우연히 맞은 것은
    근거가 아니다. 실행 식별자로 바인딩된 파라미터는 지시문이 아니라 워크아이템 행에서,
    앞 액티비티에 이어진 파라미터는 그 산출물에서 읽으므로 뺀다.
    """
    if not contexts:
        # 지시문을 모으지 않았으면 판정할 근거가 없다. 없는 근거로 막지는 않는다.
        return []
    unrecoverable: list[str] = []
    for spec in plan.parameters:
        if spec.get("runtime") or spec.get("upstream"):
            continue
        name = str(spec.get("name") or "")
        values = plan.observations.get(name)
        if not values or len(values) != len(contexts):
            continue
        judged = [
            (value, context)
            for value, context in zip(values, contexts)
            if str(context or "").strip()
        ]
        if not judged:
            continue
        if not all(_appears_in(value, context) for value, context in judged):
            unrecoverable.append(name)
        elif all(_is_input_fragment(value, context) for value, context in judged):
            unrecoverable.append(name)
    return unrecoverable


def identify_parameters(
    samples: list[list[Any]],
    contexts: list[str] | None = None,
    identities: list[dict[str, Any]] | None = None,
) -> ParameterPlan:
    """지문이 같은 표본들에서 값이 변한 자리를 파라미터로 승격한다.

    모든 표본에서 값이 같은 자리는 상수로 코드에 박는다. 우연히 같았을 가능성은
    남지만, 표본 수가 충분하면 확률이 낮고 틀렸다면 재작업 → 비활성 → 재축적으로
    자연 복구된다.

    항상 같은 값을 갖는 자리들은 하나의 파라미터로 묶는다(예: WHERE 절과 INSERT
    절에 같은 상품명이 들어가는 경우).

    ``contexts``는 표본별 워크아이템 지시문이다. 주어지면 각 파라미터가 지시문에서
    어떤 이름표 뒤에 있었는지를 함께 기록한다 — 다음 실행에서 값을 되찾는 열쇠다.

    ``identities``는 표본별 워크아이템의 식별자(행)이다. 주어지면 지시문에서 되찾을 수
    없는 실행 식별자 자리를 가려내, 이름표 대신 어느 필드에서 읽을지를 기록한다.

    값이 변한 자리라고 전부 파라미터가 되는 것은 아니다. 그 값이 표본 전부에서 **앞
    단계의 결과**에 있었다면 파라미터가 아니라 이어받는 자리다. 지시문에서 찾을 이유가
    없으므로 파라미터 승격보다 먼저 따진다.
    """
    collected, segment_kinds, spans, quoted = _slot_values(samples)
    varying = {key: values for key, values in collected.items() if len(set(map(_hashable, values))) > 1}

    def _group_key(key: tuple[int, str, int | None], values: list[Any]) -> tuple[Any, ...]:
        """묶음 기준 값 벡터.

        조각(SQL 리터럴·텍스트 토큰) 자리는 숫자로 환산해서 비교한다. 셸 명령에서
        뽑힌 토큰은 `"20"`(문자열)이고 SQL 리터럴에서 뽑힌 같은 값은 `20`(숫자)이라
        그대로 비교하면 한 값이 파라미터 둘로 갈린다. 조각 치환은 어차피 문자열로
        렌더되므로 환산해서 묶어도 실행 결과가 달라지지 않는다.
        """
        if key[2] is None:
            return tuple(map(_hashable, values))
        return tuple(_hashable(_coerce_scalar(value)) for value in values)

    # 표본 전체에서 값 벡터가 동일한 자리들을 한 파라미터로 묶는다.
    groups: dict[tuple[Any, ...], list[tuple[int, str, int | None]]] = {}
    for key, values in varying.items():
        groups.setdefault(_group_key(key, values), []).append(key)

    parameters: list[dict[str, Any]] = []
    slots: list[ParameterSlot] = []
    bindings: list[StepBinding] = []
    observations: dict[str, tuple[Any, ...]] = {}
    used: set[str] = set()

    def _unique(base: str) -> str:
        name, suffix = base, 2
        while name in used:
            name, suffix = f"{base}_{suffix}", suffix + 1
        used.add(name)
        return name

    for _vector, keys in sorted(groups.items(), key=lambda item: sorted(item[1])):
        keys.sort()
        values = varying[keys[0]]

        # 같은 값이 들어가는 자리들은 한 묶음이다. 그중 하나만 이어받고 나머지는
        # 지시문에서 찾으면 한 값이 두 경로로 갈려 서로 어긋날 수 있다. 묶음 전체가
        # 이어받을 수 있을 때만 이어받는다.
        linked = {key: step_binding(samples, key[0], values) for key in keys}
        if all(binding is not None for binding in linked.values()):
            source_index = linked[keys[0]][0]
            base = _name_for(samples, keys[0])
            name = _unique(base if not base[-1:].isdigit() else f"step_{source_index}")
            for key in keys:
                bound_source, root, bound_path, line = linked[key]
                bindings.append(
                    StepBinding(
                        call_index=key[0],
                        arg_key=key[1],
                        segment_index=key[2],
                        name=name,
                        source_index=bound_source,
                        root=root,
                        path=bound_path,
                        line=line,
                        example=values[0],
                        span=spans.get(key),
                        quoted=quoted.get(key, False),
                        values=tuple(values),
                    )
                )
            continue

        label = _shared_label(varying[keys[0]], contexts)
        # 이름은 뜻이 가장 잘 드러나는 자리에서 딴다. 자동 생성된 `content_5` 같은
        # 이름보다 옵션 이름이나 컬럼 이름에서 온 힌트가 낫다.
        named = min(keys, key=lambda key: (_name_for(samples, key).endswith(tuple("0123456789")), keys.index(key)))
        base = _name_for(samples, named)
        # 자리 번호에서 나온 이름밖에 없고 지시문 이름표가 식별자로 쓸 수 있으면 그쪽이 낫다.
        # 한글 이름표는 `string.Template` 식별자가 될 수 없어 이름으로는 못 쓰고, 값을
        # 되찾는 열쇠로만 `label` 에 남는다.
        if label and base[-1:].isdigit() and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label[0]):
            base = label[0].lower()
        runtime = _runtime_binding(base, varying[keys[0]], identities)
        if runtime and base[-1:].isdigit():
            # `query_8` 같은 자리 번호보다 `todo_id` 가 코드를 읽을 수 있게 만든다.
            base = runtime[0]
        name = _unique(base)

        observations[name] = tuple(values)
        example = values[0]
        # 텍스트 토큰은 늘 문자열로 잡히지만 실제로는 수량인 경우가 많다. 타입을
        # 숫자로 알려 두어야 다음 실행에서 입력을 숫자로 뽑아낼 수 있다.
        if keys[0][2] is not None:
            example = _coerce_scalar(example)
        ptype = _python_type_name(example)
        spec: dict[str, Any] = {"name": name, "type": ptype, "example": example}
        if runtime:
            # 실행 식별자는 워크아이템 행에서 읽는다. 이름표를 함께 남기면 실행기가
            # 지시문을 먼저 뒤져 엉뚱한 값을 집을 여지가 생긴다.
            spec["runtime"] = runtime[1]
        elif label:
            spec["label"], spec["label_position"] = label
        parameters.append(spec)
        for key in keys:
            slots.append(
                ParameterSlot(
                    call_index=key[0],
                    arg_key=key[1],
                    segment_index=key[2],
                    name=name,
                    type=ptype,
                    example=example,
                    segment_kind=segment_kinds.get((key[0], key[1]), ""),
                    span=spans.get(key),
                    quoted=quoted.get(key, False),
                )
            )

    # 이름표가 겹치면 어느 파라미터를 가리키는지 정해지지 않는다. 그런 이름표는
    # 버리고 위치·타입 기반 폴백에 맡긴다 — 틀린 값을 확신 있게 넣는 것보다 낫다.
    counts: dict[str, int] = {}
    for spec in parameters:
        if "label" in spec:
            counts[spec["label"]] = counts.get(spec["label"], 0) + 1
    for spec in parameters:
        if counts.get(spec.get("label", ""), 0) > 1:
            spec.pop("label", None)
            spec.pop("label_position", None)

    return ParameterPlan(
        parameters=tuple(parameters),
        slots=tuple(slots),
        observations=observations,
        bindings=tuple(bindings),
    )


def _hashable(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _coerce_scalar(value: Any) -> Any:
    """숫자로만 이루어진 토큰을 숫자로 본다. 그 밖에는 원래 값 그대로."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


def _python_type_name(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"
