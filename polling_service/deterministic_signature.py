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
from dataclasses import dataclass, field
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
    return SqlNormalization(signature=signature, literals=tuple(literals))


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


# 셸 행위의 종류 이름. `work_history`가 이 값을 그대로 쓴다(정의는 한 곳에 둔다).
SHELL_KIND = "shell"


def _command_shape(command: str) -> str:
    """셸 명령의 구조. 실행 프로그램과 인자 개수만 남긴다.

    일반 문자열 인자는 값이 달라도 같은 구조로 본다(`<str>`). 셸 명령은 다르다 —
    문자열 자체가 곧 수행할 행위이므로, `python3 load.py` 와 `bash clean.sh --all` 을
    같은 실행으로 보면 서로 다른 일을 하나로 굳혀 버린다.
    """
    tokens = text_segments(command)
    program = str(tokens[0].value) if tokens else ""
    return f"{program}#{len(tokens)}"


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


# 일반 문자열의 토큰. 세 갈래로 나눈다.
# 1) 따옴표로 묶인 덩어리는 통째로 한 토큰 — 셸의 `-c "UPDATE ..."` 를 쪼개면 어긋난다.
# 2) 값처럼 보이는 낱말(경로·옵션·수량·식별자). 경로 구분자와 하이픈은 낱말에 포함한다.
# 3) 그 밖의 구두점 한 글자.
# 구두점을 낱말에서 떼어내야 `노트북:` 과 `노트북` 이 같은 값으로 묶인다. 붙여 두면
# 같은 값이 자리마다 다른 파라미터로 갈라져, 호출자가 같은 값을 여러 번 넘겨야 한다.
_TOKEN = re.compile(r'"[^"]*"|\'[^\']*\'|[\w./+@-]+|[^\s\w]')
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
    """
    text = str(value)
    if not text or not context:
        return None
    index = context.find(text)
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


def identify_parameters(
    samples: list[list[Any]], contexts: list[str] | None = None
) -> ParameterPlan:
    """지문이 같은 표본들에서 값이 변한 자리를 파라미터로 승격한다.

    모든 표본에서 값이 같은 자리는 상수로 코드에 박는다. 우연히 같았을 가능성은
    남지만, 표본 수가 충분하면 확률이 낮고 틀렸다면 재작업 → 비활성 → 재축적으로
    자연 복구된다.

    항상 같은 값을 갖는 자리들은 하나의 파라미터로 묶는다(예: WHERE 절과 INSERT
    절에 같은 상품명이 들어가는 경우).

    ``contexts``는 표본별 워크아이템 지시문이다. 주어지면 각 파라미터가 지시문에서
    어떤 이름표 뒤에 있었는지를 함께 기록한다 — 다음 실행에서 값을 되찾는 열쇠다.
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
    used: set[str] = set()
    for _vector, keys in sorted(groups.items(), key=lambda item: sorted(item[1])):
        keys.sort()
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
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)

        values = varying[keys[0]]
        example = values[0]
        # 텍스트 토큰은 늘 문자열로 잡히지만 실제로는 수량인 경우가 많다. 타입을
        # 숫자로 알려 두어야 다음 실행에서 입력을 숫자로 뽑아낼 수 있다.
        if keys[0][2] is not None:
            example = _coerce_scalar(example)
        ptype = _python_type_name(example)
        spec: dict[str, Any] = {"name": name, "type": ptype, "example": example}
        if label:
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

    return ParameterPlan(parameters=tuple(parameters), slots=tuple(slots))


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
