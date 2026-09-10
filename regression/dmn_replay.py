"""core/skills/dmn_replay.py — 의사결정(DMN) 규칙 표를 코드로 평가한다.

## 왜 모델이 필요 없는가

의사결정은 프로세스와 달리 "실행해 봐야 아는" 부분이 없다. 규칙 표는 입력값을 조건과
대조해 먼저 맞는 행을 고르는 일이고, 그건 계산이다. 화면(`ProcessGPTBackend.executeBusinessRule`)
이 이미 그렇게 하고 있다 — 여기서는 같은 판정을 서버에서 하도록 옮겨 온다.

그래서 DMN 회귀 검증은 시나리오 준비·실행·채점이 전 구간 토큰 0이다.

## 화면과 같은 답을 내야 한다

판정 규칙이 화면과 갈라지면 "화면에서는 승인인데 검증은 반려" 가 되어 회귀 결과를 믿을 수
없다. 그래서 파싱(`businessRuleDmn.ts:unaryTestToOperatorAndValue`)과 매칭
(`ProcessGPTBackend.__matchesBusinessCondition`)의 동작을 그대로 옮겼다. 한 곳만 다르다 —
`not(...)` 을 파서는 `neq` 로 내는데 매처는 `ne` 만 알아서 화면에서는 그 조건이 조용히
동등 비교로 떨어진다. 여기서는 둘 다 부등호로 받는다(아래 `_NOT_EQUAL` 주석).
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

logger = logging.getLogger(__name__)

# 파서는 not(...) 을 'neq' 로 내고 매처는 'ne' 를 본다 — 둘 다 "같지 않다" 로 받는다.
_NOT_EQUAL = ("ne", "neq")

CHECK_OUTCOME_EQUALS = "outcome_equals"
# 행 번호로 규칙을 가리키는 예전 검사. 위에 행이 하나 끼어들면 아래 행이 전부 밀려서
# **동작이 그대로인데도 실패**로 잡힌다(제주도 조건을 3행에 끼워 넣자 4·5행이 그렇게 됐다).
CHECK_MATCHED_RULE = "matched_rule_index"
# 규칙의 정체(id)로 가리키는 검사. 행이 밀려도 같은 규칙이면 통과하고, 다른 규칙이
# 대신 맞으면 결론이 우연히 같아도 잡아낸다 — 자리는 바뀌어도 규칙은 그대로이기 때문이다.
CHECK_MATCHED_RULE_ID = "matched_rule_id"

_NO_MATCH = -1


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find_all(root, name: str) -> list:
    return [e for e in root.iter() if _local(e.tag) == name]


def _find_first(root, name: str):
    for e in root.iter():
        if _local(e.tag) == name:
            return e
    return None


def _text_of(node) -> str:
    if node is None:
        return ""
    child = _find_first(node, "text")
    if child is not None and child.text:
        return child.text.strip()
    return (node.text or "").strip()


def _unquote(value: str) -> str:
    s = str(value or "").strip()
    m = re.match(r'^"(.*)"$', s, re.DOTALL)
    if m:
        return m.group(1).replace('\\"', '"').replace("\\\\", "\\")
    return s


def _unary_test(test: str) -> tuple[str, Any]:
    """FEEL 단항 테스트를 (연산자, 값) 으로 가른다. businessRuleDmn.ts 와 같은 규칙."""
    t = str(test or "").strip()
    if not t or t == "-":
        # 빈 칸은 "아무 값이나" 라는 뜻이다. 화면은 gte '' 로 두는데, 그 비교는 늘 참이
        # 되지 않으므로 여기서는 명시적으로 '언제나 참' 으로 다룬다.
        return "any", ""

    m = re.match(r"^>=\s*(.+)$", t)
    if m:
        return "gte", _unquote(m.group(1))
    m = re.match(r"^<=\s*(.+)$", t)
    if m:
        return "lte", _unquote(m.group(1))
    m = re.match(r"^>\s*(.+)$", t)
    if m and not m.group(1).startswith(("date", "time")):
        return "gt", _unquote(m.group(1))
    m = re.match(r"^<\s*(.+)$", t)
    if m and not m.group(1).startswith(("date", "time")):
        return "lt", _unquote(m.group(1))
    m = re.match(r"^not\((.*)\)$", t, re.DOTALL)
    if m:
        return "neq", _unquote(m.group(1))
    m = re.match(r"^contains\(\?\s*,\s*(.*)\)$", t, re.DOTALL)
    if m:
        return "contains", _unquote(m.group(1))
    return "eq", _unquote(t)


def _as_number(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalize(value: Any) -> Any:
    """숫자로 읽히면 숫자로, 불리언 문자열이면 불리언으로."""
    if isinstance(value, bool):
        return value
    s = str(value).strip() if value is not None else ""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    n = _as_number(s)
    return n if n is not None else s


_EMPTY_TABLE = {"decision_id": "", "name": "", "inputs": [], "rules": []}


def parse_decision_tables(dmn_xml: str) -> list[dict]:
    """DMN XML 의 **모든** 결정 표를 뽑는다.

    하나의 DMN 에 결정이 여럿인 것이 보통이다(기본 혜택 · 추가 혜택 · 등급 업그레이드…).
    첫 결정만 읽으면 나머지 표는 규칙이 통째로 바뀌어도 회귀 검증이 아무 말도 하지 못한다.

    Returns:
        [{"decision_id", "name", "inputs": [{item,label,mode}], "rules": [...]}, …]
    """
    if not dmn_xml or not dmn_xml.strip():
        return []
    try:
        root = ET.fromstring(dmn_xml)
    except ET.ParseError as e:
        logger.warning("dmn-replay: XML 파싱 실패: %s", e)
        return []

    tables: list[dict] = []
    for decision in [c for c in root if _local(c.tag) == "decision"]:
        table = _find_first(decision, "decisionTable")
        if table is None:
            continue
        parsed = _parse_table(table)
        parsed["decision_id"] = decision.get("id") or ""
        parsed["name"] = decision.get("name") or ""
        tables.append(parsed)
    return tables


def find_decision_table(dmn_xml: str, decision_id: str = "", decision_name: str = "") -> dict | None:
    """시나리오가 가리키는 결정의 표. 못 찾으면 None.

    id 로 먼저 찾고, 없으면 이름으로 찾는다 — 편집기가 결정을 다시 만들면 id 는 바뀌어도
    이름은 남는 경우가 많다. 둘 다 비어 있으면(예전 시나리오) 첫 표를 쓴다.
    """
    tables = parse_decision_tables(dmn_xml)
    if not tables:
        return None
    if decision_id:
        for table in tables:
            if table.get("decision_id") == decision_id:
                return table
    if decision_name:
        for table in tables:
            if table.get("name") == decision_name:
                return table
    if decision_id or decision_name:
        return None
    return tables[0]


def parse_decision_table(dmn_xml: str) -> dict:
    """첫 결정의 표. (결정을 가리지 않는 예전 호출부를 위해 남겨 둔다)"""
    tables = parse_decision_tables(dmn_xml)
    return tables[0] if tables else dict(_EMPTY_TABLE)


def _parse_table(table) -> dict:
    """decisionTable 요소 하나에서 입력 정의와 규칙 행을 뽑는다."""
    inputs: list[dict] = []
    for idx, node in enumerate([c for c in table if _local(c.tag) == "input"]):
        expr = _find_first(node, "inputExpression")
        expr_text = _text_of(expr) if expr is not None else ""
        label = node.get("label") or ""
        item = expr_text or label or f"input_{idx + 1}"
        type_ref = (expr.get("typeRef") if expr is not None else "") or ""
        inputs.append({"item": item, "label": label or item, "mode": type_ref or "enum"})

    rules: list[dict] = []
    for node in [c for c in table if _local(c.tag) == "rule"]:
        entries = [c for c in node if _local(c.tag) == "inputEntry"]
        outputs = [c for c in node if _local(c.tag) == "outputEntry"]
        annotations = [c for c in node if _local(c.tag) == "annotationEntry"]

        conditions = []
        for i, inp in enumerate(inputs):
            test = _text_of(entries[i]) if i < len(entries) else ""
            operator, value = _unary_test(test)
            conditions.append({"key": inp["item"], "operator": operator, "value": _normalize(value)})

        outcome = _unquote(_text_of(outputs[0])) if outputs else ""
        note = _unquote(_text_of(annotations[0])) if annotations else ""
        rules.append({
            "id": node.get("id") or "",
            "conditions": conditions,
            "outcome": outcome,
            "note": note,
        })

    return {"decision_id": "", "name": "", "inputs": inputs, "rules": rules}


def _matches(condition: dict, inputs: dict) -> bool:
    """조건 하나를 입력과 대조한다 — 화면 매처와 같은 규칙."""
    operator = condition.get("operator") or "eq"
    if operator == "any":
        return True

    actual_raw = (inputs or {}).get(condition.get("key"))
    actual = _normalize(actual_raw)
    expected = condition.get("value")

    if operator in ("gt", "gte", "lt", "lte"):
        a, b = _as_number(actual), _as_number(expected)
        if a is None or b is None:
            return False
        if operator == "gt":
            return a > b
        if operator == "gte":
            return a >= b
        if operator == "lt":
            return a < b
        return a <= b

    if operator in _NOT_EQUAL:
        return actual != expected
    if operator == "contains":
        return str(expected) in str(actual if actual is not None else "")
    if operator == "notContains":
        return str(expected) not in str(actual if actual is not None else "")
    if operator == "startsWith":
        return str(actual if actual is not None else "").startswith(str(expected))
    if operator == "endsWith":
        return str(actual if actual is not None else "").endswith(str(expected))
    return actual == expected


def evaluate(table: dict, inputs: dict) -> dict:
    """입력을 규칙 표에 넣고 먼저 맞는 행을 고른다(FIRST hit policy).

    Returns:
        {"matched_rule_index": int, "outcome": str, "note": str}
        어느 행에도 맞지 않으면 index 는 -1, outcome 은 빈 문자열이다 — "매칭 없음" 은
        결과가 아니라 결과가 없다는 뜻이라, 특정 결론으로 뭉개지 않는다.
    """
    for index, rule in enumerate(table.get("rules") or []):
        conditions = rule.get("conditions") or []
        if all(_matches(c, inputs) for c in conditions):
            return {
                "matched_rule_index": index,
                # 행 번호는 위에 행이 끼어들면 밀린다 — 규칙을 가리키는 것은 id 다.
                "matched_rule_id": rule.get("id") or "",
                "outcome": rule.get("outcome") or "",
                "note": rule.get("note") or "",
            }
    return {"matched_rule_index": _NO_MATCH, "matched_rule_id": "", "outcome": "", "note": ""}


def grade(observed: dict, texts: list[str], checks: list[dict | None]) -> dict | None:
    """평가 결과를 저장된 단언과 대조한다. 전부 값 비교라 모델을 부르지 않는다."""
    if not texts or not all(checks):
        return None

    expectations = []
    for text, check in zip(texts, checks):
        kind = str(check.get("type") or "")
        value = check.get("value")
        if kind == CHECK_OUTCOME_EQUALS:
            actual = observed.get("outcome") or ""
            ok = actual == (value or "")
            evidence = f"결론 '{actual or '(매칭 없음)'}' (기대 '{value or '(매칭 없음)'}')"
        elif kind == CHECK_MATCHED_RULE_ID:
            actual_id = observed.get("matched_rule_id") or ""
            actual_index = observed.get("matched_rule_index")
            ok = actual_id == (value or "")
            if not value:
                # 아무 규칙에도 맞지 않아야 하는 시나리오.
                evidence = f"{_rule_label(actual_index)} 적용" if actual_id else "매칭 없음"
            elif ok:
                evidence = f"{_rule_label(actual_index)} · 변경 전과 같은 규칙"
            elif actual_id:
                evidence = f"{_rule_label(actual_index)} · 다른 규칙이 적용됨"
            else:
                evidence = "매칭 없음"
        elif kind == CHECK_MATCHED_RULE:
            actual = observed.get("matched_rule_index")
            ok = actual == value
            evidence = (
                f"{_rule_label(actual)} (기대 {_rule_label(value)})"
            )
        else:
            # 표로 판정할 수 없는 검사가 섞이면 통째로 판정하지 않는다.
            return None
        expectations.append({"text": text, "passed": bool(ok), "evidence": evidence})

    passed = sum(1 for e in expectations if e["passed"])
    total = len(expectations)
    return {
        "assertions": expectations,
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else None,
        "graded_by": "dmn",
    }


def _rule_label(index: Any) -> str:
    if index is None or index == _NO_MATCH:
        return "매칭 없음"
    return f"규칙 {int(index) + 1}행"


def evaluate_case(dmn_xml: str, case: dict) -> tuple[dict, dict | None]:
    """케이스 하나를 평가하고 채점한다. (관측, 채점) 을 돌려준다."""
    import json as _json

    try:
        inputs = _json.loads(case.get("prompt") or "{}")
    except Exception:
        inputs = {}
    if not isinstance(inputs, dict):
        inputs = {}

    # 시나리오는 자기가 어느 결정을 지키는지 들고 다닌다. 없으면(예전 시나리오) 첫 결정이다.
    decision_id = str(inputs.get("decision_id") or "")
    decision_name = str(inputs.get("decision_name") or "")

    tables = parse_decision_tables(dmn_xml)
    if not tables:
        # 문서를 아예 읽지 못한 것과 결정 하나가 사라진 것은 다르다.
        # 전자는 판단할 수 없고, 후자는 판단해야 한다.
        return ({"matched_rule_index": _NO_MATCH, "outcome": "", "note": "",
                 "undecided": "이 버전에서 의사결정 표를 읽지 못했습니다."}, None)

    table = find_decision_table(dmn_xml, decision_id, decision_name)

    if table is None and (decision_id or decision_name):
        # 결정이 통째로 사라진 것도 동작 변화다 — "판단 불가" 로 접으면 병합해도 되는 줄 안다.
        # 실제로도 이 결정을 부르는 쪽은 아무 결론도 받지 못한다.
        label = decision_name or decision_id
        return (
            {
                "matched_rule_index": _NO_MATCH,
                "outcome": "",
                "note": f"이 버전에는 '{label}' 결정이 없습니다.",
            },
            grade(
                {"matched_rule_index": _NO_MATCH, "outcome": "", "note": ""},
                case.get("assertions") or [],
                case.get("checks") or [],
            ),
        )

    if table is None or not table.get("rules"):
        return ({"matched_rule_index": _NO_MATCH, "outcome": "", "note": "",
                 "undecided": "이 버전에서 의사결정 표를 읽지 못했습니다."}, None)

    observed = evaluate(table, inputs.get("inputs") or inputs)
    graded = grade(observed, case.get("assertions") or [], case.get("checks") or [])
    return observed, graded
