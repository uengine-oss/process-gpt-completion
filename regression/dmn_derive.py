"""regression/dmn_derive.py — 의사결정 규칙 표에서 회귀 시나리오를 파생한다.

규칙 표 자체가 "이 조건이면 이 결론" 을 다 적어 놓았으므로, 각 행을 맞히는 입력을 만드는
일은 계산이다. 모델을 부르지 않는다.

기대값은 **지금 표가 내는 답**으로 굳힌다. 회귀 테스트가 답할 질문은 "이 변경으로 지금
되던 게 깨지는가" 이므로, 사람이 옳다고 여기는 답이 아니라 현재 실제로 나오는 답이 기준이어야
바뀐 것만 드러난다.
"""

from __future__ import annotations

import json

from typing import Any

from . import dmn_replay


# 경계값을 만들 때 숫자 조건에서 한 칸 움직이는 폭.
_EPSILON = 1


def _satisfying_value(condition: dict) -> Any:
    """이 조건을 만족시키는 값 하나."""
    operator = condition.get("operator") or "eq"
    value = condition.get("value")
    if operator == "any":
        return ""
    if operator in ("gte", "lte", "eq"):
        return value
    if operator == "gt":
        n = dmn_replay._as_number(value)
        return (n + _EPSILON) if n is not None else value
    if operator == "lt":
        n = dmn_replay._as_number(value)
        return (n - _EPSILON) if n is not None else value
    if operator in dmn_replay._NOT_EQUAL:
        # 같지 않기만 하면 되므로, 값에 표식을 붙여 확실히 다르게 만든다.
        return f"{value}-아님"
    if operator in ("contains", "startsWith", "endsWith"):
        return str(value)
    return value


def _boundary_values(condition: dict) -> list[tuple[str, Any]]:
    """숫자 경계 조건의 바로 안쪽/바깥쪽 값.

    임계값을 옮기는 변경(50만원 → 60만원)은 경계 바로 옆 입력에서만 드러난다. 표 한가운데
    값만 넣어 두면 임계값이 바뀌어도 같은 행에 맞아 회귀가 잡히지 않는다.
    """
    operator = condition.get("operator") or "eq"
    n = dmn_replay._as_number(condition.get("value"))
    if n is None or operator not in ("gte", "gt", "lte", "lt"):
        return []
    if operator in ("gte", "lt"):
        return [("경계", n), ("경계 밖", n - _EPSILON)]
    return [("경계", n), ("경계 밖", n + _EPSILON)]


_OPERATOR_TEXT = {
    "eq": "=",
    "ne": "\u2260",
    "gt": ">",
    "gte": "\u2265",
    "lt": "<",
    "lte": "\u2264",
}


def _value_text(value: Any) -> str:
    """조건 값 하나를 규칙 표에 적힌 대로 읽히게."""
    if isinstance(value, bool):
        return "참" if value else "거짓"
    if isinstance(value, (int, float)):
        # 1000000.0 을 1000000 으로 — 표에는 정수로 적혀 있다.
        return str(int(value)) if float(value).is_integer() else str(value)
    return f'"{value}"'


def _condition_text(condition: dict) -> str:
    """조건 하나를 사람이 읽는 한 마디로. (예: `고객등급 = "SILVER"`, `최근거래금액 > 1000000`)

    화면에는 입력값만 보이고 "그래서 이게 어느 규칙을 맞히는 입력인지" 는 보이지 않았다.
    조건을 못 읽으면 검증 결과가 왜 그렇게 나왔는지 표를 따로 열어 봐야 한다.
    """
    key = str(condition.get("key") or "")
    operator = str(condition.get("operator") or "eq")
    value = condition.get("value")
    if operator == "any":
        return f"{key} 무관"
    if operator in _OPERATOR_TEXT:
        return f"{key} {_OPERATOR_TEXT[operator]} {_value_text(value)}"
    if operator == "contains":
        return f"{key}에 {_value_text(value)} 포함"
    if operator == "notContains":
        return f"{key}에 {_value_text(value)} 미포함"
    if operator == "startsWith":
        return f"{key}가 {_value_text(value)} 로 시작"
    if operator == "endsWith":
        return f"{key}가 {_value_text(value)} 로 끝남"
    return f"{key} {operator} {_value_text(value)}"


def _rule_condition_texts(table: dict, rule_index: int) -> list[str]:
    """그 규칙 행이 요구하는 조건들. 행이 없으면(매칭 없음 시나리오) 빈 목록."""
    rules = table.get("rules") or []
    if rule_index is None or rule_index < 0 or rule_index >= len(rules):
        return []
    return [_condition_text(c) for c in (rules[rule_index].get("conditions") or [])]


def _tidy(value: Any) -> Any:
    """화면에 그대로 나가는 값이라 `499999.0` 같은 꼬리를 정리한다 — 표에는 정수로 적혀 있다."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _inputs_for_rule(table: dict, rule: dict) -> dict:
    """이 규칙 행을 맞히는 입력 한 벌."""
    values: dict[str, Any] = {}
    for condition in (rule.get("conditions") or []):
        key = condition.get("key")
        if not key:
            continue
        values[key] = _tidy(_satisfying_value(condition))
    # 조건이 없는 입력 열도 키는 채워 둔다 — 화면에서 무엇을 넣었는지 보이게.
    for inp in (table.get("inputs") or []):
        values.setdefault(inp["item"], "")
    return values


def _unmatched_inputs(table: dict) -> dict:
    """어느 행에도 맞지 않는 입력.

    규칙 행이 지워지거나 조건이 좁아졌을 때 "매칭 없음" 으로 떨어지는 것을 확인하는
    자리다. 만들 수 없으면(모든 값을 받는 표) 빈 dict 를 돌려주고 호출부가 건너뛴다.
    """
    values = {inp["item"]: "__없는값__" for inp in (table.get("inputs") or [])}
    if not values:
        return {}
    if dmn_replay.evaluate(table, values)["matched_rule_index"] != -1:
        return {}
    return values


def _derive_for_table(table: dict, prefix: str) -> list[dict]:
    """표 하나에서 시나리오를 만든다.

    같은 결과(맞는 행 + 결론)를 내는 입력은 하나만 남긴다. 표가 넓으면 경계값이 서로 겹쳐
    같은 검증을 여러 번 하게 되는데, 그만큼 화면이 길어지고 리뷰어가 읽지 않는다.
    """
    rules = table.get("rules") or []
    if not rules:
        return []

    seen: set[tuple] = set()
    cases: list[dict] = []

    def _add(name: str, values: dict) -> None:
        observed = dmn_replay.evaluate(table, values)
        signature = (
            observed["matched_rule_index"],
            observed["outcome"],
            json.dumps(values, sort_keys=True, ensure_ascii=False),
        )
        if signature in seen:
            return
        seen.add(signature)
        cases.append({
            "name": f"{prefix}{name}",
            "inputs": values,
            "observed": observed,
            # 이 시나리오가 어느 결정을 지키는지. 재생이 이 값으로 표를 고른다.
            "decision_id": table.get("decision_id") or "",
            "decision_name": table.get("name") or "",
            # 화면이 "이 입력이 어느 조건을 맞히는가" 를 그대로 보여줄 수 있게 남긴다.
            "conditions": _rule_condition_texts(table, observed.get("matched_rule_index")),
        })

    for index, rule in enumerate(rules):
        base_values = _inputs_for_rule(table, rule)
        _add(f"규칙 {index + 1}행 — {rule.get('outcome') or '결론 없음'}", base_values)

        for condition in (rule.get("conditions") or []):
            key = condition.get("key")
            for label, value in _boundary_values(condition):
                probe = dict(base_values)
                probe[key] = _tidy(value)
                _add(f"규칙 {index + 1}행 {key} {label}", probe)

    unmatched = _unmatched_inputs(table)
    if unmatched:
        _add("어느 규칙에도 맞지 않는 입력", unmatched)

    return cases


def derive_cases(dmn_xml: str) -> list[dict]:
    """DMN 의 **모든** 결정 표에서 시나리오를 만든다 — 모델 호출 없음.

    하나의 DMN 에 결정이 여럿인 것이 보통인데, 첫 결정만 덮으면 나머지 표는 규칙이 통째로
    바뀌어도 검증이 아무 말도 하지 못한다(실제로 `할인율 산정` 을 새로 더한 요청에서
    그 표가 검증 대상에 들지 않았다).

    결정이 둘 이상이면 이름 앞에 결정 이름을 붙인다 — 표마다 `규칙 1행` 이 있어서
    구분이 안 되면 목록에서 어느 표가 깨졌는지 읽을 수 없다.
    """
    tables = [t for t in dmn_replay.parse_decision_tables(dmn_xml) if t.get("rules")]
    if not tables:
        return []

    multi = len(tables) > 1
    cases: list[dict] = []
    for table in tables:
        label = (table.get("name") or table.get("decision_id") or "").strip()
        prefix = f"{label} · " if (multi and label) else ""
        cases.extend(_derive_for_table(table, prefix))
    return cases


def to_eval_cases(derived: list[dict]) -> list[dict]:
    """파생 시나리오를 스위트 케이스 모양으로 옮긴다.

    단언은 결론과 맞은 행 두 가지 값 비교뿐이다 — 의미 판단형이 섞이면 채점에 모델이
    끼어들어, "달라졌는지 보는" 일에 실행마다 흔들리는 판정이 들어온다.
    """
    cases: list[dict] = []
    for position, item in enumerate(derived or []):
        name = str(item.get("name") or "").strip()
        observed = item.get("observed") or {}
        if not name:
            continue
        outcome = observed.get("outcome") or ""
        index = observed.get("matched_rule_index")
        rule_id = observed.get("matched_rule_id") or ""
        rule_label = dmn_replay._rule_label(index)

        # 규칙은 **정체(id)** 로 가리킨다. 행 번호로 걸면 위에 행이 하나 끼어드는 것만으로
        # 아래 행이 전부 밀려서, 결론이 그대로인데도 깨진 것으로 잡힌다.
        # id 가 없는 표(예전 편집기 산출물)에서만 예전처럼 행 번호로 건다.
        if rule_id or index is None or index < 0:
            rule_text = f"같은 규칙이 적용된다 (변경 전 {rule_label})" if rule_id else "어느 규칙에도 맞지 않는다"
            rule_check = {"type": dmn_replay.CHECK_MATCHED_RULE_ID, "value": rule_id}
        else:
            rule_text = f"{rule_label} 이 적용된다"
            rule_check = {"type": dmn_replay.CHECK_MATCHED_RULE, "value": index}

        texts = [
            f"결론이 '{outcome}' 이다" if outcome else "어느 규칙에도 맞지 않는다",
            rule_text,
        ]
        checks = [
            {"type": dmn_replay.CHECK_OUTCOME_EQUALS, "value": outcome},
            rule_check,
        ]
        # 어느 결정을 지키는 시나리오인지 본문에 함께 싣는다 — 재생이 이 값으로 표를 고른다.
        payload: dict[str, Any] = {"inputs": item.get("inputs") or {}}
        if item.get("decision_id"):
            payload["decision_id"] = item["decision_id"]
        if item.get("decision_name"):
            payload["decision_name"] = item["decision_name"]

        # 기대값은 숫자 두 개(결론·행 번호)만으로는 읽히지 않는다 — 어느 결정의 어느 조건을
        # 맞혀서 그 결론이 나오는지까지 함께 적어야 화면이 문장으로 보여줄 수 있다.
        expected = {
            "outcome": outcome,
            "matched_rule_index": index,
            "matched_rule_id": rule_id,
            "decision_name": item.get("decision_name") or "",
            "rule_label": rule_label,
            "conditions": item.get("conditions") or [],
        }

        cases.append({
            "eval_name": name,
            "prompt": json.dumps(payload, ensure_ascii=False),
            "expected_output": json.dumps(expected, ensure_ascii=False),
            "files": [],
            "assertions": texts,
            "checks": checks,
            "position": position,
        })
    return cases


