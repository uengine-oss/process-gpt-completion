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


def _inputs_for_rule(table: dict, rule: dict) -> dict:
    """이 규칙 행을 맞히는 입력 한 벌."""
    values: dict[str, Any] = {}
    for condition in (rule.get("conditions") or []):
        key = condition.get("key")
        if not key:
            continue
        values[key] = _satisfying_value(condition)
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


def derive_cases(dmn_xml: str) -> list[dict]:
    """규칙 표에서 시나리오를 만든다 — 모델 호출 없음.

    같은 결과(맞는 행 + 결론)를 내는 입력은 하나만 남긴다. 표가 넓으면 경계값이 서로 겹쳐
    같은 검증을 여러 번 하게 되는데, 그만큼 화면이 길어지고 리뷰어가 읽지 않는다.
    """
    table = dmn_replay.parse_decision_table(dmn_xml)
    rules = table.get("rules") or []
    if not rules:
        return []

    seen: set[tuple] = set()
    cases: list[dict] = []

    def _add(name: str, values: dict) -> None:
        observed = dmn_replay.evaluate(table, values)
        signature = (observed["matched_rule_index"], observed["outcome"], json.dumps(values, sort_keys=True, ensure_ascii=False))
        if signature in seen:
            return
        seen.add(signature)
        cases.append({"name": name, "inputs": values, "observed": observed})

    for index, rule in enumerate(rules):
        base_values = _inputs_for_rule(table, rule)
        _add(f"규칙 {index + 1}행 — {rule.get('outcome') or '결론 없음'}", base_values)

        for condition in (rule.get("conditions") or []):
            key = condition.get("key")
            for label, value in _boundary_values(condition):
                probe = dict(base_values)
                probe[key] = value
                _add(f"규칙 {index + 1}행 {key} {label}", probe)

    unmatched = _unmatched_inputs(table)
    if unmatched:
        _add("어느 규칙에도 맞지 않는 입력", unmatched)

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
        texts = [
            f"결론이 '{outcome}' 이다" if outcome else "어느 규칙에도 맞지 않는다",
            f"{dmn_replay._rule_label(index)} 이 적용된다",
        ]
        checks = [
            {"type": dmn_replay.CHECK_OUTCOME_EQUALS, "value": outcome},
            {"type": dmn_replay.CHECK_MATCHED_RULE, "value": index},
        ]
        cases.append({
            "eval_name": name,
            "prompt": json.dumps({"inputs": item.get("inputs") or {}}, ensure_ascii=False),
            "expected_output": json.dumps(
                {"outcome": outcome, "matched_rule_index": index}, ensure_ascii=False
            ),
            "files": [],
            "assertions": texts,
            "checks": checks,
            "position": position,
        })
    return cases


