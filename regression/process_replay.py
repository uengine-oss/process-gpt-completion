"""core/skills/process_replay.py — 저장된 시나리오를 엔진 없이 재생해 경로를 다시 계산한다.

## 왜 엔진을 쓰지 않는가

병합 전 검증을 생성 시 검증과 같은 방식으로 돌리면 — 케이스마다 새 인스턴스를 만들고
폴링 서비스가 진행시키는 방식 — 회차마다 프로세스 실행 한 벌을 통째로 지불한다.
게이트웨이 조건은 엔진이 모델로 판정하고, serviceTask 와 에이전트 담당 액티비티는 폴링
서비스가 실제로 실행한다. 케이스 4개짜리 프로세스면 base·head 합쳐 인스턴스 8개가 돈다.

회귀 테스트가 답해야 할 질문은 "이 변경으로 **경로가 달라지는가**" 이지 "에이전트가 이번에도
좋은 산출물을 내는가" 가 아니다. 판정 결과를 시나리오가 이미 들고 있으므로(생성 시 검증에서
`todolist.gateway_decisions` 로 기록됐다), 남은 것은 그래프를 따라가는 계산뿐이다.
모델 호출 0회, 엔진 인스턴스 0개.

## 판정을 추측하지 않는다

기록되지 않은 갈림길을 만나면 그 시나리오를 **판정 불가**로 남기고 재생을 멈춘다. 경로만
보고 역산하는 방식은 두 분기가 같은 액티비티로 향할 때("고액 → 정밀검토",
"특수건 → 정밀검토") 구분이 불가능하고, 그 상태로 기준을 굳히면 깨진 변경을 "이상 없음"
으로 통과시킨다. 회귀 테스트가 조용히 거짓말을 하는 것보다 "확인 못 했다" 가 낫다.

같은 이유로 **병렬·포함 게이트웨이는 재생 대상이 아니다.** 이들은 "어느 분기를 골랐나" 가
아니라 "어디서 몇 개를 기다렸다 합치나" 의 문제라 엔진의 합류 규칙을 그대로 옮겨야 하는데,
그 사본을 여기 두면 엔진이 규칙을 고칠 때 회귀 테스트만 옛 규칙으로 통과시킨다.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 재생이 판정할 수 있는 갈림길. 나머지는 판정 불가로 남긴다.
_REPLAYABLE_GATEWAYS = ("exclusivegateway", "xor", "xorgateway")

UNDECIDED = "undecided"


class Undecidable(Exception):
    """재생으로는 다음 노드를 정할 수 없다. 사유를 그대로 결과에 남긴다."""


def _nodes(definition: dict) -> dict[str, dict]:
    """정의의 모든 노드를 id → 노드로 모은다.

    ProcessGPT 정의는 활동/게이트웨이/이벤트/서브프로세스를 각각 다른 배열에 담고, 예전
    정의는 startEvent·endEvent 를 gateways 배열에 섞어 넣기도 한다. 어느 쪽이든 같은
    맵으로 본다 — 배열 이름이 아니라 type 으로 판단한다.
    """
    out: dict[str, dict] = {}
    for key in ("activities", "gateways", "events", "subProcesses"):
        for node in (definition.get(key) or []):
            if isinstance(node, dict) and node.get("id"):
                out[str(node["id"])] = node
    return out


def _outgoing(definition: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for seq in (definition.get("sequences") or []):
        if isinstance(seq, dict) and seq.get("source"):
            out.setdefault(str(seq["source"]), []).append(seq)
    return out


def _node_type(node: dict | None) -> str:
    return str((node or {}).get("type") or "").strip().lower()


def _start_node(definition: dict, nodes: dict[str, dict]) -> str | None:
    for node_id, node in nodes.items():
        if _node_type(node) == "startevent":
            return node_id
    return None


def _pick_next(
    node_id: str,
    node: dict,
    outgoing: list[dict],
    decisions: dict,
) -> str:
    """이 노드 다음에 갈 노드 하나를 정한다."""
    if not outgoing:
        raise Undecidable(f"'{node_id}' 에서 나가는 연결이 없습니다.")

    kind = _node_type(node)
    if kind not in _REPLAYABLE_GATEWAYS and len(outgoing) == 1:
        return str(outgoing[0].get("target") or "")

    if kind not in _REPLAYABLE_GATEWAYS:
        # 병렬·포함 게이트웨이이거나, 게이트웨이가 아닌데 분기가 여럿인 정의.
        raise Undecidable(
            f"'{node_id}'({kind or '종류 미상'}) 는 재생으로 판정할 수 없는 갈림길입니다. "
            "합류 규칙이 필요한 분기는 실엔진 모드로 확인해야 합니다."
        )

    recorded = (decisions or {}).get(node_id) or {}
    selected = [str(s) for s in (recorded.get("selected") or [])]
    if not selected:
        raise Undecidable(
            f"'{node_id}' 에서 어느 분기를 골랐는지 기록이 없습니다. "
            "이 시나리오는 판정 근거 없이 만들어졌거나, 기록 이전에 저장된 것입니다."
        )
    if len(selected) > 1:
        raise Undecidable(
            f"'{node_id}' 에서 분기가 {len(selected)}개 선택된 것으로 기록돼 있습니다 — "
            "배타 게이트웨이의 기록으로 성립하지 않습니다."
        )

    seq_id = selected[0]
    for seq in outgoing:
        if str(seq.get("id")) == seq_id:
            return str(seq.get("target") or "")
    raise Undecidable(
        f"기록된 분기 '{seq_id}' 가 '{node_id}' 의 연결에 없습니다 — "
        "이 변경으로 그 분기가 사라졌습니다."
    )


def replay(definition: dict, case_inputs: dict) -> dict:
    """시나리오 하나를 정의 위에서 재생해 실행 경로를 계산한다.

    Args:
        definition: 실행용 프로세스 정의(flattened).
        case_inputs: 저장된 케이스의 입력 — `gateway_decisions` 를 쓴다.
            `activity_inputs` 는 재생에서 쓰지 않는다. 액티비티를 실제로 수행하지 않기
            때문이다(수행하면 에이전트가 붙어 회차마다 실행 비용이 든다).

    Returns:
        {"activity_order": [...], "reached_end": bool, "undecided": str|None}
        `undecided` 가 차 있으면 그 시나리오는 비교에 쓰지 않는다.
    """
    nodes = _nodes(definition)
    outgoing = _outgoing(definition)
    decisions = (case_inputs or {}).get("gateway_decisions") or {}

    start = _start_node(definition, nodes)
    if not start:
        return {"activity_order": [], "reached_end": False,
                "undecided": "정의에 startEvent 가 없어 어디서 시작할지 알 수 없습니다."}

    order: list[str] = []
    seen_nodes: set[str] = set()
    current = start
    # 노드 수의 몇 배까지만 돈다 — 되돌아가는 흐름(반려 → 재신청)이 있는 정의에서
    # 기록이 어긋나면 무한히 돌 수 있다.
    max_steps = max(16, len(nodes) * 4)

    for _ in range(max_steps):
        node = nodes.get(current)
        if node is None:
            return {"activity_order": order, "reached_end": False,
                    "undecided": f"'{current}' 노드가 정의에 없습니다 — 이 변경으로 사라졌습니다."}

        kind = _node_type(node)
        if kind == "endevent":
            return {"activity_order": order, "reached_end": True, "undecided": None}

        if kind not in ("startevent", "endevent") and not kind.endswith("gateway"):
            # 액티비티·서브프로세스만 경로에 남긴다. 게이트웨이와 이벤트는 경로가 아니라
            # 경로를 정하는 장치라, 기대 경로(생성 시 검증의 actual_order)에도 없다.
            order.append(current)

        step_key = f"{current}#{len(order)}"
        if step_key in seen_nodes:
            return {"activity_order": order, "reached_end": False,
                    "undecided": f"'{current}' 에서 같은 자리를 다시 지납니다 — 기록이 이 정의와 맞지 않습니다."}
        seen_nodes.add(step_key)

        try:
            current = _pick_next(current, node, outgoing.get(current) or [], decisions)
        except Undecidable as e:
            return {"activity_order": order, "reached_end": False, "undecided": str(e)}
        if not current:
            return {"activity_order": order, "reached_end": False,
                    "undecided": "연결의 도착 노드가 비어 있습니다."}

    return {"activity_order": order, "reached_end": False,
            "undecided": "단계 상한을 넘었습니다 — 흐름이 끝나지 않습니다."}


def grade(observed: dict, texts: list[str], checks: list[dict | None]) -> dict | None:
    """재생 결과를 저장된 단언과 대조한다. 전부 값 비교라 모델을 부르지 않는다.

    판정 불가(`undecided`)면 None 을 돌려준다 — 0점으로 매기면 "이 변경으로 깨졌다" 로
    읽혀서, 확인하지 못한 것과 깨진 것이 구분되지 않는다.
    """
    if observed.get("undecided"):
        return None
    if not texts or not all(checks):
        return None

    expectations = []
    for text, check in zip(texts, checks):
        kind = str(check.get("type") or "")
        value = check.get("value")
        if kind == "path_equals":
            actual = observed.get("activity_order") or []
            expected = [str(v) for v in (value or [])]
            ok = actual == expected
            evidence = (
                "경로 일치" if ok
                else f"기대 {' → '.join(expected) or '(없음)'} / 실제 {' → '.join(actual) or '(없음)'}"
            )
        elif kind == "reached_end":
            ok = bool(observed.get("reached_end")) == bool(value)
            evidence = f"endEvent 도달={observed.get('reached_end')} (기대 {bool(value)})"
        else:
            # 재생이 판정할 수 없는 검사가 섞여 있으면 통째로 판정하지 않는다 —
            # 반쪽짜리 채점을 합치다 어긋나는 것보다 확인 못 했다고 남기는 편이 안전하다.
            return None
        expectations.append({"text": text, "passed": bool(ok), "evidence": evidence})

    passed = sum(1 for e in expectations if e["passed"])
    total = len(expectations)
    return {
        "assertions": expectations,
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else None,
        "graded_by": "replay",
    }


def replay_case(definition: dict, case: dict) -> tuple[dict, dict | None]:
    """케이스 하나를 재생하고 채점한다. (관측, 채점) 을 돌려준다."""
    import json as _json

    try:
        inputs = _json.loads(case.get("prompt") or "{}")
    except Exception:
        inputs = {}
    observed = replay(definition, inputs if isinstance(inputs, dict) else {})
    graded = grade(observed, case.get("assertions") or [], case.get("checks") or [])
    return observed, graded
