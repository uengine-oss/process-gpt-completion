"""regression/process_derive.py — 프로세스 정의에서 회귀 시나리오를 파생한다(모델·엔진 호출 없음).

## 왜 필요한가

프로세스 시나리오는 원래 생성 시 실엔진 검증(`process_validator.ProcessValidator`)이
통과시킨 케이스를 주워 담는 방식으로만 확보됐다. 그래서 그 게이트를 거치지 않고 만들어진
프로세스 — 대부분의 기존 프로세스 — 는 병합 요청을 열어도 "비교할 시나리오가 없습니다" 로
끝나, 리뷰어가 diff 를 눈으로 보는 수밖에 없었다.

그런데 병합 전 검증은 실엔진을 쓰지 않는다. 저장된 갈림길 판정을 들고 정의 그래프를
따라가 경로를 계산할 뿐이다(`regression/process_replay.py`). 즉 시나리오에 필요한 것은
**"어느 갈림길에서 어느 분기를 골랐나" 와 그때의 경로** 뿐이고, 그 둘은 정의만 있으면
계산할 수 있다. 실행해서 얻어야 하는 값이 아니다.

그래서 의사결정(`dmn_derive`)이 규칙 표에서 시나리오를 파생하듯, 여기서는 base 정의의
배타 게이트웨이 조합을 훑어 경로를 만든다 — 모델 호출 0회, 엔진 인스턴스 0개.

## 기대값은 지금 정의가 내는 경로다

회귀 테스트가 답할 질문은 "이 변경으로 지금 되던 게 깨지는가" 다. 그래서 만든 판정을
**현재(base) 정의에 재생해 나온 경로**를 기대값으로 굳힌다. 사람이 옳다고 여기는 순서가
아니라 지금 실제로 나오는 경로가 기준이어야, 바뀐 것만 드러난다.
"""

from __future__ import annotations

import json
import logging

from . import process_replay

logger = logging.getLogger(__name__)

# 한 프로세스에서 만들 시나리오 수 상한. 갈림길이 많으면 조합이 지수로 늘어나는데, 그만큼
# 화면이 길어지고 리뷰어가 읽지 않는다. 깊이 우선으로 훑으므로 앞쪽 분기부터 채워진다.
MAX_DERIVED_CASES = 12

# 한 경로에서 같은 노드를 지나도 되는 횟수. 되돌아가는 흐름(반려 → 재신청)을 한 바퀴는
# 밟아 봐야 하고, 무제한이면 순환 정의에서 경로가 끝나지 않는다.
_MAX_NODE_VISITS = 2

# 회귀 실행기가 트레이스와 대조할 검사. 값 비교라 모델이 필요 없다
# (`process_replay.grade` 가 읽는 이름과 같아야 한다).
CHECK_PATH_EQUALS = "path_equals"
CHECK_REACHED_END = "reached_end"


def _sequence_label(definition: dict, seq: dict) -> str:
    """분기 하나를 사람이 읽을 말로. 조건문이 있으면 그것, 없으면 도착 액티비티 이름."""
    condition = str(seq.get("condition") or "").strip()
    if condition:
        return condition
    nodes = process_replay._nodes(definition)
    target = nodes.get(str(seq.get("target") or "")) or {}
    return str(target.get("name") or seq.get("target") or seq.get("id") or "")


def _case_name(definition: dict, decisions: dict, outgoing: dict) -> str:
    """고른 분기들로 시나리오 이름을 만든다.

    이름은 저장된 **판정(decisions)** 에서 만든다 — 걸어온 순서가 아니라. 되돌아가는
    흐름에서는 같은 갈림길을 두 번 지나는데 판정은 갈림길당 하나뿐이라(재생이 그렇게
    읽는다), 걸어온 순서로 이름을 지으면 "…: 미비한 경우 · …: 충분한 경우" 처럼 실제로
    재생되지 않는 조합이 이름에 남는다.

    이름은 스위트 안에서 케이스를 가리키는 키이기도 하다(deepagents `sync_cases` 가
    eval_name 으로 맞춘다). 그래서 다시 만들었을 때 같은 경로가 같은 자리를 지킨다.
    """
    if not decisions:
        return "분기 없는 기본 경로"
    nodes = process_replay._nodes(definition)
    parts = []
    for gateway_id, decision in decisions.items():
        gateway = nodes.get(gateway_id) or {}
        gateway_name = str(gateway.get("name") or gateway_id)
        selected = (decision.get("selected") or [""])[0]
        seq = next(
            (s for s in (outgoing.get(gateway_id) or []) if str(s.get("id")) == selected),
            {},
        )
        parts.append(f"{gateway_name}: {_sequence_label(definition, seq)}")
    return " · ".join(parts)


def _enumerate_paths(definition: dict) -> tuple[list[list[tuple[str, dict]]], list[str]]:
    """배타 게이트웨이 조합을 훑어 "고른 분기들" 목록을 만든다.

    Returns:
        (조합 목록, 건너뛴 사유). 각 조합은 [(게이트웨이 id, 고른 sequence), …] 이고,
        경로 계산은 호출부가 `process_replay.replay` 로 한 번 더 한다 — 여기서 따로
        경로를 만들면 채점에 쓰이는 재생과 어긋날 수 있다.
    """
    nodes = process_replay._nodes(definition)
    outgoing = process_replay._outgoing(definition)
    start = process_replay._start_node(definition, nodes)
    if not start:
        return [], ["정의에 startEvent 가 없어 어디서 시작할지 알 수 없습니다."]

    found: list[list[tuple[str, dict]]] = []
    skipped: list[str] = []
    max_steps = max(16, len(nodes) * 4)

    def walk(node_id: str, visits: dict[str, int], choices: list[tuple[str, dict]], steps: int) -> None:
        if len(found) >= MAX_DERIVED_CASES:
            return
        node = nodes.get(node_id)
        if node is None:
            skipped.append(f"'{node_id}' 노드가 정의에 없습니다.")
            return
        if steps > max_steps:
            skipped.append("흐름이 끝나지 않아 경로를 만들지 못했습니다.")
            return

        kind = process_replay._node_type(node)
        if kind == "endevent":
            found.append(list(choices))
            return

        seen = visits.get(node_id, 0) + 1
        if seen > _MAX_NODE_VISITS:
            # 되돌아가는 흐름을 한 바퀴 넘게 밟지 않는다. 이 경로는 버린다.
            return
        visits = dict(visits, **{node_id: seen})

        sequences = outgoing.get(node_id) or []
        if not sequences:
            skipped.append(f"'{node.get('name') or node_id}' 에서 나가는 연결이 없습니다.")
            return
        if len(sequences) == 1:
            walk(str(sequences[0].get("target") or ""), visits, choices, steps + 1)
            return
        if kind not in process_replay._REPLAYABLE_GATEWAYS:
            # 병렬·포함 게이트웨이는 재생 대상이 아니다(process_replay 참고). 여기서
            # 시나리오를 만들어도 검증이 판정 불가로 흘려버린다.
            skipped.append(
                f"'{node.get('name') or node_id}' 는 재생으로 판정할 수 없는 갈림길이라 건너뛰었습니다."
            )
            return
        for seq in sequences:
            if len(found) >= MAX_DERIVED_CASES:
                return
            walk(str(seq.get("target") or ""), visits, choices + [(node_id, seq)], steps + 1)

    walk(start, {}, [], 0)
    return found, skipped


def _decisions_payload(choices: list[tuple[str, dict]], outgoing: dict) -> dict:
    """고른 분기를 재생이 읽는 `gateway_decisions` 모양으로 옮긴다.

    고른 것만이 아니라 그 갈림길의 **모든 분기**를 `sequences` 에 함께 싣는다. 화면이
    "어느 조건으로 갔는지" 를 조건문으로 보여주고, 분기가 사라진 변경도 그 자리에서 읽힌다.
    """
    decisions: dict = {}
    for gateway_id, seq in choices:
        sequences = {}
        for candidate in (outgoing.get(gateway_id) or []):
            sequences[str(candidate.get("id"))] = {
                "condition": candidate.get("condition") or "",
                "target": candidate.get("target") or "",
            }
        decisions[gateway_id] = {"selected": [str(seq.get("id"))], "sequences": sequences}
    return decisions


def derive_cases(definition: dict) -> tuple[list[dict], list[str]]:
    """정의에서 회귀 시나리오를 파생한다.

    Returns:
        (파생 시나리오 목록, 건너뛴 사유 목록). 각 시나리오는
        {name, gateway_decisions, activity_order, reached_end}.
    """
    combos, skipped = _enumerate_paths(definition)
    outgoing = process_replay._outgoing(definition)

    derived: list[dict] = []
    seen_paths: set[tuple] = set()
    used_names: set[str] = set()

    for choices in combos:
        decisions = _decisions_payload(choices, outgoing)
        observed = process_replay.replay(definition, {"gateway_decisions": decisions})
        if observed.get("undecided"):
            skipped.append(str(observed["undecided"]))
            continue
        order = [str(a) for a in (observed.get("activity_order") or [])]
        if not order:
            # 액티비티를 하나도 지나지 않는 경로는 비교 기준이 되지 못한다.
            continue
        signature = (tuple(order), bool(observed.get("reached_end")))
        if signature in seen_paths:
            # 같은 경로로 수렴하는 조합이 여럿이면 하나만 남긴다 — 같은 검증을 여러 번
            # 하는 만큼 화면만 길어진다.
            continue
        seen_paths.add(signature)

        name = _case_name(definition, decisions, outgoing)
        if name in used_names:
            name = f"{name} ({len(used_names) + 1})"
        used_names.add(name)

        derived.append({
            "name": name,
            "gateway_decisions": decisions,
            "activity_order": order,
            "reached_end": bool(observed.get("reached_end")),
        })

    return derived, skipped


def to_eval_cases(derived: list[dict]) -> list[dict]:
    """파생 시나리오를 스위트 케이스 모양으로 옮긴다.

    단언은 경로와 종료 도달 두 가지 값 비교뿐이다 — 의미 판단형이 섞이면 채점에 모델이
    끼어들어, "달라졌는지 보는" 일에 실행마다 흔들리는 판정이 들어온다.
    """
    cases: list[dict] = []
    for position, item in enumerate(derived or []):
        name = str(item.get("name") or "").strip()
        order = [str(a) for a in (item.get("activity_order") or [])]
        if not name or not order:
            continue
        reached_end = bool(item.get("reached_end"))
        texts = [
            "실행 경로가 " + " → ".join(order) + " 다",
            "프로세스가 endEvent 까지 진행된다" if reached_end
            else "프로세스가 endEvent 까지 진행되지 않는다",
        ]
        checks = [
            {"type": CHECK_PATH_EQUALS, "value": order},
            {"type": CHECK_REACHED_END, "value": reached_end},
        ]
        cases.append({
            "eval_name": name,
            "prompt": json.dumps(
                {
                    # 재생은 액티비티 입력값을 쓰지 않는다(액티비티를 실제로 수행하지
                    # 않으므로). 승격된 케이스와 모양을 맞추려고 키는 남긴다.
                    "activity_inputs": {},
                    "gateway_decisions": item.get("gateway_decisions") or {},
                },
                ensure_ascii=False,
            ),
            "expected_output": json.dumps(
                {"activity_order": order, "reached_end": reached_end}, ensure_ascii=False
            ),
            # 첨부 문서를 참조하면 그 파일을 못 받아오는 순간 두 버전이 나란히 0점이 되어
            # 비교가 무너진다. 프로세스 시나리오는 입력을 본문에 다 담으므로 비워 둔다.
            "files": [],
            "assertions": texts,
            "checks": checks,
            "position": position,
        })
    return cases
