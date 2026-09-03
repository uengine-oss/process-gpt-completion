"""순방향 결정론적 코드(고착화) 생성.

에이전트가 LLM 추론으로 수행한 활동이 반복적으로 같은 구조의 작업을 낸다면, 그
경로를 고정된 파이썬 코드로 굳혀 이후에는 LLM 없이 실행할 수 있다. 이 모듈은 그
코드를 만든다.

무엇을 읽는가:
    도구 호출 목록만 보지 않는다. 워크아이템이 남긴 작업 이력 전체 — 어떤 도구를
    썼는지, 어떤 스킬 파일을 읽고 그 절차를 따랐는지, 어떤 셸 스크립트를 돌렸는지,
    어떤 파일을 만들었는지 — 를 `work_history`로 정규화해 읽는다. MCP 호출만 보면
    스킬의 셸 스크립트로 일하는 활동은 영영 고착화되지 않고, 되돌릴 대상도 놓친다.

    그 정규화는 런타임에 매이지 않는다. DeepAgents, CrewAI, 앞으로 붙을 러너가 각자
    다른 모양으로 이벤트를 남겨도 같은 행위 목록으로 읽힌다. 특정 런타임의 이벤트
    스키마를 코드에 박으면 다른 러너의 이력이 통째로 버려진다.

언제 도는가:
    워크아이템이 **DONE으로 확정되는 순간**이다. 고착화의 근거는 "이 액티비티가
    성공적으로 끝난 적이 몇 번인가"이므로, 그 판정이 내려지는 자리에 붙어야 한다.
    폴링 서비스가 체크포인트를 평가해 DONE을 결정하므로 트리거도 거기 있다
    (`polling_service/database.py`의 워크아이템 저장 훅).

    사람의 제출(`/complete`)에 붙이면 안 된다. 자율 완료(COMPLETE) 모드 에이전트
    액티비티는 그 경로를 아예 지나지 않아 영원히 고착화되지 않고, 반대로 사람이
    제출하는 폼 액티비티에서는 부수효과 이력이 없어 매번 헛돈다. 게다가 고착화는
    최적화일 뿐이라 사람의 요청을 붙잡을 이유가 없다.

이 파일은 completion과 polling_service 양쪽에 같은 내용으로 존재한다. 두 서비스는
빌드 컨텍스트가 갈려 모듈을 공유할 수 없다(`database.py`, `llm_factory.py`도 같은
사정이다). 사본이 어긋나면 만든 코드와 되돌릴 입력이 서로 다른 세계를 가리키므로,
`tests/test_shared_module_parity.py`가 두 사본이 같은지 고정한다.

이전 구현(crewai-action의 `DeterministicCodeTool`)은 실행 1건을 LLM에 보여주고
무엇이 파라미터인지 추측하게 했다. 여기서는 지문이 같은 표본 여러 건을 자리별로
대조해 값이 변한 자리를 파라미터로 승격한다. 추측이 아니라 관측이므로 생성 단계에도
LLM 호출이 없다.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import json
import logging

from database import (
    fetch_mcp_python_code,
    upsert_mcp_python_code,
    fetch_related_workitem_outputs,
    fetch_workitem_by_proc_inst_and_activity,
    fetch_workitems_by_activity,
    fetch_events_by_todo_id,
    fetch_last_deactivated_at,
)
from deterministic_template import TEMPLATE
from mcp_tool_index import build_tool_index_from_tenant
from deterministic_signature import (
    ParameterPlan,
    bind_upstream,
    compose_from_known,
    procedure_pins,
    build_output_template,
    execution_fingerprint,
    identify_parameters,
    render_template,
    unrecoverable_parameters,
)
from work_history import (
    FILE_WRITE,
    SHELL,
    Action,
    canonicalize,
    effect_actions,
    normalize_events,
    summarize,
)

logger = logging.getLogger(__name__)

# 고착화에 필요한 표본 수. 표본이 여러 건이어야 무엇이 파라미터이고 무엇이 상수인지
# 추측이 아니라 관측으로 가려낼 수 있다.
REQUIRED_SAMPLES = 3

# 표본을 몇 건까지 거슬러 볼 것인가. 최근 N건만 보면 그 안에 다른 방식으로 일한 실행이
# 하나만 섞여도 굳지 않는다 — 에이전트는 같은 활동도 매번 조금씩 다르게 푼다(문서 하나를
# 두고 `mkdir && date` · `python 힙스크립트` · `write_file` 단독이 번갈아 나왔다).
# 넓게 훑어 **같은 구조끼리 모으고**, 그중 가장 많은 무리로 굳힌다. 판정 기준은 그대로다 —
# 지문이 같은 표본이 REQUIRED_SAMPLES 건 있어야 한다. 무리 짓는 자리만 옮긴 것이다.
#
# 무한정 넓히지는 않는다. DONE 마다 이 건수만큼 이벤트를 읽으므로 비용이 그만큼 든다.
SAMPLE_SCAN_LIMIT = REQUIRED_SAMPLES * 4

# 한 활동에서 시험해 볼 표본 조합의 상한.
#
# 표본 3건을 "전부 통과해야 하는 관문"으로 쓰면, 그중 한 건에서만 값의 출처를 못 찾아도
# 활동 전체를 포기한다. 실제로 같은 문서 생성 활동이 12번을 도는 동안 매번 다른 자리가
# 걸렸다 — 4~8회차는 셸 경로, 9~11회차는 사유의 낱말과 시각. 어느 회차에도 "굳을 수 있는
# 3건"은 이미 있었는데 뽑기를 한 번만 해서 놓친 것이다.
#
# 그래서 뽑기를 탐색으로 바꾼다. 같은 무리 안에서 조합을 바꿔 가며, 값의 출처가 모두
# 정해지는 3건을 찾는다. 조합 수만큼 판정 비용이 드니 상한을 둔다.
MAX_SAMPLE_COMBINATIONS = 24

# 고착화된 코드를 쓰고도 이 횟수 이상 재작업되면 코드 자체를 의심해 비활성화한다.
# 1회 재작업은 대개 입력이 틀린 경우이므로 되돌린 뒤 새 파라미터로 재실행한다.
REWORK_DISTRUST_THRESHOLD = 2


# 고착화된 코드를 **실제로 쓴** 실행이 남기는 실행 방식. 되돌리기만 하고 재실행은
# 에이전트가 맡은 회차(`deterministic-undo-only`), 되돌리기가 막혀 재작업이 통째로
# 에이전트에게 넘어간 회차(`deterministic-skipped` / `deterministic-undo-pending`),
# 카드가 아예 없는 회차는 코드를 쓴 것이 아니다.
CODE_EXECUTION_MODES = ("deterministic", "deterministic-undo")


def _event_payload(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    data = (event or {}).get("data")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (ValueError, TypeError):
            return None
    return data if isinstance(data, dict) else None


def ran_deterministic_code(events: Optional[List[Dict[str, Any]]]) -> bool:
    """이 회차가 고착화된 코드로 돌았는가. 실행 카드의 실행 방식으로 판정한다."""
    for event in events or []:
        payload = _event_payload(event)
        if payload and str(payload.get("execution_mode") or "") in CODE_EXECUTION_MODES:
            return True
    return False


def count_reworked_code_runs(proc_inst_id: str, activity_id: str, tenant_id: str) -> int:
    """고착화된 코드를 쓰고도 재작업된 회차 수.

    재작업 횟수를 그대로 세면 안 된다. 되돌리기가 실패해 재작업 전체가 에이전트에게
    넘어간 회차까지 "코드가 틀렸다"는 증거로 세면, 코드는 한 번도 의심받을 짓을 하지
    않았는데 비활성화된다. 실제로 그렇게 됐다 — 되돌리기 인프라가 깨져 재작업이 세 번
    반복되는 동안 코드가 돈 것은 첫 회차뿐이었는데, 그 활동은 비활성화되어 새 인스턴스
    까지 전부 에이전트가 맡게 됐다.

    코드가 실제로 돈 회차만 센다. 그 회차들이 재작업됐다는 것이 곧 코드를 의심할 근거다.
    """
    items = fetch_workitem_by_proc_inst_and_activity(
        proc_inst_id, activity_id, tenant_id, recent_only=False
    )
    if items is None:
        return 0
    if not isinstance(items, list):
        items = [items]
    used = 0
    for item in items:
        todo_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else None)
        if not todo_id:
            continue
        if ran_deterministic_code(fetch_events_by_todo_id(str(todo_id))):
            used += 1
    return used


def _trace_of(todo_id: str) -> List[Action]:
    """워크아이템 하나의 작업 이력을 정규화된 행위 목록으로 읽는다."""
    return normalize_events(fetch_events_by_todo_id(todo_id))


def _form_output_of(events: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """에이전트가 마지막에 내놓은 폼 산출물을 읽는다.

    실행기는 도구 호출이 끝난 뒤 `최종 결과 반환` 작업을 하나 더 띄우고, 그 완료
    이벤트에 `{폼아이디: {필드: 값}}` 을 싣는다. 워크아이템의 산출물이 되는 값이고,
    다음 활동이 입력으로 받는 값이기도 하다.

    이벤트 타입 이름만으로 고르지 않는다. 러너마다 이름이 다르므로, **모양**으로
    고른다 — 키가 하나뿐인 dict 이고 그 값이 다시 dict 인 완료 이벤트.
    """
    for event in sorted(events or [], key=lambda e: str(e.get("timestamp") or ""), reverse=True):
        if str(event.get("crew_type") or "") != "result":
            continue
        data = event.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                continue
        if (
            isinstance(data, dict)
            and len(data) == 1
            and isinstance(next(iter(data.values())), dict)
        ):
            return data
    return None


@dataclass(frozen=True)
class Sample:
    """고착화 표본 한 건. 값이 어디서 왔는지를 가려내는 데 필요한 것들을 함께 든다."""

    # 작업 이력 전체(맥락 행위 포함). 부수효과만 남기는 일은 지문 계산 직전에 한다.
    trace: List[Action]
    # 그 워크아이템의 지시문. 파라미터 이름표를 여기서 관측한다.
    query: str
    # 에이전트가 마지막에 내놓은 폼 산출물.
    output: Optional[Dict[str, Any]]
    # 그 워크아이템의 식별자들. 실행 때 정해지는 자리를 가려내는 근거다.
    identity: Dict[str, Any]
    # 그 실행 시점에 이미 끝나 있던 앞 워크아이템들의 산출물.
    upstream: List[Dict[str, Any]] = field(default_factory=list)


def collect_samples(
    proc_def_id: str, activity_id: str, tenant_id: str, since: Optional[str]
) -> List[Sample]:
    """고착화 표본을 모은다.

    자격은 "완료(DONE)되었고 이후 재작업되지 않은 워크아이템"이다. DONE은 사람의
    승인이 아니라 완료 상태일 뿐이므로, 양성 근거는 DONE 자체가 아니라 되돌려지지
    않았다는 사실에 둔다.

    ``since``(직전 비활성 시각) 이후에 완료된 것만 센다. 이전 표본을 다시 세면
    비활성화한 코드와 동일한 코드를 곧바로 재생성해 무한 반복에 빠진다.

    최신순으로 ``SAMPLE_SCAN_LIMIT`` 건까지 모은다. 여기서 3건으로 자르지 않는 이유는
    실행 방식이 흔들리기 때문이다 — 자르고 나면 그 안에 다른 방식이 하나만 섞여도
    굳지 않는다. 같은 구조끼리 묶는 일은 호출자가 지문을 보고 한다.

    각 표본은 **작업 이력 전체**(맥락 행위 포함)와 그 워크아이템의 지시문, 그리고 그
    워크아이템의 식별자다. 부수효과만 남기는 일은 지문 계산과 코드 컴파일 직전에 한다 —
    어떤 스킬을 읽고 그 절차를 따랐는지는 생성된 코드의 출처로 함께 기록해야 하기 때문이다.

    지시문을 함께 모으는 이유는 파라미터 이름표 때문이다. 값이 관측될 때 지시문에서
    그 앞에 무엇이 적혀 있었는지를 알아야, 다음 실행의 지시문에서 같은 값을 되찾을 수
    있다.

    식별자를 함께 모으는 이유는 그 반대다. `proc_inst_id`·`todo_id` 처럼 실행 때 정해지는
    값은 지시문에 적혀 있지 않아 아무리 뒤져도 되찾을 수 없다. 그런 자리를 가려내려면
    관측된 값을 그 워크아이템 자신의 식별자와 대조해야 한다.

    앞 워크아이템의 산출물도 함께 모은다. 값은 액티비티 경계를 넘어서도 흐른다 —
    에이전트가 `get_related_workitem_outputs` 로 읽어 쓴 값은 지시문에 없어서, 그것만
    보면 "되찾을 수 없는 입력"으로 판정되어 고착화가 통째로 막힌다.
    """
    candidates = fetch_workitems_by_activity(
        proc_def_id, activity_id, tenant_id, status="DONE", since=since,
        limit=SAMPLE_SCAN_LIMIT * 2,
    ) or []

    # 같은 프로세스 인스턴스에서 재작업이 이어진 실행은 되돌려진 것으로 보고 제외한다.
    highest_rework: Dict[str, int] = {}
    for item in candidates:
        key = str(item.get("proc_inst_id") or item.get("id"))
        highest_rework[key] = max(highest_rework.get(key, 0), int(item.get("rework_count") or 0))

    samples: List[Sample] = []
    for item in candidates:
        key = str(item.get("proc_inst_id") or item.get("id"))
        if int(item.get("rework_count") or 0) < highest_rework.get(key, 0):
            continue
        events = fetch_events_by_todo_id(item.get("id"))
        trace = normalize_events(events)
        # 자격도 결과 기준으로 본다. 곁가지(미리 만든 디렉터리, 덮어쓰인 중간 파일)는
        # 재현 대상이 아니므로 그것이 실패했다고 표본을 버리지 않는다.
        effects = canonicalize(trace)
        if not effects:
            continue
        if any(action.failed for action in effects):
            # 실패한 부수효과가 섞인 실행은 흉내 낼 대상이 아니다. 이걸 표본으로 세면
            # 고착화된 코드가 **실패를 충실히 재현한다** — 그것도 매번 LLM 없이,
            # 사람이 알아채기 어려운 채로. 실제로 그렇게 굳은 적이 있다.
            logger.info(
                "표본 제외 | activity=%s todo=%s 실패한 부수효과 포함",
                activity_id, item.get("id"),
            )
            continue
        samples.append(Sample(
            trace=trace,
            query=str(item.get("query") or ""),
            output=_form_output_of(events),
            identity={
                "id": item.get("id"),
                "proc_inst_id": item.get("proc_inst_id"),
                "root_proc_inst_id": item.get("root_proc_inst_id"),
                "proc_def_id": proc_def_id,
                "activity_id": activity_id,
                "tenant_id": tenant_id,
            },
            upstream=fetch_related_workitem_outputs(
                tenant_id,
                item.get("root_proc_inst_id"),
                item.get("proc_inst_id"),
                exclude_id=item.get("id"),
                before=item.get("start_date"),
            ),
        ))
        if len(samples) >= SAMPLE_SCAN_LIMIT:
            break
    return samples


def _from_step_call(binding) -> str:
    """이어받기 한 건을 골격의 `from_step` 호출로 적는다."""
    path = json.dumps(list(binding.path), ensure_ascii=False)
    line = "None" if binding.line is None else str(binding.line)
    return (
        f"from_step(results, {binding.source_index}, "
        f"{json.dumps(binding.root)}, {path}, {line})"
    )


def _argument_expressions(
    action: Action,
    call_index: int,
    plan: ParameterPlan,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """행위 하나의 인자를 파이썬 표현식 문자열로 만든다.

    자리는 셋 중 하나다. 파라미터 자리는 `${name}` 템플릿으로 굳고, 앞 단계에서
    이어받는 자리는 `from_step(...)` 으로 굳고, 나머지는 관측된 리터럴 그대로 굳는다.

    이어받는 값들은 `linked` 로 따로 돌려준다. 실행 시점에 계산해야 하므로 문자열
    템플릿에 미리 박을 수 없고, 그 단계 바로 앞에서 한 번 구해 쓴다.
    """
    segment_slots: Dict[str, List[Tuple[int, int, str, bool]]] = {}
    whole_slots: Dict[str, str] = {}
    for slot in plan.slots:
        if slot.call_index != call_index:
            continue
        if slot.segment_index is None or slot.span is None:
            whole_slots[slot.arg_key] = slot.name
        else:
            start, end = slot.span
            segment_slots.setdefault(slot.arg_key, []).append(
                (start, end, slot.name, slot.quoted)
            )

    # 값 전체를 다시 적은 인자. 자리 대조가 아니라 이미 아는 값들의 조합으로 굳는다.
    composites = {
        composite.arg_key: composite
        for composite in plan.composites
        if composite.call_index == call_index
    }

    linked: Dict[str, str] = {}
    whole_links: Dict[str, str] = {}
    for binding in plan.bindings:
        if binding.call_index != call_index:
            continue
        if binding.arg_key in composites:
            # 합성 템플릿이 이름으로 참조한다. 자리를 따로 치환하지 않는다.
            linked[binding.name] = _from_step_call(binding)
        elif binding.segment_index is None or binding.span is None:
            # 값 전체를 이어받는 자리는 render()를 거치지 않아 원래 타입을 보존한다.
            whole_links[binding.arg_key] = _from_step_call(binding)
        else:
            linked[binding.name] = _from_step_call(binding)
            start, end = binding.span
            segment_slots.setdefault(binding.arg_key, []).append(
                (start, end, binding.name, binding.quoted)
            )

    rendered: Dict[str, str] = {}
    for arg_key in sorted(action.args):
        value = action.args[arg_key]
        if arg_key in composites:
            arguments = "inputs, linked" if linked else "inputs"
            rendered[arg_key] = (
                f"render({json.dumps(composites[arg_key].template, ensure_ascii=False)}, "
                f"{arguments})"
            )
        elif arg_key in segment_slots:
            template = render_template(value, segment_slots[arg_key])
            arguments = "inputs, linked" if linked else "inputs"
            rendered[arg_key] = (
                f"render({json.dumps(template, ensure_ascii=False)}, {arguments})"
            )
        elif arg_key in whole_links:
            rendered[arg_key] = whole_links[arg_key]
        elif arg_key in whole_slots:
            # 값 전체가 파라미터인 경우 render()를 거치지 않아 원래 타입을 보존한다.
            rendered[arg_key] = f"inputs[{json.dumps(whole_slots[arg_key])}]"
        else:
            rendered[arg_key] = json.dumps(value, ensure_ascii=False)
    return rendered, linked


def _file_call(action: Action, expressions: Dict[str, str]) -> str:
    """파일 조작 한 건을 대응하는 골격 동작 호출로 바꾼다.

    관측된 조작(op)마다 대응하는 동작이 있어야 한다. 전부 `write_file` 로 뭉뚱그리면
    삭제가 "빈 파일 만들기"로 재현되어 조용히 틀린다.

    재현할 수 없는 조작은 코드를 만들지 않고 거부한다 — 틀린 코드를 남기는 것보다
    고착화하지 않고 에이전트에게 맡기는 편이 낫다.
    """
    op = str(action.args.get("op") or "write")

    def observed(key: str) -> str:
        """관측된 값이 실제로 있는지 본다.

        표현식 문자열로 판정하면 안 된다 — 빈 경로도 `'""'` 라는 truthy한 문자열로
        렌더되어, 대상이 없는데 있는 것처럼 통과한다.
        """
        return str(action.args.get(key) or "")

    if op in ("move", "copy"):
        if not (observed("source") and observed("destination")):
            raise ValueError(f"'{action.tool}'의 원본/대상 경로를 이력에서 찾지 못했습니다.")
        return f"{op}_file({expressions['source']}, {expressions['destination']})"

    if not observed("path"):
        raise ValueError(f"'{action.tool}'의 대상 경로를 이력에서 찾지 못했습니다.")
    path = expressions["path"]
    if op == "delete":
        return f"remove_file({path})"
    if op == "mkdir":
        return f"make_dir({path})"
    if op == "edit" and "old_string" in expressions:
        return f"edit_file({path}, {expressions['old_string']}, {expressions.get('new_string', '\"\"')})"
    return f"write_file({path}, {expressions.get('content', '\"\"')})"


def _step_line(
    action: Action,
    expressions: Dict[str, str],
    tool_to_server: Dict[str, str],
    linked: Optional[Dict[str, str]] = None,
) -> str:
    """행위 종류에 맞는 실행 한 줄. 종류마다 골격의 다른 원시 동작을 쓴다.

    앞 단계에서 이어받는 값이 있으면 그 단계 바로 앞에 한 줄을 더 둔다. 인자 안에
    펼쳐 넣으면 같은 값을 여러 번 꺼내게 되고, 코드를 읽을 때 무엇을 이어받았는지도
    보이지 않는다.
    """
    prefix = ""
    if linked:
        items = ", ".join(f"{json.dumps(name)}: {expr}" for name, expr in sorted(linked.items()))
        prefix = "    linked = {" + items + "}\n"

    if action.kind == SHELL:
        command = expressions.get("command", '""')
        cwd = expressions.get("cwd", '""')
        return prefix + f"    results.append(await run_shell({command}, {cwd}))"

    if action.kind == FILE_WRITE:
        return prefix + f"    results.append(await {_file_call(action, expressions)})"

    server_key = tool_to_server.get(action.tool)
    if not server_key:
        raise ValueError(f"도구 '{action.tool}'를 제공하는 MCP 서버를 찾지 못했습니다.")
    arg_expr = "{" + ", ".join(f'"{k}": {v}' for k, v in expressions.items()) + "}"
    return prefix + (
        f'    results.append(await call_tool("{server_key}", "{action.tool}", '
        f"{arg_expr}, timeout_s=timeout_s))"
    )


def _header(todo_id: str, provenance: Dict[str, Any]) -> str:
    """생성 코드 맨 위에 붙는 출처 주석.

    이 코드가 무엇을 보고 만들어졌는지 — 어떤 스킬 절차를 따랐고 어떤 셸 스크립트를
    돌렸는지 — 를 남긴다. 나중에 그 스킬이 바뀌었을 때 코드를 의심할 근거가 된다.
    """
    lines = [f"generated_{todo_id}.py (auto-created from observed work history)"]

    def _add(label: str, values: Any) -> None:
        for value in (values or [])[:8]:
            text = " ".join(str(value).split())
            lines.append(f"  {label}: {text[:160]}")

    lines.append(f"  표본 수: {provenance.get('sample_count')}")
    lines.append(f"  행위 구성: {json.dumps(provenance.get('by_kind') or {}, ensure_ascii=False)}")
    _add("사용 도구", provenance.get("tools_used"))
    _add("읽은 스킬", provenance.get("skills_read"))
    _add("읽은 파일", provenance.get("files_read"))
    _add("실행한 셸", provenance.get("shell_commands"))
    _add("생성한 파일", provenance.get("files_written"))
    _add("위임한 서브에이전트", provenance.get("subagents"))
    return "\n# ".join(lines)


def compile_code(
    todo_id: str,
    actions: List[Action],
    tool_to_server: Dict[str, str],
    plan: ParameterPlan,
    provenance: Optional[Dict[str, Any]] = None,
) -> str:
    """대표 표본의 행위 목록과 파라미터 배치로 실행 코드를 만든다."""
    lines = []
    for index, action in enumerate(actions):
        expressions, linked = _argument_expressions(action, index, plan)
        lines.append(_step_line(action, expressions, tool_to_server, linked))

    docs = [
        f'        - {p["name"]} ({p["type"]}): example={json.dumps(p.get("example"), ensure_ascii=False)}'
        for p in plan.parameters
    ]
    return TEMPLATE.format(
        header=_header(todo_id, provenance or {}),
        steps="\n".join(lines) if lines else "    pass",
        param_docs="\n".join(docs) if docs else "        None",
    )


@dataclass(frozen=True)
class Resolution:
    """값의 출처가 모두 정해진 표본 묶음 하나."""

    samples: Tuple[Sample, ...]
    plan: ParameterPlan
    effects: Tuple[List[Action], ...]


def _resolve(samples: List[Sample]) -> Tuple[Optional[Resolution], List[str]]:
    """이 표본 묶음으로 값의 출처가 모두 정해지는지 본다.

    정해지면 계획을, 아니면 끝내 되찾지 못한 파라미터 이름들을 돌려준다. 출처를 찾는
    순서는 좁은 쪽에서 넓은 쪽이다 — 지시문에서 되찾기, 앞 액티비티의 산출물, 그리고
    이미 아는 값들의 조합으로 다시 적기.
    """
    effects = [canonicalize(sample.trace) for sample in samples]
    queries = [sample.query for sample in samples]
    identities = [sample.identity for sample in samples]

    plan = identify_parameters(effects, queries, identities)

    # 값이 어디서 오는지 모르는 파라미터가 하나라도 있으면 굳히지 않는다. 실행기의
    # 위치·타입 폴백은 언제나 무언가를 채우는 데 성공하므로, 여기서 막지 않으면
    # 엉뚱한 값으로 실제 도구를 부르는 코드가 남는다.
    unrecoverable = unrecoverable_parameters(plan, queries)
    if unrecoverable:
        # 지시문에 없다고 끝이 아니다. 값은 액티비티 경계를 넘어서도 흐른다 — 앞
        # 워크아이템의 산출물에서 같은 자리를 표본 전부에서 찾을 수 있으면, 다음
        # 실행에서도 거기서 읽으면 된다.
        plan = bind_upstream(plan, unrecoverable, [s.upstream for s in samples])
        unrecoverable = unrecoverable_parameters(plan, queries)
    if unrecoverable:
        # 되찾을 수 없는 것이 값 전체가 아니라 **합성된 문자열의 일부**일 수 있다.
        # 워크스페이스 경로가 그렇다 — 조각으로 쪼개면 토막이 남지만, 그 토막을 이루는
        # 값들은 이미 알고 있다. 아는 값들의 조합으로 다시 적을 수 있으면 굳힌다.
        #
        # 그중에는 애초에 입력이 아닌 자리도 섞여 있다. 시각 서식처럼 에이전트가 스스로
        # 정하는 자리는 되찾을 대상이 아니라 절차의 일부다 — 과반 값으로 굳힌다.
        pins = procedure_pins(plan, unrecoverable, queries)
        plan = compose_from_known(
            plan, effects, identities, unrecoverable, queries, pins
        )
        unrecoverable = unrecoverable_parameters(plan, queries)
    if unrecoverable:
        return None, unrecoverable
    return Resolution(tuple(samples), plan, tuple(effects)), []


def _search(group: List[Sample]) -> Tuple[Optional[Resolution], List[str]]:
    """한 무리 안에서 굳힐 수 있는 표본 조합을 찾는다.

    최신 표본이 든 조합부터 본다 — 최근의 방식일수록 다음 실행과 닮았을 가능성이 크다.
    """
    last_blocked: List[str] = []
    for tried, picked in enumerate(combinations(range(len(group)), REQUIRED_SAMPLES)):
        if tried >= MAX_SAMPLE_COMBINATIONS:
            break
        resolution, blocked = _resolve([group[index] for index in picked])
        if resolution is not None:
            return resolution, []
        last_blocked = blocked or last_blocked
    return None, last_blocked


def try_freeze(workitem) -> Optional[Dict[str, Any]]:
    """표본이 충분하고 지문이 일치하면 순방향 코드를 생성해 저장한다.

    워크아이템 제출 훅에서 호출한다. 이미 활성 코드가 있으면 아무것도 하지 않는다.
    실패는 전파하지 않는다 — 고착화는 최적화이지 업무 처리의 전제가 아니다.
    """
    proc_def_id = workitem.proc_def_id
    activity_id = workitem.activity_id
    tenant_id = workitem.tenant_id
    if not (proc_def_id and activity_id and tenant_id):
        return None

    existing = fetch_mcp_python_code(proc_def_id, activity_id, tenant_id)
    if existing and existing.get("code"):
        return None

    since = fetch_last_deactivated_at(proc_def_id, activity_id, tenant_id)
    collected = collect_samples(proc_def_id, activity_id, tenant_id, since)
    if len(collected) < REQUIRED_SAMPLES:
        return None

    # 같은 구조로 일한 실행끼리 모은다. 지문은 부수효과 행위로만 잰다 — 맥락 행위(스킬·
    # 파일 읽기, 조회)는 같은 일을 하면서도 실행마다 횟수가 달라지기 마련이라, 그것까지
    # 일치를 요구하면 어떤 활동도 고착화되지 않는다. 대신 맥락은 출처로 기록해 남긴다.
    groups: Dict[str, List[Sample]] = {}
    for sample in collected:
        groups.setdefault(
            execution_fingerprint(canonicalize(sample.trace)), []
        ).append(sample)

    # 큰 무리부터 본다. 같은 크기면 최신 실행이 속한 무리가 앞선다.
    ordered = sorted(groups.values(), key=lambda group: -len(group))
    if len(ordered[0]) < REQUIRED_SAMPLES:
        logger.info(
            "고착화 보류 | activity=%s 같은 지문의 표본이 %d건뿐 (훑은 표본 %d건, 지문 %d종)",
            activity_id, len(ordered[0]), len(collected), len(groups),
        )
        return None

    resolution, blocked = None, []
    for group in ordered:
        if len(group) < REQUIRED_SAMPLES:
            break
        resolution, failed = _search(group)
        if resolution is not None:
            break
        # 사유는 가장 유력한 무리(가장 크고 최신인 쪽)의 것을 남긴다. 마지막에 시도한
        # 무리의 사유를 남기면 엉뚱한 자리를 가리켜 진단을 헷갈리게 한다.
        blocked = blocked or failed
    if resolution is None:
        logger.info(
            "고착화 보류 | activity=%s 어느 표본 조합으로도 값의 출처를 정하지 못함 — "
            "되찾을 수 없는 파라미터: %s",
            activity_id, ", ".join(blocked) or "(없음)",
        )
        return None

    collected = list(resolution.samples)
    plan = resolution.plan
    effects = list(resolution.effects)
    samples = [sample.trace for sample in collected]
    queries = [sample.query for sample in collected]
    outputs = [sample.output for sample in collected]

    tool_to_server = build_tool_index_from_tenant(tenant_id)

    provenance = summarize(samples[0])
    provenance["sample_count"] = len(samples)
    provenance["effect_count"] = len(effects[0])
    provenance["fingerprint"] = execution_fingerprint(effects[0])
    # 어떤 자리가 앞 단계의 결과를 이어받는지 남긴다. 나중에 그 단계가 다른 것을 내기
    # 시작했을 때 이 코드를 의심할 근거가 된다.
    provenance["step_bindings"] = [binding.as_record() for binding in plan.bindings]
    provenance["upstream_bindings"] = [
        {"name": spec["name"], **spec["upstream"]}
        for spec in plan.parameters if spec.get("upstream")
    ]
    provenance["composed_args"] = [
        {"into": f"{composite.call_index}:{composite.arg_key}", "from": list(composite.names)}
        for composite in plan.composites
    ]

    # 재현할 수 없는 행위가 하나라도 있으면 고착화하지 않는다. 이력에 대상 경로가
    # 없는 파일 쓰기, 서버를 못 찾은 MCP 호출 등이 그렇다. 그 단계만 빼고 나머지를
    # 굳히면 세상을 절반만 바꾸는 코드가 남으므로, 활동 전체를 에이전트에게 맡긴다.
    try:
        code = compile_code(str(workitem.id), effects[0], tool_to_server, plan, provenance)
    except ValueError as exc:
        logger.info("고착화 보류 | activity=%s %s", activity_id, exc)
        return None

    # 폼 산출물까지 굳힌다. 도구 호출만 재현하면 워크아이템이 산출물 없이 완료되어
    # 다음 활동이 입력을 못 받는다 — 에이전트가 하던 일의 절반만 하는 셈이다.
    output_template = build_output_template(outputs, plan.observations)
    if output_template is None:
        logger.info(
            "산출물 템플릿 없음 | activity=%s 표본의 폼 산출물 구성이 서로 다름 "
            "— 실행 요약으로 폼을 채운다", activity_id,
        )

    record = {
        "proc_def_id": proc_def_id,
        "activity_id": activity_id,
        "tenant_id": tenant_id,
        "code": code,
        "parameters": plan.as_specification(),
        "work_history": provenance,
        "output_template": output_template,
    }
    upsert_mcp_python_code(record)
    unresolved = [
        key for key, spec in ((output_template or {}).get("fields") or {}).items()
        if spec is None
    ]
    logger.info(
        "고착화 완료 | activity=%s 표본=%d 행위=%s 파라미터=%s 이어받기=%s 산출물=%s",
        activity_id,
        len(samples),
        json.dumps(provenance.get("by_kind") or {}, ensure_ascii=False),
        [p["name"] for p in plan.parameters],
        json.dumps(
            provenance["step_bindings"]
            + provenance["upstream_bindings"]
            + provenance["composed_args"],
            ensure_ascii=False, default=str,
        ),
        "없음" if output_template is None
        else f"{(output_template.get('form_id') or '')}"
             f"{' (미해결: ' + ', '.join(unresolved) + ')' if unresolved else ''}",
    )
    return record


__all__ = [
    "REQUIRED_SAMPLES",
    "REWORK_DISTRUST_THRESHOLD",
    "Sample",
    "collect_samples",
    "compile_code",
    "freeze_on_done",
    "try_freeze",
]


def freeze_on_done(saved: Optional[Dict[str, Any]]) -> None:
    """워크아이템이 DONE으로 확정된 직후 고착화를 시도한다.

    저장소 계층의 워크아이템 저장 훅에서 부른다. 여기서 어떤 예외도 밖으로 내보내지
    않는다 — 고착화는 최적화이지 업무 처리의 전제가 아니고, 이 호출이 실패한다고
    워크아이템 저장이 되돌아가서는 안 된다.
    """
    if not saved:
        return
    if str(saved.get("status") or "").upper() != "DONE":
        return
    try:
        try_freeze(_SavedWorkitem(saved))
    except Exception:
        logger.warning(
            "고착화 건너뜀 | activity=%s", saved.get("activity_id"), exc_info=True
        )


class _SavedWorkitem:
    """저장된 워크아이템 행을 `try_freeze`가 기대하는 속성 접근으로 감싼다.

    `try_freeze`는 completion의 pydantic 워크아이템 모델을 받도록 쓰였다. 저장소
    훅에서는 dict 행밖에 없으므로 모델을 끌어오는 대신 필요한 필드만 노출한다.
    """

    __slots__ = ("id", "proc_def_id", "activity_id", "tenant_id")

    def __init__(self, row: Dict[str, Any]) -> None:
        self.id = row.get("id")
        self.proc_def_id = row.get("proc_def_id")
        self.activity_id = row.get("activity_id")
        self.tenant_id = row.get("tenant_id")
