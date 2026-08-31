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

from typing import Any, Dict, List, Optional, Tuple

import json
import logging

from database import (
    fetch_mcp_python_code,
    upsert_mcp_python_code,
    fetch_workitems_by_activity,
    fetch_events_by_todo_id,
    fetch_last_deactivated_at,
)
from deterministic_template import TEMPLATE
from mcp_tool_index import build_tool_index_from_tenant
from deterministic_signature import (
    ParameterPlan,
    execution_fingerprint,
    identify_parameters,
    render_template,
)
from work_history import (
    FILE_WRITE,
    SHELL,
    Action,
    effect_actions,
    normalize_events,
    summarize,
)

logger = logging.getLogger(__name__)

# 고착화에 필요한 표본 수. 표본이 여러 건이어야 무엇이 파라미터이고 무엇이 상수인지
# 추측이 아니라 관측으로 가려낼 수 있다.
REQUIRED_SAMPLES = 3

# 고착화된 코드를 쓰고도 이 횟수 이상 재작업되면 코드 자체를 의심해 비활성화한다.
# 1회 재작업은 대개 입력이 틀린 경우이므로 되돌린 뒤 새 파라미터로 재실행한다.
REWORK_DISTRUST_THRESHOLD = 2


def _trace_of(todo_id: str) -> List[Action]:
    """워크아이템 하나의 작업 이력을 정규화된 행위 목록으로 읽는다."""
    return normalize_events(fetch_events_by_todo_id(todo_id))


def collect_samples(
    proc_def_id: str, activity_id: str, tenant_id: str, since: Optional[str]
) -> List[Tuple[List[Action], str]]:
    """고착화 표본을 모은다.

    자격은 "완료(DONE)되었고 이후 재작업되지 않은 워크아이템"이다. DONE은 사람의
    승인이 아니라 완료 상태일 뿐이므로, 양성 근거는 DONE 자체가 아니라 되돌려지지
    않았다는 사실에 둔다.

    ``since``(직전 비활성 시각) 이후에 완료된 것만 센다. 이전 표본을 다시 세면
    비활성화한 코드와 동일한 코드를 곧바로 재생성해 무한 반복에 빠진다.

    각 표본은 **작업 이력 전체**(맥락 행위 포함)와 그 워크아이템의 지시문 한 쌍이다.
    부수효과만 남기는 일은 지문 계산과 코드 컴파일 직전에 한다 — 어떤 스킬을 읽고 그
    절차를 따랐는지는 생성된 코드의 출처로 함께 기록해야 하기 때문이다.

    지시문을 함께 모으는 이유는 파라미터 이름표 때문이다. 값이 관측될 때 지시문에서
    그 앞에 무엇이 적혀 있었는지를 알아야, 다음 실행의 지시문에서 같은 값을 되찾을 수
    있다.
    """
    candidates = fetch_workitems_by_activity(
        proc_def_id, activity_id, tenant_id, status="DONE", since=since, limit=REQUIRED_SAMPLES * 4
    ) or []

    # 같은 프로세스 인스턴스에서 재작업이 이어진 실행은 되돌려진 것으로 보고 제외한다.
    highest_rework: Dict[str, int] = {}
    for item in candidates:
        key = str(item.get("proc_inst_id") or item.get("id"))
        highest_rework[key] = max(highest_rework.get(key, 0), int(item.get("rework_count") or 0))

    samples: List[Tuple[List[Action], str]] = []
    for item in candidates:
        key = str(item.get("proc_inst_id") or item.get("id"))
        if int(item.get("rework_count") or 0) < highest_rework.get(key, 0):
            continue
        trace = _trace_of(item.get("id"))
        effects = effect_actions(trace)
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
        samples.append((trace, str(item.get("query") or "")))
        if len(samples) >= REQUIRED_SAMPLES:
            break
    return samples


def _argument_expressions(
    action: Action,
    call_index: int,
    plan: ParameterPlan,
) -> Dict[str, str]:
    """행위 하나의 인자를 파이썬 표현식 문자열로 만든다.

    파라미터 자리는 `${name}` 템플릿으로, 나머지는 관측된 리터럴 그대로 굳는다.
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

    rendered: Dict[str, str] = {}
    for arg_key in sorted(action.args):
        value = action.args[arg_key]
        if arg_key in segment_slots:
            template = render_template(value, segment_slots[arg_key])
            rendered[arg_key] = f"render({json.dumps(template, ensure_ascii=False)}, inputs)"
        elif arg_key in whole_slots:
            # 값 전체가 파라미터인 경우 render()를 거치지 않아 원래 타입을 보존한다.
            rendered[arg_key] = f"inputs[{json.dumps(whole_slots[arg_key])}]"
        else:
            rendered[arg_key] = json.dumps(value, ensure_ascii=False)
    return rendered


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


def _step_line(action: Action, expressions: Dict[str, str], tool_to_server: Dict[str, str]) -> str:
    """행위 종류에 맞는 실행 한 줄. 종류마다 골격의 다른 원시 동작을 쓴다."""
    if action.kind == SHELL:
        command = expressions.get("command", '""')
        cwd = expressions.get("cwd", '""')
        return f"    results.append(await run_shell({command}, {cwd}))"

    if action.kind == FILE_WRITE:
        return f"    results.append(await {_file_call(action, expressions)})"

    server_key = tool_to_server.get(action.tool)
    if not server_key:
        raise ValueError(f"도구 '{action.tool}'를 제공하는 MCP 서버를 찾지 못했습니다.")
    arg_expr = "{" + ", ".join(f'"{k}": {v}' for k, v in expressions.items()) + "}"
    return (
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
    lines = [
        _step_line(action, _argument_expressions(action, index, plan), tool_to_server)
        for index, action in enumerate(actions)
    ]

    docs = [
        f'        - {p["name"]} ({p["type"]}): example={json.dumps(p.get("example"), ensure_ascii=False)}'
        for p in plan.parameters
    ]
    return TEMPLATE.format(
        header=_header(todo_id, provenance or {}),
        steps="\n".join(lines) if lines else "    pass",
        param_docs="\n".join(docs) if docs else "        None",
    )


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
    samples = [trace for trace, _query in collected]
    queries = [query for _trace, query in collected]

    # 지문은 부수효과 행위로만 잰다. 맥락 행위(스킬·파일 읽기, 조회)는 같은 일을
    # 하면서도 실행마다 횟수가 달라지기 마련이라, 그것까지 일치를 요구하면 어떤
    # 활동도 고착화되지 않는다. 대신 맥락은 출처로 기록해 남긴다.
    effects = [effect_actions(sample) for sample in samples]
    if len({execution_fingerprint(sample) for sample in effects}) != 1:
        logger.info(
            "고착화 보류 | activity=%s 표본 %d건의 실행 지문이 일치하지 않음",
            activity_id, len(samples),
        )
        return None

    plan = identify_parameters(effects, queries)
    tool_to_server = build_tool_index_from_tenant(tenant_id)

    provenance = summarize(samples[0])
    provenance["sample_count"] = len(samples)
    provenance["effect_count"] = len(effects[0])
    provenance["fingerprint"] = execution_fingerprint(effects[0])

    # 재현할 수 없는 행위가 하나라도 있으면 고착화하지 않는다. 이력에 대상 경로가
    # 없는 파일 쓰기, 서버를 못 찾은 MCP 호출 등이 그렇다. 그 단계만 빼고 나머지를
    # 굳히면 세상을 절반만 바꾸는 코드가 남으므로, 활동 전체를 에이전트에게 맡긴다.
    try:
        code = compile_code(str(workitem.id), effects[0], tool_to_server, plan, provenance)
    except ValueError as exc:
        logger.info("고착화 보류 | activity=%s %s", activity_id, exc)
        return None

    record = {
        "proc_def_id": proc_def_id,
        "activity_id": activity_id,
        "tenant_id": tenant_id,
        "code": code,
        "parameters": plan.as_specification(),
        "work_history": provenance,
    }
    upsert_mcp_python_code(record)
    logger.info(
        "고착화 완료 | activity=%s 표본=%d 행위=%s 파라미터=%s",
        activity_id,
        len(samples),
        json.dumps(provenance.get("by_kind") or {}, ensure_ascii=False),
        [p["name"] for p in plan.parameters],
    )
    return record


__all__ = [
    "REQUIRED_SAMPLES",
    "REWORK_DISTRUST_THRESHOLD",
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
