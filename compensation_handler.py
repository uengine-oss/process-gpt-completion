"""보상(역방향) 코드 생성.

순방향 고착화(`deterministic_generator`)와 같은 작업 이력을 읽는다. 다른 점은 그
이력을 재현하는 대신 거꾸로 되돌린다는 것뿐이다. 그래서 이력 해석은 같은
`work_history` 정규화를 쓰고, 실행 골격도 `deterministic_template`을 공유한다.

되돌릴 대상은 MCP 호출만이 아니다. 스킬의 셸 스크립트가 만든 파일, 스크립트가 건드린
외부 상태도 함께 남는다.

**LLM을 쓰지 않는다.** 예전에는 이력을 보여 주고 되돌리는 코드를 지어내게 했는데, 그
지어내기가 곧 추측이었다 — 되돌리기가 추측이면 재작업 경로 전체가 추측 위에 선다.
지금은 `compensation.invert`가 관측에서 되돌리기 단계를 만들고, 저장되는 코드는 그
단계를 수행하는 한 벌이다(`deterministic_template.undo_script`).

되돌릴 수 없는 행위가 하나라도 섞이면 **아무것도 저장하지 않는다**. 절반만 되돌린
세상은 되돌리지 않은 세상보다 나쁘다 — 무엇이 남았는지 아무도 모르기 때문이다.
저장된 보상이 없으면 실행 런타임은 재작업 전체를 에이전트에게 넘긴다.
"""

import logging

from database import (
    fetch_mcp_python_code,
    upsert_mcp_python_code,
    fetch_events_by_todo_id,
    upsert_workitem,
    fetch_user_info_by_uid,
)
from mcp_tool_index import build_tool_index_from_tenant
from deterministic_template import undo_script
from compensation import invert
from work_history import normalize_events

logger = logging.getLogger(__name__)


async def generate_compensation(workitem, new_workitem):
    try:
        if workitem is None:
            raise Exception("Workitem is None")
        
        deterministic_code = fetch_mcp_python_code(workitem.proc_def_id, workitem.activity_id, workitem.tenant_id)
        if deterministic_code and deterministic_code.get("compensation") is not None:
            return
        
        # **이 워크아이템의** 이력만 본다. 인스턴스 전체를 가져오면 재작업하지도 않을
        # 앞 액티비티의 부수효과가 되돌리기 대상에 섞인다 — 그것들이 되돌릴 수 없으면
        # 멀쩡한 활동의 보상이 막히고, 되돌릴 수 있으면 남의 INSERT까지 지운다.
        # 실행 런타임도 같은 액티비티의 이전 회차만 보고 되돌린다(`_previous_undo_plan`).
        events = fetch_events_by_todo_id(workitem.id)
        
        if len(events) == 0:
            return
        
        # 이벤트 봉투 해석은 런타임에 매이지 않는다. crewai-action은 crew_type "action"
        # 으로, DeepAgents는 "deepagents"로 발행하고 데이터 모양도 다르다. 특정 런타임의
        # 스키마로 걸러내면 살아있는 다른 런타임의 이력이 전부 버려진다.
        trace = normalize_events(events)
        if not any(action.has_effect for action in trace):
            # 되돌릴 부수효과가 없다. 조회와 읽기만 한 활동은 보상할 것이 없다.
            return

        # 되돌릴 수 있는지는 관측이 정한다. 하나라도 되돌릴 수 없으면 저장하지 않는다 —
        # 되돌리는 일을 실제로 하지 않는 코드를 남기면, 재작업 때 되돌리기가 조용히
        # 아무것도 하지 않고 순방향 코드가 그 위에 덧씌워진다.
        steps, reasons = invert(trace)
        if reasons or not steps:
            logger.info(
                "보상 코드 없음 | activity=%s 되돌릴 수 없는 행위: %s",
                workitem.activity_id, "; ".join(reasons) or "되돌릴 부수효과 없음",
            )
            return

        compensation_code = undo_script(build_tool_index_from_tenant(workitem.tenant_id))
        if deterministic_code:
            deterministic_code["compensation"] = compensation_code
            upsert_mcp_python_code(deterministic_code)
        else:
            upsert_mcp_python_code({
                "compensation": compensation_code,
                "proc_def_id": workitem.proc_def_id,
                "activity_id": workitem.activity_id,
                "tenant_id": workitem.tenant_id
            })
        logger.info(
            "보상 코드 저장 | activity=%s 되돌리기 %d단계", workitem.activity_id, len(steps),
        )
        
        user_id = workitem.user_id
        user_name = workitem.username
        if workitem.assignees and len(workitem.assignees) > 0:
            assignee_id = workitem.assignees[0].get('endpoint')
            # endpoint는 문자열 하나이거나 목록일 수 있다. 목록만 처리하면 단일
            # 담당자 워크아이템의 user_id/username이 None으로 덮인다.
            endpoints = assignee_id if isinstance(assignee_id, list) else (
                [assignee_id] if assignee_id else []
            )
            user_list = []
            for endpoint in endpoints:
                user_info = fetch_user_info_by_uid(endpoint)
                if user_info:
                    user_list.append(user_info)
            if user_list:
                user_id = ','.join([user.get('id') for user in user_list])
                user_name = ','.join([user.get('username') for user in user_list])

        upsert_workitem({
            "id": new_workitem.get('id'),
            "status": "IN_PROGRESS",
            "user_id": user_id,
            "username": user_name,
            # 실행 런타임을 바꾸지 않는다. 퇴역한 crewai-action으로 찍으면 어떤
            # 폴링 워커도 이 워크아이템을 가져가지 않아 영구히 멈춘다.
            "agent_orch": workitem.agent_orch,
            "log": "Compensation Handling..."
        })

    except Exception as e:
        print(f"[ERROR] Failed to handle compensation: {str(e)}")
        raise Exception(f"Compensation handling failed: {str(e)}") from e


