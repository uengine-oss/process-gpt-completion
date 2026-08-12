"""프로세스 시작 지점 조회 API.

[왜 필요한가]
"이 프로세스를 실행하면 어떤 액티비티부터 시작하는가"는 실행 엔진의 판단이어야 한다.
그런데 그동안 프론트엔드가 각 화면에서 제각각 계산했고, 규칙도 서로 달랐다.
  - 어떤 화면은 sequences 에서 source == 'start_event' 인 것을 찾음 (id 문자열 하드코딩)
  - 어떤 화면은 events 에서 type == 'startEvent' 인 것만 봄 (직후에 게이트웨이가 오면 실패)
그 결과 시작 이벤트 id 가 'start_event1' 인 정의에서 조회가 실패해 activities[0] 로 폴백했고,
이 배열 순서는 실행 순서와 무관해서 "실행하면 마지막 태스크가 열리는" 문제가 발생했다.

이 API 는 엔진의 판별 로직(ProcessDefinition.find_start_event_id / find_initial_activity)을
그대로 노출해, 프론트와 엔진이 같은 답을 쓰도록 한다.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import BaseModel

from database import fetch_process_definition_by_version
from process_definition import load_process_definition


class StartActivityRequest(BaseModel):
    # definition 을 직접 넘기면 그것을 쓰고, 없으면 process_definition_id 로 조회한다.
    process_definition_id: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None
    tenant_id: Optional[str] = None
    version_tag: Optional[str] = None
    version: Optional[Any] = None
    arcv_id: Optional[str] = None


def _resolve_definition(payload: StartActivityRequest) -> Dict[str, Any]:
    if isinstance(payload.definition, dict) and payload.definition:
        return payload.definition

    if not payload.process_definition_id:
        raise HTTPException(status_code=400, detail="process_definition_id 또는 definition 이 필요합니다.")

    definition = fetch_process_definition_by_version(
        payload.process_definition_id,
        payload.version_tag,
        payload.version,
        payload.tenant_id,
        payload.arcv_id,
    )
    if not definition:
        raise HTTPException(
            status_code=404,
            detail=f"프로세스 정의를 찾을 수 없습니다: {payload.process_definition_id}",
        )
    return definition


async def get_start_activity(payload: StartActivityRequest) -> Dict[str, Any]:
    """프로세스의 시작 이벤트와 첫 번째 실행 대상 액티비티를 반환한다."""
    definition_json = _resolve_definition(payload)

    try:
        process_definition = load_process_definition(definition_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"프로세스 정의를 해석할 수 없습니다: {e}")

    start_event_id = process_definition.find_start_event_id()
    activity = process_definition.find_initial_activity()

    return {
        "startEventId": start_event_id,
        "activityId": getattr(activity, "id", None) if activity else None,
        # 화면이 곧바로 쓸 수 있도록 액티비티 원본도 함께 준다.
        "activity": activity.model_dump() if activity is not None and hasattr(activity, "model_dump") else None,
    }


def add_routes_to_app(app):
    app.add_api_route("/process-definition/start-activity", get_start_activity, methods=["POST"])
