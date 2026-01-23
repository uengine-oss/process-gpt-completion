"""
콜봇 서비스 API
Twilio 전화 서비스와 연동하여 발신자 정보를 제공하는 API
"""

import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, Query, HTTPException, Body
from database import (
    fetch_user_info_by_uid, 
    fetch_todolist_by_user_id, 
    fetch_ui_definition_by_activity_id,
    supabase_client_var,
    subdomain_var
)

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api/callbot", tags=["callbot"])


@router.get("/caller-info")
async def get_caller_info(
    from_number: Optional[str] = Query(None, description="발신 전화번호 또는 client identity"),
    user_id: Optional[str] = Query(None, description="하드코딩된 user_id (테스트용)")
):
    """
    콜봇 서비스용: 발신자 정보 조회
    
    Parameters:
    - from_number: Twilio의 From 파라미터 (전화번호 또는 client:Anonymous)
    - user_id: 테스트 환경에서 하드코딩된 user_id
    
    Returns:
    - success: 성공 여부
    - username: 사용자 이름
    - user_id: 사용자 ID
    - email: 이메일
    - tenant_id: 테넌트 ID
    - greeting: 인사말 텍스트
    """
    try:
        logger.info(f"[Callbot] Caller info request - from: {from_number}, user_id: {user_id}")
        
        # from_number가 client:Anonymous인 경우 하드코딩된 ID 사용
        if from_number == "client:Anonymous" or not from_number:
            if not user_id:
                user_id = "b425d322-63d5-4287-b5ea-a6d72fb003f5"
        
        # user_id로 조회
        if user_id:
            try:
                user_info = fetch_user_info_by_uid(user_id)
                username = user_info.get("username", "고객")
                
                logger.info(f"[Callbot] User found: {username}")
                
                return {
                    "success": True,
                    "username": username,
                    "user_id": user_info.get("id"),
                    "email": user_info.get("email"),
                    "tenant_id": user_info.get("tenant_id"),
                    "greeting": f"{username}님 안녕하세요"
                }
            except HTTPException:
                logger.warning(f"[Callbot] User not found with user_id: {user_id}")
            except Exception as e:
                logger.error(f"[Callbot] Error fetching user: {str(e)}")
        
        # 기본값 반환
        logger.info("[Callbot] Returning default guest greeting")
        return {
            "success": True,
            "username": "고객",
            "greeting": "고객님 안녕하세요"
        }
        
    except Exception as e:
        logger.error(f"[Callbot] Exception in get_caller_info: {str(e)}", exc_info=True)
        # 에러가 발생해도 기본값을 반환하여 통화가 중단되지 않도록 함
        return {
            "success": False,
            "username": "고객",
            "greeting": "안녕하세요",
            "error": str(e)
        }


@router.get("/user-todolist")
async def get_user_todolist(
    user_id: str = Query(..., description="조회할 사용자 ID"),
):
    """
    콜봇 연동용: 특정 사용자의 할 일(TODO 리스트) 반환

    Parameters:
    - user_id: 사용자 ID

    Returns:
    - success: 성공 여부
    - count: 항목 수
    - items: 요약된 todolist 항목 배열
    """
    try:
        logger.info(f"[Callbot] Todolist request - user_id: {user_id}")

        todos = fetch_todolist_by_user_id(user_id) or []

        def serialize_workitem(item):
            return {
                "id": item.id,
                "activity_name": item.activity_name,
                "status": item.status,
                "description": item.description,
                "proc_inst_id": item.proc_inst_id,
                "proc_def_id": item.proc_def_id,
                "due_date": item.due_date.isoformat() if getattr(item, "due_date", None) else None,
                "project_id": item.project_id,
            }

        items = [serialize_workitem(todo) for todo in todos]

        return {
            "success": True,
            "count": len(items),
            "items": items,
            "user_id": user_id,
        }
    except HTTPException as exc:
        logger.warning(f"[Callbot] Todolist not found for user_id: {user_id} - {exc.detail}")
        return {"success": False, "items": [], "count": 0, "user_id": user_id, "error": exc.detail}
    except Exception as e:
        logger.error(f"[Callbot] Exception in get_user_todolist: {str(e)}", exc_info=True)
        return {"success": False, "items": [], "count": 0, "user_id": user_id, "error": str(e)}


@router.get("/tasks")
async def get_user_tasks(
    user_id: str = Query(..., description="조회할 사용자 ID"),
    status_filter: str = Query("active", description="상태 필터: all, active, todo, in_progress"),
    include_overdue: bool = Query(False, description="기한 지난 업무 포함 여부")
):
    """
    콜봇 AI Function Calling용: 사용자의 진행 중인 업무 목록 조회
    
    Parameters:
    - user_id: 사용자 UUID
    - status_filter: 업무 상태 필터
      - "active": TODO + IN_PROGRESS (기본값)
      - "todo": TODO만
      - "in_progress": IN_PROGRESS만
      - "all": 모든 상태
    - include_overdue: 기한 지난 업무 포함 여부 (기본값: False)
    
    Returns:
    - success: 성공 여부
    - tasks: 업무 목록 배열
    - count: 업무 개수
    - overdue_count: 제외된 기한 지난 업무 개수
    - user_id: 조회한 사용자 ID
    """
    try:
        from datetime import datetime, timezone
        
        logger.info(f"[Callbot Tasks] Request - user_id: {user_id}, filter: {status_filter}, include_overdue: {include_overdue}")
        
        # 전체 todolist 조회
        todos = fetch_todolist_by_user_id(user_id) or []
        
        # 상태 필터링
        if status_filter == "active":
            filtered_todos = [t for t in todos if t.status in ("TODO", "IN_PROGRESS", "NEW")]
        elif status_filter == "todo":
            filtered_todos = [t for t in todos if t.status == "TODO"]
        elif status_filter == "in_progress":
            filtered_todos = [t for t in todos if t.status == "IN_PROGRESS"]
        else:  # all
            filtered_todos = todos
        
        # 기한 필터링 (기본적으로 기한 지난 업무 제외)
        now = datetime.now(timezone.utc)
        overdue_count = 0
        
        if not include_overdue:
            valid_todos = []
            for todo in filtered_todos:
                due_date = getattr(todo, "due_date", None)
                if due_date is None:
                    # due_date가 없으면 포함
                    valid_todos.append(todo)
                else:
                    # due_date가 있으면 현재 시간과 비교
                    # due_date가 naive datetime이면 UTC로 가정
                    if due_date.tzinfo is None:
                        due_date = due_date.replace(tzinfo=timezone.utc)
                    
                    if due_date >= now:
                        valid_todos.append(todo)
                    else:
                        overdue_count += 1
                        logger.debug(f"[Callbot Tasks] Filtered out overdue task: {todo.activity_name} (due: {due_date})")
            
            filtered_todos = valid_todos
            logger.info(f"[Callbot Tasks] Excluded {overdue_count} overdue tasks")
        
        def serialize_task(item):
            """업무 항목을 AI가 이해하기 쉬운 형태로 직렬화"""
            due_date = getattr(item, "due_date", None)
            return {
                "id": str(item.id) if item.id else None,
                "activity_name": item.activity_name or "업무명 없음",
                "status": item.status,
                "description": item.description or "",
                "proc_inst_id": item.proc_inst_id,
                "proc_def_id": item.proc_def_id,
                "start_date": item.start_date.isoformat() if getattr(item, "start_date", None) else None,
                "due_date": due_date.isoformat() if due_date else None,
                "project_id": str(item.project_id) if getattr(item, "project_id", None) else None,
            }
        
        tasks = [serialize_task(todo) for todo in filtered_todos]
        
        # 디버깅 로그
        logger.info(f"[Callbot Tasks] Found {len(tasks)} tasks for user {user_id} (excluded {overdue_count} overdue)")
        logger.info(f"[Callbot Tasks] Tasks data: {tasks}")
        
        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks),
            "overdue_count": overdue_count,
            "user_id": user_id,
            "filter": status_filter
        }
        
    except HTTPException as exc:
        logger.warning(f"[Callbot Tasks] Error for user_id: {user_id} - {exc.detail}")
        return {
            "success": False,
            "tasks": [],
            "count": 0,
            "overdue_count": 0,
            "user_id": user_id,
            "error": exc.detail
        }
    except Exception as e:
        logger.error(f"[Callbot Tasks] Exception: {str(e)}", exc_info=True)
        return {
            "success": False,
            "tasks": [],
            "count": 0,
            "overdue_count": 0,
            "user_id": user_id,
            "error": str(e)
        }


@router.get("/task/{task_id}")
async def get_task_detail(
    task_id: str,
):
    """
    콜봇 AI Function Calling용: 특정 업무의 상세 정보 및 폼 스키마 조회
    
    Parameters:
    - task_id: 업무(todolist) UUID
    
    Returns:
    - success: 성공 여부
    - task: 업무 정보
    - form_schema: 폼 스키마 (fields_json)
    - current_data: 현재 입력된 폼 데이터 (output)
    - reference_forms: 참조 폼 데이터 (이전 activity들의 output)
    """
    try:
        logger.info(f"[Callbot Task Detail] Request - task_id: {task_id}")
        
        supabase = supabase_client_var.get()
        if supabase is None:
            raise HTTPException(status_code=500, detail="Supabase client not configured")
        
        subdomain = subdomain_var.get()
        
        # todolist에서 업무 조회
        response = supabase.table('todolist').select('*').eq('id', task_id).eq('tenant_id', subdomain).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
        task_data = response.data[0]
        
        # 폼 스키마 조회
        form_schema = None
        if task_data.get('proc_def_id') and task_data.get('activity_id'):
            try:
                form_def = fetch_ui_definition_by_activity_id(
                    task_data['proc_def_id'],
                    task_data['activity_id'],
                    subdomain
                )
                if form_def and form_def.fields_json:
                    form_schema = form_def.fields_json
            except Exception as e:
                logger.warning(f"[Callbot Task Detail] Form schema not found: {e}")
        
        # 현재 폼 데이터
        current_data = task_data.get('output', {}) or {}
        
        # 참조 폼 데이터 조회 (같은 프로세스 인스턴스의 이전 activity들)
        reference_forms = []
        proc_inst_id = task_data.get('proc_inst_id')
        if proc_inst_id:
            try:
                # 같은 proc_inst_id의 완료된(DONE) 업무들 조회
                ref_response = supabase.table('todolist').select('*').eq(
                    'proc_inst_id', proc_inst_id
                ).eq('tenant_id', subdomain).eq('status', 'DONE').execute()
                
                if ref_response.data:
                    for ref_task in ref_response.data:
                        if ref_task.get('output'):
                            reference_forms.append({
                                "activity_name": ref_task.get('activity_name'),
                                "activity_id": ref_task.get('activity_id'),
                                "data": ref_task.get('output', {})
                            })
                    
                    logger.info(f"[Callbot Task Detail] Found {len(reference_forms)} reference forms")
            except Exception as e:
                logger.warning(f"[Callbot Task Detail] Failed to fetch reference forms: {e}")
        
        logger.info(f"[Callbot Task Detail] Task: {task_data.get('activity_name')}")
        logger.info(f"[Callbot Task Detail] Form schema fields: {len(form_schema) if form_schema else 0}")
        logger.info(f"[Callbot Task Detail] Current data keys: {list(current_data.keys())}")
        logger.info(f"[Callbot Task Detail] Reference forms: {len(reference_forms)}")
        
        return {
            "success": True,
            "task": {
                "id": task_data['id'],
                "activity_name": task_data.get('activity_name'),
                "description": task_data.get('description'),
                "status": task_data.get('status'),
                "proc_inst_id": task_data.get('proc_inst_id'),
                "proc_def_id": task_data.get('proc_def_id'),
                "activity_id": task_data.get('activity_id'),
            },
            "form_schema": form_schema,
            "current_data": current_data,
            "reference_forms": reference_forms
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Callbot Task Detail] Exception: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@router.patch("/task/{task_id}")
async def update_task_form(
    task_id: str,
    fields: Dict[str, Any] = Body(..., description="업데이트할 필드 {field_name: value}")
):
    """
    콜봇 AI Function Calling용: 업무의 폼 데이터(output) 업데이트
    
    Parameters:
    - task_id: 업무(todolist) UUID
    - fields: 업데이트할 필드 딕셔너리
    
    Returns:
    - success: 성공 여부
    - updated_fields: 업데이트된 필드 목록
    """
    try:
        logger.info(f"[Callbot Task Update] Request - task_id: {task_id}")
        logger.info(f"[Callbot Task Update] Fields: {fields}")
        
        supabase = supabase_client_var.get()
        if supabase is None:
            raise HTTPException(status_code=500, detail="Supabase client not configured")
        
        subdomain = subdomain_var.get()
        
        # 현재 업무 조회
        response = supabase.table('todolist').select('*').eq('id', task_id).eq('tenant_id', subdomain).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
        task_data = response.data[0]
        current_output = task_data.get('output', {}) or {}
        
        # ✅ 웹 형식으로 wrap (폼 이름을 최상위 키로 포함)
        # 예: {"field1": "value1"} -> {"form_id": {"field1": "value1"}}
        tool = task_data.get('tool', '')
        if tool and 'formHandler:' in tool:
            form_id = tool.split('formHandler:')[1]
            # fields가 이미 폼 이름으로 wrap되어 있지 않으면 wrap
            if form_id not in fields:
                logger.info(f"[Callbot Task Update] Wrapping fields with form_id: {form_id}")
                fields = {form_id: fields}
        
        # output 필드 업데이트 (merge)
        # 웹 형식: {"form_id": {"field1": "value1", "field2": "value2"}}
        if tool and 'formHandler:' in tool:
            form_id = tool.split('formHandler:')[1]
            if form_id in current_output and form_id in fields:
                # 기존 폼 데이터와 merge
                updated_output = {**current_output}
                updated_output[form_id] = {**current_output[form_id], **fields[form_id]}
            else:
                updated_output = {**current_output, **fields}
        else:
            updated_output = {**current_output, **fields}
        
        # 데이터베이스 업데이트
        update_response = supabase.table('todolist').update({
            'output': updated_output
        }).eq('id', task_id).eq('tenant_id', subdomain).execute()
        
        logger.info(f"[Callbot Task Update] ✅ Updated task {task_id} with fields: {list(fields.keys())}")
        logger.info(f"[Callbot Task Update] Output data: {updated_output}")
        
        return {
            "success": True,
            "updated_fields": list(fields.keys()),
            "output": updated_output
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Callbot Task Update] Exception: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


@router.post("/task/{task_id}/submit")
async def submit_task(
    task_id: str,
):
    """
    콜봇 AI Function Calling용: 업무 제출 (상태를 SUBMITTED로 변경하여 polling service가 처리)
    
    Parameters:
    - task_id: 업무(todolist) UUID
    
    Returns:
    - success: 성공 여부
    - task_id: 제출된 업무 ID
    - status: 업데이트된 상태
    
    Note:
    - 상태를 SUBMITTED로 설정하면 polling_service가 자동으로 다음 액티비티를 생성합니다
    - DONE으로 설정하면 워크플로우가 진행되지 않습니다
    """
    try:
        logger.info(f"[Callbot Task Submit] Request - task_id: {task_id}")
        
        supabase = supabase_client_var.get()
        if supabase is None:
            raise HTTPException(status_code=500, detail="Supabase client not configured")
        
        subdomain = subdomain_var.get()
        
        # 업무 존재 확인
        response = supabase.table('todolist').select('*').eq('id', task_id).eq('tenant_id', subdomain).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        
        task_data = response.data[0]
        
        # 상태 업데이트: SUBMITTED로 변경하여 polling service가 처리하도록 함
        from datetime import datetime, timezone
        update_response = supabase.table('todolist').update({
            'status': 'SUBMITTED',
            'end_date': datetime.now(timezone.utc).isoformat(),
            'retry': 0,  # retry 카운트 초기화
            'consumer': None  # consumer 해제하여 polling service가 처리 가능하도록
        }).eq('id', task_id).eq('tenant_id', subdomain).execute()
        
        logger.info(f"[Callbot Task Submit] ✅ Task {task_id} ({task_data.get('activity_name')}) submitted successfully")
        logger.info(f"[Callbot Task Submit] Status changed to SUBMITTED - polling service will create next activities")
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "SUBMITTED",
            "activity_name": task_data.get('activity_name'),
            "message": "Task submitted successfully. Polling service will process next activities."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Callbot Task Submit] Exception: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }


def add_routes_to_app(app):
    """FastAPI 앱에 콜봇 라우터 추가"""
    app.include_router(router)
