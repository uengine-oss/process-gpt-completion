"""테스트 모드 라우트 — BPMN 인앱 단위 테스트용.

설계 메모 (C:\\Users\\m6023\\.claude\\plans\\nifty-doodling-pnueli.md 참고):
- 분기 결정은 실제 엔진(폴링 서비스)이 한다. 테스트는 그 결과를 받아서 검증한다.
  → 별도 동기 실행 루프를 만들지 않고, 운영과 같은 경로(SUBMITTED → 폴링 서비스 처리)를 그대로 탄다.
- Given = { activityId: formOutputJson } — 이전 작업들의 폼 출력. DONE 워크아이템 row로 미리 심어
  대상 작업부터 실행하게 한다. (ProcessGPT엔 전역 프로세스 변수가 없으므로 작업 출력이 곧 상태)
- 매핑(전역 변수 레이어 등)이 생기면 추후 보강.

엔드포인트:
- POST /test/initiate    { process_definition_id, target_activity_id?, given?, email?, version_tag?, version? }
                         → { proc_inst_id, task_id, activity_id, seeded_activity_ids }
- POST /test/complete    { task_id, form_values?, timeout_ms? }
                         → { proc_inst_id, active_activity_ids, passed_activity_ids, process_status, outputs, instance_count, timed_out? }
- POST /test/cleanup/{proc_inst_id}
                         → { deleted: proc_inst_id }

전제: 폴링 서비스(polling_service/main.py)가 떠 있어야 /test/complete 가 실제로 진행 결과를 받는다.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta

import pytz
from fastapi import Request, HTTPException

from database import (
    supabase_client_var,
    subdomain_var,
    fetch_process_definition_by_version,
    insert_process_instance,
    upsert_workitem,
    fetch_workitem_by_id,
    fetch_todolist_by_proc_inst_id,
    fetch_process_instance,
)
from process_definition import load_process_definition


_KST = pytz.timezone("Asia/Seoul")

# 워크아이템 status 분류
_ACTIVE_STATES = {"TODO", "IN_PROGRESS", "PENDING"}
_DONE_STATES = {"DONE", "SUBMITTED", "COMPLETED"}


def _now_iso():
    return datetime.now(_KST).isoformat()


def _safe_attr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _resolve_role_default(process_definition_json, role_name):
    if not role_name:
        return None
    for role in (process_definition_json.get("roles") or []):
        if role.get("name") == role_name:
            return role.get("default") or role.get("endpoint")
    return None


# ---------------------------------------------------------------------------
# /test/initiate
# ---------------------------------------------------------------------------
async def test_initiate(input: dict):
    process_definition_id = input.get("process_definition_id")
    if not process_definition_id:
        raise HTTPException(status_code=400, detail="process_definition_id is required")

    target_activity_id = input.get("target_activity_id")
    version_tag = input.get("version_tag")
    version = input.get("version")
    given = input.get("given") or {}            # { activityId: outputJson }
    email = input.get("email")

    process_definition_json = fetch_process_definition_by_version(process_definition_id, version_tag, version)
    if not process_definition_json:
        raise HTTPException(status_code=404, detail=f"Process definition not found: {process_definition_id}")
    process_definition = load_process_definition(process_definition_json)

    # 대상 활동 (지정 없으면 초기 활동)
    if target_activity_id:
        target_activity = process_definition.find_activity_by_id(target_activity_id)
    else:
        target_activity = process_definition.find_initial_activity()
        target_activity_id = _safe_attr(target_activity, "id")
    if target_activity is None:
        raise HTTPException(status_code=400, detail=f"Activity not found: {target_activity_id}")

    tenant_id = subdomain_var.get()
    # 운영과 동일한 인스턴스 id 형식 ({proc_def_id}.{uuid}) — 폴링/정의 로딩이 정상 동작해야 함.
    process_instance_id = f"{process_definition_id.lower()}.{uuid.uuid4()}"

    insert_process_instance(
        {
            "proc_inst_id": process_instance_id,
            "proc_inst_name": _safe_attr(process_definition, "processDefinitionName") or process_definition_id,
            "proc_def_id": process_definition_id,
            "participants": [],
            "status": "RUNNING",
            "role_bindings": [],
            "current_activity_ids": [target_activity_id],
            "variables_data": [],
            "start_date": _now_iso(),
        },
        tenant_id,
    )

    # Given: 이전 작업들의 폼 출력을 DONE row로 미리 심기
    seeded = []
    if isinstance(given, dict):
        for activity_id, output in given.items():
            act = process_definition.find_activity_by_id(activity_id)
            upsert_workitem(
                {
                    "id": str(uuid.uuid4()),
                    "user_id": None,
                    "proc_inst_id": process_instance_id,
                    "proc_def_id": process_definition_id,
                    "activity_id": activity_id,
                    "activity_name": _safe_attr(act, "name") or activity_id,
                    "start_date": _now_iso(),
                    "end_date": _now_iso(),
                    "status": "DONE",
                    "output": output if isinstance(output, dict) else {"value": output},
                    "tool": _safe_attr(act, "tool"),
                    "retry": 0,
                    "consumer": None,
                    "root_proc_inst_id": process_instance_id,
                },
                tenant_id,
            )
            seeded.append(activity_id)

    # 대상 작업: TODO row.
    # - 사람 작업이면 폴링 서비스가 건드리지 않고, /test/complete 가 SUBMITTED 로 바꿔야 진행.
    # - AI/agent 작업이면 폴링이 곧바로 처리할 수도 있음(그 경우엔 /test/complete 없이 결과를 봐도 됨).
    user_email = email or _resolve_role_default(process_definition_json, _safe_attr(target_activity, "role"))
    try:
        # find_prev_activities 는 ProcessActivity 객체 리스트를 반환한다. workitem.reference_ids 는
        # JSON 직렬화되어 DB에 저장되므로 id 문자열만 추출해 둔다.
        _prev = process_definition.find_prev_activities(target_activity_id, [])
        prev_activities = [getattr(a, "id", None) for a in (_prev or []) if getattr(a, "id", None)]
    except Exception:
        prev_activities = []
    description = _safe_attr(target_activity, "description")
    instruction = _safe_attr(target_activity, "instruction")
    query = ""
    if description:
        query += f"[Description]\n{description}\n\n"
    if instruction:
        query += f"[Instruction]\n{instruction}\n\n"
    duration = _safe_attr(target_activity, "duration")
    due_date = (datetime.now(_KST) + timedelta(days=duration)).isoformat() if duration else None
    target_task_id = str(uuid.uuid4())
    upsert_workitem(
        {
            "id": target_task_id,
            "user_id": user_email,
            "proc_inst_id": process_instance_id,
            "proc_def_id": process_definition_id,
            "activity_id": target_activity_id,
            "activity_name": _safe_attr(target_activity, "name") or target_activity_id,
            "start_date": _now_iso(),
            "due_date": due_date,
            "status": "TODO",
            "assignees": None,
            "reference_ids": prev_activities,
            "duration": duration,
            "tool": _safe_attr(target_activity, "tool"),
            "output": None,
            "retry": 0,
            "consumer": None,
            "description": description,
            "query": query,
            "root_proc_inst_id": process_instance_id,
        },
        tenant_id,
    )

    return {
        "proc_inst_id": process_instance_id,
        "task_id": target_task_id,
        "activity_id": target_activity_id,
        "seeded_activity_ids": seeded,
    }


# ---------------------------------------------------------------------------
# /test/complete
# ---------------------------------------------------------------------------
def _is_subprocess_id(activity_id, process_definition):
    if not activity_id or not process_definition:
        return False
    subs = getattr(process_definition, "subProcesses", None) or []
    return any(getattr(s, "id", None) == activity_id for s in subs)


def _collect_descendant_running_active_ids(root_proc_inst_id: str, tenant_id: str):
    """root_proc_inst_id 를 가진 모든 자식 RUNNING 인스턴스의 current_activity_ids 합집합."""
    supabase = supabase_client_var.get()
    if supabase is None:
        return []
    try:
        resp = (
            supabase.table("bpm_proc_inst")
            .select("current_activity_ids,status,proc_inst_id")
            .eq("root_proc_inst_id", root_proc_inst_id)
            .eq("tenant_id", tenant_id)
            .execute()
        )
        data = resp.data or []
    except Exception:
        return []
    out = []
    for row in data:
        if (row.get("status") or "").upper() != "RUNNING":
            continue
        ids = row.get("current_activity_ids") or []
        if isinstance(ids, str):
            ids = [ids]
        for aid in ids:
            if aid:
                out.append(str(aid))
    return out


def _fetch_root_workitems(root_proc_inst_id: str, tenant_id: str):
    """루트 기준으로 부모+자식 워크아이템 모두 가져온다. (서브프로세스 내부 태스크 포함)"""
    supabase = supabase_client_var.get()
    if supabase is None:
        return []
    try:
        # root_proc_inst_id 매칭(자식들) + proc_inst_id 매칭(부모 자신)
        resp = (
            supabase.table("todolist")
            .select("*")
            .or_(f"proc_inst_id.eq.{root_proc_inst_id},root_proc_inst_id.eq.{root_proc_inst_id}")
            .eq("tenant_id", tenant_id)
            .execute()
        )
        return resp.data or []
    except Exception:
        return []


def _count_instances(root_proc_inst_id: str, tenant_id: str) -> int:
    """단위 테스트 기대값과 비교하기 위한 인스턴스 개수.
    루트 자체 + root_proc_inst_id 로 연결된 모든 자식 인스턴스 를 더한 값.
    멀티인스턴스 서브프로세스 없으면 1, N회 개입되면 1+N."""
    supabase = supabase_client_var.get()
    if supabase is None:
        return 1
    try:
        resp = (
            supabase.table("bpm_proc_inst")
            .select("proc_inst_id", count="exact")
            .or_(f"proc_inst_id.eq.{root_proc_inst_id},root_proc_inst_id.eq.{root_proc_inst_id}")
            .eq("tenant_id", tenant_id)
            .execute()
        )
        # supabase-py: 정확한 count는 resp.count, 없으면 data 길이로 폴백.
        count = getattr(resp, "count", None)
        if count is None:
            count = len(resp.data or [])
        return int(count or 0)
    except Exception:
        return 1


def _snapshot(proc_inst_id: str, tenant_id: str, process_definition=None) -> dict:
    # 부모 + 모든 자식(서브프로세스) 워크아이템을 root_proc_inst_id 기준으로 묶어서 본다.
    workitems_data = _fetch_root_workitems(proc_inst_id, tenant_id)
    passed = [
        w.get("activity_id") for w in workitems_data
        if (w.get("status") or "") in _DONE_STATES and w.get("activity_id")
    ]
    outputs = {
        w.get("activity_id"): w.get("output") for w in workitems_data
        if w.get("output") and w.get("activity_id")
    }
    try:
        inst = fetch_process_instance(proc_inst_id, tenant_id)
    except Exception:
        inst = None
    status = (_safe_attr(inst, "status") or "RUNNING")

    # 부모 + 자식 인스턴스들의 current_activity_ids 를 합쳐서 active 산출.
    # 폴링 서비스(upsert_todo_workitems)가 만든 다운스트림 placeholder 워크아이템 노이즈를
    # 피하기 위해 워크아이템 status 가 아닌 bpm_proc_inst.current_activity_ids 를 사용.
    parent_current = _safe_attr(inst, "current_activity_ids", None) or []
    if isinstance(parent_current, str):
        parent_current = [parent_current]
    descendant_current = _collect_descendant_running_active_ids(proc_inst_id, tenant_id)

    # 사용자가 wrapper(서브프로세스 노드)와 내부 태스크를 동시에 보고 싶어한다.
    # 따라서 drill-in 으로 wrapper 를 가리지 않고, 부모/자식 current 를 모두 합쳐서 노출.
    active = list(dict.fromkeys(list(parent_current or []) + list(descendant_current or [])))
    # workitem status 기반 fallback 은 폴링 서비스가 미리 만들어둔 다운스트림 placeholder
    # (status=TODO, user_id="") 가 섞여 노이즈가 심하다. current_activity_ids 가 비어있으면
    # 그냥 [] 로 둔다 (엔진이 정리 못 한 경우라도 추정 안 함).
    if (status or "").upper() == "COMPLETED":
        # 종료된 인스턴스는 active 가 있더라도 의미 없음 — 비워서 노이즈 차단.
        active = []

    return {
        "proc_inst_id": proc_inst_id,
        "active_activity_ids": active,
        "passed_activity_ids": list(dict.fromkeys(passed)),
        "process_status": status,
        "outputs": outputs,
        "instance_count": _count_instances(proc_inst_id, tenant_id),
    }


async def test_complete(input: dict):
    task_id = input.get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    form_values = input.get("form_values") or {}
    timeout_ms = int(input.get("timeout_ms") or 8000)

    tenant_id = subdomain_var.get()
    workitem = fetch_workitem_by_id(task_id, tenant_id)
    if workitem is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    proc_inst_id = workitem.proc_inst_id
    target_activity_id = workitem.activity_id

    # 서브프로세스 wrapper drill-in 을 위해 process_definition 을 한 번 로드해 둔다.
    process_definition = None
    try:
        inst_obj = fetch_process_instance(proc_inst_id, tenant_id)
        proc_def_id = _safe_attr(inst_obj, "proc_def_id", None)
        if proc_def_id:
            pdj = fetch_process_definition_by_version(proc_def_id, None, None)
            if pdj:
                process_definition = load_process_definition(pdj)
    except Exception:
        process_definition = None

    # 운영과 동일한 경로: 작업을 SUBMITTED 로 마킹 → 폴링 서비스가 그 뒤를 진행.
    # (assignee/role 처리 등 submit_workitem 의 부가 로직은 테스트엔 불필요하므로 직접 마킹.)
    wi_data = workitem.model_dump()
    wi_data["status"] = "SUBMITTED"
    wi_data["output"] = form_values
    wi_data["consumer"] = None
    for k in ("start_date", "end_date", "due_date"):
        v = wi_data.get(k)
        if v is not None and not isinstance(v, str):
            try:
                wi_data[k] = v.isoformat()
            except Exception:
                wi_data[k] = None
    upsert_workitem(wi_data, tenant_id)

    # 폴링 서비스가 인스턴스를 진행시킬 때까지 대기 — DB 상태 변화 감지.
    # 부모의 active 가 바뀐 직후엔 자식(서브프로세스) 인스턴스가 아직 생성 안 됐을 수 있다.
    # 활성 셋이 안정될 때까지 한 박자 더 보고, 서브프로세스 wrapper 만 active 인 동안에는
    # 자식 inner 가 채워질 시간을 더 확보한다 (폴링 사이클이 5~10초이기 때문).
    sub_ids: set = set()
    if process_definition:
        sub_ids = {
            getattr(s, "id", None)
            for s in (getattr(process_definition, "subProcesses", None) or [])
            if getattr(s, "id", None)
        }
    deadline = time.monotonic() + max(0.5, timeout_ms / 1000.0)
    STABILITY_TICKS = 4              # 일반 케이스: ~1.2s
    STUCK_EMPTY_TICKS = 40           # 타깃 완료 후 active 비어있는 채로 ~12s 지속되면 stuck.
    last = _snapshot(proc_inst_id, tenant_id, process_definition)
    prev_active = None
    stable_ticks = 0
    changed_once = False
    empty_after_target_done_ticks = 0
    while time.monotonic() < deadline:
        await asyncio.sleep(0.3)
        snap = _snapshot(proc_inst_id, tenant_id, process_definition)
        last = snap
        if snap["process_status"] == "COMPLETED":
            return snap
        active_list = snap["active_activity_ids"] or []
        active = tuple(sorted(active_list))
        # 엔진 한계: 서브프로세스 종료 등으로 타깃은 DONE 인데 부모 current 가 비어버리는 경우
        # 폴링이 끝없이 대기하지 않도록, 타깃 완료 후 active 가 일정 시간 비어있으면 리턴.
        if not active:
            target_done = target_activity_id in (snap["passed_activity_ids"] or [])
            if target_done:
                empty_after_target_done_ticks += 1
                if empty_after_target_done_ticks >= STUCK_EMPTY_TICKS:
                    return snap
            continue
        else:
            empty_after_target_done_ticks = 0
        if active == (target_activity_id,):
            continue
        # 서브프로세스 wrapper 만 active 면 inner 가 나올 때까지 계속 폴링한다(early return 안 함).
        # 폴링 사이클이 5~10초로 느려 wrapper 만 잠시 노출되는 구간이 길 수 있다.
        active_set = set(active_list)
        only_sub_wrapper = bool(sub_ids) and (active_set <= sub_ids)
        if only_sub_wrapper:
            continue
        # wrapper+inner 또는 일반 활동: 안정 대기 후 리턴.
        if not changed_once:
            changed_once = True
            prev_active = active
            stable_ticks = 1
            continue
        if active == prev_active:
            stable_ticks += 1
            if stable_ticks >= STABILITY_TICKS:
                return snap
        else:
            prev_active = active
            stable_ticks = 1

    last["timed_out"] = True
    return last


# ---------------------------------------------------------------------------
# /test/cleanup/{proc_inst_id}
# ---------------------------------------------------------------------------
async def test_cleanup(proc_inst_id: str):
    if not proc_inst_id:
        raise HTTPException(status_code=400, detail="proc_inst_id is required")
    tenant_id = subdomain_var.get()
    supabase = supabase_client_var.get()
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase client is not configured for this request")

    # 서브프로세스를 거친 인스턴스는 자식 인스턴스도 생긴다. 루트 기준으로 같이 정리.
    try:
        supabase.table("todolist").delete().or_(
            f"proc_inst_id.eq.{proc_inst_id},root_proc_inst_id.eq.{proc_inst_id}"
        ).eq("tenant_id", tenant_id).execute()
    except Exception:
        supabase.table("todolist").delete().eq("proc_inst_id", proc_inst_id).eq("tenant_id", tenant_id).execute()
    try:
        supabase.table("bpm_proc_inst").delete().or_(
            f"proc_inst_id.eq.{proc_inst_id},root_proc_inst_id.eq.{proc_inst_id}"
        ).eq("tenant_id", tenant_id).execute()
    except Exception:
        supabase.table("bpm_proc_inst").delete().eq("proc_inst_id", proc_inst_id).eq("tenant_id", tenant_id).execute()
    # 이벤트/소스 테이블이 있으면 함께 정리 (없으면 무시)
    for extra_table in ("events", "proc_inst_source"):
        try:
            supabase.table(extra_table).delete().eq("proc_inst_id", proc_inst_id).eq("tenant_id", tenant_id).execute()
        except Exception:
            pass
    return {"deleted": proc_inst_id}


# ---------------------------------------------------------------------------
# FastAPI handlers / route registration
# ---------------------------------------------------------------------------
async def handle_test_initiate(request: Request):
    try:
        body = await request.json()
        payload = body.get("input", body) if isinstance(body, dict) else {}
        return await test_initiate(payload or {})
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e


async def handle_test_complete(request: Request):
    try:
        body = await request.json()
        payload = body.get("input", body) if isinstance(body, dict) else {}
        return await test_complete(payload or {})
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e


async def handle_test_cleanup(proc_inst_id: str):
    try:
        return await test_cleanup(proc_inst_id)
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e)) from e


def add_routes_to_app(app):
    app.add_api_route("/test/initiate", handle_test_initiate, methods=["POST"])
    app.add_api_route("/test/complete", handle_test_complete, methods=["POST"])
    app.add_api_route("/test/cleanup/{proc_inst_id}", handle_test_cleanup, methods=["POST"])
