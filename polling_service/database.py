from supabase import create_client, Client
from pydantic import BaseModel, validator
from typing import Any, Dict, List, Optional, Set, Union
from langchain_community.vectorstores import SupabaseVectorStore
from process_definition import ProcessDefinition, load_process_definition, UIDefinition
from fastapi import HTTPException
from decimal import Decimal
from datetime import datetime, timedelta
from contextvars import ContextVar, copy_context
from dotenv import load_dotenv
from llm_factory import create_llm, create_embedding

import pytz
import socket
import os
import uuid
import json
import asyncio
import threading
import requests
from task_deadline import ensure_minimum_task_due_date


supabase_client_var = ContextVar('supabase', default=None)
subdomain_var = ContextVar('subdomain', default='localhost')
CONSUMER_FILTER = os.getenv("WORKITEM_CONSUMER")


def run_async_in_sync_context(coro):
    """
    Run coroutine safely from sync code, even when an event loop is already running.
    """
    try:
        asyncio.get_running_loop()
        has_running_loop = True
    except RuntimeError:
        has_running_loop = False

    if not has_running_loop:
        return asyncio.run(coro)

    result_holder = {"result": None, "error": None}
    ctx = copy_context()

    def _runner():
        try:
            # ContextVar(s) (e.g., supabase_client_var) do NOT automatically propagate to new threads.
            # copy_context() ensures request-scoped clients/tenant info remain available.
            result_holder["result"] = ctx.run(asyncio.run, coro)
        except Exception as exc:
            result_holder["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    if result_holder["error"] is not None:
        raise result_holder["error"]

    return result_holder["result"]


def setting_database():
    try:
        # Do not override env vars provided by container/K8s.
        # This prevents baked-in .env from breaking deployments.
        load_dotenv(override=False)

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        # Optional: attach Authorization header from env for PostgREST if provided
        extra_auth = os.getenv("SUPABASE_AUTHORIZATION")
        if extra_auth:
            try:
                # Always set raw Authorization header as-is (including Bearer ...)
                try:
                    supabase.postgrest.headers["Authorization"] = extra_auth
                except Exception:
                    try:
                        supabase.postgrest.client.headers["Authorization"] = extra_auth  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARN] Failed to apply SUPABASE_AUTHORIZATION header: {e}")
        supabase_client_var.set(supabase)
        
        print(f"[INFO] Supabase client configured successfully")
        
    except Exception as e:
        print(f"Database configuration error: {e}")


def execute_sql(sql_query):
    """
    Executes SQL query using Supabase Client API.
    
    Args:
        sql_query (str): The SQL query to execute.
        
    Returns:
        list: A list of dictionaries representing the rows returned by the query.
    """
    
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        # Supabase Client API瑜??ъ슜?섏뿬 SQL ?ㅽ뻾
        response = supabase.rpc('exec_sql', {'query': sql_query}).execute()
        
        if response.data:
            return response.data
        else:
            return "Query executed successfully"
    
    except Exception as e:
        return(f"An error occurred while executing the SQL query: {e}")
    
def execute_rpc(rpc_name, params):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.rpc(rpc_name, params).execute()
        return response.data
    except Exception as e:
        return(f"An error occurred while executing the RPC: {e}")


def fetch_process_definition(def_id, tenant_id: Optional[str] = None):
    """
    Fetches the process definition from the 'proc_def' table based on the given definition ID.
    
    Args:
        def_id (str): The ID of the process definition to fetch.
    
    Returns:
        dict: The process definition as a JSON object if found, else None.
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
    
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain


        response = supabase.table('proc_def').select('*').eq('id', def_id.lower()).eq('tenant_id', tenant_id).execute()
        
        # Check if the response contains data
        if response.data:
            # Assuming the first match is the desired one since ID should be unique
            process_definition = response.data[0].get('definition', None)
            return process_definition
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No process definition found with ID {def_id}: {e}")

def fetch_process_definition_version_by_arcv_id(def_id, arcv_id, tenant_id: Optional[str] = None):
    """
    proc_def_arcv / proc_def_version ????λ맂 ?뱀젙 arcv_id 踰꾩쟾???꾨줈?몄뒪 ?뺤쓽瑜?議고쉶?⑸땲??
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")

        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain

        # 猷⑦듃 紐⑤뱢怨??숈씪?섍쾶 proc_def_arcv 湲곗??쇰줈 議고쉶
        response = (
            supabase.table("proc_def_arcv")
            .select("*")
            .eq("proc_def_id", def_id.lower())
            .eq("arcv_id", arcv_id)
            .eq("tenant_id", tenant_id)
            .execute()
        )

        return response.data
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"No process definition version found with ID {def_id} and version {arcv_id}: {e}",
        )


def fetch_process_definition_by_version(
    def_id: str,
    version_tag: Optional[str] = None,
    version: Optional[Union[str, int]] = None,
    tenant_id: Optional[str] = None,
    arcv_id: Optional[str] = None,
):
    """
    ?ㅽ뻾 ?뺤쓽(?꾨줈?몄뒪 definition)瑜?踰꾩쟾 洹쒖튃???곕씪 議고쉶?⑸땲??

    ?명솚/紐낆떆 ?곗꽑?쒖쐞:
    - arcv_id 媛 二쇱뼱吏硫? ?대떦 proc_def_arcv 踰꾩쟾??理쒖슦?좎쑝濡?議고쉶 (湲곗〈 ?숈옉 ?좎?)
    - version_tag + version ??二쇱뼱吏硫? proc_def_version?먯꽌 ?대떦 tag/version???곗꽑 議고쉶

    湲곕낯(紐낆떆 踰꾩쟾???놁쓣 ?? ?숈옉? TS 諛⑹떇?쇰줈 ?듭씪:
    1) proc_def.prod_version ???덉쑝硫??대떦 踰꾩쟾???곗꽑 議고쉶
       - prod_version? 湲곗〈 ?댁쁺??arcv_id濡???λ릺??寃쎌슦媛 ?덉뼱 proc_def_arcv濡?癒쇱? ?쒕룄?섍퀬,
         ?ㅽ뙣 ??proc_def_version.version 留ㅼ묶???쒕룄
    2) 理쒖떊 major(version_tag='major')
    3) 理쒖떊 minor(version_tag='minor')
    4) proc_def.definition (?꾩옱 ?뺤쓽)
    """
    # 怨듯넻 紐⑤뱢濡?濡쒖쭅???꾩엫?섏뿬 以묐났 ?쒓굅
    from proc_def_versioning import fetch_process_definition_by_version_ts_style

    if not def_id:
        return None

    supabase = supabase_client_var.get()
    if supabase is None:
        raise Exception("Supabase client is not configured for this request")

    subdomain = subdomain_var.get()
    if not tenant_id:
        tenant_id = subdomain

    def fetch_arcv_rows(arcv: str) -> List[dict]:
        return fetch_process_definition_version_by_arcv_id(def_id, arcv, tenant_id) or []

    return fetch_process_definition_by_version_ts_style(
        supabase=supabase,
        def_id=def_id,
        tenant_id=tenant_id,
        version_tag=version_tag,
        version=version,
        arcv_id=arcv_id,
        fetch_arcv_rows=fetch_arcv_rows,
    )


def fetch_process_definition_latest_version(def_id, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")


        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain


        response = supabase.table('proc_def_version').select('*').eq('proc_def_id', def_id.lower()).eq('tenant_id', tenant_id).order('version', desc=True).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No process definition latest version found with ID {def_id}: {e}")


def update_proc_def_prod_version(def_id: str, prod_version: str, tenant_id: Optional[str] = None):
    """
    proc_def ?뚯씠釉붿쓽 prod_version 而щ읆???낅뜲?댄듃?⑸땲??
    諛고룷 ?뱀씤 ???대떦 踰꾩쟾???꾨줈?뺤뀡 踰꾩쟾?쇰줈 ?ㅼ젙?⑸땲??
    
    Args:
        def_id (str): ?꾨줈?몄뒪 ?뺤쓽 ID
        prod_version (str): ?꾨줈?뺤뀡 踰꾩쟾?쇰줈 ?ㅼ젙??踰꾩쟾 (arcv_id)
        tenant_id (Optional[str]): ?뚮꼳??ID
    
    Returns:
        bool: ?낅뜲?댄듃 ?깃났 ?щ?
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain
        
        response = supabase.table('proc_def').update({
            'prod_version': prod_version
        }).eq('id', def_id.lower()).eq('tenant_id', tenant_id).execute()
        
        if response.data:
            print(f"[INFO] Updated prod_version for {def_id} to {prod_version}")
            return True
        else:
            print(f"[WARN] No rows updated for proc_def {def_id}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to update prod_version for {def_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update prod_version for {def_id}: {e}")


def fetch_ui_definition(def_id):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        response = supabase.table('form_def').select('*').eq('id', def_id.lower()).eq('tenant_id', subdomain).execute()
        
        if response.data:
            # Assuming the first match is the desired one since ID should be unique
            ui_definition = UIDefinition(**response.data[0])
            return ui_definition
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No UI definition found with ID {def_id}: {e}")

def fetch_ui_definitions_by_def_id(def_id, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain
            
        response = supabase.table('form_def').select('*').eq('proc_def_id', def_id).eq('tenant_id', tenant_id).execute()
        
        if response.data and len(response.data) > 0:
            return [UIDefinition(**item) for item in response.data]
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No UI definitions found with ID {def_id}: {e}")


def fetch_ui_definition_by_activity_id(proc_def_id, activity_id, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain


        response = supabase.table('form_def').select('*').eq('proc_def_id', proc_def_id).eq('activity_id', activity_id).eq('tenant_id', tenant_id).execute()
        
        if response.data:
            # Assuming the first match is the desired one since ID should be unique
            ui_definition = UIDefinition(**response.data[0])
            return ui_definition
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No UI definition found with ID {proc_def_id}: {e}")


class ProcessInstance(BaseModel):
    proc_inst_id: str
    proc_inst_name: Optional[str] = None
    role_bindings: Optional[List[Dict[str, Any]]] = []
    current_activity_ids: Optional[List[str]] = []
    participants: Optional[List[str]] = []
    variables_data: Optional[Any] = []
    process_definition: ProcessDefinition = None  # Add a reference to ProcessDefinition
    status: str = None
    tenant_id: str
    proc_def_version: Optional[str] = None
    parent_proc_inst_id: Optional[str] = None
    root_proc_inst_id: Optional[str] = None


    class Config:
        extra = "allow"


    def __init__(self, **data):
        super().__init__(**data)
        def_id = self.get_def_id()
        tenant_id = self.tenant_id
        # proc_def_version(arcv_id)媛 ?덉쑝硫??대떦 踰꾩쟾 ?뺤쓽瑜??곗꽑 ?ъ슜?섎릺,
        # ?덇굅???곗씠???? "xxx_2.0")泥섎읆 arcv_id媛 ?꾨땶 媛믪씠 ?ㅼ뼱?????덉뼱 ?대갚???〓땲??
        definition_json = None
        if getattr(self, "proc_def_version", None):
            definition_json = fetch_process_definition_by_version(
                def_id,
                tenant_id=tenant_id,
                arcv_id=self.proc_def_version,
            )

        if not definition_json and getattr(self, "version_tag", None) and getattr(self, "version", None):
            definition_json = fetch_process_definition_by_version(
                def_id,
                tenant_id=tenant_id,
                version_tag=self.version_tag,
                version=self.version,
            )

        if not definition_json:
            definition_json = fetch_process_definition_by_version(def_id, tenant_id=tenant_id)

        if not definition_json:
            raise ValueError(
                f"Process definition not found for def_id={def_id}, tenant_id={tenant_id}, "
                f"proc_def_version={getattr(self, 'proc_def_version', None)}, "
                f"version_tag={getattr(self, 'version_tag', None)}, version={getattr(self, 'version', None)}"
            )

        self.process_definition = load_process_definition(definition_json)  # Load ProcessDefinition


    def get_def_id(self):
        # inst_id ?덉떆: "company_entrance.123e4567-e89b-12d3-a456-426614174000"
        # ?ш린??"company_entrance"媛 ?꾨줈?몄뒪 ?뺤쓽 ID?낅땲??
        return self.proc_inst_id.split(".")[0]


    def get_data(self):
        # Return all process variable values as a map
        variable_map = {}
        for variable in self.process_definition.data:
            variable_name = variable.name
            variable_map[variable_name] = getattr(self, variable_name, None)
        return variable_map
  
class WorkItem(BaseModel):
    id: str
    user_id: Optional[str]
    username: Optional[str] = None
    proc_inst_id: Optional[str] = None
    root_proc_inst_id: Optional[str] = None
    proc_def_id: Optional[str] = None
    activity_id: str
    activity_name: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    status: str
    description: Optional[str] = None
    tool: Optional[str] = None
    tenant_id: str
    reference_ids: Optional[List[str]] = []
    assignees: Optional[List[Dict[str, Any]]] = []
    duration: Optional[int] = None
    output: Optional[Dict[str, Any]] = {}
    retry: Optional[int] = 0
    consumer: Optional[str] = None
    log: Optional[str] = None
    agent_mode: Optional[str] = None
    agent_orch: Optional[str] = None
    feedback: Optional[List[Dict[str, Any]]] = []
    temp_feedback: Optional[str] = None
    execution_scope: Optional[str] = None
    rework_count: Optional[int] = 0
    project_id: Optional[str] = None
    query: Optional[str] = None
    version_tag: Optional[str] = None
    version: Optional[str] = None
    adhoc: Optional[bool] = None
    
    @validator('start_date', 'end_date', 'due_date', pre=True)
    def parse_datetime(cls, value):
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).replace(tzinfo=None)
            except ValueError:
                return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return value


    class Config:
        json_encoders = {
            datetime: lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S")
        }


def fetch_process_instance(full_id: str, tenant_id: Optional[str] = None) -> Optional[ProcessInstance]:
    try:
        if full_id == "new" or '.' not in full_id:
            return None

        if not full_id:
            raise HTTPException(status_code=404, detail="Instance Id should be provided")

        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain

        response = supabase.table('bpm_proc_inst').select("*").eq('proc_inst_id', full_id).eq('tenant_id', tenant_id).execute()

        if response.data:
            process_instance_data = response.data[0]
            process_instance = ProcessInstance(**process_instance_data)
            return process_instance
        else:
            return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    
def fetch_child_instances_by_parent(parent_proc_inst_id: str, tenant_id: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    ?뱀젙 遺紐?proc_inst_id)瑜?媛吏?紐⑤뱺 ?먯떇 ?몄뒪?댁뒪瑜?議고쉶?⑸땲??
    寃쎈웾 議고쉶: proc_inst_id, status, current_activity_ids留?諛섑솚.
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")

        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain

        response = (
            supabase.table('bpm_proc_inst')
            .select('proc_inst_id,status,current_activity_ids')
            .eq('parent_proc_inst_id', parent_proc_inst_id)
            .eq('tenant_id', tenant_id)
            .execute()
        )

        if response.data and len(response.data) > 0:
            return response.data  # List[{'proc_inst_id':..., 'status':..., 'current_activity_ids': [...]}]
        else:
            return None

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch child instances for parent {parent_proc_inst_id}: {e}") from e



def insert_process_instance(process_instance_data: dict, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")

        if not tenant_id:
            tenant_id = subdomain_var.get()
        process_instance_data['tenant_id'] = tenant_id

        return supabase.table('bpm_proc_inst').upsert(process_instance_data).execute()
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def set_participants_from_workitems(process_instance, tenant_id=None):
    """
    proc_inst_id???대떦?섎뒗 todolist??user_id?ㅼ쓣 ?뚯떛?섏뿬 participants瑜??명똿
    """
    workitems = fetch_todolist_by_proc_inst_id(process_instance.proc_inst_id, tenant_id)
    user_ids = []
    if workitems:
        for workitem in workitems:
            if workitem.user_id:
                ids = [uid.strip() for uid in workitem.user_id.split(',') if uid.strip()]
                user_ids.extend(ids)
    participants = []
    seen = set()
    for user_id in user_ids:
        if (
            user_id is not None
            and user_id != ''
            and user_id != 'undefined'
            and user_id.strip() != ''
        ):
            if user_id == 'external_customer':
                if user_id not in seen:
                    participants.append(user_id)
                    seen.add(user_id)
            else:
                assignee_info = fetch_assignee_info(user_id)
                if assignee_info['type'] not in ['unknown', 'error'] and user_id not in seen:
                    participants.append(user_id)
                    seen.add(user_id)
    process_instance.participants = participants
    return process_instance


def upsert_process_instance(process_instance: ProcessInstance, tenant_id: Optional[str] = None, definition: Optional[ProcessDefinition] = None) -> (bool, ProcessInstance):
    process_definition = process_instance.process_definition
    if process_definition is None:
        process_definition = load_process_definition(fetch_process_definition(process_instance.get_def_id(), tenant_id))
        process_instance.process_definition = process_definition
        
    if definition is not None:
        process_definition = definition

    end_activities = process_definition.find_end_activities()

    status = None
    has_active_activities = bool(process_instance.current_activity_ids)
    if end_activities:
        if has_active_activities:
            status = 'RUNNING'
        else:
            end_done = False
            for _end_activity in end_activities:
                end_workitem = fetch_workitem_by_proc_inst_and_activity(process_instance.proc_inst_id, safeget(_end_activity, 'id', ''), tenant_id)
                if end_workitem and end_workitem.status == 'DONE':
                    end_done = True
                    break
            status = 'COMPLETED' if end_done else 'RUNNING'
    else:
        if has_active_activities:
            status = 'RUNNING'
    
    # Set participants from workitems
    process_instance = set_participants_from_workitems(process_instance, tenant_id)
    participants = process_instance.participants

    process_instance_data = process_instance.dict(exclude={'process_definition'})  # Convert Pydantic model to dict
    process_instance_data = convert_decimal(process_instance_data)

    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        if not tenant_id:
            tenant_id = subdomain_var.get()
            
        process_definition_version = fetch_process_definition_latest_version(process_instance.get_def_id(), tenant_id)
        if process_definition_version:
            arcv_id = process_definition_version.get('arcv_id', None)
        else:
            arcv_id = None
        
        response = supabase.table('bpm_proc_inst').upsert({
            'proc_inst_id': process_instance.proc_inst_id,
            'proc_inst_name': process_instance.proc_inst_name,
            'current_activity_ids': process_instance.current_activity_ids,
            'participants': participants,
            'role_bindings': process_instance.role_bindings,
            'variables_data': process_instance.variables_data,
            'status': status if status else process_instance.status,
            'proc_def_id': process_instance.get_def_id(),
            'proc_def_version': arcv_id,
            'tenant_id': tenant_id,
            'end_date': datetime.now(pytz.timezone('Asia/Seoul')).isoformat() if status == 'COMPLETED' else None
        }).execute()

        success = bool(response.data)

        return success, process_instance

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def convert_decimal(data):
    for key, value in data.items():
        if isinstance(value, Decimal):
            data[key] = float(value)
    return data


def fetch_organization_chart(tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain


        response = supabase.table("configuration").select("*").eq('key', 'organization').eq('tenant_id', tenant_id).execute()
        
        # Check if the response contains data
        if response.data:
            # Assuming the first match is the desired one since ID should be unique
            value = response.data[0].get('value', None)
            organization_chart = value.get('chart', None)
            return organization_chart
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch organization chart: {e}")


def fetch_todolist_by_proc_inst_id(proc_inst_id: str, tenant_id: Optional[str] = None) -> Optional[List[WorkItem]]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain

        response = supabase.table('todolist').select("*").eq('proc_inst_id', proc_inst_id).eq('tenant_id', tenant_id).execute()
        

        if response.data:
            return [WorkItem(**item) for item in response.data]
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def fetch_workitem_by_proc_inst_and_activity(
    proc_inst_id: str, 
    activity_id: str, 
    tenant_id: Optional[str] = None, 
    use_ilike:bool = False,
    recent_only: Optional[bool] = True
) -> Optional[WorkItem]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain

        if use_ilike:
            response = supabase.table('todolist').select("*").eq('proc_inst_id', proc_inst_id).ilike('activity_id', activity_id).eq('tenant_id', tenant_id).execute()
        else:
            response = supabase.table('todolist').select("*").eq('proc_inst_id', proc_inst_id).eq('activity_id', activity_id).eq('tenant_id', tenant_id).execute()
            
        
        if response.data:
            if len(response.data) > 1 and recent_only:
                # updated_at??媛??理쒓렐?닿굅?? updated_at??媛숈쑝硫?rework_count媛 媛??????ぉ??理쒓렐 ?뚰겕?꾩씠?쒖쑝濡?媛꾩＜
                def get_recent_key(item):
                    updated_at = item.get('updated_at')
                    rework_count = item.get('rework_count', 0)
                    
                    if updated_at:
                        try:
                            if isinstance(updated_at, str):
                                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00')).replace(tzinfo=None)
                            elif hasattr(updated_at, 'replace'):
                                updated_at = updated_at.replace(tzinfo=None)
                        except (ValueError, TypeError, AttributeError):
                            updated_at = None
                    
                    return (updated_at or datetime.min, rework_count)
                
                most_recent_item = max(response.data, key=get_recent_key)
                return WorkItem(**most_recent_item)
            elif len(response.data) > 1 and not recent_only:
                return WorkItem(**response.data[0])
            elif len(response.data) == 1:
                return WorkItem(**response.data[0])
            else:
                return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    
def fetch_workitems_by_root_proc_inst_id(root_proc_inst_id: str, tenant_id: Optional[str] = None) -> Optional[List[WorkItem]]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
            
            
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain
            
        response = supabase.table('todolist').select("*").eq('root_proc_inst_id', root_proc_inst_id).eq('tenant_id', tenant_id).execute()
        
        if response.data:
            return [WorkItem(**item) for item in response.data]
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def fetch_workitems_by_proc_inst_id(proc_inst_id: str, tenant_id: Optional[str] = None) -> Optional[List[WorkItem]]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
            
            
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain
            
        response = supabase.table('todolist').select("*").eq('proc_inst_id', proc_inst_id).eq('tenant_id', tenant_id).execute()
        
        if response.data:
            return [WorkItem(**item) for item in response.data]
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    

def fetch_workitem_by_id(workitem_id: str, tenant_id: Optional[str] = None) -> Optional[WorkItem]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
            
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain
            
        response = supabase.table('todolist').select("*").eq('id', workitem_id).eq('tenant_id', tenant_id).execute()
        
        if response.data and len(response.data) > 0:
            return WorkItem(**response.data[0])
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

def fetch_workitem_with_submitted_status(limit=10) -> Optional[List[dict]]:
    try:
        pod_id = socket.gethostname()
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        # Supabase Client API瑜??ъ슜?섏뿬 ?뚰겕?꾩씠??議고쉶 諛??낅뜲?댄듃
        # 癒쇱? SUBMITTED ?곹깭?닿퀬 consumer媛 NULL???뚰겕?꾩씠?쒕뱾??議고쉶
        # ?뚮꼳???섍꼍蹂?遺꾧린 ?놁씠 ???뚮꼳??怨듯넻 議곌굔?쇰줈 議고쉶
        q = supabase.table('todolist').select('*').eq('status', 'SUBMITTED')
        if CONSUMER_FILTER:
            q = q.eq('consumer', CONSUMER_FILTER)
        else:
            q = q.is_('consumer', 'null')
        response = q.limit(limit).execute()
        
        if not response.data:
            return None
        
        # 議고쉶???뚰겕?꾩씠?쒕뱾??consumer瑜??꾩옱 pod_id濡??낅뜲?댄듃
        # ?숈떆???쒖뼱瑜??꾪빐 議곌굔遺 ?낅뜲?댄듃 ?ъ슜
        updated_workitems = []
        
        # 諛곗튂 ?낅뜲?댄듃瑜??꾪븳 ?뚰겕?꾩씠??ID 紐⑸줉
        workitem_ids = [item['id'] for item in response.data]
        
        if workitem_ids:
            try:
                # 諛곗튂 ?낅뜲?댄듃 ?쒕룄
                current_time = datetime.now().isoformat()
                q_upd = supabase.table('todolist').update({
                    'consumer': pod_id,
                    'updated_at': current_time
                }).in_('id', workitem_ids).eq('status', 'SUBMITTED')
                if CONSUMER_FILTER:
                    q_upd = q_upd.eq('consumer', CONSUMER_FILTER)
                else:
                    q_upd = q_upd.is_('consumer', 'null')
                batch_update_response = q_upd.execute()
                
                if batch_update_response.data:
                    updated_workitems = batch_update_response.data
                    print(f"[DEBUG] Successfully claimed {len(updated_workitems)} workitems for pod {pod_id}")
                else:
                    print(f"[DEBUG] No workitems were claimed in batch update")
                    
            except Exception as batch_error:
                print(f"[WARNING] Batch update failed, falling back to individual updates: {batch_error}")
                
                # 諛곗튂 ?낅뜲?댄듃媛 ?ㅽ뙣?섎㈃ 媛쒕퀎 ?낅뜲?댄듃濡??대갚
                for workitem in response.data:
                    try:
                        q_one = supabase.table('todolist').update({
                            'consumer': pod_id,
                            'updated_at': datetime.now().isoformat()
                        }).eq('id', workitem['id']).eq('status', 'SUBMITTED')
                        if CONSUMER_FILTER:
                            q_one = q_one.eq('consumer', CONSUMER_FILTER)
                        else:
                            q_one = q_one.is_('consumer', 'null')
                        update_response = q_one.execute()
                        
                        if update_response.data:
                            updated_workitems.append(update_response.data[0])
                            print(f"[DEBUG] Successfully claimed workitem {workitem['id']} for pod {pod_id}")
                        else:
                            print(f"[DEBUG] Workitem {workitem['id']} was already claimed by another pod")
                    except Exception as e:
                        print(f"[WARNING] Failed to update workitem {workitem['id']}: {e}")
                        continue

        return updated_workitems if updated_workitems else None

    except Exception as e:
        print(f"[ERROR] DB fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DB fetch failed: {str(e)}") from e


    
def fetch_workitem_with_pending_status(limit=5) -> Optional[List[dict]]:
    try:
        pod_id = socket.gethostname()
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        # PENDING ?곹깭?닿퀬 consumer媛 NULL???뚰겕?꾩씠?쒕뱾??議고쉶
        # ?뚮꼳???섍꼍蹂?遺꾧린 ?놁씠 ???뚮꼳??怨듯넻 議곌굔?쇰줈 議고쉶
        q = supabase.table('todolist').select('*').eq('status', 'PENDING')
        if CONSUMER_FILTER:
            q = q.eq('consumer', CONSUMER_FILTER)
        else:
            q = q.is_('consumer', 'null')
        response = q.limit(limit).execute()
        
        
        if not response.data:
            return None
        
        return response.data
    except Exception as e:
        print(f"[ERROR] DB fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DB fetch failed: {str(e)}") from e
    


def cleanup_stale_consumers():
    """
    ?ㅻ옒??consumer瑜??뺣━?섎뒗 ?⑥닔
    30遺??댁긽 ?낅뜲?댄듃?섏? ?딆? SUBMITTED ?곹깭???뚰겕?꾩씠?쒖쓽 consumer瑜??댁젣
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        # 30遺????쒓컙 怨꾩궛
        thirty_minutes_ago = (datetime.now() - timedelta(minutes=30)).isoformat()
        
        # ?ㅻ옒??consumer瑜?NULL濡??낅뜲?댄듃
        q = supabase.table('todolist').update({
            'consumer': None
        }).eq('status', 'SUBMITTED').not_.is_('consumer', 'null').lt('updated_at', thirty_minutes_ago)
        # CONSUMER_FILTER??NULL 痍④툒 ???뺣━ ??곸뿉???쒖쇅
        if CONSUMER_FILTER:
            q = q.neq('consumer', CONSUMER_FILTER)
        response = q.execute()
        
        if response.data:
            updated_count = len(response.data)
            print(f"[INFO] Cleaned up {updated_count} stale consumers")
        else:
            print("[INFO] No stale consumers found")

    except Exception as e:
        print(f"[ERROR] Failed to cleanup stale consumers: {str(e)}")

def upsert_workitem_completed_log(completed_workitems: List[WorkItem], process_result_data: dict, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        process_instance_id = None
        if completed_workitems:
            for completed_workitem in completed_workitems:
                if process_instance_id is None:
                    process_instance_id = completed_workitem.proc_inst_id
                user_info = fetch_assignee_info(completed_workitem.user_id)
                ui_definition = fetch_ui_definition_by_activity_id(completed_workitem.proc_def_id, completed_workitem.activity_id, tenant_id)
                form_html = ui_definition.html if ui_definition else None
                form_id = ui_definition.id if ui_definition else None
                if completed_workitem.output:
                    output = completed_workitem.output.get(form_id)
                else:
                    output = {}
                message_data = {
                    "role": "system" if user_info.get("name") == "external_customer" else "user",
                    "name": user_info.get("name"),
                    "email": user_info.get("email"),
                    "profile": user_info.get("info", {}).get("profile", ""),
                    "content": "",
                    "jsonContent": output if output else {},
                    "htmlContent": form_html if form_html else "",
                    "contentType": "html" if form_html else "text",
                    "activityId": completed_workitem.activity_id,
                    "workitemId": completed_workitem.id
                }
                upsert_chat_message(completed_workitem.proc_inst_id, message_data, tenant_id)

            description = {
                "completedActivities": process_result_data.get("completedActivities", []),
                "nextActivities": process_result_data.get("nextActivities", [])
            }
            message_json = json.dumps({
                "role": "system",
                "contentType": "json",
                "jsonContent": description
            })
            
            if process_instance_id:
                upsert_chat_message(process_instance_id, message_json, tenant_id)

    except Exception as e:
        print(f"[ERROR] upsert_workitem_completed_log: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from e

def upsert_completed_workitem(process_instance_data, process_result_data, process_definition, tenant_id: Optional[str] = None) -> List[WorkItem]:
    try:
        if not tenant_id:
            tenant_id = subdomain_var.get()

        workitems = []
        if not process_result_data['completedActivities']:
            return
        
        
        scope_name = ''
        if process_instance_data['execution_scope']:
            execution_scope = process_instance_data['execution_scope']
            scope_name =  f": ({process_instance_data.get('proc_inst_name', '')})"
        else:
            execution_scope =''
        
        for completed_activity in process_result_data['completedActivities']:
            proc_inst_id = process_instance_data['proc_inst_id']
            completed_id = completed_activity['completedActivityId']

            workitem = fetch_workitem_by_proc_inst_and_activity(proc_inst_id, completed_id, tenant_id)
            
            if workitem:
                workitem.status = completed_activity['result']
                workitem.end_date = datetime.now(pytz.timezone('Asia/Seoul'))
                user_info = fetch_assignee_info(completed_activity['completedUserEmail'])
                if user_info:
                    workitem.user_id = user_info.get('id')
                    workitem.username = user_info.get('name')
                if workitem.assignees and len(workitem.assignees) > 0:
                    for assignee in workitem.assignees:
                        if assignee.get('endpoint') and assignee.get('endpoint') == workitem.user_id:
                            assignee = {
                                'roleName': assignee.get('name'),
                                'userId': assignee.get('endpoint')
                            }
                            break
                # completed_activity is dict (not Pydantic model), use .get() instead of safeget()
                cannotProceedErrors = completed_activity.get('cannotProceedErrors', [])
                if  cannotProceedErrors and len(cannotProceedErrors) > 0:
                    workitem.log = "\n".join(f"[{error.get('type', '')}] {error.get('reason', '')}" for error in cannotProceedErrors);
            else:
                # If we get here, we're about to CREATE a new completed workitem row.
                # This is acceptable for tasks that never had a todolist row (e.g., some internal/script tasks),
                # but for user tasks it often indicates a matching bug. Log enough context to debug from container logs.
                try:
                    existing = fetch_todolist_by_proc_inst_id(proc_inst_id, tenant_id) or []
                    summary = [
                        {
                            "id": getattr(wi, "id", None),
                            "activity_id": getattr(wi, "activity_id", None),
                            "status": getattr(wi, "status", None),
                        }
                        for wi in existing
                    ]
                    print(
                        "[WARN] upsert_completed_workitem: no existing workitem found; creating new row "
                        f"tenant_id={tenant_id} proc_inst_id={proc_inst_id} completedActivityId={completed_id} "
                        f"existing_count={len(existing)} existing={summary}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[WARN] upsert_completed_workitem: failed to log existing workitems "
                        f"tenant_id={tenant_id} proc_inst_id={proc_inst_id} completedActivityId={completed_id} err={e}",
                        flush=True,
                    )

                activity = process_definition.find_activity_by_id(completed_activity['completedActivityId'])
                start_date = datetime.now(pytz.timezone('Asia/Seoul'))
                due_date = start_date + timedelta(days=safeget(activity, 'duration', 0)) if safeget(activity, 'duration', 0) else None
                assignees = []
                user_id = None
                if process_instance_data['role_bindings']:
                    role_bindings = process_instance_data['role_bindings']
                    for role_binding in role_bindings:
                        if role_binding['name'] == safeget(activity, 'role', ''):
                            user_id = ','.join(role_binding['endpoint']) if isinstance(role_binding['endpoint'], list) else role_binding['endpoint']
                            assignees.append(role_binding)
                
                user_info = None
                if completed_activity.get('completedUserEmail') and completed_activity['completedUserEmail'] != user_id:
                    user_info = fetch_assignee_info(completed_activity['completedUserEmail'])

                agent_orch = safeget(activity, 'orchestration', None)
                if agent_orch == 'none':
                    agent_orch = None
                
                log = ''
                # completed_activity is dict (not Pydantic model), use .get() instead of safeget()
                cannotProceedErrors = completed_activity.get('cannotProceedErrors', [])    
                if  cannotProceedErrors and len(cannotProceedErrors) > 0:
                    log = "\n".join(f"[{error.get('type', '')}] {error.get('reason', '')}" for error in cannotProceedErrors);
                
                if workitem and workitem.query:
                    query = workitem.query
                else:
                    query = ''
                    description = safeget(activity, 'description', '')
                    instruction = safeget(activity, 'instruction', '')
                    if description:
                        query += f"[Description]\n{description}\n\n"
                    if instruction:
                        query += f"[Instruction]\n{instruction}\n\n"
                
                workitem = WorkItem(
                    id=f"{str(uuid.uuid4())}",
                    proc_inst_id=process_instance_data['proc_inst_id'],
                    proc_def_id=process_result_data['processDefinitionId'].lower(),
                    activity_id=completed_activity['completedActivityId'],
                    activity_name= f"{safeget(activity, 'name', '')}{scope_name}",
                    user_id=user_info.get('id'),
                    username=user_info.get('name'),
                    status=completed_activity['result'],
                    tool=safeget(activity, 'tool', ''),
                    start_date=start_date,
                    end_date=datetime.now(pytz.timezone('Asia/Seoul')) if completed_activity['result'] == 'DONE' else None,
                    due_date=due_date,
                    tenant_id=tenant_id,
                    assignees=assignees,
                    duration=safeget(activity, 'duration', 0),
                    description=description,
                    query=query,
                    agent_orch=agent_orch,
                    agent_mode=safeget(activity, 'agentMode', None),
                    log=log,
                    root_proc_inst_id=process_instance_data.get('root_proc_inst_id') or process_instance_data.get('proc_inst_id'),
                    execution_scope=execution_scope,
                    version_tag=getattr(process_definition, "version_tag", None),
                    version=getattr(process_definition, "version", None),
                )
            
            
            workitem_dict = workitem.model_dump()
            workitem_dict["start_date"] = workitem.start_date.isoformat() if workitem.start_date else None
            workitem_dict["end_date"] = workitem.end_date.isoformat() if workitem.end_date else None
            workitem_dict["due_date"] = workitem.due_date.isoformat() if workitem.due_date else None
            
            process_result_data.setdefault('cancelledActivities', [])
            activity = process_definition.find_activity_by_id(completed_activity['completedActivityId'])
            if activity:
                attached_events = safeget(activity, 'attachedEvents', [])
                if attached_events:
                    for attached_event in attached_events:
                        if attached_event != completed_activity['completedActivityId']:
                            process_result_data['cancelledActivities'].append({
                                'cancelledActivityId': attached_event,
                                'cancelledUserEmail': workitem.user_id,
                                'result': 'CANCELLED'
                            })
                        
            attached_activity = process_definition.find_attached_activity(completed_activity['completedActivityId'])
            if attached_activity:
                process_result_data['cancelledActivities'].append({
                                'cancelledActivityId': safeget(attached_activity, 'id', ''),
                                'cancelledUserEmail': workitem.user_id,
                                'result': 'CANCELLED'
                            })
                attached_events = safeget(attached_activity, 'attachedEvents', [])
                if attached_events:
                    for attached_event in attached_events:
                        if attached_event != completed_activity['completedActivityId']:
                            process_result_data['cancelledActivities'].append({
                                'cancelledActivityId': attached_event,
                                'cancelledUserEmail': workitem.user_id,
                                'result': 'CANCELLED'
                            })

            supabase = supabase_client_var.get()
            if supabase is None:
                raise Exception("Supabase client is not configured for this request")
            
            workitems.append(workitem)
            
            upsert_workitem_completed_log(workitems, process_result_data, tenant_id)
            ensure_minimum_task_due_date(workitem_dict, process_instance_data.get("start_date"))
            supabase.table('todolist').upsert(workitem_dict).execute()
            
        return workitems
    except Exception as e:
        print(f"[ERROR] upsert_completed_workitem: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from e

def upsert_cancelled_workitem(process_instance_data, process_result_data, process_definition, tenant_id: Optional[str] = None) -> List[WorkItem]:
    try:
        workitems = []
        
       
        scope_name = ''
        if process_instance_data['execution_scope']:
            execution_scope = process_instance_data['execution_scope']
            scope_name =  f": ({process_instance_data.get('proc_inst_name', '')})"
        else:
            execution_scope =''
            
        for cancelled_activity in process_result_data['cancelledActivities']:
            workitem = fetch_workitem_by_proc_inst_and_activity(
                process_instance_data['proc_inst_id'],
                cancelled_activity['cancelledActivityId'],
                tenant_id
            )
            if workitem:
                workitem.status = cancelled_activity['result']
                workitem.end_date = datetime.now(pytz.timezone('Asia/Seoul'))
                workitem.user_id = cancelled_activity['cancelledUserEmail']
                if workitem.assignees and len(workitem.assignees) > 0:
                    for assignee in workitem.assignees:
                        if assignee.get('endpoint') and assignee.get('endpoint') == workitem.user_id:
                            assignee = {
                                'roleName': assignee.get('name'),
                                'userId': assignee.get('endpoint')
                            }
                            break
            else:
                activity = process_definition.find_activity_by_id(cancelled_activity['cancelledActivityId'])
                start_date = datetime.now(pytz.timezone('Asia/Seoul'))
                due_date = start_date + timedelta(days=safeget(activity, 'duration', 0)) if safeget(activity, 'duration', 0) else None
                assignees = []
                if process_instance_data['role_bindings']:
                    role_bindings = process_instance_data['role_bindings']
                    for role_binding in role_bindings:
                        if role_binding['name'] == safeget(activity, 'role', ''):
                            user_id = ','.join(role_binding['endpoint']) if isinstance(role_binding['endpoint'], list) else role_binding['endpoint']
                            assignees.append(role_binding)
                
                if cancelled_activity['cancelledUserEmail'] != user_id:
                    user_id = cancelled_activity['cancelledUserEmail']
                agent_orch = safeget(activity, 'orchestration', None)
                if agent_orch == 'none':
                    agent_orch = None
                
                if workitem and workitem.query:
                    query = workitem.query
                else:
                    query = ''
                    description = safeget(activity, 'description', '')
                    instruction = safeget(activity, 'instruction', '')
                    if description:
                        query += f"[Description]\n{description}\n\n"
                    if instruction:
                        query += f"[Instruction]\n{instruction}\n\n"
                
                workitem = WorkItem(
                    id=f"{str(uuid.uuid4())}",
                    proc_inst_id=process_instance_data['proc_inst_id'],
                    proc_def_id=process_result_data['processDefinitionId'].lower(),
                    activity_id=cancelled_activity['cancelledActivityId'],
                    activity_name= f"{safeget(activity, 'name', '')}{scope_name}",
                    user_id=user_id,
                    status="CANCELLED",
                    tool=safeget(activity, 'tool', ''),
                    start_date=start_date,
                    end_date=datetime.now(pytz.timezone('Asia/Seoul')),
                    due_date=due_date,
                    tenant_id=tenant_id,
                    assignees=assignees,
                    duration=safeget(activity, 'duration', 0),
                    description=description,
                    query=query,
                    agent_orch=agent_orch,
                    agent_mode=safeget(activity, 'agentMode', None),
                    root_proc_inst_id=process_instance_data.get('root_proc_inst_id') or process_instance_data.get('proc_inst_id'),
                    execution_scope=execution_scope,
                    version_tag=getattr(process_definition, "version_tag", None),
                    version=getattr(process_definition, "version", None),
                )
                
            workitem_dict = workitem.model_dump()
            workitem_dict["start_date"] = workitem.start_date.isoformat() if workitem.start_date else None
            workitem_dict["end_date"] = workitem.end_date.isoformat() if workitem.end_date else None
            workitem_dict["due_date"] = workitem.due_date.isoformat() if workitem.due_date else None
            
            supabase = supabase_client_var.get()
            if supabase is None:
                raise Exception("Supabase client is not configured for this request")
            ensure_minimum_task_due_date(workitem_dict, process_instance_data.get("start_date"))
            supabase.table('todolist').upsert(workitem_dict).execute()
            workitems.append(workitem)
        return workitems
            
    except Exception as e:
        print(f"[ERROR] upsert_cancelled_workitem: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from e
def safeget(obj, attr, default=None):
    return getattr(obj, attr, default)

def upsert_next_workitems(process_instance_data, process_result_data, process_definition, tenant_id: Optional[str] = None) -> List[WorkItem]:
    workitems = []
    if not tenant_id:
        tenant_id = subdomain_var.get()

    
    scope_name = ''
    if process_instance_data['execution_scope']:
        execution_scope = process_instance_data['execution_scope']
        scope_name =  f": ({process_instance_data.get('proc_inst_name', '')})"
    else:
        execution_scope =''
        
    def _is_end_event_node(node_id: str, act_data: dict = None) -> bool:
        if node_id in ("END_PROCESS", "endEvent", "end_event"):
            return True
        if act_data:
            act_type = str(act_data.get("type") or "").lower()
            if act_type == "endevent":
                return True
        if process_definition:
            gw = process_definition.find_gateway_by_id(node_id) if node_id else None
            if gw and "endevent" in str(getattr(gw, "type", "") or "").lower():
                return True
        return False

    for activity_data in process_result_data['nextActivities']:
        if _is_end_event_node(activity_data['nextActivityId'], activity_data):
            continue

        # nextUserEmail??LLM/roleBindings?먯꽌 寃곗젙??理쒖쥌 ?대떦???앸퀎??(?대찓???먮뒗 id/UUID)
        next_user_email = activity_data.get('nextUserEmail')

        workitem = fetch_workitem_by_proc_inst_and_activity(process_instance_data['proc_inst_id'], activity_data['nextActivityId'], tenant_id)

        if workitem:
            workitem.status = activity_data['result']
            workitem.end_date = datetime.now(pytz.timezone('Asia/Seoul')) if activity_data['result'] == 'DONE' else None

            # 湲곗〈 workitem??user_id媛 鍮꾩뼱 ?덇퀬(nextUserEmail留??덈뒗) 寃쎌슦, ?ш린??user_id/username??蹂닿컯
            if (not getattr(workitem, 'user_id', None)) and next_user_email:
                try:
                    user_info = fetch_assignee_info(next_user_email)
                    new_user_id = (user_info.get('id') if isinstance(user_info, dict) else None) or next_user_email
                except Exception:
                    new_user_id = next_user_email

                workitem.user_id = new_user_id

                # 理쒖쥌 user_id 湲곗??쇰줈 username ?ш퀎??
                username = ''
                if new_user_id:
                    if ',' in new_user_id:
                        usernames = []
                        user_ids = new_user_id.split(',')
                        for id in user_ids:
                            user_info_item = fetch_assignee_info(id.strip())
                            if user_info_item:
                                usernames.append(user_info_item.get('name', ''))
                        username = ','.join([name for name in usernames if name])
                    else:
                        user_info_item = fetch_assignee_info(new_user_id.strip())
                        if user_info_item:
                            username = user_info_item.get('name', '')
                workitem.username = username

            if workitem.agent_mode == None:
                workitem.agent_mode = determine_agent_mode(workitem.user_id, workitem.agent_mode)
                if workitem.agent_mode == 'COMPLETE' and (workitem.agent_orch == 'none' or workitem.agent_orch == None):
                    workitem.agent_orch = 'crewai-deep-research'
            
            # ?낅젰 ?곗씠??異붽? (?뚯씪 ?뚯떛 ?ы븿)
            try:
                # ?대깽??猷⑦봽 ?곹깭? 臾닿??섍쾶 ?덉쟾?섍쾶 鍮꾨룞湲??⑥닔 ?ㅽ뻾
                input_data = run_async_in_sync_context(
                    get_input_data_with_file_parsing(workitem.model_dump(), process_definition)
                )
            except Exception as e:
                print(f"[WARNING] ?뚯씪 ?뚯떛 以??ㅻ쪟 諛쒖깮, 湲곕낯 諛⑹떇?쇰줈 ?꾪솚: {str(e)}")
                input_data = get_input_data(workitem.model_dump(), process_definition)
            
            if input_data:
                try:
                    input_data_str = json.dumps(input_data, ensure_ascii=False)
                except Exception:
                    input_data_str = str(input_data)
                query = workitem.query
                if query and '[InputData]' in query:
                    query = query.split('[InputData]')[0] + f"[InputData]\n{input_data_str}"
                else:
                    query = f"{query}[InputData]\n{input_data_str}"
                workitem.query = query
            # print(f"[DEBUG] workitem.agent_mode: {workitem.agent_mode}")
        else:
            activity = process_definition.find_activity_by_id(activity_data['nextActivityId'])
            is_event_node = False
            if not activity:
                activity = process_definition.find_event_by_id(activity_data['nextActivityId'])
                is_event_node = activity is not None
            if activity:
                prev_activities = []
                if not is_event_node:
                    prev_activities = process_definition.find_prev_activities(safeget(activity, 'id', ''), [])
                start_date = datetime.now(pytz.timezone('Asia/Seoul'))
                
                # reference_ids ?ㅼ젙 (?댁쟾 ?≫떚鍮꾪떚 ID 紐⑸줉)
                reference_ids = []
                if prev_activities:
                    reference_ids = fetch_prev_task_ids(process_definition, safeget(activity, 'id', ''), process_instance_data['proc_inst_id'])
                    for prev_activity in prev_activities:
                        start_date = start_date + timedelta(days=safeget(prev_activity, 'duration', 0))
                
                due_date = start_date + timedelta(days=safeget(activity, 'duration', 0)) if safeget(activity, 'duration', 0) else None
                agent_mode = determine_agent_mode(next_user_email, safeget(activity, 'agentMode', None))
                agent_orch = safeget(activity, 'orchestration', None)
                if agent_orch is None or agent_orch == 'none' or agent_orch == 'None' or agent_orch == '':
                    agent_orch = None
                if agent_mode == 'COMPLETE' and (safeget(activity, 'orchestration', None) == 'none' or safeget(activity, 'orchestration', None) == None):
                    agent_orch = 'crewai-deep-research'
                
                user_id = None
                if next_user_email:
                    try:
                        user_info = fetch_assignee_info(next_user_email)
                        # fetch_assignee_info???뺤긽??????긽 id瑜?梨꾩슦吏留?
                        # ?대뼡 寃쎌슦?먮룄 next_user_email ?먯껜瑜?理쒖쥌 ?대갚?쇰줈 ?ъ슜
                        user_id = (user_info.get('id') if isinstance(user_info, dict) else None) or next_user_email
                    except Exception:
                        # fetch_assignee_info ?ㅽ뙣 ?쒖뿉??next_user_email??洹몃?濡?user_id濡??ъ슜
                        user_id = next_user_email

                
                # assignees 珥덇린????process_result_data.roleBindings瑜?癒쇱? ?뺤씤?섍퀬,
                # 留ㅼ묶???놁쑝硫?process_instance_data.role_bindings瑜?fallback?쇰줈 ?ъ슜
                assignees = []
                role_name = safeget(activity, 'role', '')
                for rb_source in [process_result_data.get('roleBindings'), process_instance_data.get('role_bindings')]:
                    if not rb_source:
                        continue
                    for role_binding in rb_source:
                        if isinstance(role_binding, dict) and role_binding.get('name') == role_name:
                            assignees.append(role_binding)
                    if assignees:
                        break
                
                # activity.agent媛 ?덉쑝硫?user_id??異붽? (以묐났 泥댄겕, ?곗꽑?쒖쐞 ?믪쓬)
                if safeget(activity, 'agent', None) is not None and safeget(activity, 'agent', None) != "":
                    agent_id = safeget(activity, 'agent', None)
                    
                    # 湲곗〈 user_id? 議곗씤 (以묐났 泥댄겕)
                    if user_id:
                        user_ids = [uid.strip() for uid in user_id.split(',') if uid.strip()]
                        if agent_id not in user_ids:
                            user_ids.insert(0, agent_id)  # activity.agent瑜?留??욎뿉 異붽? (?곗꽑?쒖쐞)
                        user_id = ','.join(user_ids)
                    else:
                        user_id = agent_id
                    
                    # assignees?먯꽌 activity.role怨??대쫫??媛숈? role_binding??endpoint瑜??뺤옣
                    role_name = safeget(activity, 'role', '')
                    updated_role_binding = False
                    for assignee in assignees:
                        if assignee.get('name') != role_name:
                            continue
                        assignee_endpoint = assignee.get('endpoint')
                        # endpoint媛 由ъ뒪?몄씤 寃쎌슦: agent_id媛 ?놁쑝硫?append
                        if isinstance(assignee_endpoint, list):
                            if agent_id not in assignee_endpoint:
                                assignee_endpoint.append(agent_id)
                                assignee['endpoint'] = assignee_endpoint
                        # endpoint媛 臾몄옄???⑥씪 媛믪씤 寃쎌슦
                        elif isinstance(assignee_endpoint, str) and assignee_endpoint.strip() != "":
                            if assignee_endpoint != agent_id:
                                assignee['endpoint'] = [assignee_endpoint, agent_id]
                        # endpoint媛 ?녾굅??鍮?寃쎌슦
                        else:
                            assignee['endpoint'] = agent_id
                        updated_role_binding = True
                        break
                    
                    # ?숈씪 role??role_binding???놁쑝硫??덈줈 ?섎굹 異붽?
                    if not updated_role_binding:
                        assignees.append({
                            "name": role_name,
                            "endpoint": agent_id
                        })
                
                # 理쒖쥌 user_id 湲곗??쇰줈 username ?ш퀎??
                username = ''
                if user_id:
                    if ',' in user_id:
                        usernames = []
                        user_ids = user_id.split(',')
                        for id in user_ids:
                            user_info_item = fetch_assignee_info(id.strip())
                            if user_info_item:
                                usernames.append(user_info_item.get('name', ''))
                        username = ','.join([name for name in usernames if name])
                    else:
                        user_info = fetch_assignee_info(user_id.strip())
                        if user_info:
                            username = user_info.get('name', '')
                
                if workitem and workitem.query:
                    query = workitem.query
                else:
                    query = ''
                    description = safeget(activity, 'description', '')
                    instruction = safeget(activity, 'instruction', '')
                    if description:
                        query += f"[Description]\n{description}\n\n"
                    if instruction:
                        query += f"[Instruction]\n{instruction}\n\n"
                
                if agent_mode is not None and agent_mode != "none" and agent_mode != "None" and agent_mode != "":
                    agent_mode = agent_mode.upper()
                else:
                    agent_mode = None
                
                workitem = WorkItem(
                    id=str(uuid.uuid4()),
                    reference_ids=reference_ids if prev_activities else [],
                    proc_inst_id=process_instance_data['proc_inst_id'],
                    proc_def_id=process_result_data['processDefinitionId'].lower(),
                    activity_id=safeget(activity, 'id', ''),
                    activity_name= f"{safeget(activity, 'name', '')}{scope_name}",
                    user_id=user_id,
                    username=username,
                    status=activity_data['result'],
                    start_date=start_date,
                    due_date=due_date,
                    tool=safeget(activity, 'tool', ''),
                    tenant_id=tenant_id,
                    assignees=assignees if assignees else [],
                    agent_mode=agent_mode,
                    description=description,
                    query=query,
                    agent_orch=agent_orch,
                    root_proc_inst_id=process_instance_data.get('root_proc_inst_id') or process_instance_data.get('proc_inst_id'),
                    execution_scope=execution_scope,
                    version_tag=getattr(process_definition, "version_tag", None),
                    version=getattr(process_definition, "version", None),
                )
                
                # ?덈줈 ?앹꽦??workitem?먮룄 ?낅젰 ?곗씠??異붽? (?뚯씪 ?뚯떛 ?ы븿)
                try:
                    # ?대깽??猷⑦봽 ?곹깭? 臾닿??섍쾶 ?덉쟾?섍쾶 鍮꾨룞湲??⑥닔 ?ㅽ뻾
                    input_data = run_async_in_sync_context(
                        get_input_data_with_file_parsing(workitem.model_dump(), process_definition)
                    )
                except Exception as e:
                    print(f"[WARNING] ?뚯씪 ?뚯떛 以??ㅻ쪟 諛쒖깮, 湲곕낯 諛⑹떇?쇰줈 ?꾪솚: {str(e)}")
                    input_data = get_input_data(workitem.model_dump(), process_definition)
                
                print(f"[DEBUG] input_data: {input_data}")

                if input_data:
                    try:
                        input_data_str = json.dumps(input_data, ensure_ascii=False)
                    except Exception:
                        input_data_str = str(input_data)
                    
                    print(f"[DEBUG] input_data_str: {input_data_str}")
                    if workitem.query and '[InputData]' in workitem.query:
                        workitem.query = workitem.query.split('[InputData]')[0] + f"[InputData]\n{input_data_str}"
                    else:
                        workitem.query = f"{workitem.query}\n[InputData]\n{input_data_str}"
                    
                    print(f"[INFO] ??workitem???낅젰 ?곗씠??異붽? ?꾨즺 (?뚯씪 ?뚯떛 ?ы븿): {workitem.id}")
        
        try:
            if workitem:
                workitem_dict = workitem.model_dump()
                workitem_dict["start_date"] = workitem.start_date.isoformat() if workitem.start_date else None
                workitem_dict["end_date"] = workitem.end_date.isoformat() if workitem.end_date else None
                workitem_dict["due_date"] = workitem.due_date.isoformat() if workitem.due_date else None

                # TEMP: 肄쒕큸 ?뚯뒪?몄퐫?쒕뱶 ?ㅼ쓬 ?낅Т ?앹꽦 ???뱀젙 ?대떦??frcp9408@gmail.com)?먭쾶 釉뚮씪?곗? 肄??몃━嫄?
                try:
                    target_email = "frcp9408@gmail.com"
                    trigger_url = "https://monitor-faithful-slightly.ngrok-free.app/call/client"
                    trigger_identity = "browser-user"
                    status_val = (workitem_dict.get("status") or "").upper()
                    assignee_id = workitem_dict.get("user_id")

                    print(f"[TwilioTrigger][next] status={status_val} assignee_id={assignee_id} activity={workitem.activity_id}")

                    if status_val in ("IN_PROGRESS", "TODO", "NEW") and isinstance(assignee_id, str):
                        assignee_email = ""
                        try:
                            user_row = fetch_user_info(assignee_id)
                            assignee_email = (user_row.get("email") or "").lower()
                            print(f"[TwilioTrigger][next] resolved via user lookup -> email={assignee_email}")
                        except Exception as e:
                            print(f"[TwilioTrigger][next] user lookup failed ({assignee_id}): {e}")
                            assignee_info = fetch_assignee_info(assignee_id) or {}
                            assignee_email = (assignee_info.get("email") or assignee_id or "").lower()
                            print(f"[TwilioTrigger][next] resolved via email fallback -> email={assignee_email}")

                        if assignee_email == target_email and trigger_url:
                            try:
                                resp = requests.post(trigger_url, json={"identity": trigger_identity}, timeout=5)
                                resp.raise_for_status()
                                print(f"[TwilioTrigger][next] fired for {assignee_email} -> {trigger_url}")
                            except Exception as exc:
                                print(f"[TwilioTrigger][next] Failed trigger for {assignee_email}: {exc}")
                        else:
                            print(f"[TwilioTrigger][next] skipped: email mismatch or no trigger_url (email={assignee_email})")
                    else:
                        print(f"[TwilioTrigger][next] skipped: status={status_val}, assignee={assignee_id}")
                except Exception as exc:
                    print(f"[TwilioTrigger][next] Skipped trigger logic: {exc}")

                # browser-automation-agent??寃쎌슦 ?곸꽭??description ?앹꽦
                # if workitem.agent_orch == 'browser-automation-agent':
                #     print(f"[DEBUG] Generating browser automation description for workitem: {workitem.id}")
                #     try:
                #         updated_query = _generate_browser_automation_description(
                #             process_instance_data, workitem.id, tenant_id
                #         )
                #         if updated_query and updated_query != workitem.query:
                #             workitem_dict["query"] = updated_query
                #     except Exception as e:
                #         print(f"[ERROR] Failed to generate browser automation description: {str(e)}")

                supabase = supabase_client_var.get()
                if supabase is None:
                    raise Exception("Supabase client is not configured for this request")
                ensure_minimum_task_due_date(workitem_dict, process_instance_data.get("start_date"))
                supabase.table('todolist').upsert(workitem_dict).execute()
                workitems.append(workitem)
        except Exception as e:
            print(f"[ERROR] upsert_next_workitems: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e)) from e


    return workitems


def fetch_prev_task_ids(process_definition, current_activity_id: str, proc_inst_id: str) -> List[str]:
    """
    ?꾩옱 ?뚯뒪?ъ쓽 ?쒗???뺣낫瑜??댁슜??諛붾줈 吏곸쟾 ?뚯뒪?ъ쓽 ID 紐⑸줉??諛섑솚?⑸땲??
    
    Args:
        process_definition: ?꾨줈?몄뒪 ?뺤쓽 媛앹껜
        current_activity_id: ?꾩옱 ?뚯뒪?ъ쓽 ID
        proc_inst_id: ?꾨줈?몄뒪 ?몄뒪?댁뒪 ID
    
    Returns:
        List[str]: 吏곸쟾 ?뚯뒪?ъ쓽 activity ID 紐⑸줉
    """
    prev_task_ids = []
    prev_activities = process_definition.find_immediate_prev_activities(current_activity_id)
    
    if prev_activities:
        # ?댁쟾 ?≫떚鍮꾪떚?ㅼ쓽 activity_id瑜??섏쭛
        for prev_activity in prev_activities:
            prev_task_ids.append(prev_activity.id)
    
    return prev_task_ids


def upsert_todo_workitems(process_instance_data, process_result_data, process_definition, tenant_id: Optional[str] = None):
    try:
        if not tenant_id:
            tenant_id = subdomain_var.get()


        initial_activity = next((activity for activity in process_definition.activities if process_definition.is_starting_activity(activity.id)), None)
        if not initial_activity:
            initial_activity = process_definition.find_initial_activity()
        
        scope_name = ''
        if process_instance_data['execution_scope']:
            execution_scope = process_instance_data['execution_scope']
            scope_name =  f": ({process_instance_data.get('proc_inst_name', '')})"
        else:
            execution_scope =''

        next_activities = process_definition.find_next_activities(initial_activity.id, True)
        for activity in next_activities:
            if safeget(activity, 'type', '') == 'endEvent':
                continue
            if safeget(activity, 'type', '') == 'adHocSubProcess':
                continue
            
            prev_activities = process_definition.find_prev_activities(activity.id, [])
            start_date = datetime.now(pytz.timezone('Asia/Seoul'))
        
            if prev_activities:
                # ?숈씪??srcTrg瑜?媛吏??≫떚鍮꾪떚??以?duration??媛????寃껊쭔 ?④린湲?
                srcTrg_groups = {}
                for prev_activity in prev_activities:
                    if prev_activity.srcTrg not in srcTrg_groups:
                        srcTrg_groups[prev_activity.srcTrg] = []
                    srcTrg_groups[prev_activity.srcTrg].append(prev_activity)
                # duration??媛?????≫떚鍮꾪떚留??좏깮
                filtered_activities = []
                for activities in srcTrg_groups.values():
                    max_duration_activity = max(activities, key=lambda x: x.duration if x.duration is not None else 0)
                    filtered_activities.append(max_duration_activity)
                
                reference_ids = fetch_prev_task_ids(process_definition, safeget(activity, 'id', ''), process_instance_data['proc_inst_id'])
                
                for prev_activity in filtered_activities:
                    # duration ?ㅺ? ?덉뼱??媛믪씠 None ?????덉뼱(?앹꽦???뺤쓽??duration 誘몄??? timedelta(days=None) ?щ옒??諛⑹?.
                    start_date = start_date + timedelta(days=(safeget(prev_activity, 'duration', 0) or 0))

            due_date = start_date + timedelta(days=(safeget(activity, 'duration', 0) or 0)) if safeget(activity, 'duration', 0) else None
            workitem = fetch_workitem_by_proc_inst_and_activity(process_instance_data['proc_inst_id'], safeget(activity, 'id', ''), tenant_id)
            if not workitem:
                user_id = ""
                assignees = []
                if process_result_data['roleBindings']:
                    role_bindings = process_result_data['roleBindings']
                    for role_binding in role_bindings:
                        if role_binding['name'] == safeget(activity, 'role', ''):
                            user_id = ','.join(role_binding['endpoint']) if isinstance(role_binding['endpoint'], list) else role_binding['endpoint']
                            assignees.append(role_binding)
                
                agent_mode = determine_agent_mode(user_id, safeget(activity, 'agentMode', None))
                agent_orch = safeget(activity, 'orchestration', None)
                if agent_orch is None or agent_orch == 'none' or agent_orch == 'None' or agent_orch == '':
                    agent_orch = None
                if agent_mode == 'COMPLETE' and (safeget(activity, 'orchestration', None) == 'none' or safeget(activity, 'orchestration', None) == None):
                    agent_orch = 'crewai-deep-research'
                    
                if agent_mode is not None and agent_mode != "none" and agent_mode != "None" and agent_mode != "":
                    agent_mode = agent_mode.upper()
                else:
                    agent_mode = None

                status = "TODO"
                
                if workitem and workitem.query:
                    query = workitem.query
                else:
                    query = ''
                    description = safeget(activity, 'description', '')
                    instruction = safeget(activity, 'instruction', '')
                    if description:
                        query += f"[Description]\n{description}\n\n"
                    if instruction:
                        query += f"[Instruction]\n{instruction}\n\n"

                # tool 寃곗젙: activity type??'task'媛 ?ы븿?섍퀬 tool??鍮꾩뼱?덉쑝硫?'defaultForm' ?ъ슜
                activity_tool = safeget(activity, 'tool', '')
                activity_type = safeget(activity, 'type', '').lower()
                if not activity_tool and 'task' in activity_type:
                    activity_tool = 'formHandler:defaultForm'
                    
                # activity.agent媛 ?덉쑝硫?user_id??異붽? (以묐났 泥댄겕, ?곗꽑?쒖쐞 ?믪쓬)
                if safeget(activity, 'agent', None) is not None and safeget(activity, 'agent', None) != "":
                    agent_id = safeget(activity, 'agent', None)
                    
                    # 湲곗〈 user_id? 議곗씤 (以묐났 泥댄겕)
                    if user_id:
                        user_ids = [uid.strip() for uid in user_id.split(',') if uid.strip()]
                        if agent_id not in user_ids:
                            user_ids.insert(0, agent_id)  # activity.agent瑜?留??욎뿉 異붽? (?곗꽑?쒖쐞)
                        user_id = ','.join(user_ids)
                    else:
                        user_id = agent_id
                    
                    # assignees?먯꽌 activity.role怨??대쫫??媛숈? role_binding??endpoint瑜??뺤옣
                    role_name = safeget(activity, 'role', '')
                    updated_role_binding = False
                    for assignee in assignees:
                        if assignee.get('name') != role_name:
                            continue
                        assignee_endpoint = assignee.get('endpoint')
                        # endpoint媛 由ъ뒪?몄씤 寃쎌슦: agent_id媛 ?놁쑝硫?append
                        if isinstance(assignee_endpoint, list):
                            if agent_id not in assignee_endpoint:
                                assignee_endpoint.append(agent_id)
                                assignee['endpoint'] = assignee_endpoint
                        # endpoint媛 臾몄옄???⑥씪 媛믪씤 寃쎌슦
                        elif isinstance(assignee_endpoint, str) and assignee_endpoint.strip() != "":
                            if assignee_endpoint != agent_id:
                                assignee['endpoint'] = [assignee_endpoint, agent_id]
                        # endpoint媛 ?녾굅??鍮?寃쎌슦
                        else:
                            assignee['endpoint'] = agent_id
                        updated_role_binding = True
                        break
                    
                    # ?숈씪 role??role_binding???놁쑝硫??덈줈 ?섎굹 異붽?
                    if not updated_role_binding:
                        assignees.append({
                            "name": role_name,
                            "endpoint": agent_id
                        })
                
                # 理쒖쥌 user_id 湲곗??쇰줈 username ?ш퀎??
                username = ''
                if user_id:
                    if ',' in user_id:
                        usernames = []
                        user_ids = user_id.split(',')
                        for id in user_ids:
                            user_info = fetch_assignee_info(id.strip())
                            if user_info:
                                usernames.append(user_info.get('name', ''))
                        username = ','.join([name for name in usernames if name])
                    else:
                        user_info = fetch_assignee_info(user_id.strip())
                        if user_info:
                            username = user_info.get('name', '')

                workitem = WorkItem(
                    id=f"{str(uuid.uuid4())}",
                    reference_ids=reference_ids if prev_activities else [],
                    proc_inst_id=process_instance_data['proc_inst_id'],
                    proc_def_id=process_result_data['processDefinitionId'].lower(),
                    activity_id=safeget(activity, 'id', ''),
                    activity_name= f"{safeget(activity, 'name', '')}{scope_name}",
                    user_id=user_id,
                    username=username,
                    status=status,
                    tool=activity_tool,
                    start_date=start_date,
                    due_date=due_date,
                    tenant_id=tenant_id,
                    assignees=assignees if assignees else [],
                    duration=safeget(activity, 'duration', 0),
                    agent_mode=agent_mode,
                    description=description,
                    query=query,
                    agent_orch=agent_orch,
                    root_proc_inst_id=process_instance_data.get('root_proc_inst_id') or process_instance_data.get('proc_inst_id'),
                    execution_scope=execution_scope,
                    version_tag=getattr(process_definition, "version_tag", None),
                    version=getattr(process_definition, "version", None),
                )
                workitem_dict = workitem.model_dump()
                workitem_dict["start_date"] = workitem.start_date.isoformat() if workitem.start_date else None
                workitem_dict["end_date"] = workitem.end_date.isoformat() if workitem.end_date else None
                workitem_dict["due_date"] = workitem.due_date.isoformat() if workitem.due_date else None

                supabase = supabase_client_var.get()
                if supabase is None:
                    raise Exception("Supabase client is not configured for this request")
                ensure_minimum_task_due_date(workitem_dict, process_instance_data.get("start_date"))
                supabase.table('todolist').upsert(workitem_dict).execute()
    except Exception as e:
        print(f"[ERROR] upsert_todo_workitems: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from e


def _generate_browser_automation_description(
    process_instance_data: dict, 
    current_workitem_id, 
    tenant_id: str
) -> str:
    """
    browser-automation-agent???곸꽭??description???앹꽦?⑸땲??
    """
    try:
        # ?댁쟾 workitem?ㅼ쓣 媛?몄????ъ슜???붿껌?ы빆怨??꾨줈?몄뒪 ?먮쫫 ?뚯븙
        all_workitems = fetch_workitems_by_proc_inst_id(process_instance_data['proc_inst_id'], tenant_id)

        form_data = fetch_ui_definition_by_activity_id(process_instance_data['proc_def_id'], process_instance_data['current_activity_ids'][0], tenant_id)
        
        # ?댁쟾, ?꾩옱, ?댄썑 workitem ?뺣낫 遺꾩꽍 (status 湲곕컲)
        done_workitems = []
        current_workitem = None
        next_workitems = []
        
        if all_workitems:
            for workitem in all_workitems:
                workitem_info = {
                    "activity_name": workitem.activity_name,
                    "description": workitem.description,
                    "status": workitem.status,
                    "output": workitem.output,
                    "activity_id": workitem.activity_id
                }
                
                if workitem.status in ['DONE', 'COMPLETED', 'SUBMITTED']:
                    done_workitems.append(workitem_info)
                elif workitem.id == current_workitem_id:
                    current_workitem = workitem_info
                else:
                    next_workitems.append(workitem_info)
        
        # LLM???ъ슜?섏뿬 ?곸꽭??description ?앹꽦
        prompt_template = """
?뱀떊? browser-automation-agent(browser-use)媛 ??釉뚮씪?곗?瑜??듯빐 ?묒뾽???섑뻾?????덈룄濡??곸꽭???④퀎蹂??ㅻ챸???앹꽦?섎뒗 AI?낅땲??

=== ?꾩옱 ?묒뾽 ===
{current_workitem}

=== ?꾩옱 ?묒뾽??寃곌낵濡??낅젰?섏뼱?쇳븷(湲곕??섎뒗 寃곌낵媛? ?낅젰 ???곗씠?곗엯?덈떎. ?????곗씠?곕? 梨꾩썙?ｌ쓣 ???덈뒗 寃곌낵瑜??산린 ?꾪븳 ?곸꽭???ㅻ챸???앹꽦?섏꽭?? ===
{form_data}

=== ?댁쟾 ?묒뾽??===
{done_workitems}

=== ?댄썑 ?묒뾽??===
{next_workitems}

=== 遺꾩꽍 ?붽뎄?ы빆 ===
1. ?댁쟾 ?묒뾽?먯꽌 ?ъ슜?먭? ?낅젰??援ъ껜?곸씤 ?댁슜怨?吏移⑥쓣 ?뚯븙?섏꽭??
2. ?댄썑 ?묒뾽?먯꽌 ?대뼡 寃곌낵臾쇱씠 ?꾩슂?쒖? ?뚯븙?섏꽭??
3. ?꾩옱 ?묒뾽???꾩껜 ?꾨줈?몄뒪?먯꽌 ?대뼡 ??븷???섎뒗吏 ?댄빐?섏꽭??
4. ?댄썑 ?묒뾽??URL ?쒓났?대굹 ?뚯씪 ?ㅼ슫濡쒕뱶?쇰㈃, ?꾩옱 ?묒뾽?먯꽌 ?대떦 寃곌낵臾쇱쓣 ?살뼱?대뒗 ?④퀎瑜??ы븿?섏꽭??

=== ?앹꽦 ?붽뎄?ы빆 ===
- browser-use媛 ??釉뚮씪?곗?瑜??듯빐 ?섑뻾?????덈뒗 援ъ껜?곸씤 ?④퀎蹂??ㅻ챸???앹꽦?섏꽭??
- 媛??④퀎???ㅽ뻾 媛?ν븯怨?紐낇솗?댁빞 ?⑸땲??
- ?댁쟾 ?묒뾽???낅젰 ?댁슜???쒖슜?섏꽭??
- ?댄썑 ?묒뾽???꾩슂??寃곌낵臾쇱쓣 ?앹꽦?섎뒗 ?④퀎瑜??ы븿?섏꽭??
- ?붽뎄?ы빆 諛??ъ슜?먯쓽 ?낅젰, 吏移⑥쓣 諛섏쁺?섏뿬 ?앹꽦?섎릺, 遺덊븘?뷀븳 ?ㅻ챸? ?ы븿?섏? 留덉꽭?? ?덈? ?ㅼ뼱 濡쒓렇???붽뎄媛 ?녿떎硫?濡쒓렇???ㅻ챸? ?ы븿?섏? 留덉꽭??
- ?ъ슜?먭? ?댁쟾???낅젰???낅젰媛믪씠 ?덈떎硫?洹?媛믪쓣 ?쒖슜?섏뿬 ?앹꽦?섏꽭?? ?낅젰媛믪씠 ?곸꽭?섍쾶 ?묒꽦?섏뼱 ?덈떎硫?議곌툑??蹂댁셿留뚰븯???ъ슜?섎릺, ?낅젰媛믪씠 紐낇솗?섏? ?딅떎硫??낅젰媛믪쓣 ?쒖슜?섏뿬 ?ㅻ챸???곸꽭?섍쾶 ?묒꽦?섏꽭??

?뺤떇:
1. [?④퀎紐?: [援ъ껜?곸씤 ?섑뻾 諛⑸쾿]
2. [?④퀎紐?: [援ъ껜?곸씤 ?섑뻾 諛⑸쾿]
...

?덉떆: 
- ?낅젰媛? (?묒냽二쇱냼 ?쒓났)?ш린???뚯씪 ?ㅼ슫濡쒕뱶 ?댁쨾
- 吏移? ?뱀젙 ?ъ씠?몄뿉 ?묒냽?섏뿬 ?뚯씪???ㅼ슫濡쒕뱶

??寃쎌슦 ?ㅻ챸:
1. ?뱀젙 ?ъ씠???묒냽二쇱냼)???묒냽
2. ?묒냽???쒖떆?섎뒗 ?앹뾽李??뺤씤 ???앹뾽李??リ린
3. ?ㅼ슫濡쒕뱶 ?뱀뀡??李얠븘 ?대룞
4. ?ㅼ슫濡쒕뱶 踰꾪듉 ?대┃
5. ?뚯씪 ?ㅼ슫濡쒕뱶 ?대┃ ???ㅼ슫濡쒕뱶 ???뚯씪 ?뺤씤
6. 醫낅즺

?낅젰媛믪씠 蹂대떎 ?곸꽭??寃쎌슦:
- ?낅젰媛? 
1. https://www.g2b.go.kr/ 二쇱냼???섎씪?ν꽣 ?묒냽 ???쒖떆?섎뒗 ?앹뾽???덈떎硫??앹뾽 紐⑤몢 ?リ린
2. ?곷떒 寃???뗫낫湲??꾩씠肄??대┃ ??"ai" 寃??
3. 議곕떖?뺣낫 寃?됯껐怨?以??낆같 怨듦퀬??"?붾낫湲?媛 ?꾨땶 ?ъ뾽紐?"CMS ?낅Т DX ?뚮옯???대?吏遺꾩꽍 AI 援ъ텞 異붿쭊 愿??IT?먯썝 ?꾩엯")???대┃?섏뿬 ?곸꽭 ?섏씠吏濡??대룞
4. ?섎떒?쇰줈 ?ㅽ겕濡ㅽ븯??"?뚯씪泥⑤?" ?뱀뀡??李얘퀬 ?쒖씪 ?곷떒???덈뒗 ?뚯씪紐??덉떆: 	?쒖븞?붿껌??CMS ?낅Т DX ?뚮옯???대?吏遺꾩꽍 AI 援ъ텞 異붿쭊 愿??IT?먯썝 ?꾩엯).hwpx) ?대┃
5. 醫낅즺
- 吏移? ?뱀젙 ?ъ씠?몄뿉 ?묒냽?섏뿬 ?뚯씪???ㅼ슫濡쒕뱶

??寃쎌슦 ?ㅻ챸:
1. https://www.g2b.go.kr/ 二쇱냼???섎씪?ν꽣 ?묒냽
2. ?묒냽???쒖떆?섎뒗 ?앹뾽李??뺤씤 ???앹뾽李??リ린
3. ?곷떒 寃???뗫낫湲??꾩씠肄??대┃ ??"ai" 寃??
4. 議곕떖?뺣낫 寃?됯껐怨?以??낆같 怨듦퀬??"?붾낫湲?媛 ?꾨땶 ?ъ뾽紐?"CMS ?낅Т DX ?뚮옯???대?吏遺꾩꽍 AI 援ъ텞 異붿쭊 愿??IT?먯썝 ?꾩엯")???대┃?섏뿬 ?곸꽭 ?섏씠吏濡??대룞
5. ?섎떒?쇰줈 ?ㅽ겕濡ㅽ븯??"?뚯씪泥⑤?" ?뱀뀡??李얘퀬 ?쒖씪 ?곷떒???덈뒗 ?뚯씪紐??덉떆: ?쒖븞?붿껌??CMS ?낅Т DX ?뚮옯???대?吏遺꾩꽍 AI 援ъ텞 異붿쭊 愿??IT?먯썝 ?꾩엯).hwpx) ?대┃
6. ?뚯씪 ?ㅼ슫濡쒕뱶 ?대┃ ???ㅼ슫濡쒕뱶 ???뚯씪 ?뺤씤
7. 醫낅즺

???덉떆泥섎읆 ?낅젰媛믨낵 吏移⑥쓣 諛섏쁺?섏뿬 ?곸꽭???④퀎蹂??ㅻ챸(query)???앹꽦?댁＜?몄슂.
?뚯씪 ?ㅼ슫濡쒕뱶 諛??숈옉 寃곌낵???꾩옱 ?④퀎???낅젰 ???뺤떇??留욊쾶 ?앹꽦?섎씪怨?紐낆떆?섏뿬?쇳븿. 
?앹꽦??寃곌낵瑜??낅젰 ?쇱뿉 梨꾩썙?ｋ뒗 ?숈옉? browser-use 醫낅즺 ?댄썑???섑뻾?섎룄濡?援ы쁽?섏뼱 ?덇린?뚮Ц???쇰???以꾩씠湲??꾪빐 ?낅젰 ?쇱쓣 梨꾩썙?ｌ쑝?쇰뒗 ?ㅻ챸? ?ы븿?섏? 留덉꽭??

遺덊븘?뷀븳 ?ㅻ챸??理쒖냼?뷀븯怨??ъ슜?먯쓽 ?낅젰, 吏移? ?꾩옱 諛??댁쟾, ?댄썑 紐⑤뱺 activity??紐⑹쟻, ?섎룄瑜??뚯븙?섏뿬 
理쒕????ъ슜?먯쓽 紐⑹쟻??留욌뒗 ?숈옉???섑뻾?쒗궗 ???덈뒗 ?곸꽭???④퀎蹂??ㅻ챸(query)???앹꽦?댁＜?몄슂:
"""

        # print(f"[DEBUG] current_workitem: {current_workitem}")
        # print(f"[DEBUG] done_workitems: {done_workitems}")
        # print(f"[DEBUG] next_workitems: {next_workitems}")
        # print(f"[DEBUG] form_data: {form_data.fields_json}")
        # print(f"[DEBUG] str form_data: {str(form_data.fields_json)}")


        prompt = prompt_template.format(
            current_workitem=current_workitem,
            done_workitems=done_workitems,
            next_workitems=next_workitems,
            form_data=str(form_data.fields_json) if form_data and form_data.fields_json else "NULL"
        )
        
        # LLM ?몄텧
        model = create_llm(streaming=True, temperature=0)
        response = model.invoke(prompt)
        
        # ?묐떟?먯꽌 ?④퀎蹂??ㅻ챸 異붿텧
        if hasattr(response, 'content'):
            description = response.content
        else:
            description = str(response)
        
        return description.strip()
        
    except Exception as e:
        print(f"[ERROR] Failed to generate browser automation description: {str(e)}")
        # 湲곕낯 description 諛섑솚
        return None


def upsert_workitem(workitem_data: dict, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        ensure_minimum_task_due_date(workitem_data)
        if "start_date" in workitem_data and workitem_data["start_date"]:
            if not isinstance(workitem_data["start_date"], str):
                workitem_data["start_date"] = workitem_data["start_date"].isoformat()
        if "end_date" in workitem_data and workitem_data["end_date"]:
            if not isinstance(workitem_data["end_date"], str):
                workitem_data["end_date"] = workitem_data["end_date"].isoformat()
        if "due_date" in workitem_data and workitem_data["due_date"]:
            if not isinstance(workitem_data["due_date"], str):
                workitem_data["due_date"] = workitem_data["due_date"].isoformat()

        if not tenant_id:
            tenant_id = subdomain_var.get()
        workitem_data["tenant_id"] = tenant_id

        return supabase.table('todolist').upsert(workitem_data).execute()
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def delete_workitem(workitem_id: str, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        if not tenant_id:
            tenant_id = subdomain_var.get()


        supabase.table('todolist').delete().eq('id', workitem_id).eq('tenant_id', tenant_id).execute()
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
def upsert_chat_message(chat_room_id: str, data: Any, tenant_id: Optional[str] = None) -> None:
    """
    梨꾪똿 硫붿떆吏瑜?upsert?섎뒗 ?⑥닔
    
    Args:
        chat_room_id: 梨꾪똿諛?ID
        data: 硫붿떆吏 ?곗씠??(dict ?먮뒗 str) - role ?꾨뱶 ?ы븿
        tenant_id: ?뚮꼳??ID
    """
    try:
        current_timestamp = int(datetime.now(pytz.timezone('Asia/Seoul')).timestamp() * 1000)
        
        # data媛 臾몄옄?댁씤 寃쎌슦 JSON?쇰줈 ?뚯떛
        if isinstance(data, str):
            message_data = json.loads(data)
        else:
            message_data = data
        
        # role???놁쑝硫?湲곕낯媛??ㅼ젙
        if "role" not in message_data:
            message_data["role"] = "system"
        
        # timestamp媛 ?놁쑝硫?異붽?
        if "timeStamp" not in message_data:
            message_data["timeStamp"] = current_timestamp

        if not tenant_id:
            tenant_id = subdomain_var.get()

        # 梨꾪똿 ?꾩씠???곗씠??援ъ꽦
        chat_item_data = {
            "id": chat_room_id,
            "uuid": str(uuid.uuid4()),
            "messages": message_data,
            "tenant_id": tenant_id
        }

        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")

        supabase.table("chats").upsert(chat_item_data).execute()
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

def fetch_user_info(email: str) -> Dict[str, str]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.table("users").select("*").eq('email', email).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        else:
            response = supabase.table("users").select("*").eq('id', email).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            else:
                raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def fetch_assignee_info(assignee_id: str) -> Dict[str, str]:
    """
    ?대떦???뺣낫瑜?李얜뒗 ?⑥닔
    ?대떦?먭? ?좎??몄? ?먯씠?꾪듃?몄? ?먮떒?섍퀬 ?곸젅???뺣낫瑜?諛섑솚?⑸땲??
    
    Args:
        assignee_id: ?대떦??ID (?대찓???먮뒗 ?먯씠?꾪듃 ID)
    
    Returns:
        ?대떦???뺣낫 ?뺤뀛?덈━
    """
    try:
        try:
            user_info = fetch_user_info(assignee_id)
            type = "user"
            if user_info.get("is_agent") == True:
                type = "agent"
            return {
                "type": type,
                "id": user_info.get("id", assignee_id),
                "name": user_info.get("username", assignee_id),
                "email": user_info.get("email", assignee_id),
                "info": user_info
            }
        except HTTPException as user_error:
            if user_error.status_code == 500 or user_error.status_code == 404:
                return {
                    "type": "unknown",
                    "id": assignee_id,
                    "name": assignee_id,
                    "email": assignee_id,
                    "info": {}
                }
            else:
                raise user_error
    except Exception as e:
        return {
            "type": "error",
            "id": assignee_id,
            "name": assignee_id,
            "email": assignee_id,
            "info": {},
            "error": str(e)
        }


def determine_agent_mode(user_id: str, agent_mode: Optional[str] = None) -> Optional[str]:
    """
    ?ъ슜??ID? ?≫떚鍮꾪떚???먯씠?꾪듃 紐⑤뱶瑜?湲곕컲?쇰줈 ?곸젅???먯씠?꾪듃 紐⑤뱶瑜?寃곗젙?⑸땲??
    
    Args:
        user_id: ?ъ슜??ID (?쇳몴濡?援щ텇???щ윭 ID 媛??
        agent_mode: ?≫떚鍮꾪떚?먯꽌 ?ㅼ젙???먯씠?꾪듃 紐⑤뱶
    
    Returns:
        寃곗젙???먯씠?꾪듃 紐⑤뱶 (None, "DRAFT", "COMPLETE")
    """
    # ?≫떚鍮꾪떚?먯꽌 紐낆떆?곸쑝濡??먯씠?꾪듃 紐⑤뱶媛 ?ㅼ젙??寃쎌슦
    if agent_mode is not None:
        if agent_mode.lower() not in ["none", "null"]:
            mode = agent_mode.upper()
            return mode

    # user_id媛 ?놁쑝硫?None 諛섑솚
    if not user_id:
        return None
    
    # ?щ윭 ?ъ슜??ID媛 ?덈뒗 寃쎌슦
    if ',' in user_id:
        user_ids = user_id.split(',')
        has_user = False
        has_agent = False
        
        for user_id in user_ids:
            assignee_info = fetch_assignee_info(user_id)
            if assignee_info['type'] == "user":
                has_user = True
            elif assignee_info['type'] == "agent":
                has_agent = True
        
        # ?ъ슜???먯씠?꾪듃 議고빀?대㈃ DRAFT
        if has_user and has_agent:
            return "DRAFT"
        # ?먯씠?꾪듃留??덉쑝硫?COMPLETE
        elif has_agent and not has_user:
            return "COMPLETE"
        # ?ъ슜?먮쭔 ?덉쑝硫?None
        elif has_user and not has_agent:
            return None
    
    # ?⑥씪 ?ъ슜??ID??寃쎌슦
    else:
        assignee_info = fetch_assignee_info(user_id)
        if assignee_info['type'] == "agent":
            return "COMPLETE"
        elif assignee_info['type'] == "user":
            return None
    
    return None


def get_vector_store():
    supabase = supabase_client_var.get()
    if supabase is None:
        raise Exception("Supabase client is not configured")
    
    embeddings = create_embedding()
    
    return SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )


def fetch_tenant_mcp_config(tenant_id: str) -> Optional[Dict[str, Any]]:
    """
    ?뚮꼳?몄쓽 MCP ?ㅼ젙??議고쉶?⑸땲??
    
    Args:
        tenant_id (str): ?뚮꼳??ID
        
    Returns:
        Optional[Dict[str, Any]]: MCP ?ㅼ젙 ?뺣낫 ?먮뒗 None
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.table('tenants').select('mcp').eq('id', tenant_id).execute()
        
        if response.data and len(response.data) > 0:
            mcp_config = response.data[0].get('mcp', {})
            return mcp_config if mcp_config else None
        else:
            print(f"[WARNING] No tenant found with ID: {tenant_id}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to fetch tenant MCP config: {str(e)}")
        return None


def get_field_value(field_info: str, process_definition: Any, process_instance_id: str, tenant_id: str):
    """
    ?곗텧臾쇱뿉???뱀젙 ?꾨뱶??媛믪쓣 異붿텧 (援ъ“ 蹂寃?理쒖냼??
    - (1) ?꾩옱 ?몄뒪?댁뒪 ?⑥씪 議고쉶 ??媛??덉쑝硫??⑥씪媛믪쑝濡?諛섑솚
    - (2) 猷⑦듃 ?몄뒪?댁뒪 ?⑥씪 議고쉶(+洹몃９ ?몃뜳?? ??媛??덉쑝硫??⑥씪媛믪쑝濡?諛섑솚
    - (3) 猷⑦듃 湲곗? ?ㅺ굔 議고쉶(fetch_workitems_by_root_proc_inst_id)
         ???꾨? 諛곗뿴濡?紐⑥븘 { form_id: { field_id: ["<scope>:<value>", ...] } } ?뺥깭濡?諛섑솚
    """
    try:
        field_value: Dict[str, Any] = {}

        process_definition_id = process_definition.processDefinitionId
        split_field_info = field_info.split('.')
        form_id = split_field_info[0]
        field_id = split_field_info[1]

        # ---- form_id -> activity_id 留ㅽ븨 (?ㅻ떒怨? ----
        # 1) activity.tool 吏곸젒 鍮꾧탳
        # 2) activity.tool ?먯꽌 'formHandler:' prefix ?쒓굅 ??鍮꾧탳
        # 3) form_id ??'_form' ?묐? ?쒓굅 + PD ID prefix ?쒓굅 ???앸?遺??좏겙 留ㅼ묶
        # 4) 理쒗썑 fallback: 媛숈? ?몄뒪?댁뒪 紐⑤뱺 ?뚰겕?꾩씠???쒗쉶?섏뿬 output ?덉뿉 form_id ?ㅺ? ?덈뒗吏 ?뺤씤
        def _norm_tool(t: Any) -> str:
            try:
                s = str(t or "").strip()
            except Exception:
                s = ""
            if s.startswith("formHandler:"):
                s = s[len("formHandler:"):]
            return s

        activities = list(getattr(process_definition, "activities", None) or [])
        activity_id: Optional[str] = None

        activity_id = next((a.id for a in activities if getattr(a, "tool", None) == form_id), None)

        if not activity_id:
            activity_id = next((a.id for a in activities if _norm_tool(getattr(a, "tool", None)) == form_id), None)

        if not activity_id:
            base = form_id[:-len("_form")] if form_id.endswith("_form") else form_id
            if process_definition_id and base.startswith(f"{process_definition_id}_"):
                base = base[len(process_definition_id) + 1:]
            for a in activities:
                aid = getattr(a, "id", None)
                if not aid:
                    continue
                if base == aid or base.endswith(f"_{aid}"):
                    activity_id = aid
                    break
            if not activity_id and base:
                last_token = base.split("_")[-1]
                if any(getattr(a, "id", None) == last_token for a in activities):
                    activity_id = last_token

        def _out(wi: Any) -> Optional[dict]:
            return getattr(wi, "output", None) or (wi.get("output") if isinstance(wi, dict) else None)

        def _val_from_form(out: dict) -> Optional[Any]:
            form = out.get(form_id)
            if isinstance(form, dict):
                return form.get(field_id)
            return None

        def _to_int(v: Any, default: int = 0) -> int:
            try:
                s = str(v).strip()
                return int(s) if s != "" else default
            except Exception:
                return default

        def _ci_equal(a: Optional[str], b: Optional[str]) -> bool:
            return (a or "").lower() == (b or "").lower()

        # activity_id 紐?李얠? 寃쎌슦 - 媛숈? ?몄뒪?댁뒪 紐⑤뱺 ?뚰겕?꾩씠?쒖뿉??output[form_id] 吏곸젒 ?먯깋
        if not activity_id:
            try:
                wi_all = fetch_workitems_by_proc_inst_id(process_instance_id, tenant_id) or []
            except Exception:
                wi_all = []
            for wi in wi_all:
                out = _out(wi)
                if isinstance(out, dict) and isinstance(out.get(form_id), dict):
                    val = _val_from_form(out)
                    if val is not None:
                        field_value[form_id] = { field_id: val }
                        return field_value
            return None

        # (1) ?꾩옱 ?몄뒪?댁뒪 ?⑥씪 議고쉶
        workitem = fetch_workitem_by_proc_inst_and_activity(process_instance_id, activity_id, tenant_id, True)
        if workitem:
            out = _out(workitem)
            if out:
                val = _val_from_form(out)
                if val is not None:
                    field_value[form_id] = { field_id: val }
                    return field_value
        else:
            # activity_id 留ㅽ븨? ?먯?留??뚰겕?꾩씠?쒖씠 洹?ID 濡????≫엺 耳?댁뒪 -> output-key ?ㅼ틪
            try:
                wi_all = fetch_workitems_by_proc_inst_id(process_instance_id, tenant_id) or []
            except Exception:
                wi_all = []
            for wi in wi_all:
                out = _out(wi)
                if isinstance(out, dict) and isinstance(out.get(form_id), dict):
                    val = _val_from_form(out)
                    if val is not None:
                        field_value[form_id] = { field_id: val }
                        return field_value

        # ?몄뒪?댁뒪 ?뺣낫
        instance = fetch_process_instance(process_instance_id, tenant_id)
        root_proc_inst_id = getattr(instance, "root_proc_inst_id", None) or (instance.get("root_proc_inst_id") if isinstance(instance, dict) else None)
        exec_scope_raw = getattr(instance, "execution_scope", None) or (instance.get("execution_scope") if isinstance(instance, dict) else None)
        exec_scope = _to_int(exec_scope_raw, 0)

        # (2) 猷⑦듃 ?몄뒪?댁뒪 ?⑥씪 議고쉶(+洹몃９ ?몃뜳??
        workitem_root = fetch_workitem_by_proc_inst_and_activity(root_proc_inst_id, activity_id, tenant_id, True)
        if workitem_root:
            out = _out(workitem_root)
            if out:
                val = _val_from_form(out)
                if val is not None:
                    field_value[form_id] = { field_id: val }
                    return field_value
                form = out.get(form_id)
                if isinstance(form, dict):
                    for _, item_value in form.items():
                        try:
                            candidate = item_value[exec_scope][field_id]
                            if candidate is not None:
                                field_value[form_id] = { field_id: candidate }
                                return field_value
                        except Exception:
                            pass

        workitems = fetch_workitems_by_root_proc_inst_id(root_proc_inst_id, tenant_id)
        if not workitems:
            return None

        filtered: List[Any] = []
        for wi in workitems:
            wi_act = getattr(wi, "activity_id", None) or (wi.get("activity_id") if isinstance(wi, dict) else None)
            if _ci_equal(wi_act, activity_id):
                filtered.append(wi)
        if not filtered:
            # 理쒗썑 fallback: activity_id 留ㅼ묶 ?ㅽ뙣?대룄 output ?덉뿉 form_id ?ㅺ? ?덈뒗 ?뚰겕?꾩씠??吏곸젒 ?먯깋
            for wi in workitems:
                out = _out(wi)
                if isinstance(out, dict) and isinstance(out.get(form_id), dict):
                    val = _val_from_form(out)
                    if val is not None:
                        field_value[form_id] = { field_id: val }
                        return field_value
            return None

        def _sort_key(wi: Any):
            scope = _to_int(getattr(wi, "execution_scope", None) or (wi.get("execution_scope") if isinstance(wi, dict) else None), 10**9)
            missing = 1 if scope == 10**9 else 0
            return (missing, scope)

        filtered.sort(key=_sort_key)

        values: List[str] = []
        for wi in filtered:
            out = _out(wi)
            if not out:
                continue
            val = _val_from_form(out)
            if val is not None:
                scope_i = _to_int(getattr(wi, "execution_scope", None) or (wi.get("execution_scope") if isinstance(wi, dict) else None), 0)
                values.append(f"{scope_i}:{val}")

        if values:
            field_value[form_id] = { field_id: values }
            return field_value

        return None

    except Exception as e:
        print(f"[ERROR] Failed to get output field value for {field_info}: {str(e)}")
        return None


def group_fields_by_form(field_values: dict) -> dict:
    """
    ?꾨뱶 媛믩뱾???쇰퀎濡?洹몃９?뷀븯??怨듯넻 ?⑥닔
    
    Args:
        field_values: {'form_id.field_name': {'form_id': {'field_name': value}}, ...} ?뺥깭???뺤뀛?덈━
    
    Returns:
        {'form_id': {'field_name': value, ...}, ...} ?뺥깭??洹몃９?붾맂 ?뺤뀛?덈━
    """
    form_groups = {}
    
    for field_key, field_value in field_values.items():
        if field_value is None:
            continue
            
        form_id = field_key.split('.')[0]
        if form_id not in form_groups:
            form_groups[form_id] = {}
        
        field_id = field_key.split('.')[1] if '.' in field_key else field_key
        
        if isinstance(field_value, dict) and form_id in field_value:
            actual_value = field_value[form_id].get(field_id)
            if actual_value is not None:
                form_groups[form_id][field_id] = actual_value
    
    return {form_id: fields for form_id, fields in form_groups.items() if fields}

def get_input_data(workitem: dict, process_definition: Any):
    """
    ?뚰겕?꾩씠???ㅽ뻾???꾩슂???낅젰 ?곗씠??異붿텧
    - ?쒕룞???ㅼ젙??inputData媛 ?덉쑝硫??대떦 ?꾨뱶媛믪쓣 ?ъ슜
    - inputData媛 ?녾퀬 泥?踰덉㎏ ?쒕룞???꾨땲硫? ?댁쟾 ?쒕룞????異쒕젰 ?곗씠?곕? fallback?쇰줈 ?ъ슜
    """
    try:
        activity_id = workitem.get('activity_id')
        activity = process_definition.find_activity_by_id(activity_id)

        if not activity:
            print(f"[WARNING][get_input_data] activity not found for activity_id={activity_id}")
            return None

        input_data = {}
        input_fields = activity.inputData
        if len(input_fields) != 0:
            field_values = {}
            for input_field in input_fields:
                field_value = get_field_value(input_field, process_definition, workitem.get('proc_inst_id'), workitem.get('tenant_id'))
                if field_value is not None:
                    field_values[input_field] = field_value

            grouped_data = group_fields_by_form(field_values)
            input_data.update(grouped_data)

        if not input_data and not process_definition.is_starting_activity(activity_id):
            input_data = _get_prev_activity_form_data(workitem, process_definition)

        return input_data

    except Exception as e:
        print(f"[ERROR] Failed to get selected info for {workitem.get('id')}: {str(e)}")
        return None


def _get_prev_activity_form_data(workitem: dict, process_definition: Any) -> dict:
    """
    ?댁쟾 ?쒕룞???뚰겕?꾩씠??output?먯꽌 ???곗씠?곕? ?섏쭛?섏뿬 諛섑솚?쒕떎.
    泥?踰덉㎏ ?쒕룞???꾨땶??inputData ?ㅼ젙???녿뒗 寃쎌슦??fallback?쇰줈 ?ъ슜.
    """
    activity_id = workitem.get('activity_id')
    proc_inst_id = workitem.get('proc_inst_id')
    tenant_id = workitem.get('tenant_id')

    if not proc_inst_id or not activity_id:
        return {}

    prev_activities = process_definition.find_immediate_prev_activities(activity_id)
    if not prev_activities:
        return {}

    merged: dict = {}
    for prev_act in prev_activities:
        prev_workitem = fetch_workitem_by_proc_inst_and_activity(
            proc_inst_id, prev_act.id, tenant_id
        )
        if not prev_workitem or not prev_workitem.output:
            continue
        output = prev_workitem.output
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except Exception:
                continue
        if isinstance(output, dict):
            merged.update(output)

    return merged


async def get_input_data_with_file_parsing(workitem: dict, process_definition: Any):
    """
    ?뚰겕?꾩씠???ㅽ뻾???꾩슂???낅젰 ?곗씠??異붿텧
    inputData???대? ?뚯떛???뚯씪 ?댁슜???ы븿?섏뼱 ?덉쓬
    crewai-action??寃쎌슦 10000???댁긽?대㈃ ?붿빟 泥섎━
    """
    try:
        from document_parser import summarize_text
        
        # 湲곕낯 ?낅젰 ?곗씠??媛?몄삤湲?
        input_data = get_input_data(workitem, process_definition)
        
        print(f"[DEBUG] get_input_data_with_file_parsing - workitem: {workitem.get('id')}")
        
        if not input_data:
            print(f"[WARNING] No input_data found")
            return None
        
        # agent_orch媛 crewai-action??寃쎌슦?먮쭔 ?붿빟 泥섎━
        agent_orch = workitem.get('agent_orch', '')
        if agent_orch != 'crewai-action':
            print(f"[DEBUG] agent_orch is not crewai-action ({agent_orch}), skipping summarization")
            return input_data
        
        # input_data瑜?臾몄옄?대줈 蹂?섑븯??湲몄씠 ?뺤씤
        try:
            input_data_str = json.dumps(input_data, ensure_ascii=False)
        except Exception:
            input_data_str = str(input_data)
        
        input_data_length = len(input_data_str)
        print(f"[DEBUG] input_data length: {input_data_length}")
        
        # 10000???댁긽??寃쎌슦 ?붿빟
        if input_data_length > 10000:
            print(f"[INFO] input_data exceeds 10000 characters ({input_data_length}), summarizing...")
            
            try:
                # 湲곗〈 ?붿빟 濡쒖쭅 ?ъ슜
                summarized_str = await summarize_text(input_data_str, max_length=5000)
                print(f"[INFO] Successfully summarized input_data from {input_data_length} to {len(summarized_str)} characters")
                
                # ?붿빟???띿뒪?몃? JSON?쇰줈 ?뚯떛 ?쒕룄
                try:
                    return json.loads(summarized_str)
                except json.JSONDecodeError:
                    # JSON ?뚯떛 ?ㅽ뙣 ??臾몄옄???뺥깭濡?諛섑솚
                    return {"summarized_content": summarized_str}
                    
            except Exception as e:
                print(f"[ERROR] Failed to summarize input_data: {str(e)}")
                import traceback
                traceback.print_exc()
                # ?붿빟 ?ㅽ뙣 ???먮낯 諛섑솚
                return input_data
        
        return input_data

    except Exception as e:
        print(f"[ERROR] Failed to get input data with file parsing for {workitem.get('id')}: {str(e)}")
        import traceback
        traceback.print_exc()
        # ?먮윭 諛쒖깮??湲곕낯 ?낅젰 ?곗씠?곕씪??諛섑솚
        return get_input_data(workitem, process_definition)

