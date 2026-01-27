from supabase import create_client, Client
from pydantic import BaseModel, validator
from typing import Any, Dict, List, Optional, Set, Union
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from process_definition import ProcessDefinition, load_process_definition, UIDefinition
from fastapi import HTTPException
from decimal import Decimal
from datetime import datetime, timedelta
from contextvars import ContextVar
from dotenv import load_dotenv
from llm_factory import create_llm

import pytz
import socket
import os
import uuid
import json
import asyncio


supabase_client_var = ContextVar('supabase', default=None)
subdomain_var = ContextVar('subdomain', default='localhost')
CONSUMER_FILTER = os.getenv("WORKITEM_CONSUMER")


def setting_database():
    try:
        load_dotenv(override=True)

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
        
        # Supabase Client API를 사용하여 SQL 실행
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
    proc_def_arcv / proc_def_version 에 저장된 특정 arcv_id 버전의 프로세스 정의를 조회합니다.
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")

        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain

        # 루트 모듈과 동일하게 proc_def_arcv 기준으로 조회
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
    실행 정의(프로세스 definition)를 버전 규칙에 따라 조회합니다.

    호환/명시 우선순위:
    - arcv_id 가 주어지면: 해당 proc_def_arcv 버전을 최우선으로 조회 (기존 동작 유지)
    - version_tag + version 이 주어지면: proc_def_version에서 해당 tag/version을 우선 조회

    기본(명시 버전이 없을 때) 동작은 TS 방식으로 통일:
    1) proc_def.prod_version 이 있으면 해당 버전을 우선 조회
       - prod_version은 기존 운영상 arcv_id로 저장되는 경우가 있어 proc_def_arcv로 먼저 시도하고,
         실패 시 proc_def_version.version 매칭도 시도
    2) 최신 major(version_tag='major')
    3) 최신 minor(version_tag='minor')
    4) proc_def.definition (현재 정의)
    """
    # 공통 모듈로 로직을 위임하여 중복 제거
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
    proc_def 테이블의 prod_version 컬럼을 업데이트합니다.
    배포 승인 시 해당 버전을 프로덕션 버전으로 설정합니다.
    
    Args:
        def_id (str): 프로세스 정의 ID
        prod_version (str): 프로덕션 버전으로 설정할 버전 (arcv_id)
        tenant_id (Optional[str]): 테넌트 ID
    
    Returns:
        bool: 업데이트 성공 여부
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
    variables_data: Optional[List[Dict[str, Any]]] = []
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
        # proc_def_version(arcv_id)가 있으면 해당 버전 정의를 우선 사용하되,
        # 레거시 데이터(예: "xxx_2.0")처럼 arcv_id가 아닌 값이 들어올 수 있어 폴백을 둡니다.
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
        # inst_id 예시: "company_entrance.123e4567-e89b-12d3-a456-426614174000"
        # 여기서 "company_entrance"가 프로세스 정의 ID입니다.
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
    특정 부모(proc_inst_id)를 가진 모든 자식 인스턴스를 조회합니다.
    경량 조회: proc_inst_id, status, current_activity_ids만 반환.
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
    proc_inst_id에 해당하는 todolist의 user_id들을 파싱하여 participants를 세팅
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

    end_activity = process_definition.find_end_activity()
    
    status = None
    if end_activity:
        end_workitem = fetch_workitem_by_proc_inst_and_activity(process_instance.proc_inst_id, safeget(end_activity, 'id', ''), tenant_id)
        if end_workitem:
            if end_workitem.status == 'DONE':
                status = 'COMPLETED'
            else:
                status = 'RUNNING'
        else:
            status = 'RUNNING'
    else:
        if process_instance.current_activity_ids and len(process_instance.current_activity_ids) != 0:
            if end_activity and safeget(end_activity, 'id', '') in process_instance.current_activity_ids:
                status = 'COMPLETED'
            else:
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
                # updated_at이 가장 최근이거나, updated_at이 같으면 rework_count가 가장 큰 항목을 최근 워크아이템으로 간주
                def get_recent_key(item):
                    updated_at = item.get('updated_at')
                    rework_count = item.get('rework_count', 0)
                    
                    if updated_at:
                        try:
                            if isinstance(updated_at, str):
                                updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00')).replace(tzinfo=None)
                            elif hasattr(updated_at, 'replace'):
                                updated_at = updated_at.replace(tzinfo=None)
                        except:
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
        
        # Supabase Client API를 사용하여 워크아이템 조회 및 업데이트
        # 먼저 SUBMITTED 상태이고 consumer가 NULL인 워크아이템들을 조회
        env = os.getenv("ENV")
        if env == 'dev':
            q = supabase.table('todolist').select('*').eq('status', 'SUBMITTED').eq('tenant_id', 'uengine')
            if CONSUMER_FILTER:
                q = q.eq('consumer', CONSUMER_FILTER)
            else:
                q = q.is_('consumer', 'null')
            response = q.limit(limit).execute()
        else:
            q = supabase.table('todolist').select('*').eq('status', 'SUBMITTED').neq('tenant_id', 'uengine')
            if CONSUMER_FILTER:
                q = q.eq('consumer', CONSUMER_FILTER)
            else:
                q = q.is_('consumer', 'null')
            response = q.limit(limit).execute()
        
        if not response.data:
            return None
        
        # 조회된 워크아이템들의 consumer를 현재 pod_id로 업데이트
        # 동시성 제어를 위해 조건부 업데이트 사용
        updated_workitems = []
        
        # 배치 업데이트를 위한 워크아이템 ID 목록
        workitem_ids = [item['id'] for item in response.data]
        
        if workitem_ids:
            try:
                # 배치 업데이트 시도
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
                
                # 배치 업데이트가 실패하면 개별 업데이트로 폴백
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
        
        env = os.getenv("ENV")
        if env == 'dev':
            q = supabase.table('todolist').select('*').eq('status', 'PENDING').eq('tenant_id', 'uengine')
            if CONSUMER_FILTER:
                q = q.eq('consumer', CONSUMER_FILTER)
            else:
                q = q.is_('consumer', 'null')
            response = q.limit(limit).execute()
        else:
            q = supabase.table('todolist').select('*').eq('status', 'PENDING').neq('tenant_id', 'uengine')
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
    오래된 consumer를 정리하는 함수
    30분 이상 업데이트되지 않은 SUBMITTED 상태의 워크아이템의 consumer를 해제
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        # 30분 전 시간 계산
        thirty_minutes_ago = (datetime.now() - timedelta(minutes=30)).isoformat()
        
        # 오래된 consumer를 NULL로 업데이트
        q = supabase.table('todolist').update({
            'consumer': None
        }).eq('status', 'SUBMITTED').not_.is_('consumer', 'null').lt('updated_at', thirty_minutes_ago)
        # CONSUMER_FILTER는 NULL 취급 → 정리 대상에서 제외
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
            workitem = fetch_workitem_by_proc_inst_and_activity(
                process_instance_data['proc_inst_id'],
                completed_activity['completedActivityId'],
                tenant_id
            )
            
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
                activity = process_definition.find_activity_by_id(completed_activity['completedActivityId'])
                start_date = datetime.now(pytz.timezone('Asia/Seoul'))
                due_date = start_date + timedelta(days=safeget(activity, 'duration', 0)) if safeget(activity, 'duration', 0) else None
                assignees = []
                if process_instance_data['role_bindings']:
                    role_bindings = process_instance_data['role_bindings']
                    for role_binding in role_bindings:
                        if role_binding['name'] == safeget(activity, 'role', ''):
                            user_id = ','.join(role_binding['endpoint']) if isinstance(role_binding['endpoint'], list) else role_binding['endpoint']
                            assignees.append(role_binding)
                
                user_info = None
                if completed_activity['completedUserEmail'] != user_id:
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
                    root_proc_inst_id=process_instance_data['root_proc_inst_id'],
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
                    root_proc_inst_id=process_instance_data['root_proc_inst_id'],
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
        
    for activity_data in process_result_data['nextActivities']:
        if activity_data['nextActivityId'] in ["END_PROCESS", "endEvent", "end_event"]:
            continue
        
        workitem = fetch_workitem_by_proc_inst_and_activity(process_instance_data['proc_inst_id'], activity_data['nextActivityId'], tenant_id)
        
        if workitem:
            workitem.status = activity_data['result']
            workitem.end_date = datetime.now(pytz.timezone('Asia/Seoul')) if activity_data['result'] == 'DONE' else None
            if workitem.agent_mode == None:
                workitem.agent_mode = determine_agent_mode(workitem.user_id, workitem.agent_mode)
                if workitem.agent_mode == 'COMPLETE' and (workitem.agent_orch == 'none' or workitem.agent_orch == None):
                    workitem.agent_orch = 'crewai-deep-research'
            
            # 입력 데이터 추가 (파일 파싱 포함)
            try:
                # 비동기 함수를 동기 컨텍스트에서 실행
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 이벤트 루프가 있으면 새 태스크로 실행
                    import nest_asyncio
                    nest_asyncio.apply()
                    input_data = loop.run_until_complete(
                        get_input_data_with_file_parsing(workitem.model_dump(), process_definition)
                    )
                else:
                    input_data = asyncio.run(
                        get_input_data_with_file_parsing(workitem.model_dump(), process_definition)
                    )
            except Exception as e:
                print(f"[WARNING] 파일 파싱 중 오류 발생, 기본 방식으로 전환: {str(e)}")
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
            if activity:
                prev_activities = process_definition.find_prev_activities(safeget(activity, 'id', ''), [])
                start_date = datetime.now(pytz.timezone('Asia/Seoul'))
                if prev_activities:
                    for prev_activity in prev_activities:
                        start_date = start_date + timedelta(days=safeget(prev_activity, 'duration', 0))
                due_date = start_date + timedelta(days=safeget(activity, 'duration', 0)) if safeget(activity, 'duration', 0) else None
                agent_mode = determine_agent_mode(activity_data['nextUserEmail'], safeget(activity, 'agentMode', None))
                agent_orch = safeget(activity, 'orchestration', None)
                if agent_orch is None or agent_orch == 'none' or agent_orch == 'None' or agent_orch == '':
                    agent_orch = None
                if agent_mode == 'COMPLETE' and (safeget(activity, 'orchestration', None) == 'none' or safeget(activity, 'orchestration', None) == None):
                    agent_orch = 'crewai-deep-research'
                
                user_info = fetch_assignee_info(activity_data['nextUserEmail'])
                user_id = user_info.get('id')
                
                # assignees 초기화
                assignees = []
                if process_result_data.get('roleBindings'):
                    role_bindings = process_result_data['roleBindings']
                    for role_binding in role_bindings:
                        if role_binding['name'] == safeget(activity, 'role', ''):
                            assignees.append(role_binding)
                
                # activity.agent가 있으면 user_id에 추가 (중복 체크, 우선순위 높음)
                if safeget(activity, 'agent', None) is not None and safeget(activity, 'agent', None) != "":
                    agent_id = safeget(activity, 'agent', None)
                    
                    # 기존 user_id와 조인 (중복 체크)
                    if user_id:
                        user_ids = [uid.strip() for uid in user_id.split(',') if uid.strip()]
                        if agent_id not in user_ids:
                            user_ids.insert(0, agent_id)  # activity.agent를 맨 앞에 추가 (우선순위)
                        user_id = ','.join(user_ids)
                    else:
                        user_id = agent_id
                    
                    # assignees에 activity.agent 추가 (중복 체크)
                    agent_already_in_assignees = False
                    for assignee in assignees:
                        assignee_endpoint = assignee.get('endpoint', '')
                        if isinstance(assignee_endpoint, list):
                            if agent_id in assignee_endpoint:
                                agent_already_in_assignees = True
                                break
                        elif assignee_endpoint == agent_id:
                            agent_already_in_assignees = True
                            break
                    
                    if not agent_already_in_assignees:
                        # activity.agent에 해당하는 role_binding 찾기 또는 생성
                        agent_role_binding = None
                        if process_result_data.get('roleBindings'):
                            for rb in process_result_data['roleBindings']:
                                rb_endpoint = rb.get('endpoint', '')
                                if isinstance(rb_endpoint, list):
                                    if agent_id in rb_endpoint:
                                        agent_role_binding = rb
                                        break
                                elif rb_endpoint == agent_id:
                                    agent_role_binding = rb
                                    break
                        
                        # role_binding을 찾지 못했으면 activity.role을 사용하여 생성
                        if not agent_role_binding:
                            agent_role_binding = {
                                "name": safeget(activity, 'role', ''),
                                "endpoint": agent_id,
                                "default": agent_id
                            }
                        
                        assignees.append(agent_role_binding)
                
                # 최종 user_id 기준으로 username 재계산
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
                    root_proc_inst_id=process_instance_data['root_proc_inst_id'],
                    execution_scope=execution_scope,
                    version_tag=getattr(process_definition, "version_tag", None),
                    version=getattr(process_definition, "version", None),
                )
                
                # 새로 생성된 workitem에도 입력 데이터 추가 (파일 파싱 포함)
                try:
                    # 비동기 함수를 동기 컨텍스트에서 실행
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 이미 실행 중인 이벤트 루프가 있으면 새 태스크로 실행
                        import nest_asyncio
                        nest_asyncio.apply()
                        input_data = loop.run_until_complete(
                            get_input_data_with_file_parsing(workitem.model_dump(), process_definition)
                        )
                    else:
                        input_data = asyncio.run(
                            get_input_data_with_file_parsing(workitem.model_dump(), process_definition)
                        )
                except Exception as e:
                    print(f"[WARNING] 파일 파싱 중 오류 발생, 기본 방식으로 전환: {str(e)}")
                    input_data = get_input_data(workitem.model_dump(), process_definition)
                
                if input_data:
                    try:
                        input_data_str = json.dumps(input_data, ensure_ascii=False)
                    except Exception:
                        input_data_str = str(input_data)
                    
                    if workitem.query and '[InputData]' in workitem.query:
                        workitem.query = workitem.query.split('[InputData]')[0] + f"[InputData]\n{input_data_str}"
                    else:
                        workitem.query = f"{workitem.query}\n[InputData]\n{input_data_str}"
                    
                    print(f"[INFO] 새 workitem에 입력 데이터 추가 완료 (파일 파싱 포함): {workitem.id}")
        
        try:
            if workitem:
                workitem_dict = workitem.model_dump()
                workitem_dict["start_date"] = workitem.start_date.isoformat() if workitem.start_date else None
                workitem_dict["end_date"] = workitem.end_date.isoformat() if workitem.end_date else None
                workitem_dict["due_date"] = workitem.due_date.isoformat() if workitem.due_date else None

                # browser-automation-agent인 경우 상세한 description 생성
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
                supabase.table('todolist').upsert(workitem_dict).execute()
                workitems.append(workitem)
        except Exception as e:
            print(f"[ERROR] upsert_next_workitems: {str(e)}")
            raise HTTPException(status_code=404, detail=str(e)) from e


    return workitems


def fetch_prev_task_ids(process_definition, current_activity_id: str, proc_inst_id: str) -> List[str]:
    """
    현재 테스크의 시퀀스 정보를 이용해 바로 직전 테스크의 ID 목록을 반환합니다.
    
    Args:
        process_definition: 프로세스 정의 객체
        current_activity_id: 현재 테스크의 ID
        proc_inst_id: 프로세스 인스턴스 ID
    
    Returns:
        List[str]: 직전 테스크의 activity ID 목록
    """
    prev_task_ids = []
    prev_activities = process_definition.find_immediate_prev_activities(current_activity_id)
    
    if prev_activities:
        # 이전 액티비티들의 activity_id를 수집
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
            
            prev_activities = process_definition.find_prev_activities(activity.id, [])
            start_date = datetime.now(pytz.timezone('Asia/Seoul'))
        
            if prev_activities:
                # 동일한 srcTrg를 가진 액티비티들 중 duration이 가장 큰 것만 남기기
                srcTrg_groups = {}
                for prev_activity in prev_activities:
                    if prev_activity.srcTrg not in srcTrg_groups:
                        srcTrg_groups[prev_activity.srcTrg] = []
                    srcTrg_groups[prev_activity.srcTrg].append(prev_activity)
                # duration이 가장 큰 액티비티만 선택
                filtered_activities = []
                for activities in srcTrg_groups.values():
                    max_duration_activity = max(activities, key=lambda x: x.duration if x.duration is not None else 0)
                    filtered_activities.append(max_duration_activity)
                
                reference_ids = fetch_prev_task_ids(process_definition, safeget(activity, 'id', ''), process_instance_data['proc_inst_id'])
                
                for prev_activity in filtered_activities:
                    start_date = start_date + timedelta(days=safeget(prev_activity, 'duration', 0))
            
            due_date = start_date + timedelta(days=safeget(activity, 'duration', 0)) if safeget(activity, 'duration', 0) else None
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

                # tool 결정: activity type에 'task'가 포함되고 tool이 비어있으면 'defaultForm' 사용
                activity_tool = safeget(activity, 'tool', '')
                activity_type = safeget(activity, 'type', '').lower()
                if not activity_tool and 'task' in activity_type:
                    activity_tool = 'formHandler:defaultForm'
                    
                # activity.agent가 있으면 user_id에 추가 (중복 체크, 우선순위 높음)
                if safeget(activity, 'agent', None) is not None and safeget(activity, 'agent', None) != "":
                    agent_id = safeget(activity, 'agent', None)
                    
                    # 기존 user_id와 조인 (중복 체크)
                    if user_id:
                        user_ids = [uid.strip() for uid in user_id.split(',') if uid.strip()]
                        if agent_id not in user_ids:
                            user_ids.insert(0, agent_id)  # activity.agent를 맨 앞에 추가 (우선순위)
                        user_id = ','.join(user_ids)
                    else:
                        user_id = agent_id
                    
                    # assignees에 activity.agent 추가 (중복 체크)
                    agent_already_in_assignees = False
                    for assignee in assignees:
                        assignee_endpoint = assignee.get('endpoint', '')
                        if isinstance(assignee_endpoint, list):
                            if agent_id in assignee_endpoint:
                                agent_already_in_assignees = True
                                break
                        elif assignee_endpoint == agent_id:
                            agent_already_in_assignees = True
                            break
                    
                    if not agent_already_in_assignees:
                        # activity.agent에 해당하는 role_binding 찾기 또는 생성
                        agent_role_binding = None
                        if process_result_data.get('roleBindings'):
                            for rb in process_result_data['roleBindings']:
                                rb_endpoint = rb.get('endpoint', '')
                                if isinstance(rb_endpoint, list):
                                    if agent_id in rb_endpoint:
                                        agent_role_binding = rb
                                        break
                                elif rb_endpoint == agent_id:
                                    agent_role_binding = rb
                                    break
                        
                        # role_binding을 찾지 못했으면 activity.role을 사용하여 생성
                        if not agent_role_binding:
                            agent_role_binding = {
                                "name": safeget(activity, 'role', ''),
                                "endpoint": agent_id,
                                "default": agent_id
                            }
                        
                        assignees.append(agent_role_binding)
                
                # 최종 user_id 기준으로 username 재계산
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
                    root_proc_inst_id=process_instance_data['root_proc_inst_id'],
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
    browser-automation-agent용 상세한 description을 생성합니다.
    """
    try:
        # 이전 workitem들을 가져와서 사용자 요청사항과 프로세스 흐름 파악
        all_workitems = fetch_workitems_by_proc_inst_id(process_instance_data['proc_inst_id'], tenant_id)

        form_data = fetch_ui_definition_by_activity_id(process_instance_data['proc_def_id'], process_instance_data['current_activity_ids'][0], tenant_id)
        
        # 이전, 현재, 이후 workitem 정보 분석 (status 기반)
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
        
        # LLM을 사용하여 상세한 description 생성
        prompt_template = """
당신은 browser-automation-agent(browser-use)가 웹 브라우저를 통해 작업을 수행할 수 있도록 상세한 단계별 설명을 생성하는 AI입니다.

=== 현재 작업 ===
{current_workitem}

=== 현재 작업에 결과로 입력되어야할(기대하는 결과값) 입력 폼 데이터입니다. 이 폼 데이터를 채워넣을 수 있는 결과를 얻기 위한 상세한 설명을 생성하세요. ===
{form_data}

=== 이전 작업들 ===
{done_workitems}

=== 이후 작업들 ===
{next_workitems}

=== 분석 요구사항 ===
1. 이전 작업에서 사용자가 입력한 구체적인 내용과 지침을 파악하세요
2. 이후 작업에서 어떤 결과물이 필요한지 파악하세요
3. 현재 작업이 전체 프로세스에서 어떤 역할을 하는지 이해하세요
4. 이후 작업이 URL 제공이나 파일 다운로드라면, 현재 작업에서 해당 결과물을 얻어내는 단계를 포함하세요

=== 생성 요구사항 ===
- browser-use가 웹 브라우저를 통해 수행할 수 있는 구체적인 단계별 설명을 생성하세요
- 각 단계는 실행 가능하고 명확해야 합니다
- 이전 작업의 입력 내용을 활용하세요
- 이후 작업에 필요한 결과물을 생성하는 단계를 포함하세요
- 요구사항 및 사용자의 입력, 지침을 반영하여 생성하되, 불필요한 설명은 포함하지 마세요. 예를 들어 로그인 요구가 없다면 로그인 설명은 포함하지 마세요.
- 사용자가 이전에 입력한 입력값이 있다면 그 값을 활용하여 생성하세요. 입력값이 상세하게 작성되어 있다면 조금의 보완만하여 사용하되, 입력값이 명확하지 않다면 입력값을 활용하여 설명을 상세하게 작성하세요.

형식:
1. [단계명]: [구체적인 수행 방법]
2. [단계명]: [구체적인 수행 방법]
...

예시: 
- 입력값: (접속주소 제공)여기서 파일 다운로드 해줘
- 지침: 특정 사이트에 접속하여 파일을 다운로드

인 경우 설명:
1. 특정 사이트(접속주소)에 접속
2. 접속시 표시되는 팝업창 확인 후 팝업창 닫기
3. 다운로드 섹션을 찾아 이동
4. 다운로드 버튼 클릭
5. 파일 다운로드 클릭 후 다운로드 된 파일 확인
6. 종료

입력값이 보다 상세한 경우:
- 입력값: 
1. https://www.g2b.go.kr/ 주소의 나라장터 접속 후 표시되는 팝업이 있다면 팝업 모두 닫기
2. 상단 검색 돋보기 아이콘 클릭 후 "ai" 검색
3. 조달정보 검색결과 중 입찰 공고의 "더보기"가 아닌 사업명("CMS 업무 DX 플랫폼 이미지분석 AI 구축 추진 관련 IT자원 도입")을 클릭하여 상세 페이지로 이동
4. 하단으로 스크롤하여 "파일첨부" 섹션을 찾고 제일 상단에 있는 파일명(예시: 	제안요청서(CMS 업무 DX 플랫폼 이미지분석 AI 구축 추진 관련 IT자원 도입).hwpx) 클릭
5. 종료
- 지침: 특정 사이트에 접속하여 파일을 다운로드

인 경우 설명:
1. https://www.g2b.go.kr/ 주소의 나라장터 접속
2. 접속시 표시되는 팝업창 확인 후 팝업창 닫기
3. 상단 검색 돋보기 아이콘 클릭 후 "ai" 검색
4. 조달정보 검색결과 중 입찰 공고의 "더보기"가 아닌 사업명("CMS 업무 DX 플랫폼 이미지분석 AI 구축 추진 관련 IT자원 도입")을 클릭하여 상세 페이지로 이동
5. 하단으로 스크롤하여 "파일첨부" 섹션을 찾고 제일 상단에 있는 파일명(예시: 제안요청서(CMS 업무 DX 플랫폼 이미지분석 AI 구축 추진 관련 IT자원 도입).hwpx) 클릭
6. 파일 다운로드 클릭 후 다운로드 된 파일 확인
7. 종료

위 예시처럼 입력값과 지침을 반영하여 상세한 단계별 설명(query)을 생성해주세요.
파일 다운로드 및 동작 결과는 현재 단계의 입력 폼 형식에 맞게 생성하라고 명시하여야함. 
생성된 결과를 입력 폼에 채워넣는 동작은 browser-use 종료 이후에 수행하도록 구현되어 있기때문에 혼란을 줄이기 위해 입력 폼을 채워넣으라는 설명은 포함하지 마세요.

불필요한 설명을 최소화하고 사용자의 입력, 지침, 현재 및 이전, 이후 모든 activity의 목적, 의도를 파악하여 
최대한 사용자의 목적에 맞는 동작을 수행시킬 수 있는 상세한 단계별 설명(query)을 생성해주세요:
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
        
        # LLM 호출
        model = create_llm(model="gpt-4o", streaming=True, temperature=0)
        response = model.invoke(prompt)
        
        # 응답에서 단계별 설명 추출
        if hasattr(response, 'content'):
            description = response.content
        else:
            description = str(response)
        
        return description.strip()
        
    except Exception as e:
        print(f"[ERROR] Failed to generate browser automation description: {str(e)}")
        # 기본 description 반환
        return None


def upsert_workitem(workitem_data: dict, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
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
    채팅 메시지를 upsert하는 함수
    
    Args:
        chat_room_id: 채팅방 ID
        data: 메시지 데이터 (dict 또는 str) - role 필드 포함
        tenant_id: 테넌트 ID
    """
    try:
        current_timestamp = int(datetime.now(pytz.timezone('Asia/Seoul')).timestamp() * 1000)
        
        # data가 문자열인 경우 JSON으로 파싱
        if isinstance(data, str):
            message_data = json.loads(data)
        else:
            message_data = data
        
        # role이 없으면 기본값 설정
        if "role" not in message_data:
            message_data["role"] = "system"
        
        # timestamp가 없으면 추가
        if "timeStamp" not in message_data:
            message_data["timeStamp"] = current_timestamp

        if not tenant_id:
            tenant_id = subdomain_var.get()

        # 채팅 아이템 데이터 구성
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
    담당자 정보를 찾는 함수
    담당자가 유저인지 에이전트인지 판단하고 적절한 정보를 반환합니다.
    
    Args:
        assignee_id: 담당자 ID (이메일 또는 에이전트 ID)
    
    Returns:
        담당자 정보 딕셔너리
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
    사용자 ID와 액티비티의 에이전트 모드를 기반으로 적절한 에이전트 모드를 결정합니다.
    
    Args:
        user_id: 사용자 ID (쉼표로 구분된 여러 ID 가능)
        agent_mode: 액티비티에서 설정된 에이전트 모드
    
    Returns:
        결정된 에이전트 모드 (None, "DRAFT", "COMPLETE")
    """
    # 액티비티에서 명시적으로 에이전트 모드가 설정된 경우
    if agent_mode is not None:
        if agent_mode.lower() not in ["none", "null"]:
            mode = agent_mode.upper()
            return mode

    # user_id가 없으면 None 반환
    if not user_id:
        return None
    
    # 여러 사용자 ID가 있는 경우
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
        
        # 사용자+에이전트 조합이면 DRAFT
        if has_user and has_agent:
            return "DRAFT"
        # 에이전트만 있으면 COMPLETE
        elif has_agent and not has_user:
            return "COMPLETE"
        # 사용자만 있으면 None
        elif has_user and not has_agent:
            return None
    
    # 단일 사용자 ID인 경우
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
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", deployment="text-embedding-3-small")
    
    return SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )


def fetch_tenant_mcp_config(tenant_id: str) -> Optional[Dict[str, Any]]:
    """
    테넌트의 MCP 설정을 조회합니다.
    
    Args:
        tenant_id (str): 테넌트 ID
        
    Returns:
        Optional[Dict[str, Any]]: MCP 설정 정보 또는 None
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
    산출물에서 특정 필드의 값을 추출 (구조 변경 최소화)
    - (1) 현재 인스턴스 단일 조회 → 값 있으면 단일값으로 반환
    - (2) 루트 인스턴스 단일 조회(+그룹 인덱싱) → 값 있으면 단일값으로 반환
    - (3) 루트 기준 다건 조회(fetch_workitems_by_root_proc_inst_id)
         → 전부 배열로 모아 { form_id: { field_id: ["<scope>:<value>", ...] } } 형태로 반환
    """
    try:
        field_value: Dict[str, Any] = {}

        process_definition_id = process_definition.processDefinitionId
        split_field_info = field_info.split('.')
        form_id = split_field_info[0]
        field_id = split_field_info[1]
        activity_id = next((activity.id for activity in process_definition.activities if activity.tool == form_id), None)
        if not activity_id:
            activity_id = form_id.replace("_form", "").replace(f"{process_definition_id}_", "")

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

        # (1) 현재 인스턴스 단일 조회
        workitem = fetch_workitem_by_proc_inst_and_activity(process_instance_id, activity_id, tenant_id, True)
        if workitem:
            out = _out(workitem)
            if out:
                val = _val_from_form(out)
                if val is not None:
                    field_value[form_id] = { field_id: val }
                    return field_value

        # 인스턴스 정보
        instance = fetch_process_instance(process_instance_id, tenant_id)
        root_proc_inst_id = getattr(instance, "root_proc_inst_id", None) or (instance.get("root_proc_inst_id") if isinstance(instance, dict) else None)
        exec_scope_raw = getattr(instance, "execution_scope", None) or (instance.get("execution_scope") if isinstance(instance, dict) else None)
        exec_scope = _to_int(exec_scope_raw, 0)

        # (2) 루트 인스턴스 단일 조회(+그룹 인덱싱)
        workitem_root = fetch_workitem_by_proc_inst_and_activity(root_proc_inst_id, activity_id, tenant_id, True)
        if workitem_root:
            out = _out(workitem_root)
            if out:
                # (a) 직접 필드
                val = _val_from_form(out)
                if val is not None:
                    field_value[form_id] = { field_id: val }
                    return field_value
                # (b) 그룹형: form 내부 item_value[exec_scope][field_id]
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
    필드 값들을 폼별로 그룹화하는 공통 함수
    
    Args:
        field_values: {'form_id.field_name': {'form_id': {'field_name': value}}, ...} 형태의 딕셔너리
    
    Returns:
        {'form_id': {'field_name': value, ...}, ...} 형태의 그룹화된 딕셔너리
    """
    form_groups = {}
    
    for field_key, field_value in field_values.items():
        if not field_value:
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
    워크아이템 실행에 필요한 입력 데이터 추출
    """
    try:
        activity_id = workitem.get('activity_id')
        activity = process_definition.find_activity_by_id(activity_id)

        if not activity:
            return None
        
        input_data = {}
        input_fields = activity.inputData
        if len(input_fields) != 0:
            # 각 필드의 값을 가져오기
            field_values = {}
            for input_field in input_fields:
                field_value = get_field_value(input_field, process_definition, workitem.get('proc_inst_id'), workitem.get('tenant_id'))
                if field_value:
                    field_values[input_field] = field_value
            
            # 폼별로 그룹화
            grouped_data = group_fields_by_form(field_values)
            input_data.update(grouped_data)

        return input_data

    except Exception as e:
        print(f"[ERROR] Failed to get selected info for {workitem.get('id')}: {str(e)}")
        return None


async def get_input_data_with_file_parsing(workitem: dict, process_definition: Any):
    """
    워크아이템 실행에 필요한 입력 데이터 추출
    inputData에 이미 파싱된 파일 내용이 포함되어 있음
    crewai-action인 경우 10000자 이상이면 요약 처리
    """
    try:
        from document_parser import summarize_text
        
        # 기본 입력 데이터 가져오기
        input_data = get_input_data(workitem, process_definition)
        
        print(f"[DEBUG] get_input_data_with_file_parsing - workitem: {workitem.get('id')}")
        
        if not input_data:
            print(f"[WARNING] No input_data found")
            return None
        
        # agent_orch가 crewai-action인 경우에만 요약 처리
        agent_orch = workitem.get('agent_orch', '')
        if agent_orch != 'crewai-action':
            print(f"[DEBUG] agent_orch is not crewai-action ({agent_orch}), skipping summarization")
            return input_data
        
        # input_data를 문자열로 변환하여 길이 확인
        try:
            input_data_str = json.dumps(input_data, ensure_ascii=False)
        except Exception:
            input_data_str = str(input_data)
        
        input_data_length = len(input_data_str)
        print(f"[DEBUG] input_data length: {input_data_length}")
        
        # 10000자 이상인 경우 요약
        if input_data_length > 10000:
            print(f"[INFO] input_data exceeds 10000 characters ({input_data_length}), summarizing...")
            
            try:
                # 기존 요약 로직 사용
                summarized_str = await summarize_text(input_data_str, max_length=5000)
                print(f"[INFO] Successfully summarized input_data from {input_data_length} to {len(summarized_str)} characters")
                
                # 요약된 텍스트를 JSON으로 파싱 시도
                try:
                    return json.loads(summarized_str)
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 문자열 형태로 반환
                    return {"summarized_content": summarized_str}
                    
            except Exception as e:
                print(f"[ERROR] Failed to summarize input_data: {str(e)}")
                import traceback
                traceback.print_exc()
                # 요약 실패 시 원본 반환
                return input_data
        
        return input_data

    except Exception as e:
        print(f"[ERROR] Failed to get input data with file parsing for {workitem.get('id')}: {str(e)}")
        import traceback
        traceback.print_exc()
        # 에러 발생시 기본 입력 데이터라도 반환
        return get_input_data(workitem, process_definition)