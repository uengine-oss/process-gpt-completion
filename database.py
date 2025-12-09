import os
from supabase import create_client, Client
from supabase.client import AsyncClient, create_async_client
from pydantic import BaseModel, validator
from typing import Any, Dict, List, Optional, Union
import uuid
from process_definition import ProcessDefinition, load_process_definition, UIDefinition
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import HTTPException
from decimal import Decimal
from datetime import datetime, timedelta
import pytz
from contextvars import ContextVar
from dotenv import load_dotenv
import socket
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

db_config_var = ContextVar('db_config', default={})
supabase_client_var = ContextVar('supabase', default=None)
subdomain_var = ContextVar('subdomain', default='localhost')


def setting_database():
    try:
        if os.getenv("ENV") != "production":
            load_dotenv(override=True)

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        supabase_client_var.set(supabase)
        
        db_config = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT")
        }
        db_config_var.set(db_config)
        
    except Exception as e:
        print(f"Database configuration error: {e}")

setting_database()

async def update_tenant_id(subdomain):
    try:
        if not subdomain:
            raise Exception("Unable to configure Tenant ID.")
        subdomain_var.set(subdomain)
    except Exception as e:
        print(f"An error occurred: {e}")




def load_sql_from_file(file_path):
    """Load SQL commands from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:  # UTF-8 인코딩으로 파일을 열기
        return file.read()


# def update_db():
#     try:
#         db_config = db_config_var.get()
#         # Establish a connection to the database
#         connection = psycopg2.connect(**db_config)
#         cursor = connection.cursor(cursor_factory=RealDictCursor)
      
#         # Load SQL from file
#         sql_query = load_sql_from_file('update_db_sql.txt')
      
#         cursor.execute(sql_query)
#         connection.commit()
      
#         return "Tables created successfully."
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         if connection:
#             connection.close()
     
def insert_usage(usage_data: dict):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise HTTPException(status_code=500, detail="Supabase 클라이언트가 요청에 대해 구성되지 않았습니다.")
        
        if not usage_data:
            raise HTTPException(status_code=400, detail="사용량 데이터가 제공되지 않았습니다.")
        
        if not usage_data.get('serviceId'):
            raise HTTPException(status_code=400, detail="서비스 ID가 제공되지 않았습니다.")
        
        if not usage_data.get('userId'):
            raise HTTPException(status_code=400, detail="사용자 ID가 제공되지 않았습니다.")
            
        if not usage_data.get('startAt'):
            raise HTTPException(status_code=400, detail="사용 시작 시점이 제공되지 않았습니다.")
            
        if not usage_data.get('usage'):
            raise HTTPException(status_code=400, detail="메타데이터 제공되지 않았습니다.")
    
        if not usage_data.get('tenantId'):
            usage_data['tenantId'] = subdomain_var.get()
    

        # Procedure 호출
        # return supabase.rpc("insert_usage_from_payload", usage_data).execute()
        return supabase.rpc("insert_usage_from_payload", {"p_payload": usage_data}).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"사용량 삽입 중 오류가 발생했습니다: {e}")
    


def db_client_signin(user_info: dict):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.auth.sign_in_with_password({ "email": user_info.get('email'), "password": user_info.get('password') })
        supabase.auth.set_session(response.session.access_token, response.session.refresh_token)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while signing in: {e}")


def execute_sql(sql_query):
    """
    Connects to a PostgreSQL database and executes the given SQL query.
    
    Args:
        sql_query (str): The SQL query to execute.
        
    Returns:
        list: A list of dictionaries representing the rows returned by the query.
    """
    
    try:
        db_config = db_config_var.get()
        # Establish a connection to the database
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Execute the SQL query
        cursor.execute(sql_query)
        
        # If the query was a SELECT statement, fetch the results
        if sql_query.strip().upper().startswith("SELECT"):
            result = cursor.fetchall()
        else:
            # Commit the transaction if the query modified the database
            connection.commit()
            result = "Table Created"
        
        return result
    
    except Exception as e:
        return(f"An error occurred while executing the SQL query: {e}")
    
    finally:
        # Close the cursor and connection to clean up


        if connection:
            connection.close()


def fetch_all_process_definitions():
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        response = supabase.table('proc_def').select('*').eq('tenant_id', subdomain).execute()
        
        return response.data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching process definitions: {e}")




def fetch_all_process_definition_ids():
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        response = supabase.table('proc_def').select('id').eq('tenant_id', subdomain).execute()
        
        return response.data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching process definition ids: {e}")




def generate_create_statement_for_table(table_name):
    """
    Generates a CREATE TABLE statement for the given table name by fetching its current schema.
    
    Args:
        table_name (str): The name of the table for which to generate the CREATE statement.
        
    Returns:
        str: A CREATE TABLE statement as a string, or an error message if the operation fails.
    """
    
    try:
        db_config = db_config_var.get()
        # Establish a connection to the database
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        
        # Fetch the table schema
        cursor.execute(f"SELECT column_name, data_type, character_maximum_length FROM information_schema.columns WHERE table_name = '{table_name}'")
        columns = cursor.fetchall()
        
        if not columns:
            return f"No existing table"
        
        # Generate the CREATE TABLE statement
        create_statement = f"CREATE TABLE {table_name} (\n"
        for column in columns:
            column_name, data_type, max_length = column
            column_def = f"{column_name} {data_type}"
            if max_length:
                column_def += f"({max_length})"
            create_statement += f"    {column_def},\n"
        
        # Remove the last comma and add the closing parenthesis
        create_statement = create_statement.rstrip(',\n') + "\n);"
        
        return create_statement
    
    except Exception as e:
        return(f"An error occurred while generating CREATE statement for table {table_name}: {e}")
    
    finally:
        # Close the cursor and connection to clean up
        if connection:
            connection.close()


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


def fetch_process_definition_by_version(
    def_id: str,
    version_tag: Optional[str] = None,
    version: Optional[Union[str, int]] = None,
    tenant_id: Optional[str] = None,
    arcv_id: Optional[str] = None,
):
    """
    version_tag / version 기준으로 프로세스 정의를 조회하는 헬퍼.

    - version_tag == "major": proc_def_version(버전 아카이브 뷰/테이블)에서 조회
      * version 이 주어지면 해당 version 레코드 우선 조회
      * 없으면 최신 version(내림차순) 사용
    - version_tag == "minor" 또는 None/기타: 현재 proc_def 테이블에서 조회

    가능한 경우 기존 헬퍼(fetch_process_definition, fetch_process_definition_latest_version)를 재사용합니다.
    """
    tag = (version_tag or "").lower()

    if arcv_id:
        try:
            versions = fetch_process_definition_version_by_arcv_id(def_id, arcv_id, tenant_id)
            if versions and len(versions) > 0:
                row = versions[0]
                definition = row.get("definition", None)
                if isinstance(definition, dict):
                    if "version" not in definition:
                        definition["version"] = row.get("version")
                    if "version_tag" not in definition:
                        definition["version_tag"] = "major"
                return definition
        except Exception:
            # arcv_id 조회 실패 시 아래 공통 로직으로 폴백
            pass

    # minor 이거나 태그가 없으면 현재 정의에서 조회 (기존 동작 유지)
    if tag != "major":
        definition = fetch_process_definition(def_id, tenant_id)
        # definition에 요청 기반 버전 메타 정보 주입
        if isinstance(definition, dict):
            if "version" not in definition:
                definition["version"] = version
            if "version_tag" not in definition:
                # 태그가 없으면 minor 취급
                definition["version_tag"] = version_tag or "minor"
        return definition

    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")

        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain

        # 1) version 이 명시된 경우: 해당 버전 우선 조회
        if version is not None:
            response = (
                supabase.table("proc_def_version")
                .select("*")
                .eq("proc_def_id", def_id.lower())
                .eq("tenant_id", tenant_id)
                .eq("version", str(version))
                .execute()
            )
            if response.data and len(response.data) > 0:
                row = response.data[0]
                definition = row.get("definition", None)
                if isinstance(definition, dict):
                    if "version" not in definition:
                        definition["version"] = row.get("version")
                    if "version_tag" not in definition:
                        definition["version_tag"] = "major"
                return definition

        # 2) 명시된 버전이 없거나, 해당 버전이 없으면 최신 버전 사용
        latest = fetch_process_definition_latest_version(def_id, tenant_id)
        if latest:
            definition = latest.get("definition", None)
            if isinstance(definition, dict):
                if "version" not in definition:
                    definition["version"] = latest.get("version")
                if "version_tag" not in definition:
                    definition["version_tag"] = "major"
            return definition

        # 3) 아카이브에도 없으면 기본 정의로 폴백
        definition = fetch_process_definition(def_id, tenant_id)
        if isinstance(definition, dict):
            if "version" not in definition:
                definition["version"] = version
            if "version_tag" not in definition:
                definition["version_tag"] = "major"
        return definition

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"No process definition found with ID {def_id} for version_tag={version_tag}, version={version}: {e}",
        )


def upsert_process_definition(definition: dict, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        if not tenant_id:
            tenant_id = subdomain_var.get()
        
        process_definition_id = definition.get('id')
        definition['tenant_id'] = tenant_id
        
        process_definition = supabase.table('proc_def').select('*').eq('id', process_definition_id).eq('tenant_id', tenant_id).execute()
        
        if process_definition.data:
            existing_data = process_definition.data[0]
            definition['uuid'] = existing_data.get('uuid')
            definition['bpmn'] = existing_data.get('bpmn')
            definition['isdeleted'] = existing_data.get('isdeleted', False)
            return supabase.table('proc_def').upsert(definition).execute()
        else:
            definition.pop('uuid', None)
            return supabase.table('proc_def').insert(definition).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error upserting process definition with ID {process_definition_id}: {e}")


def fetch_process_definition_versions(def_id, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain


        response = supabase.table('proc_def_version').select('*').eq('proc_def_id', def_id.lower()).eq('tenant_id', tenant_id).execute()
        
        return response.data
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No process definition version found with ID {def_id}: {e}")


def fetch_process_definition_version_by_arcv_id(def_id, arcv_id, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")  


        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain


        response = supabase.table('proc_def_arcv').select('*').eq('proc_def_id', def_id.lower()).eq('arcv_id', arcv_id).eq('tenant_id', tenant_id).execute()
        
        return response.data
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No process definition version found with ID {def_id} and version {arcv_id}: {e}")


def fetch_process_definition_latest_version(def_id, tenant_id: Optional[str] = None):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")


        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain


        response = supabase.table('proc_def_version').select('*').eq('proc_def_id', def_id.lower()).eq('tenant_id', tenant_id).order('version', desc=True).execute()
        
        if response.data:
            return response.data[0]
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No process definition latest version found with ID {def_id}: {e}")


def fetch_all_ui_definition():
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        response = supabase.table('form_def').select('*').eq('tenant_id', subdomain).execute()
        
        if response.data:
            return response.data
        else:
            return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching UI definitions: {e}")


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
    execution_scope: Optional[str] = None


    class Config:
        extra = "allow"


    def __init__(self, **data):
        super().__init__(**data)
        def_id = self.get_def_id()
        tenant_id = self.tenant_id
        # proc_def_version(arcv_id)가 있으면 해당 버전 정의를 우선 사용
        definition_json = fetch_process_definition_by_version(
            def_id,
            tenant_id=tenant_id,
            arcv_id=self.proc_def_version,
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
    root_proc_inst_id: Optional[str] = None
    query: Optional[str] = None
    
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
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


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


def upsert_process_instance(process_instance: ProcessInstance, tenant_id: Optional[str] = None) -> (bool, ProcessInstance):
    if 'END_PROCESS' in process_instance.current_activity_ids or 'endEvent' in process_instance.current_activity_ids or 'end_event' in process_instance.current_activity_ids or process_instance.status == 'COMPLETED':
        process_instance.current_activity_ids = []
        status = 'COMPLETED'
    else:
        status = 'RUNNING'
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
            'participants': process_instance.participants,
            'role_bindings': process_instance.role_bindings,
            'variables_data': process_instance.variables_data,
            'status': status,
            'proc_def_id': process_instance.get_def_id(),
            'proc_def_version': arcv_id,
            'tenant_id': tenant_id
        }).execute()
        success = bool(response.data)
        return success, process_instance
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def fetch_table_columns(table_name: str) -> List[str]:
    """
    Fetches the column names of a given table from the database.
    
    Args:
        table_name (str): The name of the table to fetch columns from.
    
    Returns:
        List[str]: A list of column names.
    """
    try:
        db_config = db_config_var.get()
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
        columns = cursor.fetchall()
        return [column[0] for column in columns]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch columns for table {table_name}: {e}")
    finally:
        if connection:
            connection.close()


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


def fetch_process_instance_list(user_id: str, process_definition_id: Optional[str] = None) -> Optional[List[ProcessInstance]]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if process_definition_id:
            response = supabase.table('bpm_proc_inst').select("*").eq('tenant_id', subdomain).eq('proc_def_id', process_definition_id).filter('participants', 'cs', '{' + user_id + '}').execute()
        else:
            response = supabase.table('bpm_proc_inst').select("*").eq('tenant_id', subdomain).filter('participants', 'cs', '{' + user_id + '}').execute()
        
        if response.data:
            return [ProcessInstance(**item) for item in response.data]
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def fetch_todolist_by_user_id(user_id: str) -> Optional[List[WorkItem]]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")


        subdomain = subdomain_var.get()
        response = supabase.table('todolist').select("*").eq('user_id', user_id).eq('tenant_id', subdomain).execute()
        
        if response.data:
            return [WorkItem(**item) for item in response.data]
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


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
    recent_only: Optional[bool] = True
) -> Union[WorkItem, List[WorkItem], None]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain


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
                return [WorkItem(**item) for item in response.data]
            elif len(response.data) == 1:
                return WorkItem(**response.data[0])
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

        next_activities = [activity for activity in process_definition.activities if activity.id != initial_activity.id]
        for activity in next_activities:
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
                
                reference_ids = fetch_prev_task_ids(process_definition, activity.id, process_instance_data['proc_inst_id'])
                
                for prev_activity in filtered_activities:
                    start_date = start_date + timedelta(days=prev_activity.duration)
            
            due_date = start_date + timedelta(days=activity.duration) if activity.duration else None
            workitem = fetch_workitem_by_proc_inst_and_activity(process_instance_data['proc_inst_id'], activity.id, tenant_id)
            if not workitem:
                user_id = ""
                assignees = []
                if process_result_data['roleBindings']:
                    role_bindings = process_result_data['roleBindings']
                    for role_binding in role_bindings:
                        if role_binding['name'] == activity.role:
                            if isinstance(role_binding['endpoint'], list):
                                user_id = ','.join(role_binding['endpoint'])
                            else:
                                user_id = role_binding['endpoint']
                            assignees.append(role_binding)
                
                agent_mode = None
                if activity.agentMode is not None and activity.agentMode != "":
                    if activity.agentMode != "none" and activity.agentMode != "None":
                        agent_mode = activity.agentMode.upper()
                elif activity.agentMode is None and user_id:
                    assignee_info = fetch_assignee_info(user_id)
                
                workitem = WorkItem(
                    id=f"{str(uuid.uuid4())}",
                    reference_ids=reference_ids if prev_activities else [],
                    proc_inst_id=process_instance_data['proc_inst_id'],
                    proc_def_id=process_result_data['processDefinitionId'].lower(),
                    activity_id=activity.id,
                    activity_name=activity.name,
                    user_id=user_id,
                    status="TODO",
                    tool=activity.tool,
                    start_date=start_date,
                    due_date=due_date,
                    tenant_id=tenant_id,
                    assignees=assignees if assignees else [],
                    duration=activity.duration,
                    agent_mode=agent_mode,
                    description=activity.description,
                    agent_orch=activity.orchestration
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


import json


class ChatMessage(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    email: Optional[str] = None
    image: Optional[str] = None
    content: Optional[str] = None
    timeStamp: Optional[int] = None
    jsonContent: Optional[Any] = None
    htmlContent: Optional[str] = None
    contentType: Optional[str] = None

class ChatItem(BaseModel):
    id: str
    uuid: str
    messages: Optional[ChatMessage] = None
    tenant_id: str


def fetch_chat_history(chat_room_id: str) -> List[ChatItem]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")


        subdomain = subdomain_var.get()
        response = supabase.table("chats").select("*").eq('id', chat_room_id).eq('tenant_id', subdomain).execute()


        chatHistory = []
        for chat in response.data:
            chat.pop('jsonContent', None)
            chatHistory.append(ChatItem(**chat))
        return chatHistory
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def upsert_chat_message(chat_room_id: str, data: Any, is_system: bool, tenant_id: Optional[str] = None, is_agent: Optional[bool] = False) -> None:
    try:
        if is_agent:
            message = ChatMessage(
                name=data["name"],
                role="agent",
                content=data["content"],
                jsonContent=data["jsonData"] if "jsonData" in data else None,
                htmlContent=data["html"] if "html" in data else None,
                timeStamp=int(datetime.now(pytz.timezone('Asia/Seoul')).timestamp() * 1000),
            )
        else:
            if is_system:
                if isinstance(data, str):
                    json_data = json.loads(data)
                else:
                    json_data = data
                message = ChatMessage(
                    name="system",
                    role="system",
                    email="system@uengine.org",
                    image="",
                    content=json_data["description"],
                    contentType="html" if "html" in json_data else "text",
                    jsonContent=json_data["jsonData"] if "jsonData" in json_data else None,
                    htmlContent=json_data["html"] if "html" in json_data else None,
                    timeStamp=int(datetime.now(pytz.timezone('Asia/Seoul')).timestamp() * 1000)
                )
            else:
                if data["email"] == "external_customer":
                    name = "외부 고객"
                else:
                    user_info = fetch_user_info(data["email"])
                    name = user_info["username"]
                message = ChatMessage(
                    name=name,
                    role="user",
                    email=data["email"],
                    image="",
                    content=data["command"],
                    timeStamp=int(datetime.now(pytz.timezone('Asia/Seoul')).timestamp() * 1000)
                )

        if not tenant_id:
            tenant_id = subdomain_var.get()


        chat_item = ChatItem(
            id=chat_room_id,
            uuid=str(uuid.uuid4()),
            messages=message,
            tenant_id=tenant_id
        )
        chat_item_dict = chat_item.model_dump()


        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")


        supabase.table("chats").upsert(chat_item_dict).execute();
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
                type = user_info.get("agent_type")
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


from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore


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


def update_user_admin(input):
    try:
        user_id = input.get('user_id')
        user_info = input.get('user_info')
        supabase = supabase_client_var.get()
        
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.auth.admin.update_user_by_id(user_id, user_info)
        return response


    except Exception as e:
        raise HTTPException(status_code=e.status, detail=str(e)) from e

def invite_user(input):
    try:
        supabase = supabase_client_var.get()
     
        email = input.get("email")
        is_admin = input.get("is_admin")

        tenant_id = input.get('tenant_id') if input.get('tenant_id') else subdomain_var.get()
        user_id = None
        response = None
        redirect_url = None

        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        try:
            is_user_exist = fetch_user_info(email)
        except HTTPException as e:
            is_user_exist = None
        
        if is_user_exist:
            # 기존 사용자인 경우
            user_id = is_user_exist['id']
            
            # 기존 사용자를 위한 초대 이메일 발송 (실패해도 사용자 초대는 진행)
            try:
                send_existing_user_invitation_email(email, tenant_id)
            except Exception as email_error:
                print(f"Warning: Failed to send invitation email, but continuing with user invitation: {email_error}")
            
        else:
            # 신규 사용자인 경우 - Supabase의 초대 메일 사용 (비밀번호 설정 링크 포함)
            redirect_url = f"https://{tenant_id}.process-gpt.io/auth/initial-setting"
            response = supabase.auth.admin.invite_user_by_email(
                email,
                {
                    "redirect_to": redirect_url
                }
            )
            if response.user:
                user_id = response.user.id

        if user_id:
            supabase.table("users").insert({
                "id": user_id,
                "email": email,
                "username": email.split('@')[0],
                "role": 'user',
                "is_admin": is_admin,
                "tenant_id": tenant_id
            }).execute()
        
        return {
            "success": True,
            "message": f"Invitation sent to {email}",
            "redirect_url": redirect_url,
            "user_id": user_id
        }
        
    except Exception as e:
        print(f"Error inviting user: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to invite user: {str(e)}") from e


def set_initial_info(input):
    try:
        supabase = supabase_client_var.get()
        
        user_id = input.get("user_id")
        user_name = input.get("user_name")
        password = input.get("password")
        tenant_id = subdomain_var.get()
        
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
        
        if not password:
            raise HTTPException(status_code=400, detail="Password is required")
        
        # 관리자 권한으로 사용자 비밀번호 업데이트
        response = supabase.auth.admin.update_user_by_id(
            user_id,
            {
                "password": password
            }
        )
        
        print(f"Initial password set for user: {user_id}")
        print(f"Response: {response}")

        supabase.table("users").update({
            "username": user_name
        }).eq('id', user_id).eq('tenant_id', tenant_id).execute()
        
        return {
            "success": True,
            "message": "Initial setting has been completed successfully",
            "user_id": user_id
        }
        
    except Exception as e:
        print(f"Error setting initial password: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set initial password: {str(e)}") from e

def create_user(input):
    try:
        
        supabase = supabase_client_var.get()
     
        username = input.get("username")
        email = input.get("email")
        role = input.get("role")
        tenant_id = subdomain_var.get()

        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        try:
            is_user_exist = fetch_user_info(email)
        except HTTPException as e:
            is_user_exist = None
        
        if is_user_exist:
            supabase.table("users").insert({
                "id": is_user_exist["id"],
                "email": email,
                "username": username,
                "role": role,
                "tenant_id": tenant_id
            }).execute()
        else:
            response = supabase.auth.admin.create_user({
                "email": email,
                "username": username,
                "password": "000000",
                "email_confirm": True,
                "app_metadata": {
                    "tenant_id": tenant_id
                }
            })
            
            if response.user:
                supabase.table("users").insert({
                    "id": response.user.id,
                    "email": email,
                    "username": username,
                    "role": role,
                    "tenant_id": tenant_id
                }).execute()
                return response
            else:
                raise HTTPException(status_code=404, detail="User creation failed")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def update_user(input):
    try:
        user_id = input.get('user_id')
        user_info = input.get('user_info')
        supabase = supabase_client_var.get()
        
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.table("users").update(user_info).eq('id', user_id).execute()
        return response
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def fetch_user_info_by_uid(uid: str) -> Dict[str, str]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.table("users").select("*").eq('id', uid).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def check_tenant_owner(tenant_id: str, uid: str) -> bool:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.table("tenants").select("*").eq('id', tenant_id).eq('owner', uid).execute()
        if response.data:
            return True
        else:
            return False
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

def send_existing_user_invitation_email(email: str, tenant_id: str) -> bool:
    """
    기존 사용자를 다른 테넌트에 초대하는 이메일을 발송하는 함수
    """
    try:
        # SMTP 설정
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = os.getenv("SMTP_PORT")
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if not all([smtp_server, smtp_port, smtp_username, smtp_password]):
            print("SMTP configuration is incomplete")
            return False
        
        # 테넌트 URL 생성
        if tenant_id == "localhost":
            tenant_url = "http://localhost:8088"
        else:
            tenant_url = f"https://{tenant_id}.process-gpt.io"
        
        # 이메일 템플릿
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Process GPT 테넌트 초대</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f8fafc;">
    <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 30px; text-align: center;">
            <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 600;">
                🎉 새로운 테넌트에 초대되었습니다
            </h1>
        </div>
        
        <!-- Content -->
        <div style="padding: 40px 30px;">
            <h2 style="color: #2d3748; margin: 0 0 20px 0; font-size: 22px; font-weight: 600;">
                {tenant_id}
            </h2>
            
            <p style="color: #4a5568; line-height: 1.6; margin: 0 0 30px 0; font-size: 16px;">
                <strong>{tenant_id}</strong> 테넌트에 초대되었습니다.<br>
                아래 버튼을 클릭하여 테넌트에 접속하실 수 있습니다.
            </p>
            
            <!-- CTA Button -->
            <div style="text-align: center; margin: 40px 0;">
                <a href="{tenant_url}" 
                   style="display: inline-block; 
                          background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
                          color: #ffffff; 
                          text-decoration: none; 
                          padding: 16px 32px; 
                          border-radius: 8px; 
                          font-weight: 600; 
                          font-size: 16px;
                          box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4);
                          transition: all 0.3s ease;">
                    🚀 테넌트 접속하기
                </a>
            </div>
            
            <div style="background-color: #f7fafc; padding: 20px; border-radius: 8px; border-left: 4px solid #4f46e5;">
                <p style="color: #2d3748; margin: 0; font-size: 14px; line-height: 1.5;">
                    <strong>💡 안내사항:</strong><br>
                    • 기존 계정으로 로그인하시면 됩니다<br>
                    • 문의사항이 있으시면 관리자(help@uengine.org)에게 연락해주세요
                </p>
            </div>
        </div>
        
        <!-- Footer -->
        <div style="background-color: #f8fafc; padding: 30px; text-align: center; border-top: 1px solid #e2e8f0;">
            <p style="color: #718096; margin: 0; font-size: 14px;">
                Process GPT Team<br>
                <span style="color: #a0aec0;">이 이메일은 자동으로 발송되었습니다.</span>
            </p>
        </div>
        
    </div>
</body>
</html>
        """
        
        # 이메일 메시지 생성
        msg = MIMEMultipart()
        msg['From'] = 'noreply@process-gpt.io'
        msg["Reply-To"] = "help@uengine.org"
        msg['To'] = email
        msg['Subject'] = f'[Process GPT] {tenant_id} 테넌트에 초대되었습니다'
        msg.attach(MIMEText(html_template, 'html', 'utf-8'))
        
        # SMTP를 통해 이메일 발송
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        print(f"Invitation email sent to {email} for tenant {tenant_id}")
        return True
        
    except Exception as e:
        print(f"Failed to send invitation email: {e}")
        return False


def upsert_process_instance_source(source_data: dict):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        supabase.table("proc_inst_source").upsert(source_data).execute()
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


def fetch_events_by_todo_id(todo_id: str) -> Optional[List[Dict[str, Any]]]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.table('events').select("*").eq('todo_id', todo_id).order('timestamp', desc=True).execute()
        
        if response.data and len(response.data) > 0:
            return response.data

        return []
    except Exception as e:
        print(f"[ERROR] Failed to fetch events by todo_id: {str(e)}")
        return None


def fetch_events_by_proc_inst_id(proc_inst_id: str) -> Optional[List[Dict[str, Any]]]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.table('events').select("*").eq('proc_inst_id', proc_inst_id).order('timestamp', desc=True).execute()
        
        if response.data and len(response.data) > 0:
            return response.data

        return []
    except Exception as e:
        print(f"[ERROR] Failed to fetch events by proc_inst_id: {str(e)}")
        return None


def fetch_events_by_proc_inst_id_until_activity(
    proc_def_id: str,
    proc_inst_id: str, 
    target_activity_id: str, 
    tenant_id: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    특정 프로세스 인스턴스의 이벤트를 특정 액티비티까지만 가져옵니다.
    프로세스 정의의 sequences를 사용하여 target_activity_id와 그 이전 액티비티들의 
    이벤트만 가져옵니다.
    
    Args:
        proc_def_id: 프로세스 정의 ID
        proc_inst_id: 프로세스 인스턴스 ID
        target_activity_id: 목표 액티비티 ID (이 액티비티까지만 이벤트를 가져옴)
        tenant_id: 테넌트 ID (선택사항)
    
    Returns:
        필터링된 이벤트 목록
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        subdomain = subdomain_var.get()
        if not tenant_id:
            tenant_id = subdomain
        
        # 1. 프로세스 정의 가져오기 (인스턴스에 저장된 버전 우선 사용)
        instance = fetch_process_instance(proc_inst_id, tenant_id)
        arcv_id = None
        if instance and getattr(instance, "proc_def_version", None):
            arcv_id = instance.proc_def_version

        process_definition_json = fetch_process_definition_by_version(
            proc_def_id,
            tenant_id=tenant_id,
            arcv_id=arcv_id,
        )
        if not process_definition_json:
            print(f"[ERROR] Process definition not found for {proc_def_id}")
            return []
        
        process_definition = load_process_definition(process_definition_json)
        
        # 2. target_activity와 모든 이전 액티비티들을 프로세스 정의에서 찾기
        target_activity_ids = set([target_activity_id])
        prev_activities = process_definition.find_prev_activities(target_activity_id, [])
        
        for prev_activity in prev_activities:
            target_activity_ids.add(prev_activity.id)
        
        if not target_activity_ids:
            return []
        
        # 3. 해당 proc_inst_id의 todolist에서 target_activity_ids에 해당하는 항목들만 가져오기
        # activity_id별로 status가 DONE인 것 중 rework_count가 가장 큰 워크아이템 선택
        todolist_response = supabase.table('todolist').select("id, activity_id, rework_count, status").eq(
            'proc_inst_id', proc_inst_id
        ).eq('tenant_id', tenant_id).execute()
        
        if not todolist_response.data:
            return []
        
        # activity_id별로 status가 DONE인 것 중 rework_count가 가장 큰 워크아이템 선택
        activity_todo_map = {}
        for todo in todolist_response.data:
            activity_id = todo.get('activity_id')
            status = todo.get('status')
            
            # target_activity_ids에 포함된 액티비티만 선택
            if activity_id not in target_activity_ids:
                continue
            
            # status가 DONE이 아니면 제외
            if status != 'DONE':
                continue
                
            rework_count = todo.get('rework_count', 0)
            
            if activity_id not in activity_todo_map:
                activity_todo_map[activity_id] = todo
            else:
                # rework_count가 더 큰 것으로 업데이트 (재작업된 것)
                existing_rework_count = activity_todo_map[activity_id].get('rework_count', 0)
                if rework_count > existing_rework_count:
                    activity_todo_map[activity_id] = todo
        
        # 4. 수집된 todo_id들로 이벤트 가져오기
        target_todo_ids = [todo.get('id') for todo in activity_todo_map.values()]
        
        if not target_todo_ids:
            return []
        
        # 5. 해당 todo_id들에 대한 이벤트만 가져오기
        events_response = supabase.table('events').select("*").eq(
            'proc_inst_id', proc_inst_id
        ).in_('todo_id', target_todo_ids).order('timestamp', desc=True).execute()
        
        if events_response.data and len(events_response.data) > 0:
            return events_response.data
        
        return []
    except Exception as e:
        print(f"[ERROR] Failed to fetch events by proc_inst_id until activity: {str(e)}")
        return None


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

def fetch_mcp_python_code(proc_def_id: str, activity_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        
        response = supabase.table('mcp_python_code').select('*').eq('proc_def_id', proc_def_id).eq('activity_id', activity_id).eq('tenant_id', tenant_id).order('created_at', desc=True).limit(1).execute()
        if response.data and len(response.data) > 0:
            return response.data[0]
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

def upsert_mcp_python_code(record: Dict[str, Any]):
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured for this request")
        return supabase.table("mcp_python_code").upsert(record).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e