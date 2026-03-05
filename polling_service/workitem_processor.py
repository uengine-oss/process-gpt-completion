from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.output_parsers.json import SimpleJsonOutputParser
from llm_factory import create_llm
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Tuple
import json
import re
import uuid
import requests
import os
import asyncio
from dotenv import load_dotenv
from datetime import datetime
from fastapi import HTTPException
import threading
import queue
import time
import ast

from database import (
    fetch_process_definition_by_version, fetch_process_instance, fetch_ui_definition,
    fetch_ui_definition_by_activity_id, fetch_ui_definitions_by_def_id, fetch_user_info, fetch_assignee_info, 
    fetch_workitem_by_proc_inst_and_activity, upsert_process_instance, 
    upsert_completed_workitem, upsert_next_workitems, upsert_chat_message, 
    upsert_todo_workitems, upsert_workitem, ProcessInstance,
    fetch_todolist_by_proc_inst_id, execute_rpc, upsert_cancelled_workitem, insert_process_instance,
    fetch_child_instances_by_parent, fetch_organization_chart, fetch_workitems_by_root_proc_inst_id,
    get_field_value, group_fields_by_form, get_input_data, update_proc_def_prod_version
)
from process_definition import load_process_definition
from code_executor import execute_python_code
from smtp_handler import generate_email_template, send_email
from mcp_processor import mcp_processor


if os.getenv("ENV") != "production":
    load_dotenv(override=True)

# ------------------------------------------------------------
# Helpers: annotate output data with UI field display names
# ------------------------------------------------------------
def _build_field_text_map_from_ui_definition(ui_def: Any) -> Dict[str, str]:
    """Build a map of field_key -> display text from a single UIDefinition.

    Safely handles missing or malformed `fields_json`.
    """
    try:
        fields = getattr(ui_def, "fields_json", None) or ui_def.get("fields_json")  # type: ignore[attr-defined]
    except Exception:
        fields = None

    key_to_text: Dict[str, str] = {}
    if isinstance(fields, list):
        for item in fields:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            text = item.get("text")
            if isinstance(key, str) and isinstance(text, str):
                key_to_text[key] = text
    return key_to_text


def _extract_alias_from_html(html: Optional[str]) -> Optional[str]:
    try:
        if not isinstance(html, str) or not html:
            return None
        m = re.search(r'alias\s*=\s*"([^"]+)"', html)
        if m:
            return m.group(1)
    except Exception:
        return None
    return None


def add_field_names_by_activity(
    output_data: Dict[str, Any],
    activity_id: str,
    ui_definitions: Optional[List[Any]],
) -> Dict[str, Any]:
    """Return a copy of `output_data` where each known field key becomes
    {"name": <display text>, "value": <original value>} for the given activity.

    - Looks up the UIDefinition matching `activity_id` from `ui_definitions`.
    - For keys present in `output_data` and in `fields_json`, wraps the value into {name, value}.
    - Does not mutate the original dict.
    """
    annotated: Dict[str, Any] = dict(output_data or {})
    if not activity_id or not ui_definitions:
        return annotated

    # Find matching UI by activity_id
    ui_for_activity: Optional[Any] = None
    for ui in ui_definitions or []:
        try:
            if getattr(ui, "activity_id", None) == activity_id or (
                isinstance(ui, dict) and ui.get("activity_id") == activity_id
            ):
                ui_for_activity = ui
                break
        except Exception:
            continue

    if not ui_for_activity:
        return annotated

    key_to_text = _build_field_text_map_from_ui_definition(ui_for_activity)
    if not key_to_text:
        return annotated

    def _wrap(value: Any, display_name: str) -> Dict[str, Any]:
        # If already in desired shape, update name and keep value as-is
        if isinstance(value, dict) and "name" in value and "value" in value:
            wrapped = dict(value)
            wrapped["name"] = display_name
            return wrapped
        return {"name": display_name, "value": value}

    for k in list(output_data.keys()):
        if isinstance(k, str) and k.startswith("__"):
            continue

        value = output_data[k]
        display = key_to_text.get(k, k)

        # If value is a list of dicts, annotate each dict's fields by key and wrap regardless of mapping
        if isinstance(value, list):
            transformed_list = _annotate_list_elements_with_field_names(value, ui_definitions)
            annotated[k] = {"name": display, "value": transformed_list}
            continue

        # If value is a dict, annotate its inner keys and wrap regardless of mapping
        if isinstance(value, dict):
            item_dict = _annotate_dict_with_field_names(value, ui_definitions)
            annotated[k] = {"name": display, "value": item_dict}
            continue

        # Scalar -> wrap if mapping exists; otherwise leave scalar as-is
        if k in key_to_text:
            annotated[k] = {"name": display, "value": value}

    # Remove legacy map if present
    if "__names__" in annotated:
        try:
            del annotated["__names__"]
        except Exception:
            pass
    return annotated


def add_field_name_by_key(
    output_data: Dict[str, Any],
    field_key: str,
    ui_definitions: Optional[List[Any]],
) -> Dict[str, Any]:
    """Return a copy of `output_data` with a single field wrapped as
    {"name": <display text>, "value": <original value>}.

    - Scans all provided `ui_definitions` to find the first  field whose `key` matches `field_key`.
    - Does not mutate the original dict.
    """
    annotated: Dict[str, Any] = dict(output_data or {})
    if not field_key or not ui_definitions:
        return annotated

    display_text: Optional[str] = None
    for ui in ui_definitions or []:
        key_to_text = _build_field_text_map_from_ui_definition(ui)
        if field_key in key_to_text:
            display_text = key_to_text[field_key]
            break

    if field_key in annotated:
        original_value = annotated[field_key]
        display_name = display_text if display_text else field_key

        # Already wrapped: update name and recursively annotate only its "value"
        if isinstance(original_value, dict) and ("name" in original_value and "value" in original_value):
            wrapped = dict(original_value)
            wrapped["name"] = display_name
            inner = wrapped.get("value")
            if isinstance(inner, list):
                wrapped["value"] = _annotate_list_elements_with_field_names(inner, ui_definitions)
            elif isinstance(inner, dict):
                wrapped["value"] = _annotate_dict_with_field_names(inner, ui_definitions)
            annotated[field_key] = wrapped
            return annotated

        # If list: annotate each dict element's inner keys by scanning ui_definitions
        if isinstance(original_value, list):
            transformed_list = _annotate_list_elements_with_field_names(original_value, ui_definitions)
            annotated[field_key] = {"name": display_name, "value": transformed_list}
            return annotated

        # If dict: annotate its inner keys, then wrap
        if isinstance(original_value, dict):
            item = _annotate_dict_with_field_names(original_value, ui_definitions)
            annotated[field_key] = {"name": display_name, "value": item}
            return annotated

        # Scalar: simple wrap
        annotated[field_key] = {"name": display_name, "value": original_value}

    return annotated

# ------------------------------------------------------------
# Helpers: recursive annotation with cycle guard
# ------------------------------------------------------------
def _annotate_list_elements_with_field_names(
    value_list: list[Any],
    ui_definitions: Optional[List[Any]],
    _seen: Optional[set[int]] = None,
    _depth: int = 0,
    _max_depth: int = 50,
) -> list[Any]:
    """
    Potentially deep recursion helper for list elements used by:
    - add_field_names_by_activity
    - add_field_name_by_key
    """
    if _seen is None:
        _seen = set()
    # depth guard
    if _depth > _max_depth:
        return value_list
    oid = id(value_list)
    if oid in _seen:
        return value_list
    _seen.add(oid)

    transformed_list: list[Any] = []
    for elem in value_list:
        if isinstance(elem, dict):
            item_dict = dict(elem)
            for inner_key in list(item_dict.keys()):
                item_dict = add_field_name_by_key(item_dict, inner_key, ui_definitions)
            transformed_list.append(item_dict)
        else:
            transformed_list.append(elem)
    return transformed_list


def _annotate_dict_with_field_names(
    value_dict: Dict[str, Any],
    ui_definitions: Optional[List[Any]],
    _seen: Optional[set[int]] = None,
    _depth: int = 0,
    _max_depth: int = 50,
) -> Dict[str, Any]:
    """
    Potentially deep recursion helper for dict values used by:
    - add_field_names_by_activity
    - add_field_name_by_key
    """
    if _seen is None:
        _seen = set()
    if _depth > _max_depth:
        return value_dict
    oid = id(value_dict)
    if oid in _seen:
        return value_dict
    _seen.add(oid)

    item_dict = dict(value_dict)
    for inner_key in list(item_dict.keys()):
        item_dict = add_field_name_by_key(item_dict, inner_key, ui_definitions)
    return item_dict


def iter_reference_scalars_extractor(
    d,
    prefix: str = "",
    acc: Optional[list] = None,
    limit: int = 6,
    _seen: Optional[set[int]] = None,
    _depth: int = 0,
    _max_depth: int = 50,
) -> list:
    """
    Extract up to 'limit' scalar references from nested dict/list for logging or reference info.
    Split out from run_completed_determination to isolate recursion-prone logic.
    """
    if acc is None:
        acc = []
    if len(acc) >= limit:
        return acc
    if _seen is None:
        _seen = set()
    if _depth > _max_depth:
        return acc
    if isinstance(d, dict):
        oid = id(d)
        if oid in _seen:
            return acc
        _seen.add(oid)
        for k, v in d.items():
            if len(acc) >= limit:
                break
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, (str, int, float, bool)) or v is None:
                acc.append({"key": key, "value": v})
            elif isinstance(v, dict):
                iter_reference_scalars_extractor(v, key, acc, limit, _seen=_seen, _depth=_depth + 1, _max_depth=_max_depth)
            elif isinstance(v, list) and v and isinstance(v[0], (str, int, float, bool)):
                acc.append({"key": key, "value": v[:5]})
    return acc

def collect_ui_field_keys(ui_defs: Optional[List[Any]]) -> set[str]:
    """
    Collect all field keys from given UI definitions' fields_json.
    Returns a set of string keys. Tolerates malformed inputs.
    """
    keys: set[str] = set()
    try:
        RESERVED = {"", "name", "value"}
        for ui in ui_defs or []:
            mapping = _build_field_text_map_from_ui_definition(ui)
            for k in mapping.keys():
                if isinstance(k, str) and (k not in RESERVED):
                    keys.add(k)
    except Exception:
        # Best-effort; ignore failures
        pass
    return keys


def apply_field_name_annotation_recursively(
    obj: Any,
    ui_definitions: Optional[List[Any]],
    field_keys: Optional[set[str]] = None,
    _seen: Optional[set[int]] = None,
) -> Any:
    """
    Recursively annotate objects using UI field names while guarding against circular references.
    - Wraps matching keys via add_field_name_by_key
    - Traverses dicts and lists
    - Uses object id-based visited set to prevent infinite recursion on cyclic graphs
    """
    if field_keys is None:
        field_keys = collect_ui_field_keys(ui_definitions)
    if _seen is None:
        _seen = set()

    # Primitive short-circuit
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    oid = id(obj)
    if oid in _seen:
        # Already visited; return as-is to avoid cycles
        return obj
    _seen.add(oid)

    if isinstance(obj, dict):
        RESERVED = {"name", "value"}
        annotated: dict[str, Any] = dict(obj)
        # If already wrapped node, recurse only into its "value"
        if "name" in annotated and "value" in annotated:
            v = annotated.get("value")
            if isinstance(v, (dict, list)):
                annotated["value"] = apply_field_name_annotation_recursively(v, ui_definitions, field_keys, _seen)
            return annotated
        # First, wrap known field keys present at this level (skip reserved)
        for fk in field_keys:
            if fk in RESERVED or fk == "":
                continue
            if fk in annotated:
                annotated = add_field_name_by_key(annotated, fk, ui_definitions)
        # Then recurse into children (skip reserved keys)
        for k, v in list(annotated.items()):
            if k in RESERVED:
                continue
            if isinstance(v, (dict, list)):
                annotated[k] = apply_field_name_annotation_recursively(v, ui_definitions, field_keys, _seen)
        return annotated

    if isinstance(obj, list):
        return [apply_field_name_annotation_recursively(v, ui_definitions, field_keys, _seen) for v in obj]

    # Fallback: return original for unsupported types
    return obj

# LLM 객체 생성 (공통 팩토리 사용)
model = create_llm(model="gpt-4o", streaming=True, temperature=0)

# parser 생성
class CustomJsonOutputParser(SimpleJsonOutputParser):
    def parse(self, text: str) -> dict:
        # Multiple parsing strategies to handle various response formats
        
        # Strategy 1: Extract JSON from markdown code blocks
        json_patterns = [
            r'```json\n(.*?)\n```',  # Standard markdown JSON
            r'```\n(.*?)\n```',      # Generic code block
            r'```(.*?)```',           # Code block without newlines
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    continue
        
        # Strategy 2: Try to find JSON object directly in the text
        # Look for content that starts with { and ends with }
        json_start = text.find('{')
        json_end = text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_content = text[json_start:json_end + 1]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Clean up common LLM artifacts and try again
        cleaned_text = text.strip()
        # Remove common prefixes
        prefixes_to_remove = [
            "Here is the JSON output based on the provided information and process definition:",
            "Here is the JSON response:",
            "The result is:",
            "JSON output:",
            "Response:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned_text.startswith(prefix):
                cleaned_text = cleaned_text[len(prefix):].strip()
        
        # Try parsing the cleaned text
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Try to extract and fix common JSON formatting issues
        # Remove any text before the first { and after the last }
        first_brace = cleaned_text.find('{')
        last_brace = cleaned_text.rfind('}')
        
        if first_brace != -1 and last_brace != -1:
            json_content = cleaned_text[first_brace:last_brace + 1]
            try:
                return json.loads(json_content)
            except json.JSONDecodeError as e:
                # Try to fix common issues
                fixed_content = self._fix_common_json_issues(json_content)
                try:
                    return json.loads(fixed_content)
                except json.JSONDecodeError:
                    pass
        
        raise ValueError(f"Could not parse JSON from text: {text[:200]}...")
    
    def _fix_common_json_issues(self, json_content: str) -> str:
        """Fix common JSON formatting issues from LLM responses"""
        # Remove trailing commas before closing brackets/braces
        json_content = re.sub(r',(\s*[}\]])', r'\1', json_content)
        
        # Fix unquoted property names
        json_content = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', json_content)
        
        # Fix single quotes to double quotes
        json_content = json_content.replace("'", '"')
        
        # Fix boolean values
        json_content = re.sub(r':\s*true\s*([,}])', r': true\1', json_content)
        json_content = re.sub(r':\s*false\s*([,}])', r': false\1', json_content)
        
        # Fix missing quotes around string values
        json_content = re.sub(r':\s*([^"][^,}\]]*[^"\s,}\]])', r': "\1"', json_content)
        
        # Fix newlines and special characters in strings
        json_content = json_content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        
        # Fix unescaped quotes within strings
        json_content = re.sub(r'([^\\])"([^"]*?)([^\\])"', r'\1"\2\\"\3"', json_content)
        
        return json_content

parser = CustomJsonOutputParser()


prompt_completed = PromptTemplate.from_template(
"""
You are a BPMN Completion Extractor.

Goal:
- 이번 스텝에서 완료된 액티비티/서브프로세스/이벤트를 표시한다.

Inputs:
Process Definition:
- activities: {activities}
- gateways: {gateways}
- events: {events}
- sequences: {sequences}
- attached_activities: {attached_activities}
- subProcesses: {subProcesses}

Current Step:
- activity_id: {activity_id}
- user: {user_email}
- submitted_output: {output}

Runtime Context:
- output: {output}
- previous_outputs: {previous_outputs}
- today: {today}
- gateway_condition_data: {gateway_condition_data}
- sequence_conditions: {sequence_conditions}
- instance_name_pattern: {instance_name_pattern}

Instructions:
1) 기본 완료 조건
- submitted_output 이 activities 의 checkpoints 만족하는지를 기준으로 결과를 "DONE" 과 "PENDING" 중에서 출력한다.
- checkpoints 가 없으면 출력 결과를 "DONE" 으로 출력한다.
- 현재 activity_id를 type="activity" 으로 completedActivities에 추가한다.

2) Instance Name
- Use instance_name_pattern if provided; otherwise fallback to "processDefinitionId.key" from submitted_output, with total length ≤ 20 characters.

3) Output
- 반드시 아래 JSON만 출력한다. 추가 설명 금지.


```json
{{
  "completedActivities": [
    {{
      "completedActivityId": "activity_or_event_id",
      "completedActivityName": "name_if_available",
      "completedUserEmail": "{user_email}",
      "type": "activity" | "event",
      "expression": "cron expression if event",
      "dueDate": "YYYY-MM-DD if event",
      "result": "DONE | PENDING", // PENDING when cannotProceedErrors exist
      "description": "완료된 활동에 대한 설명 (Korean)",
      "cannotProceedErrors": [
        {{
          "type": "PROCEED_CONDITION_NOT_MET" | "SYSTEM_ERROR" | "DATA_FIELD_NOT_EXIST",
          "reason": "설명 (Korean)"
        }}
      ]
    }}
  ],
}}
"""
)


# Pydantic model for process execution

class ProceedError(BaseModel):
    type: str
    reason: str
class Activity(BaseModel):
    nextActivityId: Optional[str] = None
    nextActivityName: Optional[str] = None
    nextUserEmail: Optional[str] = None
    result: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    expression: Optional[str] = None

class CompletedActivity(BaseModel):
    completedActivityId: Optional[str] = None
    completedActivityName: Optional[str] = None
    completedUserEmail: Optional[str] = None
    result: Optional[str] = None
    description: Optional[str] = None
    cannotProceedErrors: Optional[List[ProceedError]] = None

class FieldMapping(BaseModel):
    key: str
    name: str
    value: Any

class ProcessResult(BaseModel):
    instanceId: str
    instanceName: str
    fieldMappings: Optional[List[FieldMapping]] = None
    nextActivities: Optional[List[Activity]] = None
    completedActivities: Optional[List[CompletedActivity]] = None
    processDefinitionId: str
    result: Optional[str] = None

# upsert 디바운스 큐 및 쓰레드 정의 (파일 상단에 위치)
upsert_queue = queue.Queue()

def upsert_worker():
    last_upsert_time = 0
    last_item = None
    DEBOUNCE_SEC = 1  # 0.5초에 한 번만 upsert
    while True:
        try:
            item, tenant_id = upsert_queue.get(timeout=DEBOUNCE_SEC)
            last_item = (item, tenant_id)
            upsert_queue.task_done()
        except queue.Empty:
            pass  # 큐가 비어있으면 넘어감
        now = time.time()
        if last_item and (now - last_upsert_time) >= DEBOUNCE_SEC:
            upsert_workitem(last_item[0], last_item[1])
            last_upsert_time = now
            last_item = None

# 프로그램 시작 시 한 번만 실행
threading.Thread(target=upsert_worker, daemon=True).start()

def initialize_role_bindings(process_result_json: dict) -> list:
    """Initialize role_bindings from process_result_json"""
    existing_role_bindings = process_result_json.get("roleBindings", [])
    initial_role_bindings = []
    if existing_role_bindings:
        for rb in existing_role_bindings:
            role_binding = {
                "name": rb.get("name"),
                "endpoint": rb.get("endpoint"),
                "resolutionRule": rb.get("resolutionRule")
            }
            initial_role_bindings.append(role_binding)
    return initial_role_bindings

def check_external_customer_and_send_email(activity_obj, process_instance, process_definition):
    """
    Check that the next activity's role is assigned to external customer.
    If the next activity's role is assigned to external customer, send an email to the external customer.
    """
    try:
        # Determine if the role is for an external customer
        role_name = activity_obj.role
        role_info = next((role for role in process_definition.roles if role.name == role_name), None)
        
        if role_info and role_info.endpoint == "external_customer":
            customer_email = None
            workitems = fetch_todolist_by_proc_inst_id(process_instance.proc_inst_id)
            completed_workitems = [workitem for workitem in workitems if workitem.status == "DONE"]
            completed_outputs = [workitem.output for workitem in completed_workitems]
            for output in completed_outputs:
                if output:
                    try:
                        output_json = json.loads(output) if isinstance(output, str) else output
                        # output_json이 딕셔너리인지 확인
                        if isinstance(output_json, dict):
                            # 각 폼 필드에서 customer_email 찾기
                            for form_key, form_data in output_json.items():
                                if isinstance(form_data, dict) and "customer_email" in form_data:
                                    customer_email = form_data["customer_email"]
                                    break
                            # customer_email을 찾았으면 루프 종료
                            if customer_email:
                                break
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"[WARNING] Failed to parse output JSON: {e}")
                        continue
            
            if customer_email:
                if (process_instance.tenant_id == "localhost"):
                    base_url = "http://localhost:8088/external-forms"
                else:
                    tenant_id = process_instance.tenant_id
                    base_url = f"https://{tenant_id}.process-gpt.io/external-forms"
                
                proc_def_id = process_definition.processDefinitionId
                proc_inst_id = process_instance.proc_inst_id
                external_form_id = activity_obj.tool.replace("formHandler:", "")
                activity_id = activity_obj.id
                
                external_form_url = f"{base_url}/{external_form_id}?process_definition_id={proc_def_id}&activity_id={activity_id}&process_instance_id={proc_inst_id}"
                
                additional_info = {
                    "support_email": "help@uengine.org"
                }
                
                # 이메일 템플릿 생성
                email_template = generate_email_template(activity_obj, external_form_url, additional_info)
                title = f"'{activity_obj.name}' 를 진행해주세요."
                print(f"Sending email to {customer_email} with title {title}")
                # 이메일 전송
                send_email(subject=title, body=email_template, to_email=customer_email)
                
                return True
            else:
                print(f"No customer email found for {process_instance.proc_inst_id}")
                return False
    except Exception as e:
        # Log the error but don't stop the process
        print(f"Failed to send notification to external customer: {str(e)}")
        return False

def _create_or_get_process_instance(process_result: ProcessResult, process_result_json: dict, tenant_id: Optional[str] = None) -> ProcessInstance:
    """Create new process instance or get existing one"""
    if not fetch_process_instance(process_result.instanceId, tenant_id):
        if process_result.instanceId == "new" or '.' not in process_result.instanceId:
            instance_id = f"{process_result.processDefinitionId.lower()}.{str(uuid.uuid4())}"
        else:
            instance_id = process_result.instanceId
        return ProcessInstance(
            proc_inst_id=instance_id,
            proc_inst_name=f"{process_result.instanceName}",
            role_bindings=initialize_role_bindings(process_result_json),
            current_activity_ids=[],
            participants=[],
            variables_data=[],
            status="RUNNING",
            tenant_id=tenant_id,
            root_proc_inst_id=instance_id
        )
    else:
        process_instance = fetch_process_instance(process_result.instanceId, tenant_id)
        if process_instance.status == "NEW" and process_instance.parent_proc_inst_id == None:
            process_instance.proc_inst_name = process_result.instanceName
            process_instance.root_proc_inst_id = process_result.instanceId
        return process_instance

def _update_process_variables(process_instance: ProcessInstance, field_mappings: List[FieldMapping]):
    """Update process instance variables from field mappings"""
    if not field_mappings:
        return
    
    # Ensure variables_data is initialized
    if process_instance.variables_data is None:
        process_instance.variables_data = []
    
    for data_change in field_mappings:
        form_entry = next((item for item in process_instance.variables_data 
                          if isinstance(item["value"], dict) and data_change.key in item["value"]), None)
        
        if form_entry:
            form_entry["value"][data_change.key] = data_change.value
        else:
            variable = {
                "key": data_change.key,
                "name": data_change.name,
                "value": data_change.value
            }
            existing_variable = next((item for item in process_instance.variables_data 
                                    if item["key"] == data_change.key), None)
            if existing_variable:
                existing_variable.update(variable)
            else:
                process_instance.variables_data.append(variable)

def _process_next_activities(process_instance: ProcessInstance, process_result: ProcessResult, 
                           process_result_json: dict, process_definition):
    """Process next activities"""
    # Ensure current_activity_ids is initialized
    if process_instance.current_activity_ids is None:
        process_instance.current_activity_ids = []

    # 1) 먼저 "이번 호출에서 완료된 활동"은 current에서 제거한다.
    #    (nextActivities 계산/역할 해석이 실패해 비어도, 완료된 활동이 계속 current로 남아
    #     UI에서 '완료됨' + '진행중'으로 중복 표시되는 문제를 방지)
    completed_ids: set[str] = set()
    try:
        for ca in (process_result.completedActivities or []):
            try:
                cid = getattr(ca, "completedActivityId", None)
                cres = (getattr(ca, "result", None) or "").upper()
                if cid and cres in ("DONE", "SUBMITTED", "COMPLETED"):
                    completed_ids.add(str(cid))
            except Exception:
                continue
    except Exception:
        completed_ids = set()

    if completed_ids:
        process_instance.current_activity_ids = [
            aid for aid in (process_instance.current_activity_ids or [])
            if str(aid) not in completed_ids
        ]

    # 2) nextActivities로부터 새 current 후보를 누적한다(기존 current는 유지 + 유니크).
    next_acts = process_result.nextActivities or []
    for activity in next_acts:
        next_id = getattr(activity, "nextActivityId", None)
        if next_id in ["endEvent", "END_PROCESS", "end_event"]:
            # 종료로 향하면 current는 비워 종료 처리로 위임
            process_instance.current_activity_ids = []
            break

        # 게이트웨이인 경우: 이벤트 게이트웨이는 그대로, 그 외는 다음 액티비티로 확장
        if next_id and process_definition.find_gateway_by_id(next_id):
            if getattr(activity, "type", None) == "event":
                if str(next_id) not in [str(x) for x in process_instance.current_activity_ids]:
                    process_instance.current_activity_ids.append(next_id)
                activity_obj = process_definition.find_activity_by_id(next_id)
                check_external_customer_and_send_email(activity_obj, process_instance, process_definition)
                continue

            expanded = process_definition.find_next_activities(next_id, True) or []
            if expanded:
                # process_result_json에 gateway 대신 확장된 nextActivities를 반영
                process_result_json["nextActivities"] = []
                next_activity_dicts = [
                    Activity(
                        nextActivityId=act.id,
                        nextUserEmail=getattr(activity, "nextUserEmail", None),
                        result="IN_PROGRESS",
                    ).model_dump()
                    for act in expanded
                ]
                process_result_json["nextActivities"].extend(next_activity_dicts)

                for act in expanded:
                    if act.id and str(act.id) not in [str(x) for x in process_instance.current_activity_ids]:
                        process_instance.current_activity_ids.append(act.id)
                    # Check external customer and send email for expanded activities
                    activity_obj = process_definition.find_activity_by_id(act.id)
                    check_external_customer_and_send_email(activity_obj, process_instance, process_definition)
                continue

            # 확장 결과가 없으면 nextActivities를 비워둔다(다만 완료된 current가 살아남지 않도록 위에서 제거됨)
            process_result_json["nextActivities"] = []
            continue

        # 일반 액티비티
        if next_id and str(next_id) not in [str(x) for x in process_instance.current_activity_ids]:
            process_instance.current_activity_ids.append(next_id)

        # Check external customer and send email
        activity_obj = process_definition.find_activity_by_id(next_id)
        check_external_customer_and_send_email(activity_obj, process_instance, process_definition)

def _process_sub_processes(process_instance: ProcessInstance, process_result: ProcessResult, process_result_json: dict, process_definition):
    _SENTINEL = object()

    def collect_participants(role_bindings):
        participants = []
        last = _SENTINEL
        for rb in role_bindings or []:
            endpoint = rb.get("endpoint")
            if isinstance(endpoint, list):
                participants.extend(endpoint)
                if endpoint:
                    last = endpoint[-1]
            elif endpoint:
                participants.append(endpoint)
                last = endpoint
        return participants, last

    def create_initial_workitem(child_def, child_proc_inst_id, child_proc_def_id, role_bindings, endpoint, process_instance, execution_scope):
        start_event = next((gw for gw in (child_def.gateways or []) if getattr(gw, 'type', None) == 'startEvent'), None)
        
        root_proc_inst_id = process_instance.root_proc_inst_id
        if root_proc_inst_id == None:
            root_proc_inst_id = process_instance.proc_inst_id
            
        if start_event:
            start_date = datetime.now().isoformat()
            workitem_data = {
                "id": str(uuid.uuid4()),
                "user_id": endpoint,
                "username": None,
                "proc_inst_id": child_proc_inst_id,
                "proc_def_id": child_proc_def_id,
                "activity_id": start_event.id,
                "activity_name": start_event.name or 'Start',
                "start_date": start_date,
                "due_date": None,
                "status": "SUBMITTED",
                "assignees": role_bindings,
                "reference_ids": [],
                "duration": None,
                "tool": None,
                "output": {},
                "retry": 0,
                "consumer": None,
                "description": start_event.description or '',
                "tenant_id": process_instance.tenant_id,
                "root_proc_inst_id": root_proc_inst_id,
                "execution_scope": execution_scope,
            }
            upsert_workitem(workitem_data, process_instance.tenant_id)
            print(f"[INFO] Created startEvent workitem for child: {child_proc_inst_id} -> {start_event.id}")
        else:
            initial_act = child_def.find_initial_activity() if child_def else None
            if not initial_act:
                print(f"[WARN] No initial activity found for child process '{child_proc_def_id}'")
                return
            start_date = datetime.now().isoformat()
            due_date = None
            if initial_act.duration:
                try:
                    from datetime import timedelta
                    due_date = (datetime.now() + timedelta(days=initial_act.duration)).isoformat()
                except Exception:
                    due_date = None
            workitem_data = {
                "id": str(uuid.uuid4()),
                "user_id": endpoint,
                "username": None,
                "proc_inst_id": child_proc_inst_id,
                "proc_def_id": child_proc_def_id,
                "activity_id": initial_act.id,
                "activity_name": initial_act.name,
                "start_date": start_date,
                "due_date": due_date,
                "status": "SUBMITTED",
                "assignees": role_bindings,
                "reference_ids": [],
                "duration": initial_act.duration,
                "tool": initial_act.tool,
                "output": {},
                "retry": 0,
                "consumer": None,
                "description": initial_act.description,
                "tenant_id": process_instance.tenant_id,
                "root_proc_inst_id": root_proc_inst_id,
            }
            upsert_workitem(workitem_data, process_instance.tenant_id)
            print(f"[INFO] Created initial activity workitem for child: {child_proc_inst_id} -> {initial_act.id}")

    def resolve_multi_instance_count(activity, process_result_json):
        raw = getattr(activity, 'multiInstanceCount', None)
        if raw is None:
            try:
                na = process_result_json.get('nextActivities') or []
                target = next((x for x in na if x.get('nextActivityId') == activity.nextActivityId), None)
                if target:
                    raw = target.get('multiInstanceCount')
            except Exception:
                raw = None
        try:
            cnt = int(str(raw)) if raw is not None else 1
        except Exception:
            cnt = 1
        return 1 if cnt < 1 else cnt
    
    def resolve_multi_instance_reason(activity, process_result_json):
        raw = getattr(activity, 'multiInstanceReason', None)
        if raw is None:
            try:
                na = process_result_json.get('nextActivities') or []
                target = next((x for x in na if x.get('nextActivityId') == activity.nextActivityId), None)
                if target:
                    raw = target.get('multiInstanceReason')
            except Exception:
                raw = None
        return raw

    for activity in process_result.nextActivities or []:
        if activity.type != "subProcess":
            continue
        
        prev_activities = process_definition.find_immediate_prev_activities(activity.nextActivityId)
        for prev_activity in prev_activities:
            for completed_activity in process_result.completedActivities:
                if completed_activity.completedActivityId == prev_activity.id:
                    completed_activity.result = "PENDING"
                    break
            for completed_activity_json in process_result_json.get("completedActivities", []):
                if completed_activity_json.get("completedActivityId") == prev_activity.id:
                    completed_activity_json["result"] = "PENDING"
                    break
        
        next_sub_process = process_definition.find_next_sub_process(activity.nextActivityId)
        if not next_sub_process:
            next_sub_process = process_definition.find_sub_process_by_id(activity.nextActivityId)
        if not next_sub_process:
            continue

        try:
            child_def = process_definition.build_subprocess_definition(next_sub_process.id)
        except Exception as e:
            print(f"[ERROR] Failed to build subprocess definition for '{next_sub_process.id}': {e}")
            continue

        child_proc_def_id = child_def.processDefinitionId or f"{process_instance.process_definition.processDefinitionId}.{next_sub_process.id}"

        role_bindings = process_instance.role_bindings or []
        participants, last_endpoint = collect_participants(role_bindings)
        endpoint = last_endpoint if last_endpoint is not _SENTINEL else None

        mi_count = resolve_multi_instance_count(activity, process_result_json)
        mi_reasons = resolve_multi_instance_reason(activity, process_result_json)
        execution_scope = 0
    
        root_proc_inst_id = process_instance.root_proc_inst_id
        if root_proc_inst_id == None:
            root_proc_inst_id = process_instance.proc_inst_id

        for i in range(mi_count):
            mi_reason = mi_reasons[i] if mi_reasons else ""
            child_proc_inst_id = f"{str(child_proc_def_id).lower()}.{str(uuid.uuid4())}"
            try:
                process_instance_data = {
                    "proc_inst_id": child_proc_inst_id,
                    "proc_inst_name": f"{mi_reason}:{execution_scope}",
                    "proc_def_id": child_proc_def_id,
                    "participants": participants,
                    "status": "NEW",
                    "role_bindings": role_bindings,
                    "start_date": datetime.now().isoformat(),
                    "tenant_id": process_instance.tenant_id,
                    "parent_proc_inst_id": process_instance.proc_inst_id,
                    "root_proc_inst_id": root_proc_inst_id,
                    "execution_scope": execution_scope
                }
                insert_process_instance(process_instance_data, process_instance.tenant_id)
                print(f"[INFO] Spawned child instance: {child_proc_inst_id} (parent={process_instance.proc_inst_id})")
            except Exception as e:
                print(f"[ERROR] Failed to insert child process instance '{child_proc_inst_id}': {e}")
                continue

            try:
                create_initial_workitem(child_def, child_proc_inst_id, child_proc_def_id, role_bindings, endpoint, process_instance, execution_scope)
                execution_scope += 1
            except Exception as e:
                print(f"[ERROR] Failed to create initial workitem for child '{child_proc_inst_id}': {e}")
                continue

def _execute_script_tasks(process_instance: ProcessInstance, process_result: ProcessResult, 
                         process_result_json: dict, process_definition):
    """Execute script tasks in next activities"""
    for activity in process_result.nextActivities:
        activity_obj = process_definition.find_activity_by_id(activity.nextActivityId)
        if activity_obj and activity_obj.type == "scriptTask":
            env_vars = {}
            # Ensure variables_data is not None
            if process_instance.variables_data:
                for variable in process_instance.variables_data:
                    if variable["value"] is None:
                        continue
                    if isinstance(variable["value"], list):
                        variable["value"] = ', '.join(map(str, variable["value"]))
                    if isinstance(variable["value"], dict):
                        variable["value"] = json.dumps(variable["value"])
                    env_vars[variable["key"]] = variable["value"]
            
            result = execute_python_code(activity_obj.pythonCode, env_vars=env_vars)
            
            if result.returncode != 0:
                # Script task execution error
                process_instance.current_activity_ids = [activity.id for activity in process_definition.find_next_activities(activity_obj.id)]
                process_result_json["result"] = result.stderr
            else:
                process_result_json["result"] = result.stdout
                # Script task execution success
                process_instance.current_activity_ids = [
                    act_id for act_id in process_instance.current_activity_ids
                    if act_id != activity_obj.id
                ]
                    
                process_result_json["nextActivities"] = [
                    Activity(**act) for act in process_result_json.get("nextActivities", [])
                    if act.get("nextActivityId") != activity_obj.id
                ]
                completed_activity = CompletedActivity(
                    completedActivityId=activity_obj.id,
                    completedUserEmail=activity.nextUserEmail,
                    result="DONE"
                )
                completed_activity_dict = completed_activity.model_dump()
                process_result_json["completedActivities"].append(completed_activity_dict)
        else:
            result = f"Next activity {activity.nextActivityId} is not a ScriptActivity or not found."
            process_result_json["result"] = result

def _register_event(process_instance: ProcessInstance, process_result: ProcessResult, 
                   process_result_json: dict, process_definition):
    """Register intermediate events when process instance is in WAITING status"""
    try:
        print(f"[DEBUG] Starting event registration for process instance: {process_instance.proc_inst_id}")
        
        # Find intermediate events in current process state
        events = []
        
        # Check current activity IDs for intermediate events
        if process_result.nextActivities:
            for activity in process_result.nextActivities:
                # Check if activity is an intermediate event (gateway with event type)
                gateway = process_definition.find_gateway_by_id(activity.nextActivityId)
                if gateway:
                    events.append({
                        'event_id': gateway.id,
                        'event_name': gateway.name,
                        'event_type': gateway.type,
                        'condition': gateway.condition,
                        'expression': activity.expression,
                        'process_id': process_instance.proc_inst_id,
                        'properties': gateway.properties
                    })
                    print(f"[DEBUG] Found intermediate event: {gateway.id} of type {gateway.type}")
        
        # Register events if found
        if events:
            for event in events:
                _register_single_event(process_instance, event, process_result_json)
                print(f"[INFO] Registered intermediate event: {event['event_id']}")
        else:
            print(f"[DEBUG] No intermediate events found for process instance: {process_instance.proc_inst_id}")
            
    except Exception as e:
        print(f"[ERROR] Failed to register events for process instance {process_instance.proc_inst_id}: {str(e)}")
        # Don't raise exception to avoid breaking the main process flow
        import traceback
        print(traceback.format_exc())
def _is_intermediate_event(gateway) -> bool:
    """Check if gateway represents an intermediate event"""
    intermediate_event_types = [
        "intermediateThrowEvent",
        "intermediateCatchEvent", 
        "timerIntermediateEvent",
        "messageIntermediateEvent",
        "signalIntermediateEvent",
        "conditionalIntermediateEvent",
        "linkIntermediateEvent",
        "escalationIntermediateEvent",
        "errorIntermediateEvent",
        "cancelIntermediateEvent",
        "compensationIntermediateEvent"
    ]
    
    return gateway.type in intermediate_event_types
def _register_single_event(process_instance: ProcessInstance, event: dict, process_result_json: dict):
    """Register a single intermediate event - Implementation placeholder"""
    # TODO: Implement actual event registration logic here
    # This could involve:
    # - Creating event listeners for timer events
    # - Setting up message subscriptions for message events  
    # - Registering signal handlers for signal events
    # - Setting up conditional checks for conditional events
    # - Storing event metadata in database
    
    print(f"[PLACEHOLDER] Event registration logic for {event['event_type']} event {event['event_id']} goes here")
    
    # Example structure for what the implementation might look like:
    _register_timer_event(process_instance, event)
    # elif event['event_type'] == 'messageIntermediateEvent':
    #     _register_message_event(process_instance, event)
    # elif event['event_type'] == 'signalIntermediateEvent':
    #     _register_signal_event(process_instance, event)
    # else:
    #     _register_generic_event(process_instance, event)
    
def _register_timer_event(process_instance: ProcessInstance, event: dict):
    """Register a timer intermediate event"""
    print(f"[INFO] Registering timer intermediate event: {event['event_id']}")
    if event['expression']:
        job_name = f"{event['process_id']}_{event['event_id']}"
        cron_expr = event['expression']
        params = {
            "p_job_name": job_name,
            "p_cron_expr": cron_expr,
            "p_input": {
                "proc_inst_id": event['process_id'],
                "activity_id": event['event_id']
            }
        }
        result = execute_rpc("register_cron_intermidiated", params)
    return result
def _persist_process_data(process_instance: ProcessInstance, process_result: ProcessResult, 
                         process_result_json: dict, process_definition, tenant_id: Optional[str] = None):
    """Persist process data to database"""
    # Upsert workitems
    upsert_todo_workitems(process_instance.model_dump(), process_result_json, process_definition, tenant_id)
    completed_workitems = upsert_completed_workitem(process_instance.model_dump(), process_result_json, process_definition, tenant_id)
    upsert_cancelled_workitem(process_instance.model_dump(), process_result_json, process_definition, tenant_id)
    next_workitems = upsert_next_workitems(process_instance.model_dump(), process_result_json, process_definition, tenant_id)
    
    # Upsert process instance
    if process_instance.status == "NEW":
        process_instance.proc_inst_name = process_result.instanceName
    _, process_instance = upsert_process_instance(process_instance, tenant_id, process_definition)
    
    # Update process_result_json
    process_result_json["instanceId"] = process_instance.proc_inst_id
    process_result_json["instanceName"] = process_instance.proc_inst_name
    process_result_json["workitemIds"] = [workitem.id for workitem in next_workitems] if next_workitems else []
    
def _check_service_tasks(process_instance: ProcessInstance, process_result_json: dict, process_definition):
    try:
        for activity in process_result_json.get("nextActivities", []):
            activity_obj = process_definition.find_activity_by_id(activity.get("nextActivityId"))
            if activity_obj and activity_obj.type == "serviceTask":
                next_workitem = fetch_workitem_by_proc_inst_and_activity(process_instance.proc_inst_id, activity_obj.id, process_instance.tenant_id)
                if next_workitem:
                    upsert_workitem({
                        "id": next_workitem.id,
                        "status": "SUBMITTED",
                    }, process_instance.tenant_id)
    except Exception as e:
        print(f"[ERROR] Failed to check service tasks: {str(e)}")
        raise e
    
def execute_next_activity(process_result_json: dict, tenant_id: Optional[str] = None) -> str:
    try:
        process_result = ProcessResult(**process_result_json)
        
        # Create or get process instance
        process_instance = _create_or_get_process_instance(process_result, process_result_json, tenant_id)
        process_definition = process_instance.process_definition
        
        if process_instance.parent_proc_inst_id:
            parent_process_instance = fetch_process_instance(process_instance.parent_proc_inst_id, tenant_id)
            parent_sub_processes = parent_process_instance.current_activity_ids
            for parent_sub_process_id in parent_sub_processes:
                parent_sub_process = process_definition.find_sub_process_by_id(parent_sub_process_id)
                if parent_sub_process:
                    process_definition = parent_process_instance.process_definition.build_subprocess_definition(parent_sub_process_id)
                    break
        
        # Update process variables
        _update_process_variables(process_instance, process_result.fieldMappings)
        
        
        # Process next activities
        _process_next_activities(process_instance, process_result, process_result_json, process_definition)
        
        # Process sub processes
        _process_sub_processes(process_instance, process_result, process_result_json, process_definition)
        
        # Execute script tasks
        _execute_script_tasks(process_instance, process_result, process_result_json, process_definition)
        
        # Persist data
        _persist_process_data(process_instance, process_result, process_result_json, process_definition, tenant_id)
        
        # Regester event
        _register_event(process_instance, process_result, process_result_json, process_definition)
        
        # Check service tasks
        _check_service_tasks(process_instance, process_result_json, process_definition)
        
        # Progress parent if all children completed
        _progress_parent_if_all_children_completed(process_instance.proc_inst_id, tenant_id)
        
        return json.dumps(process_result_json)
    except Exception as e:
        message_json = json.dumps({"role": "system", "content": str(e)})
        upsert_chat_message(process_instance.proc_inst_id, message_json, tenant_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
    
def _progress_parent_if_all_children_completed(current_proc_inst_id: str, tenant_id: Optional[str] = None):
    """
    현재 인스턴스의 부모가 있고, 부모의 모든 자식 인스턴스가 종료(기본: COMPLETED)되면
    부모 인스턴스의 current_activity_ids 중 subProcess 활동의 워크아이템을 SUBMITTED로 바꿔 재개를 트리거한다.
    """
    try:
        child_inst = fetch_process_instance(current_proc_inst_id, tenant_id)
        if not child_inst or not getattr(child_inst, "parent_proc_inst_id", None):
            return
        parent_id = child_inst.parent_proc_inst_id

        children = fetch_child_instances_by_parent(parent_id, tenant_id) or []
        if not children:
            return

        terminal_statuses = {"COMPLETED"}
        if any((c.get("status") not in terminal_statuses) for c in children):
            return

        parent_inst = fetch_process_instance(parent_id, tenant_id)
        if not parent_inst:
            return

        parent_def = getattr(parent_inst, "process_definition", None)
        if not parent_def:
            print(f"[WARN] Parent process_definition not loaded for {parent_id}")
            return

        for act_id in (parent_inst.current_activity_ids or []):
            if parent_def.find_sub_process_by_id(act_id):
                workitem = fetch_workitem_by_proc_inst_and_activity(parent_id, act_id, tenant_id)
                if workitem and getattr(workitem, "status", None) != "SUBMITTED":
                    upsert_workitem({"id": workitem.id, "status": "SUBMITTED"}, tenant_id)
                    print(f"[INFO] Parent({parent_id}) subprocess workitem {workitem.id} -> SUBMITTED")
    except Exception as e:
        print(f"[ERROR] Parent progression check failed for {current_proc_inst_id}: {e}")



MEMENTO_SERVICE_URL = os.getenv("MEMENTO_SERVICE_URL", "http://memento-service:8005")

def process_output(workitem, tenant_id):
    try:
        if workitem["output"] is None or workitem["output"] == {}:
            return
        url = f"{MEMENTO_SERVICE_URL}/process-output"
        response = requests.post(url, json={
            "workitem_id": workitem["id"],
            "tenant_id": tenant_id
        })
        return response.json()
    except Exception as e:
        print(f"[ERROR] Error in process_output for workitem {workitem.get('id', 'unknown')}: {str(e)}")
        return None



def get_workitem_position(workitem: dict) -> Tuple[bool, bool]:
    """
    워크아이템이 프로세스 정의에서 첫 번째 또는 마지막 워크아이템인지 판별
    startEvent와 연결된 액티비티가 첫 번째, endEvent와 연결된 액티비티가 마지막
    
    Returns:
        Tuple[bool, bool]: (is_first, is_last)
    """
    proc_inst_id = workitem.get('proc_inst_id')
    proc_def_id = workitem.get('proc_def_id')
    activity_id = workitem.get('activity_id')
    tenant_id = workitem.get('tenant_id')
    
    if not proc_inst_id or proc_inst_id == "new" or not proc_def_id or not activity_id:
        return False, False
    
    try:
        # 프로세스 정의 조회
        # 1순위: 워크아이템에 저장된 version_tag / version
        version_tag = workitem.get('version_tag')
        version = workitem.get('version')

        # 2순위: 인스턴스에 저장된 proc_def_version(arcv_id)
        # - workitem에 버전(version_tag/version)이 명시된 경우에는 arcv_id를 넘기지 않아
        #   "workitem 버전 우선" 규칙이 깨지지 않도록 합니다.
        process_instance = fetch_process_instance(proc_inst_id, tenant_id)
        arcv_id = None
        if not version_tag and not version:
            if process_instance and getattr(process_instance, "proc_def_version", None):
                arcv_id = process_instance.proc_def_version

        process_definition_json = fetch_process_definition_by_version(
            proc_def_id,
            version_tag,
            version,
            tenant_id,
            arcv_id,
        )
        process_definition = load_process_definition(process_definition_json)
        
        # 첫 번째 액티비티 확인 (startEvent와 연결된 액티비티)
        is_first = process_definition.is_starting_activity(activity_id)
        
        # 마지막 액티비티 확인 (endEvent와 연결된 액티비티)
        end_activity = process_definition.find_end_activity()
        is_last = end_activity and end_activity.id == activity_id
        
        return is_first, is_last
        
    except Exception as e:
        print(f"[ERROR] Failed to determine workitem position for {workitem.get('id')}: {str(e)}")
        return False, False

def update_instance_status_on_error(workitem: dict, is_first: bool, is_last: bool):
    """
    예외 발생 시 인스턴스 상태를 업데이트
    """
    proc_inst_id = workitem.get('proc_inst_id')
    if not proc_inst_id or proc_inst_id == "new":
        return
    
    try:
        if is_first:
            process_instance = fetch_process_instance(proc_inst_id, workitem.get('tenant_id'))
            if process_instance:
                process_instance.status = "RUNNING"
                upsert_process_instance(process_instance, workitem.get('tenant_id'))
                print(f"[INFO] Updated instance {proc_inst_id} status to RUNNING due to first workitem failure")
        
        elif is_last:
            process_instance = fetch_process_instance(proc_inst_id, workitem.get('tenant_id'))
            if process_instance:
                process_instance.status = "COMPLETED"
                upsert_process_instance(process_instance, workitem.get('tenant_id'))
                print(f"[INFO] Updated instance {proc_inst_id} status to COMPLETED due to last workitem failure")
                
    except Exception as e:
        print(f"[ERROR] Failed to update instance status for {proc_inst_id}: {str(e)}")

from typing import Any, Dict, List, Optional


def get_gateway_condition_data(workitem: dict, process_definition: Any, gateway_id: str):
    """
    워크아이템 실행에 필요한 게이트웨이 조건 데이터 추출
    """
    try:
        gateway = process_definition.find_gateway_by_id(gateway_id)
        if not gateway:
            return None
        
        condition_data = {}
        if gateway.conditionData:
            process_instance_id = workitem.get('proc_inst_id')
            # 각 필드의 값을 가져오기
            field_values = {}
            for condition_field in gateway.conditionData:
                field_value = get_field_value(condition_field, process_definition, process_instance_id, workitem.get('tenant_id'))
                if field_value:
                    field_values[condition_field] = field_value
            
            # 폼별로 그룹화
            grouped_data = group_fields_by_form(field_values)
            condition_data.update(grouped_data)

        return condition_data
    except Exception as e:
        print(f"[ERROR] Failed to get gateway condition data for {workitem.get('id')}: {str(e)}")
        return None
    
def get_sequence_condition_data(process_definition: Any, current_activity_id: str, next_activities: List[str]):
    """
    워크아이템 실행에 필요한 시퀀스 조건 데이터 추출
    - current_activity_id에서 시작하여 next_activities 중 어느 하나에 도달할 때까지의 경로에 포함된
      모든 시퀀스의 properties를 수집한다 (게이트웨이는 건너가되, 해당 시퀀스는 수집 대상).
    """
    try:
        sequence_condition_data = {}
        if not process_definition or not hasattr(process_definition, "sequences"):
            return sequence_condition_data

        targets: set = set(next_activities or [])
        visited_nodes: set = set()
        visited_sequences: set = set()
        stack: List[str] = [current_activity_id]

        while stack:
            node_id = stack.pop()
            if node_id in visited_nodes:
                continue
            visited_nodes.add(node_id)

            stop_here = node_id in targets

            if stop_here:
                continue

            for seq in process_definition.sequences or []:
                if getattr(seq, "source", None) != node_id:
                    continue
                if getattr(seq, "id", None) in visited_sequences:
                    continue
                visited_sequences.add(seq.id)

                properties = getattr(seq, "properties", None)
                if properties:
                    try:
                        properties_json = json.loads(properties)
                        sequence_condition_data[seq.id] = properties_json
                    except Exception:
                        pass
        
                if seq.name:
                    sequence_condition_data.setdefault(seq.id, {})["name"] = seq.name
                    
                if not stop_here:
                    next_node = getattr(seq, "target", None)
                    if next_node and next_node not in visited_nodes:
                        stack.append(next_node)

        return sequence_condition_data
    except Exception as e:
        print(f"[ERROR] Failed to get sequence condition data: {str(e)}")
        return None
    
async def run_prompt_and_parse(prompt_tmpl, chain_input, workitem, tenant_id, parser, merged_log=None, log_prefix="[LLM]", enable_logging=True):
    log_text = merged_log + ""
    collected_text = ""
    num_of_chunk = 0

    async for chunk in model.astream(prompt_tmpl.format(**chain_input)):
        token = chunk.content
        collected_text += token
        log_text += token

        # 실시간 로그 적재 (enable_logging이 True일 때만)
        if enable_logging:
            upsert_queue.put((
                {
                    "id": workitem['id'],
                    "log": f"{log_prefix} {log_text}"
                },
                tenant_id
            ))
            num_of_chunk += 1
            if num_of_chunk % 10 == 0:
                upsert_workitem({"id": workitem['id'], "log": log_text}, tenant_id)

    # 파싱 리트라이
    parsed_output = None
    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            parsed_output = parser.parse(collected_text)
            break
        except Exception as parse_error:
            retry_count += 1
            print(f"[WARNING] JSON parsing attempt {retry_count} failed for workitem {workitem['id']}: {str(parse_error)}")

            if retry_count >= max_retries:
                print(f"[ERROR] All JSON parsing attempts failed. Raw response: {collected_text[:500]}...")
                upsert_workitem({
                    "id": workitem['id'],
                    "status": "PENDING",
                    "log": f"JSON parsing failed after {max_retries} attempts: {str(parse_error)}"
                }, tenant_id)
                error_message = json.dumps({
                    "role": "system",
                    "content": f"JSON 파싱 오류가 발생했습니다: {str(parse_error)}"
                })
                upsert_chat_message(workitem['proc_inst_id'], error_message, tenant_id)
                raise parse_error

            await asyncio.sleep(0.5)

    if parsed_output is None:
        raise Exception("Failed to parse JSON response after all retry attempts")

    return parsed_output, log_text



async def _evaluate_sequence_conditions(model, parser, process_definition, all_workitem_input_data, workitem_input_data, sequence_condition_data, ui_definitions):
    sequence_condition_data = sequence_condition_data or {}
    nl_condition_sequences = []

    for sequence in process_definition.sequences or []:
        condition_data = sequence_condition_data.get(sequence.id)
        if not isinstance(condition_data, dict):
            continue

        expr = condition_data.get("conditionFunction")
        if isinstance(expr, str) and expr.strip():
            eval_contexts: list[dict] = []

            # NEW: Support scoped condition function syntax: "<form_key>: <expression>"
            expr_text = expr.strip()
            scoped_context = None
            if ":" in expr_text:
                try:
                    prefix, rhs = expr_text.split(":", 1)
                    prefix = prefix.strip()
                    rhs = rhs.strip()
                    if prefix and isinstance(all_workitem_input_data, dict):
                        maybe_ctx = all_workitem_input_data.get(prefix)
                        if isinstance(maybe_ctx, dict):
                            scoped_context = maybe_ctx
                            expr = rhs
                except Exception:
                    pass

            seen = set()

            def _collect_contexts(value):
                if isinstance(value, dict):
                    obj_id = id(value)
                    if obj_id in seen:
                        return
    
                    seen.add(obj_id)
                    eval_contexts.append(value)
    
                    for nested in value.values():
                        _collect_contexts(nested)
    
                elif isinstance(value, list):
                    for nested in value:
                        _collect_contexts(nested)

            if scoped_context is not None:
                eval_contexts.append(scoped_context)
            else:
                if all_workitem_input_data:
                    _collect_contexts(all_workitem_input_data)

                if not eval_contexts:
                    eval_contexts.append({})

            condition_eval = False
            last_error: Exception | None = None
            evaluated = False

            for context in eval_contexts:
                try:
                    result = bool(eval(expr, {"__builtins__": {}}, context))
                except Exception as e:
                    last_error = e
                else:
                    evaluated = True
                    if result:
                        condition_eval = True
                        break

            if not condition_eval and last_error and not evaluated:
                print(f"[WARN] conditionFunction eval failed on {sequence.id}: {last_error}")

            _set_condition_eval(sequence_condition_data, sequence.id, condition_eval)
            continue

        condition_text = condition_data.get("condition")
        if isinstance(condition_text, str) and condition_text.strip():
            nl_condition_sequences.append((sequence.id, condition_text.strip()))

    if nl_condition_sequences:
        await _evaluate_nl_conditions(model, parser, all_workitem_input_data, workitem_input_data, nl_condition_sequences, sequence_condition_data, ui_definitions)


def _set_condition_eval(sequence_condition_data, seq_id, condition_met, reason=None):
    def _to_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("true", "yes", "y", "1"): return True
            if v in ("false", "no", "n", "0", "none", "null", ""): return False
        return False

    entry = sequence_condition_data.setdefault(seq_id, {})
    entry["conditionEval"] = _to_bool(condition_met)
    if isinstance(reason, str) and reason.strip():
        entry["conditionReason"] = reason.strip()


async def _evaluate_nl_conditions(model, parser, all_workitem_input_data, workitem_input_data, nl_condition_sequences, sequence_condition_data, ui_definitions):
    ui_field_keys = collect_ui_field_keys(ui_definitions)
    all_workitem_input_data = apply_field_name_annotation_recursively(all_workitem_input_data, ui_definitions, ui_field_keys)
    workitem_input_data = apply_field_name_annotation_recursively(workitem_input_data, ui_definitions, ui_field_keys)
    def _normalize(obj):
        if isinstance(obj, dict):
            return {str(k): _normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_normalize(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    runtime_context = {"current_output": _normalize(all_workitem_input_data), "previous_outputs": _normalize(workitem_input_data)}
    # Build NL conditions with priority: condition > name (from sequence_condition_data)
    conditions_payload = []
    existing_ids = set()
    try:
        for seq_id, text in nl_condition_sequences:
            if isinstance(text, str) and text.strip():
                conditions_payload.append({"sequenceId": seq_id, "condition": text})
                existing_ids.add(str(seq_id))
    except Exception:
        # Fallback: tolerate unexpected shapes
        pass

    if isinstance(sequence_condition_data, dict):
        for sid, sdata in sequence_condition_data.items():
            sid_str = str(sid)
            if sid_str in existing_ids:
                continue
            cond_text = None
            name_text = None
            try:
                cond_text = sdata.get("condition") if isinstance(sdata, dict) else None
            except Exception:
                cond_text = None
            try:
                name_text = sdata.get("name") if isinstance(sdata, dict) else None
            except Exception:
                name_text = None

            if isinstance(cond_text, str) and cond_text.strip():
                conditions_payload.append({"sequenceId": sid, "condition": cond_text.strip()})
                existing_ids.add(sid_str)
            elif isinstance(name_text, str) and name_text.strip():
                conditions_payload.append({"sequenceId": sid, "condition": name_text.strip()})
                existing_ids.add(sid_str)

    chain_input_text = {
        "instruction": "You are a BPMN sequence condition evaluator. Use the runtime context JSON and determine whether each natural-language condition is satisfied.",
        "outputFormat": {"results": [{"sequenceId": "...", "conditionMet": True, "reason": "optional explanation"}]},
        "runtimeContext": runtime_context,
        "conditions": conditions_payload
    }

    prompt_tmpl = PromptTemplate.from_template('{chain_input_text}')
    chain_input = {"chain_input_text": json.dumps(chain_input_text, ensure_ascii=False)}

    try:
        response_text = ''
        async for chunk in model.astream(prompt_tmpl.format(**chain_input)):
            token = getattr(chunk, 'content', None)
            if token:
                response_text += token
    except Exception as e:
        print(f"[WARN] condition prompt failed: {e}")
        return

    parsed_response = None
    try:
        parsed_response = json.loads(response_text)
    except Exception:
        try:
            parsed_response = parser.parse(response_text)
        except Exception as parse_error:
            print(f"[WARN] condition prompt parse failed: {parse_error}")
            return

    results = []
    if isinstance(parsed_response, dict):
        for key in ("results", "sequenceResults", "evaluations"):
            value = parsed_response.get(key)
            if isinstance(value, list):
                results = value
                break

    updated_ids: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        seq_id = item.get("sequenceId") or item.get("sequence_id")
        if not seq_id:
            continue
        condition_met = item.get("conditionMet")
        if condition_met is None:
            condition_met = item.get("met")
        if condition_met is None:
            condition_met = item.get("result")
        _set_condition_eval(sequence_condition_data, seq_id, condition_met, item.get("reason"))
        updated_ids.add(seq_id)

    for seq_id, _ in nl_condition_sequences:
        if seq_id not in updated_ids:
            _set_condition_eval(sequence_condition_data, seq_id, False)


# NEW: Minimal timer event expression checker
async def check_event_expression(next_activity_payloads: list[dict], chain_input_next: dict) -> list[dict]:
    """
    Fill 'expression' (or 'dueDate') for timer events among next_activity_payloads using a minimal prompt.
    - Identifies candidates where type == 'event', target event type contains 'timer', and 'expression' is empty.
    - Uses only provided chain_input_next data (events/output/previous_outputs/today) to infer values.
    - Returns updated payload list without raising on errors.
    """
    try:
        if not isinstance(next_activity_payloads, list) or not next_activity_payloads:
            return next_activity_payloads

        events_def = chain_input_next.get("events") or []

        def _get(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _find_event(ev_id: str):
            for e in events_def:
                if _get(e, "id") == ev_id:
                    return e
            return None

        def _is_timer_event(ev_id: str) -> bool:
            e = _find_event(ev_id)
            if not e:
                return False
            # 1) direct 'type' contains 'timer'
            t = str(_get(e, "type") or "").lower()
            if "timer" in t:
                return True
            # 2) BPMN definition type
            dt = str(_get(e, "definitionType") or "").lower()
            if "timer" in dt:
                return True
            # 3) properties include explicit expression
            props = _get(e, "properties")
            if isinstance(props, str):
                try:
                    props = json.loads(props)
                except Exception:
                    props = None
            if isinstance(props, dict) and props.get("expression"):
                return True
            return False

        # Select candidates
        candidates = [p for p in next_activity_payloads or []
                      if isinstance(p, dict)
                      and (p.get("type") == "event")
                      and not p.get("expression")
                      and p.get("nextActivityId")
                      and _is_timer_event(p.get("nextActivityId"))]
        if not candidates:
            return next_activity_payloads

        candidate_events = []  # unresolved candidates for LLM
        resolved_expressions: dict[str, dict] = {}  # ev_id -> {"expression": str, "dueDate": str}

        for p in candidates:
            ev_id = p.get("nextActivityId")
            ev = _find_event(ev_id)
            if ev is None:
                # No metadata; fall back later with name from payload if any
                candidate_events.append({"id": ev_id, "nlText": p.get("nextActivityName") or ""})
            else:
                # Parse properties (may be JSON string)
                props = _get(ev, "properties")
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except Exception:
                        props = None
                # Precedence: expression > expressionNL > name
                explicit_expr = None
                if isinstance(props, dict):
                    explicit_expr = props.get("expression") or None
                # Some models may carry expression under 'condition'
                if not explicit_expr:
                    cond = _get(ev, "condition")
                    if isinstance(cond, dict):
                        explicit_expr = cond.get("expression") or cond.get("cron")
                    elif isinstance(cond, str) and cond.strip():
                        explicit_expr = cond.strip()
                if isinstance(explicit_expr, str) and explicit_expr.strip():
                    resolved_expressions[_get(ev, "id")] = {
                        "expression": explicit_expr.strip(),
                        "dueDate": ""
                    }
                    continue
                # If no explicit expression, prefer expressionNL; else fall back to name/description
                nl_text = None
                if isinstance(props, dict):
                    nl_text = props.get("expressionNL")
                if not isinstance(nl_text, str) or not nl_text.strip():
                    nl_text = _get(ev, "name") or _get(ev, "description") or (p.get("nextActivityName") or "")
                candidate_events.append({
                    "id": _get(ev, "id"),
                    "name": _get(ev, "name"),
                    "type": _get(ev, "type"),
                    "nlText": nl_text or ""
                })

        runtime_context = {
            "today": chain_input_next.get("today"),
            "output": chain_input_next.get("output"),
            "previous_outputs": chain_input_next.get("previous_outputs") or {},
        }

        # If every candidate is already resolved via explicit expression, just apply and return
        if not candidate_events and resolved_expressions:
            for p in next_activity_payloads:
                ev_id = p.get("nextActivityId")
                if ev_id and ev_id in resolved_expressions:
                    expr = resolved_expressions[ev_id].get("expression")
                    due  = resolved_expressions[ev_id].get("dueDate")
                    if expr:
                        p["expression"] = expr
                    if due:
                        p["dueDate"] = due
            return next_activity_payloads

        chain_input_text = {
            "instruction": (
                "당신은 BPMN 타이머 이벤트 플래너입니다. 각 후보의 자연어(nlText)를 "
                "간결한 cron 표현식(우선) 또는 YYYY-MM-DD 형식의 dueDate로 변환하세요. "
                "runtimeContext만 사용하며, 해석 불가하면 비워두세요."
            ),
            "outputFormat": {"timers": [{"id": "...", "expression": "", "dueDate": ""}]},
            "runtimeContext": runtime_context,
            "nlCandidates": candidate_events,
        }

        prompt_tmpl = PromptTemplate.from_template('{chain_input_text}')
        chain_input = {"chain_input_text": json.dumps(chain_input_text, ensure_ascii=False)}

        response_text = ""
        async for chunk in model.astream(prompt_tmpl.format(**chain_input)):
            token = getattr(chunk, 'content', None)
            if token:
                response_text += token

        # Parse
        try:
            parsed = json.loads(response_text)
        except Exception:
            try:
                parsed = parser.parse(response_text)
            except Exception as parse_error:
                print(f"[WARN] check_event_expression parse failed: {parse_error}")
                # Even if LLM parsing failed, apply any resolved expressions
                for p in next_activity_payloads:
                    ev_id = p.get("nextActivityId")
                    if ev_id and ev_id in resolved_expressions:
                        expr = resolved_expressions[ev_id].get("expression")
                        due  = resolved_expressions[ev_id].get("dueDate")
                        if expr:
                            p["expression"] = expr
                        if due:
                            p["dueDate"] = due
                return next_activity_payloads

        timers = None
        if isinstance(parsed, dict):
            for key in ("timers", "events", "results"):
                val = parsed.get(key)
                if isinstance(val, list):
                    timers = val
                    break
        if not isinstance(timers, list):
            # Apply pre-resolved expressions only
            for p in next_activity_payloads:
                ev_id = p.get("nextActivityId")
                if ev_id and ev_id in resolved_expressions:
                    expr = resolved_expressions[ev_id].get("expression")
                    due  = resolved_expressions[ev_id].get("dueDate")
                    if expr:
                        p["expression"] = expr
                    if due:
                        p["dueDate"] = due
            return next_activity_payloads

        timer_map: dict[str, dict] = {}
        for item in timers:
            if not isinstance(item, dict):
                continue
            ev_id = item.get("id") or item.get("eventId") or item.get("nextActivityId")
            if not ev_id:
                continue
            timer_map[ev_id] = {
                "expression": item.get("expression") or "",
                "dueDate": item.get("dueDate") or "",
            }

        for p in next_activity_payloads:
            ev_id = p.get("nextActivityId")
            # Apply explicit expressions first
            if ev_id and ev_id in resolved_expressions:
                expr = resolved_expressions[ev_id].get("expression")
                due  = resolved_expressions[ev_id].get("dueDate")
                if expr:
                    p["expression"] = expr
                if due:
                    p["dueDate"] = due
                continue
            # Otherwise, apply LLM-derived timers if available
            if ev_id and ev_id in timer_map:
                expr = timer_map[ev_id].get("expression")
                due  = timer_map[ev_id].get("dueDate")
                if expr:
                    p["expression"] = expr
                if due:
                    p["dueDate"] = due

        return next_activity_payloads
    except Exception as e:
        print(f"[WARN] check_event_expression failed: {e}")
        return next_activity_payloads


# NEW: Minimal subprocess multi-instance planner
async def check_subprocess_expression(next_activity_payloads: list[dict], chain_input_next: dict) -> list[dict]:
    """
    For next activities that are subprocesses, infer multiInstanceCount and multiInstanceReason
    from provided output/previous_outputs with a minimal prompt. Defaults to 1 and [] when unknown.

    Collection path precedence for deriving multi instances:
    1) determinationCode
    2) foreachVariable (numeric)
    3) foreachVariable (natural-language/path)
    4) name
    비어있거나 해석 불가하면 다음 순서로 계속 시도합니다.
    """
    try:
        if not isinstance(next_activity_payloads, list) or not next_activity_payloads:
            return next_activity_payloads

        sub_defs = chain_input_next.get("subProcesses") or []

        def _get(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _find_sub(sp_id: str):
            for s in sub_defs:
                if _get(s, "id") == sp_id:
                    return s
            return None

        # Select candidates
        candidates = [p for p in next_activity_payloads or []
                      if isinstance(p, dict)
                      and (p.get("type") == "subProcess")
                      and p.get("nextActivityId")]
        if not candidates:
            return next_activity_payloads

        # Extract ordered collection hints: determinationCode -> foreachVariable(numeric) -> foreachVariable(nl) -> name
        # Returns list of (hint_type, value)
        def _extract_collection_hints(sd: Any) -> list[tuple[str, str]]:
            if sd is None:
                return []

            def _from_props(keys: tuple[str, ...]) -> str | None:
                props = _get(sd, "properties") or _get(sd, "uengineProperties")
                props_json = None
                if isinstance(props, str):
                    try:
                        props_json = json.loads(props)
                    except Exception:
                        # Try regex extraction for common keys
                        try:
                            # determinationCode
                            m = re.search(r'"(determinationCode)"\s*:\s*"([^"]+)"', props, re.IGNORECASE)
                            if m:
                                return m.group(2).strip()
                            # foreachVariable variants
                            m = re.search(r'"(forEachVariable|foreachVariable)"\s*:\s*"([^"]+)"', props, re.IGNORECASE)
                            if m:
                                return m.group(2).strip()
                        except Exception:
                            pass
                elif isinstance(props, dict):
                    props_json = props
                if isinstance(props_json, dict):
                    for k in keys:
                        v = props_json.get(k)
                        if isinstance(v, str) and v.strip():
                            return v.strip()
                return None

            hints: list[tuple[str, str]] = []

            # 1) determinationCode (primary)
            det = _get(sd, "determinationCode")
            if isinstance(det, str) and det.strip():
                hints.append(("determinationCode", det.strip()))
            else:
                det_props = _from_props(("determinationCode",))
                if isinstance(det_props, str) and det_props.strip():
                    hints.append(("determinationCode", det_props.strip()))

            # 2) foreachVariable numeric, 3) foreachVariable nl
            fv_val: str | None = None
            for key in ("foreachVariable", "forEachVariable", "foreach_variable", "for_each_variable"):
                v = _get(sd, key)
                if isinstance(v, str) and v.strip():
                    fv_val = v.strip()
                    break
            if not fv_val:
                fv_props = _from_props(("foreachVariable", "forEachVariable", "foreach_variable", "for_each_variable"))
                if isinstance(fv_props, str) and fv_props.strip():
                    fv_val = fv_props.strip()
            if fv_val:
                # try numeric first
                try:
                    cnt_val = int(str(fv_val))
                except Exception:
                    cnt_val = None
                if isinstance(cnt_val, int) and cnt_val > 0:
                    hints.append(("foreach_numeric", str(cnt_val)))
                else:
                    hints.append(("foreach_nl", fv_val))

            # 4) name (last)
            nm = _get(sd, "name")
            if isinstance(nm, str) and nm.strip():
                hints.append(("name", nm.strip()))

            return hints

        def _traverse_path(root: Any, path: str) -> Any:
            cur = root
            if path is None or path == "":
                return cur
            for seg in str(path).split('.'):
                if isinstance(cur, dict):
                    if seg in cur:
                        cur = cur[seg]
                    else:
                        return None
                elif isinstance(cur, list):
                    try:
                        idx = int(seg)
                        if 0 <= idx < len(cur):
                            cur = cur[idx]
                        else:
                            return None
                    except Exception:
                        return None
                else:
                    return None
            return cur

        def _resolve_collection_from_foreach(foreach_var: str, ctx: dict) -> list | None:
            # Prefer aggregated inputs when a form:section style key is used
            all_inputs = ctx.get("all_workitem_input_data") or {}
            output = ctx.get("output") or {}
            previous = ctx.get("previous_outputs") or {}

            # Support "formKey:section.path" style (e.g., vip_newsletter_process_activity_1en8e0l_form:vip_info_section)
            if ":" in str(foreach_var):
                form_key, rest = foreach_var.split(":", 1)
                form_key = form_key.strip()
                rest = rest.strip()
                base = all_inputs.get(form_key)
                if base is not None:
                    val = _traverse_path(base, rest)
                    if isinstance(val, list):
                        return val
                    if isinstance(val, dict):
                        for v in val.values():
                            if isinstance(v, list):
                                return v
                for root in (output, previous):
                    base = root.get(form_key) if isinstance(root, dict) else None
                    if base is not None:
                        val = _traverse_path(base, rest)
                        if isinstance(val, list):
                            return val
                        if isinstance(val, dict):
                            for v in val.values():
                                if isinstance(v, list):
                                    return v
                return None

            for root in (all_inputs, output, previous):
                val = _traverse_path(root, foreach_var)
                if isinstance(val, list):
                    return val
                if isinstance(val, dict):
                    for v in val.values():
                        if isinstance(v, list):
                            return v
            return None

        def _summarize_reason(item: Any) -> str:
            def _clip_value(s: Any, limit: int = 7) -> str:
                try:
                    t = str(s)
                except Exception:
                    t = f"{s}"
                return t if len(t) <= limit else t[:limit]
            try:
                if isinstance(item, dict):
                    pairs = []
                    for k in sorted(item.keys()):
                        v = item[k]
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            pairs.append(f"{k}={_clip_value(v)}")
                    if pairs:
                        return ", ".join(pairs)
                    return _clip_value(json.dumps(item, ensure_ascii=False))
                if isinstance(item, list):
                    preview = ", ".join(_clip_value(x) for x in item[:5])
                    return _clip_value(f"list[{len(item)}]: {preview}")
                return _clip_value(item)
            except Exception:
                try:
                    return _clip_value(json.dumps(item, ensure_ascii=False))
                except Exception:
                    return _clip_value(item)

        collection_hint_map: dict[str, list[tuple[str, str]]] = {}
        for p in candidates:
            sp_id = p.get("nextActivityId")
            sd = _find_sub(sp_id)
            hints = _extract_collection_hints(sd)
            if hints:
                collection_hint_map[sp_id] = hints

        handled_ids: set[str] = set()
        for p in candidates:
            sp_id = p.get("nextActivityId")
            hints = collection_hint_map.get(sp_id) or []
            if not hints:
                continue
            resolved = False
            # Try hints in order; accept only if non-empty when path-based
            for hint_type, hint_val in hints:
                if hint_type == "foreach_numeric":
                    try:
                        cnt = int(str(hint_val))
                    except Exception:
                        cnt = 0
                    if cnt > 0:
                        p["multiInstanceCount"] = str(cnt)
                        p["multiInstanceReason"] = [""] * cnt
                        handled_ids.add(sp_id)
                        resolved = True
                        break
                    continue

                collection = _resolve_collection_from_foreach(hint_val, chain_input_next)
                if isinstance(collection, list) and len(collection) > 0:
                    cnt = len(collection)
                    reasons = [_summarize_reason(elem) for elem in collection]
                    if len(reasons) < cnt:
                        reasons += [""] * (cnt - len(reasons))
                    elif len(reasons) > cnt:
                        reasons = reasons[:cnt]
                    p["multiInstanceCount"] = str(cnt)
                    p["multiInstanceReason"] = reasons
                    handled_ids.add(sp_id)
                    resolved = True
                    break
            if resolved:
                continue

        if handled_ids and all((p.get("nextActivityId") in handled_ids) for p in candidates):
            return next_activity_payloads

        candidate_subs = []
        for p in candidates:
            sp_id = p.get("nextActivityId")
            if sp_id in handled_ids:
                continue
            sd = _find_sub(sp_id)
            if sd is None:
                candidate_subs.append({"id": sp_id, "name": sp_id})
            else:
                candidate_subs.append({
                    "id": _get(sd, "id"),
                    "name": _get(sd, "name") or sp_id,
                    "description": _get(sd, "description") or "",
                })

        if not candidate_subs:
            return next_activity_payloads
        
        runtime_output = chain_input_next.get("output")
        runtime_previous_outputs = chain_input_next.get("previous_outputs") or {}

        try:
            ui_definitions = chain_input_next.get("ui_definitions")
            activity_id = chain_input_next.get("activity_id")

            if ui_definitions and activity_id:
                if isinstance(runtime_output, dict):
                    runtime_output = add_field_names_by_activity(runtime_output, activity_id, ui_definitions)
                if isinstance(runtime_previous_outputs, dict):
                    # Prefer form alias as the name when possible
                    for _k in list(runtime_previous_outputs.keys()):
                        display_name = None
                        try:
                            # find a ui_def whose id matches the form key
                            matched_ui = None
                            for ui in ui_definitions or []:
                                uid = getattr(ui, "id", None) or (ui.get("id") if isinstance(ui, dict) else None)
                                if uid == _k:
                                    matched_ui = ui
                                    break
                            if matched_ui is not None:
                                display_name = _extract_alias_from_html(getattr(matched_ui, "html", None) or (matched_ui.get("html") if isinstance(matched_ui, dict) else None))
                        except Exception:
                            display_name = None

                        # Wrap the top-level form key
                        runtime_previous_outputs = add_field_name_by_key(runtime_previous_outputs, _k, ui_definitions)
                        if display_name:
                            try:
                                cur = runtime_previous_outputs.get(_k)
                                if isinstance(cur, dict) and "name" in cur:
                                    cur["name"] = display_name
                            except Exception:
                                pass

                ui_field_keys = collect_ui_field_keys(ui_definitions)
                runtime_output = apply_field_name_annotation_recursively(runtime_output, ui_definitions, ui_field_keys)
                runtime_previous_outputs = apply_field_name_annotation_recursively(runtime_previous_outputs, ui_definitions, ui_field_keys)
        except Exception:
            # Best-effort enrichment; ignore failures
            pass

        runtime_context = {
            "output": runtime_output,
            "previous_outputs": runtime_previous_outputs,
            "instance_name_pattern": chain_input_next.get("instance_name_pattern") or "",
        }

        chain_input_text = {
            "instruction": (
                "당신은 BPMN 서브프로세스 멀티인스턴스 플래너입니다. 각 후보 서브프로세스에 대해 "
                "runtimeContext의 값만을 사용하여 multiInstanceCount(정수)와 multiInstanceReason(문자열 배열)을 결정하세요. "
                "배열 크기 등 명확한 근거가 있을 때만 count>1로 하며, 불명확하면 count=1과 빈 배열을 반환합니다. "
                "Reason은 배열이고 count의 숫자만큼 있으며 각 항목은 해당 인스턴스를 식별/설명할 수 있게 한국어로 간결 요약하세요(객체/JSON은 핵심 필드 요약)."
            ),
            "outputFormat": {"subprocesses": [{"id": "...", "multiInstanceCount": 1, "multiInstanceReason": [""]}]},
            "runtimeContext": runtime_context,
            "candidateSubprocesses": candidate_subs,
        }

        prompt_tmpl = PromptTemplate.from_template('{chain_input_text}')
        chain_input = {"chain_input_text": json.dumps(chain_input_text, ensure_ascii=False)}

        response_text = ""
        async for chunk in model.astream(prompt_tmpl.format(**chain_input)):
            token = getattr(chunk, 'content', None)
            if token:
                response_text += token

        # Parse
        try:
            parsed = json.loads(response_text)
        except Exception:
            try:
                parsed = parser.parse(response_text)
            except Exception as parse_error:
                print(f"[WARN] check_subprocess_expression parse failed: {parse_error}")
                return next_activity_payloads

        subs = None
        if isinstance(parsed, dict):
            for key in ("subprocesses", "nextActivities", "results"):
                val = parsed.get(key)
                if isinstance(val, list):
                    subs = val
                    break
        if not isinstance(subs, list):
            return next_activity_payloads

        sub_map: dict[str, dict] = {}
        for item in subs:
            if not isinstance(item, dict):
                continue
            sp_id = item.get("id") or item.get("subprocessId") or item.get("nextActivityId")
            if not sp_id or sp_id in handled_ids:
                continue
            cnt_raw = item.get("multiInstanceCount")
            try:
                cnt = int(str(cnt_raw)) if cnt_raw is not None else 1
            except Exception:
                cnt = 1
            if cnt < 1:
                cnt = 1
            reasons = item.get("multiInstanceReason")
            if not isinstance(reasons, list):
                reasons = []
            # Normalize reasons to strings
            norm_reasons = []
            for r in reasons[:cnt]:
                if isinstance(r, (dict, list)):
                    try:
                        norm_reasons.append(json.dumps(r, ensure_ascii=False))
                    except Exception:
                        norm_reasons.append(str(r))
                else:
                    norm_reasons.append(str(r))
            # pad/trim to count
            if len(norm_reasons) < cnt:
                norm_reasons += [""] * (cnt - len(norm_reasons))
            else:
                norm_reasons = norm_reasons[:cnt]

            sub_map[sp_id] = {
                "multiInstanceCount": str(cnt),
                "multiInstanceReason": norm_reasons,
            }

        for p in next_activity_payloads:
            sp_id = p.get("nextActivityId")
            if sp_id and sp_id in sub_map:
                p["multiInstanceCount"] = sub_map[sp_id]["multiInstanceCount"]
                p["multiInstanceReason"] = sub_map[sp_id]["multiInstanceReason"]

        return next_activity_payloads
    except Exception as e:
        print(f"[WARN] check_subprocess_expression failed: {e}")
        return next_activity_payloads


async def check_task_status(next_activity_payloads: list[dict], chain_input_next: dict) -> list[dict]:
    try:
        if not isinstance(next_activity_payloads, list) or not next_activity_payloads:
            return next_activity_payloads

        activity_id = chain_input_next.get("activity_id")
        sequences = chain_input_next.get("sequences") or []
        gateways  = chain_input_next.get("gateways")  or []
        branch_merged_workitems = chain_input_next.get("branch_merged_workitems") or []

        DONE_STATES = {"DONE", "SUBMITTED", "COMPLETED"}

        def _get(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _norm_id(v):
            return str(v) if v is not None else None

        seqs_by_source: dict[str, list] = {}
        seqs_by_target: dict[str, list] = {}
        for s in sequences:
            src = _norm_id(_get(s, "source") or _get(s, "sourceRef"))
            tgt = _norm_id(_get(s, "target") or _get(s, "targetRef"))
            if src:
                seqs_by_source.setdefault(src, []).append(s)
            if tgt:
                seqs_by_target.setdefault(tgt, []).append(s)

        gateway_index: dict[str, dict] = {}
        for g in gateways:
            gid = _norm_id(_get(g, "id"))
            if gid:
                gateway_index[gid] = g

        # Treat branches that are already included in next_activity_payloads as DONE
        consider_done_ids: set[str] = set()
        for p in (next_activity_payloads or []):
            nid = _norm_id(p.get("nextActivityId"))
            if nid:
                consider_done_ids.add(nid)

        def _is_gateway(node_id: str) -> bool:
            return node_id in gateway_index

        def _gw_type(node_id: str) -> str:
            g = gateway_index.get(node_id) or {}
            t = (_get(g, "type") or "").lower()
            if "exclusive" in t or t in ("xor", "xorgateway"):
                return "exclusive"
            if "inclusive" in t or t in ("or", "orgateway"):
                return "inclusive"
            if "parallel" in t or t in ("and", "andgateway"):
                return "parallel"
            return t or "unknown"

        def _seq_exists(src: str, tgt: str) -> bool:
            for s in seqs_by_source.get(src, []) or []:
                st = _norm_id(_get(s, "target") or _get(s, "targetRef"))
                if st == tgt:
                    return True
            return False

        def _classify_path(current_id: str, target_id: str):
            """
            Returns (path_type, gateway_id, join_or_split)
            - path_type: "direct" | "via_gateway" | "unknown"
            - gateway_id: id or None
            - join_or_split: "join" | "split" | None (only if via_gateway)
            """
            if _seq_exists(current_id, target_id):
                return "direct", None, None

            for s in seqs_by_source.get(current_id, []) or []:
                gw = _norm_id(_get(s, "target") or _get(s, "targetRef"))
                if gw and _is_gateway(gw) and _seq_exists(gw, target_id):
                    incomings = len(seqs_by_target.get(gw, []) or [])
                    outgoings = len(seqs_by_source.get(gw, []) or [])
                    join_or_split = None
                    if incomings >= 2 and outgoings == 1:
                        join_or_split = "join"
                    elif incomings == 1 and outgoings >= 2:
                        join_or_split = "split"
                    return "via_gateway", gw, join_or_split

            return "unknown", None, None

        def _all_parallel_done() -> bool:
            if not branch_merged_workitems:
                return True
            has_in_progress = False
            for wi in branch_merged_workitems:
                aid = _norm_id(_get(wi, "activity_id") or _get(wi, "activityId"))
                # If this branch is already planned in next activities, treat as DONE
                if aid in consider_done_ids:
                    continue
                st = (_get(wi, "status") or "").upper()
                if st == "IN_PROGRESS":
                    has_in_progress = True
                    break
                if st not in DONE_STATES:
                    return False
            return not has_in_progress

        def _no_in_progress_in_parallel() -> bool:
            for wi in branch_merged_workitems:
                aid = _norm_id(_get(wi, "activity_id") or _get(wi, "activityId"))
                # If this branch is already planned in next activities, ignore its status
                if aid in consider_done_ids:
                    continue
                st = (_get(wi, "status") or "").upper()
                if st == "IN_PROGRESS":
                    return False
            return True

        filtered: list[dict] = []
        cur_id = _norm_id(activity_id)

        for p in next_activity_payloads:
            nid = _norm_id(p.get("nextActivityId"))
            if not cur_id or not nid:
                filtered.append(p)
                continue

            path_type, gw_id, join_or_split = _classify_path(cur_id, nid)

            keep = True
            if path_type in ("direct", "unknown"):
                keep = _all_parallel_done()
            elif path_type == "via_gateway":
                gtype = _gw_type(gw_id)
                if join_or_split == "join":
                    if gtype == "parallel":
                        keep = _all_parallel_done()
                    elif gtype == "inclusive":
                        keep = _no_in_progress_in_parallel()
                    elif gtype == "exclusive":
                        keep = True
                    else:
                        keep = _all_parallel_done()
                else:
                    keep = True

            if keep:
                filtered.append(p)

        return filtered
    except Exception as e:
        print(f"[WARN] check_task_status failed: {e}")
        return next_activity_payloads


async def check_role_binding(next_activity_payloads: list[dict], chain_input_next: dict) -> list[dict]:
    """
    Use a minimal LLM prompt to resolve nextUserEmail for next activities based on roleBindings,
    organizations, and runtime inputs. This only fills missing nextUserEmail values and keeps
    existing values/items intact (no filtering or removal).

    Rules encoded in the prompt:
    - roleBindings: default assignments; if role not present or endpoint empty, infer from inputs/orgs.
    - If roleBindings.endpoint is an array, join with ',' for nextUserEmail.
    - If role is external_customer, use an email found within inputs only (otherwise exclude).
    - Try similar-key lookup from output/previous_outputs (e.g., role-like names, '*email*').
    - If an org member is found by name/email, use its id as nextUserEmail.
    - If unresolved, exclude silently.
    """
    try:
        if not isinstance(next_activity_payloads, list) or not next_activity_payloads:
            return next_activity_payloads

        activities = chain_input_next.get("activities") or []
        role_bindings = chain_input_next.get("role_bindings") or []
        organizations = chain_input_next.get("organizations") or []
        output = chain_input_next.get("output") or {}
        previous_outputs = chain_input_next.get("previous_outputs") or {}

        def _get(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        # Collect candidate activities that need user resolution (skip events)
        candidates: list[dict] = []
        for p in next_activity_payloads:
            if not isinstance(p, dict):
                continue
            if (p.get("type") or "").lower() == "event":
                continue
            # Only resolve when nextUserEmail is missing/blank
            if isinstance(p.get("nextUserEmail"), str) and p.get("nextUserEmail").strip():
                continue
            if p.get("nextUserEmail") not in (None, "") and not isinstance(p.get("nextUserEmail"), str):
                continue
            act_id = p.get("nextActivityId")
            if not act_id:
                continue
            activity_obj = None
            for a in activities:
                if _get(a, "id") == act_id:
                    activity_obj = a
                    break
            candidates.append({
                "id": act_id,
                "name": _get(activity_obj, "name") or act_id,
                "role": _get(activity_obj, "role") or "",
            })

        if not candidates:
            return next_activity_payloads

        chain_input_text = {
            "instruction": (
                "당신은 역할 할당기입니다. 아래 규칙을 따르세요:\n"
                "1) roleBindings에 현재 액티비티의 role이 있고 endpoint가 존재하면 사용합니다.\n"
                "   - endpoint가 배열이면 ','로 조인하여 nextUserEmail로 설정합니다.\n"
                "2) endpoint가 비어있거나 해당 role이 없으면, output/previous_outputs에서 역할명과 유사한 키 또는 'email' 관련 값을 찾으세요.\n"
                "   - 값이 이메일이면 그대로 사용하거나, 이름/이메일로 organizations에서 구성원을 찾아 해당 id를 nextUserEmail로 설정합니다.\n"
                "3) role이 external_customer인 경우 입력 구조 내에서 email을 찾을 수 있을 때만 사용합니다(없으면 제외).\n"
                "4) 유효 email/id를 결정할 수 없으면 오류 없이 해당 항목을 제외합니다."
            ),
            "outputFormat": {"assignments": [{"id": "...", "nextUserEmail": "", "keep": True}]},
            "candidates": candidates,
            "roleBindings": role_bindings,
            "organizations": organizations,
            "runtimeContext": {
                "output": output,
                "previous_outputs": previous_outputs,
            },
        }

        prompt_tmpl = PromptTemplate.from_template('{chain_input_text}')
        chain_input = {"chain_input_text": json.dumps(chain_input_text, ensure_ascii=False)}

        response_text = ""
        async for chunk in model.astream(prompt_tmpl.format(**chain_input)):
            token = getattr(chunk, 'content', None)
            if token:
                response_text += token

        try:
            parsed = json.loads(response_text)
        except Exception:
            try:
                parsed = parser.parse(response_text)
            except Exception as parse_error:
                print(f"[WARN] check_role_binding parse failed: {parse_error}")
                return next_activity_payloads

        assignments = None
        if isinstance(parsed, dict):
            for key in ("assignments", "results", "nextActivities"):
                val = parsed.get(key)
                if isinstance(val, list):
                    assignments = val
                    break
        if not isinstance(assignments, list):
            return next_activity_payloads

        # Build map id -> nextUserEmail (non-empty only)
        amap: dict[str, str] = {}
        for item in assignments:
            if not isinstance(item, dict):
                continue
            cid = item.get("id") or item.get("nextActivityId")
            if not cid:
                continue
            email_or_id = item.get("nextUserEmail")
            if isinstance(email_or_id, str) and email_or_id.strip():
                amap[str(cid)] = email_or_id.strip()

        # Fill only missing nextUserEmail; preserve all entries
        for p in next_activity_payloads:
            if not isinstance(p, dict):
                continue
            if (p.get("type") or "").lower() == "event":
                continue
            existing = p.get("nextUserEmail")
            if isinstance(existing, str) and existing.strip():
                continue
            if existing not in (None, "") and not isinstance(existing, str):
                continue
            nid = p.get("nextActivityId")
            if nid and nid in amap:
                p["nextUserEmail"] = amap[nid]

        return next_activity_payloads
    except Exception as e:
        print(f"[WARN] check_role_binding failed: {e}")
        return next_activity_payloads

def run_completed_determination(completed_json, chain_input_completed):
    CHECKPOINTS_REQUIRED = False
    CONDITIONLESS_PROCEEDS = True
    HONOR_SYSTEM_ERROR = True
    CHECKPOINTS_MODE = "ALL"

    if not isinstance(completed_json, dict):
        completed_json = {}
    completed_json.setdefault("completedActivities", [])

    activity_id = chain_input_completed.get("activity_id")
    user_email = chain_input_completed.get("user_email")
    output = chain_input_completed.get("output") or {}
    previous_outputs = chain_input_completed.get("previous_outputs") or {}
    sequences = chain_input_completed.get("sequences") or []
    sequence_conditions = chain_input_completed.get("sequence_conditions") or {}
    activities = chain_input_completed.get("activities") or []
    gateways = chain_input_completed.get("gateways") or []

    def obj_to_dict(x):
        if isinstance(x, dict):
            return x
        d = {}
        for k in dir(x):
            if k.startswith("_"):
                continue
            try:
                v = getattr(x, k)
            except Exception:
                continue
            if callable(v):
                continue
            d[k] = v
        return d

    def safe_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            s = x.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    v = json.loads(s)
                    return v if isinstance(v, dict) else {}
                except Exception:
                    return {}
        return {}

    def safe_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    v = json.loads(s)
                    return v if isinstance(v, list) else []
                except Exception:
                    return []
        return []

    def get_activity_meta(items, aid):
        for a in items:
            d = obj_to_dict(a)
            if d.get("id") == aid:
                props = safe_dict(d.get("properties") or d.get("uengineProperties"))
                cps_raw = d.get("checkpoints")
                if cps_raw is None:
                    cps_raw = props.get("checkpoints")
                cps = safe_list(cps_raw)
                name = d.get("name") or (props.get("name") if isinstance(props, dict) else "") or ""
                return name, cps, d
        return "", [], None

    def dot_get(root, path):
        cur = root
        if not path:
            return True, cur
        for seg in str(path).split("."):
            if isinstance(cur, dict):
                if seg in cur:
                    cur = cur[seg]
                else:
                    return False, None
            elif isinstance(cur, list):
                try:
                    idx = int(seg)
                except:
                    return False, None
                if 0 <= idx < len(cur):
                    cur = cur[idx]
                else:
                    return False, None
            else:
                return False, None
        return True, cur

    def to_number(x):
        try:
            if isinstance(x, bool):
                return x
            return float(x)
        except:
            return x

    def cmp_values(lv, op, rv):
        if op == "==": return lv == rv
        if op == "!=": return lv != rv
        if op in (">", ">=", "<", "<="):
            ln, rn = to_number(lv), to_number(rv)
            if isinstance(ln, (int, float)) and isinstance(rn, (int, float)):
                if op == ">": return ln > rn
                if op == ">=": return ln >= rn
                if op == "<": return ln < rn
                if op == "<=": return ln <= rn
            return False
        if op == "in":
            try: return lv in rv
            except: return False
        if op == "not in":
            try: return lv not in rv
            except: return False
        if op == "contains":
            try: return rv in lv
            except: return False
        return False

    def parse_literal(s):
        if isinstance(s, str) and ((s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"'))):
            return s[1:-1]
        try:
            return int(s)
        except:
            try:
                return float(s)
            except:
                if isinstance(s, str):
                    sl = s.lower()
                    if sl == "true": return True
                    if sl == "false": return False
                    if sl == "null": return None
                return s

    def eval_predicate(pred, ctx):
        if isinstance(pred, dict):
            field = pred.get("field")
            op = pred.get("op", "==")
            rv = pred.get("value")
            ok, lv = dot_get(ctx, field)
            if not ok:
                return False, ("DATA_FIELD_NOT_EXIST", f"필드 없음: {field}")
            good = cmp_values(lv, op, rv)
            return (good, None if good else ("PROCEED_CONDITION_NOT_MET", f"{field} {op} {rv} 불만족"))
        if isinstance(pred, str):
            txt = pred.strip()
            for op in [" not in ", " contains ", ">=", "<=", "==", "!=", ">", "<", " in "]:
                if op in txt:
                    left, right = txt.split(op, 1)
                    left, right = left.strip(), right.strip()
                    if op.strip() in ("in", "not in") and (right.startswith("[") and right.endswith("]")):
                        try:
                            rv = json.loads(right)
                        except:
                            rv = right
                    elif op.strip() in ("in", "not in") and "," in right:
                        rv = [parse_literal(x.strip()) for x in right.split(",")]
                    else:
                        rv = parse_literal(right)
                    ok, lv = dot_get(ctx, left)
                    if not ok:
                        return False, ("DATA_FIELD_NOT_EXIST", f"필드 없음: {left}")
                    good = cmp_values(lv, op.strip(), rv)
                    return (good, None if good else ("PROCEED_CONDITION_NOT_MET", f"{left} {op.strip()} {rv} 불만족"))
            # 단순 필드명인 경우: ctx.checkpoints 배열에 포함되어 있는지 확인
            ok, checkpoints_list = dot_get(ctx, "checkpoints")
            if ok and isinstance(checkpoints_list, list):
                # checkpoints 배열에 pred가 포함되어 있는지 확인
                good = txt in checkpoints_list
                return (good, None if good else ("PROCEED_CONDITION_NOT_MET", f"체크포인트 '{txt}'가 체크되지 않았습니다"))
            
            # 기존 로직: 필드 존재 여부와 truthy 값 확인
            ok, lv = dot_get(ctx, txt)
            if not ok:
                return False, ("DATA_FIELD_NOT_EXIST", f"필드 없음: {txt}")
            good = bool(lv)
            return (good, None if good else ("PROCEED_CONDITION_NOT_MET", f"{txt} 값이 falsy"))
        return False, ("SYSTEM_ERROR", "지원되지 않는 체크포인트 타입")

    def outgoing_sequence_objs(seqs, aid):
        outs = []
        for s in seqs:
            d = obj_to_dict(s)
            src = d.get("sourceRef") or d.get("source")
            if src == aid:
                outs.append(d)
        return outs

    def iter_reference_scalars(d, prefix="", acc=None, limit=6):
        return iter_reference_scalars_extractor(d, prefix=prefix, acc=acc, limit=limit)

    def normalize_gateway_type(g):
        t = (g.get("type") or g.get("gatewayType") or "").lower()
        if "gateway" not in t:
            return None
        if "exclusive" in t or t in ("xor", "xorgateway"):
            return "exclusive"
        if "inclusive" in t or t in ("or", "orgateway"):
            return "inclusive"
        if "parallel" in t or t in ("and", "andgateway"):
            return "parallel"
        return None

    activity_name, checkpoints_raw, _ = get_activity_meta(activities, activity_id)
    checkpoints = checkpoints_raw if isinstance(checkpoints_raw, list) else safe_list(checkpoints_raw)

    checkpoint_errors = []
    checkpoints_ok = True
    # if not checkpoints:
    #     checkpoints_ok = not CHECKPOINTS_REQUIRED
    # else:
    #     results = []
    #     for pred in checkpoints:
    #         ok, err = eval_predicate(pred, output)
    #         results.append(ok)
    #         if not ok and err:
    #             checkpoint_errors.append({"type": err[0], "reason": err[1]})
    #     checkpoints_ok = all(results) if CHECKPOINTS_MODE == "ALL" else any(results)

    gateway_map = {}
    for g in gateways:
        gd = obj_to_dict(g)
        gt = normalize_gateway_type(gd)
        if gt:
            gateway_map[gd.get("id")] = {"raw": gd, "type": gt}

    seqs_from_activity = outgoing_sequence_objs(sequences, activity_id)

    def seq_eval_state(seq_id, is_gateway_edge=False):
        sc = sequence_conditions.get(seq_id)
        if isinstance(sc, dict) and "conditionEval" in sc:
            return True if sc.get("conditionEval") else False
        if isinstance(sc, dict) and ("conditionFunction" in sc):
            return None
        return (None if is_gateway_edge else bool(CONDITIONLESS_PROCEEDS))

    used_unknown = False
    allowed_direct = False
    allowed_via_gateway = False

    for s in seqs_from_activity:
        sid = s.get("id")
        tgt = s.get("targetRef") or s.get("target")
        leg_state = seq_eval_state(sid, is_gateway_edge=False)
        if not tgt:
            if leg_state is not False:
                allowed_direct = True
            if leg_state is None:
                used_unknown = True
            continue

        gw = gateway_map.get(tgt)
        if not gw:
            if leg_state is not False:
                allowed_direct = True
            if leg_state is None:
                used_unknown = True
            continue
        if leg_state is False:
            continue

        g_type = gw["type"]
        g_out = outgoing_sequence_objs(sequences, tgt)
        states = []
        for gs in g_out:
            gsid = gs.get("id")
            states.append(seq_eval_state(gsid, is_gateway_edge=True))

        if not states:
            gw_ok = False
        else:
            cnt_true = sum(1 for st in states if st is True)
            cnt_unknown = sum(1 for st in states if st is None)
            if g_type == "parallel":
                gw_ok = (cnt_true == len(states))
                if gw_ok is False and cnt_unknown > 0:
                    used_unknown = True
            else:
                gw_ok = (cnt_true >= 1) or (cnt_unknown >= 1)
                if gw_ok and cnt_true == 0 and cnt_unknown >= 1:
                    used_unknown = True

        if gw_ok:
            allowed_via_gateway = True

    if seqs_from_activity:
        sequences_ok = allowed_direct or allowed_via_gateway
    else:
        sequences_ok = bool(CONDITIONLESS_PROCEEDS)

    cannot = []
    if not checkpoints_ok:
        cannot.extend(checkpoint_errors or [{"type": "PROCEED_CONDITION_NOT_MET", "reason": "체크포인트 불만족"}])
    if checkpoints_ok and not sequences_ok:
        cannot.append({"type": "PROCEED_CONDITION_NOT_MET", "reason": "시퀀스/게이트웨이 조건 불만족"})

    has_system_error = any(e.get("type") == "SYSTEM_ERROR" for e in cannot) if HONOR_SYSTEM_ERROR else False
    result = "DONE" if (checkpoints_ok and sequences_ok and not has_system_error) else "PENDING"

    reference_info = iter_reference_scalars(previous_outputs, limit=6)

    entry = {
        "completedActivityId": activity_id,
        "completedActivityName": activity_name,
        "completedUserEmail": user_email,
        "type": "activity",
        "expression": None,
        "dueDate": None,
        "result": result,
        "description": "체크포인트·시퀀스·게이트웨이 조건 기반 자동 판정",
        "cannotProceedErrors": cannot
    }

    if not used_unknown and result == "DONE":
        replaced = False
        for i, ca in enumerate(completed_json["completedActivities"]):
            if ca.get("completedActivityId") == activity_id:
                seen = {((e or {}).get("type"), (e or {}).get("reason")) for e in (ca.get("cannotProceedErrors") or [])}
                for e in (entry.get("cannotProceedErrors") or []):
                    key = ((e or {}).get("type"), (e or {}).get("reason"))
                    if key not in seen:
                        ca.setdefault("cannotProceedErrors", []).append(e)
                        seen.add(key)
                ca["result"] = "DONE"
                for k in ("completedActivityName", "completedUserEmail", "type", "expression", "dueDate", "description"):
                    if not ca.get(k) and entry.get(k):
                        ca[k] = entry[k]
                replaced = True
                break
        if not replaced:
            completed_json["completedActivities"].append(entry)
    else:
        completed_json["completedActivities"] = [
            ca for ca in completed_json["completedActivities"]
            if ca.get("completedActivityId") != activity_id
        ]

    return completed_json




def resolve_next_activity_payloads(
    process_definition,
    activity_id: str,
    workitem: dict,
    sequence_condition_data: dict | None,
) -> list[dict[str, Any]]:
    """Derive next activity payloads from process definition and evaluated conditions."""
    if not process_definition:
        return []

    role_bindings_for_next = workitem.get("assignees", []) or []

    # 현재 워크아이템의 assignees 정보가 없는 경우,
    # bpm_proc_inst.role_bindings 컬럼 값을 폴백으로 사용
    if not role_bindings_for_next:
        try:
            proc_inst_id = workitem.get("proc_inst_id")
            tenant_id = workitem.get("tenant_id")
            if proc_inst_id and tenant_id:
                proc_inst = fetch_process_instance(proc_inst_id, tenant_id)
                if proc_inst is not None:
                    rb = getattr(proc_inst, "role_bindings", None)
                    # role_bindings가 문자열로 저장되어 있을 가능성도 고려
                    if isinstance(rb, str):
                        try:
                            rb = json.loads(rb)
                        except Exception:
                            rb = None
                    if isinstance(rb, list):
                        role_bindings_for_next = rb
        except Exception as e:
            print(f"[WARN] Failed to fallback role_bindings from process instance: {e}")

    def _extract_endpoint(binding: dict | None) -> str | None:
        if not isinstance(binding, dict):
            return None
        endpoint = binding.get("endpoint")
        if isinstance(endpoint, list):
            return endpoint[0] if endpoint else None
        if isinstance(endpoint, str) and endpoint.strip():
            return endpoint
        return None

    def _resolve_next_user_email(node, node_type: str) -> Optional[str]:
        role_name = getattr(node, "role", None) if node else None
        if isinstance(role_name, str):
            for binding in role_bindings_for_next:
                if isinstance(binding, dict) and binding.get("name") == role_name:
                    endpoint = _extract_endpoint(binding)
                    if endpoint:
                        return endpoint
                    break
        # For events, default to system; otherwise return None so prompt-based resolver can fill it later
        if node_type == "event":
            return "system"
        return None

    sequences_all = list(getattr(process_definition, "sequences", []) or [])

    def _parse_seq_properties(seq: Any) -> dict:
        props = getattr(seq, "properties", None) or getattr(seq, "uengineProperties", None)
        if isinstance(props, dict):
            return props
        if isinstance(props, str):
            try:
                return json.loads(props)
            except Exception:
                pass
        return {}

    def _normalize_gateway_type(gw_obj: Any) -> str | None:
        gw_type = (getattr(gw_obj, "type", None) or "").lower()
        if "exclusive" in gw_type or gw_type in ("xor", "xorgateway"):
            return "exclusive"
        if "inclusive" in gw_type or gw_type in ("or", "orgateway"):
            return "inclusive"
        if "parallel" in gw_type or gw_type in ("and", "andgateway"):
            return "parallel"
        return (gw_type or None)

    def _sequence_condition_state(seq_id: str | None) -> Optional[bool]:
        # Treat missing or empty condition data as True per policy: "empty counts as true"
        if not isinstance(sequence_condition_data, dict):
            return True
        if not seq_id:
            return True
        sc = sequence_condition_data.get(seq_id)
        if not isinstance(sc, dict):
            return True
        if "conditionEval" in sc:
            try:
                return True if sc.get("conditionEval") else False
            except Exception:
                # If evaluation value is malformed, consider it False (explicit but invalid)
                return False
        # No explicit evaluation -> treat as True
        return True

    def _sequence_condition_allows(seq_id: str | None) -> bool:
        if not seq_id or not isinstance(sequence_condition_data, dict):
            return True
        sc = sequence_condition_data.get(seq_id)
        if isinstance(sc, dict) and "conditionEval" in sc:
            return bool(sc.get("conditionEval"))
        return True

    def _allowed_targets_from(source_id: str) -> list[str]:
        # If source is a gateway, apply gateway-specific rules
        gateway_obj = process_definition.find_gateway_by_id(source_id)
        if gateway_obj:
            gw_type = _normalize_gateway_type(gateway_obj)
            out_seqs: list[Any] = []
            for seq in sequences_all:
                src = getattr(seq, "source", None) or getattr(seq, "sourceRef", None)
                if src == source_id:
                    out_seqs.append(seq)

            # Exclusive (XOR): choose exactly one
            if gw_type == "exclusive":
                true_seqs: list[Any] = []
                unknown_seqs: list[Any] = []
                default_seq: Any | None = None

                for s in out_seqs:
                    sid = getattr(s, "id", None)
                    state = _sequence_condition_state(sid)
                    props = _parse_seq_properties(s)
                    is_default = False
                    try:
                        # tolerate various keys/cases
                        for k in ("default", "isDefault", "defaultFlow", "default_flow"):
                            if isinstance(props.get(k), bool) and props.get(k):
                                is_default = True
                                break
                    except Exception:
                        pass
                    if is_default and default_seq is None:
                        default_seq = s
                    if state is True:
                        true_seqs.append(s)
                    elif state is None:
                        unknown_seqs.append(s)

                chosen: list[Any] = []
                if len(true_seqs) == 1:
                    chosen = true_seqs
                elif len(true_seqs) > 1:
                    # Use priority if available; if equal and no default, deterministically pick first
                    def _priority(seq_obj: Any) -> tuple[int, int]:
                        props = _parse_seq_properties(seq_obj)
                        prio = props.get("priority")
                        try:
                            prio_val = int(str(prio)) if prio is not None else 0
                        except Exception:
                            prio_val = 0
                        # stable tie-breaker by id
                        sid = getattr(seq_obj, "id", "") or ""
                        try:
                            sid_key = int(re.sub(r"\D", "", sid) or 0)
                        except Exception:
                            sid_key = 0
                        return (prio_val, sid_key)

                    # Sort ascending (lower number => higher priority)
                    sorted_true = sorted(true_seqs, key=_priority)
                    # Deterministic selection even if all priorities are equal and no default
                    chosen = [sorted_true[0]]
                else:
                    # No explicit True
                    if default_seq is not None:
                        chosen = [default_seq]
                    else:
                        chosen = []  # exclude (no stall globally; just this branch)

                targets: list[str] = []
                for s in chosen:
                    tgt = getattr(s, "target", None) or getattr(s, "targetRef", None)
                    if tgt:
                        targets.append(tgt)
                return targets

            # Other gateway types: fall back to non-strict filtering
            targets: list[str] = []
            for seq in out_seqs:
                if _sequence_condition_allows(getattr(seq, "id", None)):
                    tgt = getattr(seq, "target", None) or getattr(seq, "targetRef", None)
                    if tgt:
                        targets.append(tgt)
            return targets

        # Non-gateway source: apply condition filter as-is
        targets: list[str] = []
        for seq in sequences_all:
            source_ref = getattr(seq, "source", None) or getattr(seq, "sourceRef", None)
            if source_ref == source_id and _sequence_condition_allows(getattr(seq, "id", None)):
                target_ref = getattr(seq, "target", None) or getattr(seq, "targetRef", None)
                if target_ref:
                    targets.append(target_ref)
        return targets

    def _collect_next_nodes() -> list[tuple[str, Any]]:
        collected: list[tuple[str, Any]] = []
        visited_nodes: set[str] = set()
        visited_gateways: set[str] = set()

        def _record(node_type: str, node_obj: Any) -> None:
            node_id = getattr(node_obj, "id", None)
            if not node_id or node_id in visited_nodes:
                return
            visited_nodes.add(node_id)
            collected.append((node_type, node_obj))

        def _visit(target_id: str | None) -> None:
            if not target_id:
                return
            activity_obj = process_definition.find_activity_by_id(target_id)
            if activity_obj:
                _record("activity", activity_obj)
                return
            sub_process_obj = process_definition.find_sub_process_by_id(target_id)
            if sub_process_obj:
                _record("subProcess", sub_process_obj)
                return
            event_obj = process_definition.find_event_by_id(target_id)
            if event_obj and getattr(event_obj, "type", None):
                _record("event", event_obj)
                return
            gateway_obj = process_definition.find_gateway_by_id(target_id)
            if gateway_obj:
                if target_id in visited_gateways:
                    return
                visited_gateways.add(target_id)
                for downstream_id in _allowed_targets_from(target_id):
                    _visit(downstream_id)
                return

        for initial_target in _allowed_targets_from(activity_id):
            _visit(initial_target)
        return collected

    resolved_next_nodes = _collect_next_nodes()
    next_activity_payloads: list[dict[str, Any]] = []

    for node_type, node_obj in resolved_next_nodes:
        node_id = getattr(node_obj, "id", None)
        if not node_id:
            continue
        node_name = getattr(node_obj, "name", "") or node_id
        description = getattr(node_obj, "description", "") or ""
        next_type_value = getattr(node_obj, "type", None) or node_type
        expression_value = None
        if node_type == "event":
            condition_value = getattr(node_obj, "condition", None)
            if isinstance(condition_value, dict):
                expression_value = (
                    condition_value.get("expression")
                    or condition_value.get("cron")
                )
            elif isinstance(condition_value, str):
                expression_value = condition_value

        activity_payload = Activity(
            nextActivityId=node_id,
            nextActivityName=node_name,
            nextUserEmail=_resolve_next_user_email(node_obj, node_type),
            result="IN_PROGRESS",
            description=description,
            type=next_type_value,
            expression=expression_value,
        ).model_dump()
        next_activity_payloads.append(activity_payload)

    return next_activity_payloads

def inject_boundary_events_as_next(
    next_activity_payloads: list[dict[str, Any]],
    attached_activities: list[dict],
    process_definition: Any | None = None,
    events: list[Any] | None = None,
) -> list[dict[str, Any]]:
    """
    다음 액티비티 후보에 boundary(Attached) 이벤트를 포함시킨다.
    - 이미 존재하는 nextActivityId는 중복 추가하지 않는다.
    - 이벤트 payload는 type='event'로 추가하고, condition에서 expression을 유도한다.
    - nextUserEmail은 시스템 이벤트로 간주하여 'system'으로 설정한다.
    """
    try:
        if not isinstance(next_activity_payloads, list):
            next_activity_payloads = []
        if not isinstance(attached_activities, list) or not attached_activities:
            return next_activity_payloads

        existing_ids: set[str] = set()
        for p in next_activity_payloads:
            try:
                nid = p.get("nextActivityId")
                if isinstance(nid, str) and nid:
                    existing_ids.add(nid)
            except Exception:
                continue

        def _get(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _extract_expression(ev) -> str | None:
            cond = _get(ev, "condition")
            if isinstance(cond, dict):
                expr = cond.get("expression") or cond.get("cron")
                if isinstance(expr, str) and expr.strip():
                    return expr.strip()
            elif isinstance(cond, str) and cond.strip():
                return cond.strip()
            return None

        def _find_event_by_id(eid: str):
            if not eid:
                return None
            # Prefer process_definition API
            try:
                if process_definition:
                    ev = process_definition.find_event_by_id(eid)
                    if ev:
                        return ev
            except Exception:
                pass
            # Fallback to provided events list
            try:
                if isinstance(events, list):
                    for e in events:
                        if _get(e, "id") == eid:
                            return e
            except Exception:
                pass
            return None

        for item in attached_activities:
            try:
                events = _get(item, "attached_events") or []
                for ev in events:
                    # attached_events가 문자열 ID일 수 있음
                    if isinstance(ev, (str, int)):
                        ev_id = str(ev)
                        ev_obj = _find_event_by_id(ev_id)
                    else:
                        ev_id = _get(ev, "id")
                        ev_obj = ev

                    if not ev_id or ev_id in existing_ids:
                        continue
                    # boundary 성격 판별: 객체가 있으면 type에서 판단, 없으면 attached 목록이므로 boundary로 간주
                    ev_type_text = str(_get(ev_obj, "type") or "").lower() if ev_obj is not None else "boundary"
                    if "boundary" not in ev_type_text:
                        # attached 목록이지만 타입이 명시적으로 boundary가 아닐 경우만 스킵
                        # (대부분 attached는 boundary이므로, 타입 정보 없으면 포함)
                        if ev_obj is not None:
                            continue

                    ev_name = _get(ev_obj, "name") or ev_id
                    description = _get(ev_obj, "description") or ""
                    expression_value = _extract_expression(ev_obj) if ev_obj is not None else None
                    payload = {
                        "nextActivityId": ev_id,
                        "nextActivityName": ev_name,
                        "nextUserEmail": "system",
                        "result": "IN_PROGRESS",
                        "description": description,
                        "type": "event",
                        "expression": expression_value,
                    }
                    next_activity_payloads.append(payload)
                    existing_ids.add(ev_id)
            except Exception:
                continue

        return next_activity_payloads
    except Exception:
        return next_activity_payloads


def _handle_deploy_approval(
    activity_id: str,
    process_definition: Any,
    output: Dict[str, Any],
    all_workitem_input_data: Dict[str, Any],
    tenant_id: str
) -> bool:
    """
    배포 승인 처리: isDeploy가 true인 activity가 완료되고 승인 상태일 때
    inputData에서 bpmn-uengine-field를 찾아 해당 프로세스의 prod_version을 업데이트합니다.
    
    Args:
        activity_id: 현재 완료된 activity ID
        process_definition: 프로세스 정의 객체
        output: 현재 워크아이템의 출력 데이터
        all_workitem_input_data: 모든 워크아이템 입력 데이터
        tenant_id: 테넌트 ID
    
    Returns:
        bool: 배포 처리 성공 여부
    """
    try:
        # 1. activity에서 isDeploy 속성 확인
        activity = process_definition.find_activity_by_id(activity_id)
        if not activity:
            print(f"[DEBUG] _handle_deploy_approval: Activity {activity_id} not found")
            return False
        
        # 디버깅: activity 객체의 속성 확인
        print(f"[DEBUG] _handle_deploy_approval: Activity found - id={activity.id}, name={activity.name}")
        print(f"[DEBUG] _handle_deploy_approval: hasattr isDeploy={hasattr(activity, 'isDeploy')}")
        
        is_deploy = getattr(activity, 'isDeploy', None)
        print(f"[DEBUG] _handle_deploy_approval: isDeploy value={is_deploy}, type={type(is_deploy)}")
        
        if not is_deploy:
            print(f"[DEBUG] _handle_deploy_approval: isDeploy is falsy ({is_deploy}), skipping")
            return False
        
        print(f"[INFO] Activity {activity_id} has isDeploy=true, checking approval status...")
        
        # 2. output에서 review_result가 approved인지 확인
        review_result = None
        
        # output이 dict인 경우 review_result 찾기
        if isinstance(output, dict):
            # 직접 review_result 확인
            review_result = output.get('review_result')
            
            # 중첩된 구조에서 찾기
            if not review_result:
                for key, value in output.items():
                    if isinstance(value, dict):
                        review_result = value.get('review_result')
                        if review_result:
                            break
        
        if review_result != 'accept':
            print(f"[INFO] Review result is '{review_result}', not 'approved'. Skipping deploy.")
            return False
        
        print(f"[INFO] Review result is 'approved', processing deployment...")
        
        # 3. inputData에서 bpmn-uengine-field 찾기
        input_data = getattr(activity, 'inputData', None) or []
        bpmn_field_path = None
        
        for input_item in input_data:
            if isinstance(input_item, str) and 'bpmn-uengine-field' in input_item:
                bpmn_field_path = input_item
                break
        
        if not bpmn_field_path:
            print(f"[WARN] No bpmn-uengine-field found in inputData for activity {activity_id}")
            return False
        
        # 4. all_workitem_input_data에서 해당 필드 값 추출
        # bpmn_field_path 형식: "form_id.bpmn-uengine-field"
        parts = bpmn_field_path.split('.')
        if len(parts) < 2:
            print(f"[WARN] Invalid bpmn-uengine-field path: {bpmn_field_path}")
            return False
        
        form_id = parts[0]
        field_name = '.'.join(parts[1:])  # bpmn-uengine-field
        
        bpmn_data = None
        
        # all_workitem_input_data에서 폼 데이터 찾기
        if isinstance(all_workitem_input_data, dict):
            form_data = all_workitem_input_data.get(form_id)
            if isinstance(form_data, dict):
                bpmn_data = form_data.get(field_name) or form_data.get('bpmn-uengine-field')
        
        if not bpmn_data:
            print(f"[WARN] No BPMN data found at {bpmn_field_path}")
            return False
        
        # 5. bpmn_data에서 processDefinitionId와 version 추출
        if isinstance(bpmn_data, str):
            try:
                bpmn_data = json.loads(bpmn_data)
            except json.JSONDecodeError:
                print(f"[WARN] Failed to parse BPMN data as JSON")
                return False
        
        if not isinstance(bpmn_data, dict):
            print(f"[WARN] BPMN data is not a dict: {type(bpmn_data)}")
            return False
        
        target_proc_def_id = bpmn_data.get('definition_id')
        target_version = bpmn_data.get('version')
        
        if not target_proc_def_id:
            print(f"[WARN] No processDefinitionId found in BPMN data")
            return False
        
        if not target_version:
            print(f"[WARN] No version found in BPMN data, cannot update prod_version")
            return False
        
        # 6. prod_version 업데이트
        print(f"[INFO] Updating prod_version for {target_proc_def_id} to {target_version}")
        result = update_proc_def_prod_version(target_proc_def_id, str(target_version), tenant_id)
        
        if result:
            print(f"[INFO] Successfully updated prod_version for {target_proc_def_id} to {target_version}")
        else:
            print(f"[WARN] Failed to update prod_version for {target_proc_def_id}")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Failed to handle deploy approval: {str(e)}")
        return False


async def handle_workitem(workitem):
    is_first, is_last = get_workitem_position(workitem)

    if workitem.get('retry', 0) >= 3:
        update_instance_status_on_error(workitem, is_first, is_last)
        return

    activity_id = workitem['activity_id']
    process_definition_id = workitem['proc_def_id']
    process_instance_id = workitem['proc_inst_id']
    tenant_id = workitem['tenant_id']

    # 1순위: 워크아이템에 저장된 버전 정보
    version_tag = workitem.get('version_tag')
    version = workitem.get('version')

    # 2순위: 인스턴스에 저장된 proc_def_version(arcv_id)
    # - workitem에 버전(version_tag/version)이 명시된 경우에는 arcv_id를 넘기지 않아
    #   "workitem 버전 우선" 규칙이 깨지지 않도록 합니다.
    process_instance = fetch_process_instance(process_instance_id, tenant_id)
    arcv_id = None
    if not version_tag and not version:
        if process_instance and getattr(process_instance, "proc_def_version", None):
            arcv_id = process_instance.proc_def_version

    process_definition_json = fetch_process_definition_by_version(
        process_definition_id,
        version_tag,
        version,
        tenant_id,
        arcv_id,
    )
    process_definition = load_process_definition(process_definition_json)

    if workitem['user_id'] != "external_customer":
        if workitem['user_id'] and ',' in workitem['user_id']:
            user_ids = workitem['user_id'].split(',')
            user_info = []
            for user_id in user_ids:
                assignee_info = fetch_assignee_info(user_id)
                user_info.append({
                    "name": assignee_info.get("name", user_id),
                    "email": assignee_info.get("email", user_id),
                    "type": assignee_info.get("type", "unknown"),
                    "info": assignee_info.get("info", {})
                })
        else:
            assignee_info = fetch_assignee_info(workitem['user_id'])
            user_info = {
                "name": assignee_info.get("name", workitem['user_id']),
                "email": assignee_info.get("email", workitem['user_id']),
                "type": assignee_info.get("type", "unknown"),
                "info": assignee_info.get("info", {})
            }
    else:
        user_info = {
            "name": "외부 고객",
            "type": "external_customer",
            "email": workitem['user_id'],
            "info": {}
        }

    today = datetime.now().strftime("%Y-%m-%d")

    ui_definition = fetch_ui_definition_by_activity_id(process_definition_id, activity_id, tenant_id)
    ui_definitions = fetch_ui_definitions_by_def_id(process_definition_id, tenant_id)
    
    output = {}
    if workitem.get('output') and isinstance(workitem['output'], str):
        try:
            output = json.loads(workitem['output'])
        except Exception:
            output = {}
    else:
        output = workitem.get('output') or {}

    form_id = ui_definition.id if ui_definition else None
    if form_id and isinstance(output, dict) and output.get(form_id):
        output = output.get(form_id)

    try:
        next_activities = []
        next_near_activities = []
        gateway_condition_data = None
        sequence_condition_data = None
        
        if process_definition:
            next_activities = [activity.id for activity in process_definition.find_next_activities(activity_id, True)]
            next_near_activities = [activity.id for activity in process_definition.find_near_next_activities(activity_id, True)]
            for act_id in next_activities:
                if process_definition.find_gateway_by_id(act_id):
                    try:
                        gateway_condition_data = get_gateway_condition_data(workitem, process_definition, act_id)
                    except Exception as e:
                        print(f"[ERROR] Failed to get gateway condition data for {workitem.get('id')}: {str(e)}")
                        gateway_condition_data = None
                        
            sequence_condition_data = get_sequence_condition_data(process_definition, activity_id, next_near_activities)
                        
        workitem_input_data = None
        all_workitem_input_data = {}
        try:
            workitem_input_data = get_input_data(workitem, process_definition)
            all_workitem_input_data = get_all_input_data(workitem, process_definition)
        except Exception as e:
            print(f"[ERROR] Failed to get selected info for {workitem.get('id')}: {str(e)}")

        sequence_condition_data = sequence_condition_data or {}
        await _evaluate_sequence_conditions(model, parser, process_definition, all_workitem_input_data, workitem_input_data, sequence_condition_data, ui_definitions)

        attached_activities = []
        for next_activity in next_near_activities:
            activity = process_definition.find_activity_by_id(next_activity)
            if activity and getattr(activity, 'attachedEvents', None):
                attached_activities.append({
                    "activity_id": activity.id,
                    "attached_events": activity.attachedEvents
                })
                
        input_activities = []
        for activity in process_definition.activities:
            input_activities.append({
                "id": activity.id,
                "type": activity.type,
                "role": activity.role,
                "description": activity.description,
                "inputData": activity.inputData,
                "checkpoints": activity.checkpoints
            })
        input_gateways = []
        for gateway in process_definition.gateways:
            input_gateways.append({
                "id": gateway.id,
                "type": gateway.type
            })
        input_sequences = []
        for sequence in process_definition.sequences:
            input_sequences.append({
                "id": sequence.id,
                "source": sequence.source,
                "target": sequence.target
            })

        if not workitem['user_id'] or ',' not in workitem['user_id']:
            user_email_for_prompt = workitem['user_id']
        else:
            user_email_for_prompt = ','.join(workitem['user_id'].split(','))
            
        # instance_name = process_definition_json.get("processDefinitionName") + "_" + workitem['id']
        process_instance = fetch_process_instance(process_instance_id, tenant_id)
        if process_instance and process_instance.proc_inst_name != process_definition_json.get("processDefinitionName"):
            instance_name = process_instance.proc_inst_name
        else:
            instance_name = process_definition_json.get("processDefinitionName") + "_" + process_instance_id.split('.')[1]

        completed_json = {
            "instanceId": process_instance_id,
            "instanceName": instance_name,
            "processDefinitionId": process_definition_id,
            "fieldMappings": [],
            "roleBindings": workitem.get('assignees', []),
            "completedActivities": [],
            "nextActivities": [],
            "cancelledActivities": []
        };

        merged_workitems_from_step = []
                    
        target_containers = process_definition.find_target_containers(activity_id)
        if target_containers:
            for target_container in target_containers:
                block = process_definition.find_block(target_container)
                if block:
                    source_containers = block.node_ids
                    for source_container in source_containers:
                        merged_workitems = fetch_workitem_by_proc_inst_and_activity(process_instance_id, source_container, tenant_id)
                        if merged_workitems:
                            merged_item = {
                                "activity_id": merged_workitems.activity_id,
                                "activity_name": merged_workitems.activity_name,
                                "status": merged_workitems.status,
                            }
                            merged_workitems_from_step.append(merged_item)

        chain_input_completed = {
            "activities": process_definition.activities,
            "gateways": process_definition_json.get('gateways', []),
            "events": process_definition_json.get('events', []),
            "subProcesses": process_definition.subProcesses,
            "sequences": process_definition.sequences,
            "role_bindings": workitem.get('assignees', []),

            "instance_id": process_instance_id,
            "instance_name_pattern": process_definition_json.get("instanceNamePattern") or "null",
            "process_definition_id": process_definition_id,
            "activity_id": activity_id,
            "user_email": user_email_for_prompt,
            "output": output,
            "today": today,
            "previous_outputs": workitem_input_data,
            "gateway_condition_data": gateway_condition_data,
            "attached_activities": attached_activities,
            "sequence_conditions": sequence_condition_data
        }
        
        completed_json = run_completed_determination(completed_json, chain_input_completed)

        if len(completed_json["completedActivities"]) == 0:
            llm_completed_json, completed_log = await run_prompt_and_parse(
                prompt_completed, chain_input_completed, workitem, tenant_id, parser, "", log_prefix="[COMPLETED]", enable_logging=True
            )
            # Merge only expected keys to preserve instanceId/name/definitionId, etc.
            completed_json["completedActivities"] = llm_completed_json.get("completedActivities", [])
            
        if len(completed_json["completedActivities"]) > 0:
            completed_activities_from_step = (
                completed_json.get("completedActivities")
                or completed_json.get("completedActivitiesDelta")
                or []
            )
    
            organizations = fetch_organization_chart(tenant_id)
            next_activity_payloads = resolve_next_activity_payloads(
                process_definition,
                activity_id,
                workitem,
                sequence_condition_data,
            )



            chain_input_next = {
                "activities": process_definition.activities,
                "gateways": process_definition_json.get('gateways', []),
                "events": process_definition_json.get('events', []),
                "subProcesses": process_definition.subProcesses,
                "sequences": process_definition.sequences,
                "instance_id": process_instance_id,
                "activity_id": activity_id,
                "process_definition_id": process_definition_id,
                "output": output,
                "ui_definitions": ui_definitions,
                "next_activities": next_near_activities,
                "role_bindings": workitem.get('assignees', []),
                "organizations": organizations,
                "instance_name_pattern": process_definition_json.get("instanceNamePattern") or "",
                "today": today,
                "previous_outputs": workitem_input_data,
                "all_workitem_input_data": all_workitem_input_data,
                "branch_merged_workitems": merged_workitems_from_step,
                "completedActivities": completed_activities_from_step,
                "attached_activities": attached_activities,
                "sequence_conditions": sequence_condition_data
            }
            
            next_activity_payloads = inject_boundary_events_as_next(
                next_activity_payloads,
                attached_activities,
                process_definition,
                process_definition_json.get('events', [])
            )
            
            next_activity_payloads = await check_event_expression(next_activity_payloads, chain_input_next)
        
            next_activity_payloads = await check_subprocess_expression(next_activity_payloads, chain_input_next)

            next_activity_payloads = await check_task_status(next_activity_payloads, chain_input_next)
            
            next_activity_payloads = await check_role_binding(next_activity_payloads, chain_input_next)

            completed_json["nextActivities"] = next_activity_payloads

            # 배포 승인 처리: isDeploy가 true이고 승인 상태인 경우 prod_version 업데이트
            _handle_deploy_approval(
                activity_id,
                process_definition,
                output,
                all_workitem_input_data,
                tenant_id
            )

            execute_next_activity(completed_json, tenant_id)
            
            process_output(workitem, tenant_id)

    except Exception as e:
        print(f"[ERROR] Error in handle_workitem for workitem {workitem['id']}: {str(e)}")
        raise e


async def handle_service_workitem(workitem):
    """
    서비스 업무를 처리하는 함수
    """
    # 워크아이템 위치 판별
    is_first, is_last = get_workitem_position(workitem)

    if workitem['retry'] >= 3:
        # 예외 발생 시 인스턴스 상태 업데이트
        update_instance_status_on_error(workitem, is_first, is_last)
        return

    def extract_tool_results_from_agent_messages(messages):
        """
        LangChain agent의 메시지 리스트에서 도구 실행 결과만 추출하여
        {tool_name: {status, ...}} 형태의 딕셔너리로 반환
        """
        tool_results = {}
        for msg in messages:
            # ToolMessage: content가 JSON 문자열일 수 있음
            if hasattr(msg, "name") and hasattr(msg, "content"):
                try:
                    content = msg.content
                    if content and (content.startswith("{") or content.startswith("[")):
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "status" in parsed:
                            tool_results[msg.name] = parsed
                        elif isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict) and "status" in item:
                                    tool_results[msg.name] = item
                except Exception:
                    continue
            # AIMessage: additional_kwargs에 tool_calls가 있을 수 있음
            elif hasattr(msg, "additional_kwargs"):
                tool_calls = msg.additional_kwargs.get("tool_calls", [])
                for call in tool_calls:
                    tool_name = call.get("function", {}).get("name")
                    arguments = call.get("function", {}).get("arguments")
                    if tool_name and arguments:
                        try:
                            args = json.loads(arguments)
                            tool_results[tool_name] = args
                        except Exception:
                            tool_results[tool_name] = arguments
        return tool_results

    try:
        print(f"[DEBUG] Starting service workitem processing for: {workitem['id']}")
        
        agent_id = workitem['user_id']
        tenant_id = workitem['tenant_id']
        agent_info = None
        if not agent_id:
            print(f"[ERROR] No agent ID found in workitem: {workitem['id']}")
            upsert_workitem({
                "id": workitem['id'],
                "log": "No agent ID found"
            }, tenant_id)
            return
        
        if agent_id and ',' in agent_id:
            agent_ids = workitem['user_id'].split(',')
            for agent_id in agent_ids:
                assignee_info = fetch_assignee_info(agent_id)
                if assignee_info and assignee_info.get("type") == "agent":
                    agent_info = fetch_user_info(agent_id)
                    break
        else:
            assignee_info = fetch_assignee_info(agent_id)
            if assignee_info and assignee_info.get("type") == "agent":
                agent_info = fetch_user_info(agent_id)

        if not agent_info:
            print(f"[ERROR] Agent not found: {agent_id}")
            upsert_workitem({
                "id": workitem['id'],
                "log": f"Agent not found: {agent_id}"
            }, tenant_id)
            return

        results = await mcp_processor.execute_mcp_tools(workitem, agent_info, tenant_id)
        messages = results.get("messages", [])
        
        if messages:
            tool_results = extract_tool_results_from_agent_messages(messages)
        else:
            tool_results = {}

        if not tool_results:
            print(f"[ERROR] MCP tools execution failed: No tool results found")
            upsert_workitem({
                "id": workitem['id'],
                "log": "MCP tools execution failed: No tool results found"
            }, tenant_id)
            # return

        error_count = 0
        success_count = 0
        result_summary = []
        
        for tool_name, result in tool_results.items():
            if isinstance(result, dict) and result.get("status") == "success":
                success_count += 1
                connection_type = result.get("connection_type", "unknown")
                result_summary.append(f"{tool_name} ({connection_type}): 성공")
            else:
                error_count += 1
                connection_type = result.get("connection_type", "unknown") if isinstance(result, dict) else "unknown"
                error_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                result_summary.append(f"{tool_name} ({connection_type}): 실패 - {error_msg}")
        
        if error_count == 0:
            log_message = f"모든 MCP 도구 실행 완료: {', '.join(result_summary)}"
        elif success_count > 0:
            log_message = f"일부 MCP 도구 실행 완료: {', '.join(result_summary)}"
        else:
            log_message = f"모든 MCP 도구 실행 실패: {', '.join(result_summary)}"
        
        upsert_workitem({
            "id": workitem['id'],
            "status": "DONE",
            "log": log_message,
            "output": tool_results
        }, tenant_id)
        
        # 채팅 메시지 추가
        def summarize_agent_messages(messages):
            lines = []
            for msg in messages:
                role = getattr(msg, 'role', None) or getattr(msg, 'name', None) or msg.__class__.__name__
                content = getattr(msg, 'content', None)
                if content:
                    lines.append(f"[{role}] {content}")
            return '\n'.join(lines)

        summarized_messages = summarize_agent_messages(messages)
        # 채팅 메시지 추가
        def get_last_ai_message_content(messages):
            last_content = ""
            for msg in reversed(messages):
                if msg.__class__.__name__ == "AIMessage" and hasattr(msg, "content"):
                    last_content = msg.content
                    break
            return last_content

        last_ai_content = get_last_ai_message_content(messages)
        message_data = {
            "role": "system",
            "content": last_ai_content,
            "jsonContent": tool_results
        }
        upsert_chat_message(workitem['proc_inst_id'], message_data, tenant_id)
        
        # 리소스 정리
        await mcp_processor.cleanup()
                
    except Exception as e:
        print(f"[ERROR] Error in handle_service_workitem for workitem {workitem['id']}: {str(e)}")
        
        # 에러 상태로 워크아이템 업데이트
        upsert_workitem({
            "id": workitem['id'],
            "log": f"Service workitem processing failed: {str(e)}"
        }, workitem['tenant_id'])
        
        # 에러 메시지를 채팅에 추가
        error_message = json.dumps({
            "role": "system",
            "content": f"서비스 업무 처리 중 오류가 발생했습니다: {str(e)}"
        })
        upsert_chat_message(workitem['proc_inst_id'], error_message, workitem['tenant_id'])
        
        # 리소스 정리
        try:
            await mcp_processor.cleanup()
        except:
            pass
        
        raise e
    
async def handle_pending_workitem(workitem):
    """
    [규칙]
    - 대상: 부모 워크아이템이 subProcess이고 상태가 PENDING일 때만 동작.
    - 스캔 후, 자식 전체에 SUBMITTED가 하나도 없으면 부모 PENDING 워크아이템을 DONE으로 전환.
    - 자식 인스턴스의 상태는 변경하지 않는다(승격 금지).
    """
    try:
        wid = workitem.get('id')
        proc_def_id = workitem.get('proc_def_id')
        tenant_id   = workitem.get('tenant_id')
        parent_proc_inst_id = workitem.get('proc_inst_id')

        if not all([wid, proc_def_id, tenant_id, parent_proc_inst_id]):
            print(f"[WARN] handle_pending_workitem: insufficient keys in workitem id={wid}")
            return

        # 부모 워크아이템이 PENDING일 때만 실행
        if (workitem.get('status') or '').upper() != 'PENDING':
            print(f"[DEBUG] handle_pending_workitem: parent workitem is not PENDING (id={wid})")
            return

        # 1순위: 워크아이템에 저장된 버전 정보
        version_tag = workitem.get('version_tag')
        version = workitem.get('version')

        # 2순위: 인스턴스에 저장된 proc_def_version(arcv_id)
        # - workitem에 버전(version_tag/version)이 명시된 경우에는 arcv_id를 넘기지 않아
        #   "workitem 버전 우선" 규칙이 깨지지 않도록 합니다.
        process_instance = fetch_process_instance(workitem.get('proc_inst_id'), tenant_id)
        arcv_id = None
        if not version_tag and not version:
            if process_instance and getattr(process_instance, "proc_def_version", None):
                arcv_id = process_instance.proc_def_version

        process_definition_json = fetch_process_definition_by_version(
            proc_def_id,
            version_tag,
            version,
            tenant_id,
            arcv_id,
        )
        process_definition = load_process_definition(process_definition_json)
        activity = process_definition.find_activity_by_id(workitem.get('activity_id'))
        if not activity:
            print(f"[ERROR] handle_pending_workitem: Activity not found: {workitem.get('activity_id')}")
            return

        child_instances = fetch_child_instances_by_parent(parent_proc_inst_id, tenant_id) or []
        if not child_instances:
            print(f"[DEBUG] No child instances for parent {parent_proc_inst_id}")
            return

        any_submitted_left = False
        total_children = 0
        total_items_scanned = 0
        total_items_closed  = 0

        for child in child_instances:
            total_children += 1
            child_id = child.get("proc_inst_id") if isinstance(child, dict) \
                       else getattr(child, "proc_inst_id", None)
            if not child_id:
                continue

            child_items = fetch_todolist_by_proc_inst_id(child_id) or []
            for ci in child_items:
                total_items_scanned += 1
                status = (ci.get("status") if isinstance(ci, dict) else getattr(ci, "status", None)) or ""

                su = status.upper()
                if su == "SUBMITTED":
                    any_submitted_left = True
                    continue

        if not any_submitted_left:
            try:
                upsert_workitem({"id": wid, "status": "DONE"}, tenant_id)
                print(f"[INFO] Parent pending workitem {wid} -> DONE "
                      f"(children={total_children}, scanned={total_items_scanned}, closed={total_items_closed})")
            except Exception as e:
                print(f"[ERROR] Failed to mark parent workitem {wid} DONE: {e}")
        else:
            print(f"[DEBUG] SUBMITTED remains in children; keep parent PENDING "
                  f"(children={total_children}, scanned={total_items_scanned}, closed={total_items_closed})")

    except Exception as e:
        print(f"[ERROR] Error in handle_pending_workitem for workitem {workitem.get('id')}: {str(e)}")
        raise e

def get_all_input_data(workitem: dict, process_definition: Any) -> Dict[str, Any]:

    """

    루트 프로세스 인스턴스 기준으로 모든 워크아이템을 조회하여

    - 같은 activity_id가 중복인 경우 start_date가 가장 최신인 것만 유지

    - 추출된 output을 폼 key 기반으로 모아 반환



    Returns:

        Dict[str, Any]: 최신 워크아이템들의 output 목록 (미싱 데이터/오류는 제외)

    """

    try:
        tenant_id = workitem.get('tenant_id')
        proc_inst_id = workitem.get('proc_inst_id')
        root_proc_inst_id = workitem.get('root_proc_inst_id')

        if not root_proc_inst_id and proc_inst_id and tenant_id:

            try:
                inst = fetch_process_instance(proc_inst_id, tenant_id)

                root_proc_inst_id = (
                    getattr(inst, 'root_proc_inst_id', None)
                    or (inst.get('root_proc_inst_id') if isinstance(inst, dict) else None)
                )

            except Exception:
                root_proc_inst_id = None

        if not tenant_id or not root_proc_inst_id:
            return {}

        workitems = fetch_workitems_by_root_proc_inst_id(root_proc_inst_id, tenant_id) or []

        def _get(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _is_numeric_key(k: Any) -> bool:
            try:
                if isinstance(k, (int, float)):
                    return True
                if isinstance(k, str) and k.isdigit():
                    return True
            except Exception:
                pass
            return False

        def _resolve_form_key_for_workitem(wi: Any) -> str:
            act_id = _get(wi, 'activity_id') or _get(wi, 'activityId')
            proc_def_id = _get(wi, 'proc_def_id') or _get(wi, 'procDefId') or workitem.get('proc_def_id')
            ten = _get(wi, 'tenant_id') or workitem.get('tenant_id')
            try:
                ui_def = fetch_ui_definition_by_activity_id(proc_def_id, act_id, ten)
                form_key = getattr(ui_def, 'id', None) or (ui_def.get('id') if isinstance(ui_def, dict) else None)
                if form_key and isinstance(form_key, str):
                    return form_key
            except Exception:
                pass
            return str(act_id) if act_id is not None else 'unknown_form'


        cur_scope_raw = workitem.get('execution_scope') or workitem.get('executionScope')
        try:
            cur_scope = int(str(cur_scope_raw)) if cur_scope_raw is not None else None
        except Exception:
            cur_scope = cur_scope_raw

        def _norm_scope(v):
            if v is None or v == "":
                return None
            try:
                return int(str(v))
            except Exception:
                return str(v)

        def _scope_of(obj):
            return _norm_scope(_get(obj, 'execution_scope') or _get(obj, 'executionScope'))

        if cur_scope is not None:
            workitems = [wi for wi in workitems if (_scope_of(wi) is None) or (_scope_of(wi) == cur_scope)]

        def _parse_dt(s: str) -> datetime:
            try:
                return datetime.fromisoformat(s)
            except Exception:
                try:
                    return datetime.strptime(s[:19], '%Y-%m-%dT%H:%M:%S')
                except Exception:
                    return datetime.min

        latest_by_activity: dict[str, Any] = {}

        for wi in workitems:
            act_id = _get(wi, 'activity_id') or _get(wi, 'activityId')
            if not act_id:
                continue

            start_date = str(_get(wi, 'start_date') or _get(wi, 'startDate') or '')
            prev = latest_by_activity.get(act_id)

            if not prev:
                latest_by_activity[act_id] = wi
            else:
                prev_sd = str(_get(prev, 'start_date') or _get(prev, 'startDate') or '')
                if _parse_dt(start_date) >= _parse_dt(prev_sd):
                    latest_by_activity[act_id] = wi

        selected = list(latest_by_activity.values())
        selected.sort(key=lambda x: _parse_dt(str(_get(x, 'start_date') or _get(x, 'startDate') or '')))

        outputs: Dict[str, Any] = {}

        def _register_output(key, value):
            if key is None:
                return False

            key_str = str(key).strip()
            if not key_str:
                return False

            outputs[key_str] = value
            return True

        for wi in selected:
            out = _get(wi, 'output')
            if out in (None, '', {}):
                continue

            if isinstance(out, str):
                try:
                    out = json.loads(out)
                except Exception:
                    continue
                
            if isinstance(out, dict):
                registered = False
                try:
                    if len(out) == 1:
                        only_key, only_val = next(iter(out.items()))
                        reg_key = _resolve_form_key_for_workitem(wi) if _is_numeric_key(only_key) else only_key
                        registered = _register_output(reg_key, only_val)

                    if not registered:
                        for k, v in out.items():
                            if isinstance(v, dict) and ('form' in str(k).lower() or 'Form' in str(k)):
                                reg_key = _resolve_form_key_for_workitem(wi) if _is_numeric_key(k) else k
                                if _register_output(reg_key, v):
                                    registered = True
                                    break
                                
                    if not registered:
                        keys = list(out.keys())
                        if keys and all(_is_numeric_key(k) for k in keys):
                            form_key = _resolve_form_key_for_workitem(wi)
                            registered = _register_output(form_key, out)

                    if not registered:
                        act_key = _get(wi, 'activity_id') or _get(wi, 'activityId')
                        if not act_key and out:
                            first_key = next(iter(out.keys()), None)
                            act_key = _resolve_form_key_for_workitem(wi) if _is_numeric_key(first_key) else first_key

                        _register_output(act_key, out)
                except Exception:
                    act_key = _get(wi, 'activity_id') or _get(wi, 'activityId')
                    _register_output(act_key, out)
            else:
                act_key = _get(wi, 'activity_id') or _get(wi, 'activityId')
                _register_output(act_key, out)

        return outputs
    except Exception as e:
        print(f"[ERROR] Failed to get all input data for {workitem.get('id')}: {str(e)}")
        return {}

