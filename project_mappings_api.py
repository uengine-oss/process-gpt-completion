import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from psycopg2.extras import Json, RealDictCursor

from dashboard_api import _activity_props, _activities, _as_dict, _as_list, _system_items
from database import db_config_var, subdomain_var


TABLE_NAME = "project_task_mappings"


class LinkedSystem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    system_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("system_id", "systemId", "id"),
    )
    system_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("system_name", "systemName", "name"),
    )

    @field_validator("system_id", "system_name", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class TaskMappingItem(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    task_id: str = Field(
        validation_alias=AliasChoices("task_id", "taskId", "activity_id", "activityId", "id"),
    )
    task_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("task_name", "taskName", "activity_name", "activityName", "name"),
    )
    linked_systems: List[LinkedSystem] = Field(
        default_factory=list,
        validation_alias=AliasChoices("linked_systems", "linkedSystems", "systems"),
    )

    @field_validator("task_id", mode="before")
    @classmethod
    def _require_task_id(cls, value: Any) -> str:
        text = "" if value is None else str(value).strip()
        if not text:
            raise ValueError("task_id is required")
        return text

    @field_validator("task_name", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("linked_systems", mode="before")
    @classmethod
    def _normalize_linked_systems_input(cls, value: Any) -> List[Any]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        normalized = []
        for item in items:
            if isinstance(item, str):
                normalized.append({"system_name": item})
            else:
                normalized.append(item)
        return normalized


class ProjectMappingCommitRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    project_id: str = Field(
        validation_alias=AliasChoices("project_id", "projectId", "Project_ID", "id"),
    )
    process_id: str = Field(
        validation_alias=AliasChoices("process_id", "processId", "proc_def_id", "procDefId", "processDefinitionId"),
    )
    process_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("process_name", "processName", "proc_def_name", "procDefName"),
    )
    task_list: List[TaskMappingItem] = Field(
        validation_alias=AliasChoices("task_list", "taskList", "tasks"),
    )
    linked_systems: List[LinkedSystem] = Field(
        default_factory=list,
        validation_alias=AliasChoices("linked_systems", "linkedSystems", "systems"),
    )

    @field_validator("project_id", "process_id", mode="before")
    @classmethod
    def _require_text(cls, value: Any) -> str:
        text = "" if value is None else str(value).strip()
        if not text:
            raise ValueError("value is required")
        return text

    @field_validator("process_name", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("task_list", mode="before")
    @classmethod
    def _normalize_task_list_input(cls, value: Any) -> List[Any]:
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    @field_validator("linked_systems", mode="before")
    @classmethod
    def _normalize_linked_systems_input(cls, value: Any) -> List[Any]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        normalized = []
        for item in items:
            if isinstance(item, str):
                normalized.append({"system_name": item})
            else:
                normalized.append(item)
        return normalized

class ProjectMappingResponse(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    tenant_id: Optional[str] = None
    project_id: Optional[str] = None
    process_id: Optional[Union[str, List[str]]] = None
    process_name: Union[str, List[str]]
    task_list: List[TaskMappingItem]
    linked_systems: List[LinkedSystem]


def add_routes_to_app(app: FastAPI):
    app.add_api_route("/api/v1/projects/mappings", commit_project_mapping, methods=["POST"])
    app.add_api_route(
        "/api/v1/projects/{project_id}/mappings",
        get_project_mapping,
        methods=["GET"],
        response_model=ProjectMappingResponse,
    )


async def commit_project_mapping(body: ProjectMappingCommitRequest):
    tenant_id = _get_current_tenant_id()
    return await asyncio.to_thread(_commit_project_mapping, tenant_id, body)


async def get_project_mapping(project_id: str):
    tenant_id = _get_current_tenant_id()
    result = await asyncio.to_thread(_get_project_mapping, tenant_id, project_id)
    if not result:
        raise HTTPException(status_code=404, detail="Project mapping not found")
    return result


def _get_current_tenant_id() -> str:
    tenant_id = (subdomain_var.get() or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is not resolved for this request")
    return tenant_id


def _connect_db():
    db_config = db_config_var.get() or {}
    required_keys = ("dbname", "user", "host", "port")
    missing_keys = [key for key in required_keys if not str(db_config.get(key) or "").strip()]
    if missing_keys:
        raise HTTPException(status_code=500, detail=f"Database configuration is missing: {', '.join(missing_keys)}")
    try:
        return psycopg2.connect(**db_config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to connect to database: {exc}") from exc


def _ensure_table(cursor) -> None:
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id bigserial PRIMARY KEY,
            tenant_id text NOT NULL,
            project_id text NOT NULL,
            proc_def_id text NOT NULL,
            process_name text,
            task_id text NOT NULL,
            task_name text,
            linked_systems jsonb NOT NULL DEFAULT '[]'::jsonb,
            task_payload jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, project_id, proc_def_id, task_id)
        )
        """
    )
    cursor.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {TABLE_NAME}_tenant_project_idx
            ON {TABLE_NAME} (tenant_id, project_id)
        """
    )
    cursor.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {TABLE_NAME}_tenant_proc_def_idx
            ON {TABLE_NAME} (tenant_id, proc_def_id)
        """
    )


def _commit_project_mapping(tenant_id: str, body: ProjectMappingCommitRequest) -> Dict[str, Any]:
    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)

        process_row = _fetch_process_row(cursor, tenant_id, body.process_id)
        if not process_row:
            raise HTTPException(status_code=404, detail="Process not found")

        process_name = body.process_name or process_row.get("name") or body.process_id
        definition_task_lookup = _definition_task_lookup(process_row.get("definition"))
        request_level_systems = _linked_system_models_to_dicts(body.linked_systems)

        cursor.execute(
            f"""
            DELETE FROM {TABLE_NAME}
            WHERE tenant_id = %(tenant_id)s
              AND project_id = %(project_id)s
              AND proc_def_id = %(proc_def_id)s
            """,
            {
                "tenant_id": tenant_id,
                "project_id": body.project_id,
                "proc_def_id": body.process_id,
            },
        )

        now = datetime.utcnow().isoformat()
        for task in body.task_list:
            definition_task = definition_task_lookup.get(task.task_id, {})
            linked_systems = _linked_system_models_to_dicts(task.linked_systems)
            if not linked_systems:
                linked_systems = list(definition_task.get("linked_systems") or [])
            if not linked_systems:
                linked_systems = list(request_level_systems)
            linked_systems = _dedupe_linked_systems(linked_systems)
            task_name = task.task_name or definition_task.get("task_name") or task.task_id

            cursor.execute(
                f"""
                INSERT INTO {TABLE_NAME}
                    (tenant_id, project_id, proc_def_id, process_name, task_id, task_name,
                     linked_systems, task_payload, created_at, updated_at)
                VALUES
                    (%(tenant_id)s, %(project_id)s, %(proc_def_id)s, %(process_name)s, %(task_id)s, %(task_name)s,
                     %(linked_systems)s, %(task_payload)s, %(created_at)s, %(updated_at)s)
                ON CONFLICT (tenant_id, project_id, proc_def_id, task_id) DO UPDATE SET
                    process_name = EXCLUDED.process_name,
                    task_name = EXCLUDED.task_name,
                    linked_systems = EXCLUDED.linked_systems,
                    task_payload = EXCLUDED.task_payload,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "tenant_id": tenant_id,
                    "project_id": body.project_id,
                    "proc_def_id": body.process_id,
                    "process_name": process_name,
                    "task_id": task.task_id,
                    "task_name": task_name,
                    "linked_systems": Json(linked_systems),
                    "task_payload": Json(task.model_dump(mode="json")),
                    "created_at": now,
                    "updated_at": now,
                },
            )

        result = _get_project_mapping_with_cursor(cursor, tenant_id, body.project_id)
        connection.commit()
        return result or {
            "tenant_id": tenant_id,
            "project_id": body.project_id,
            "process_id": body.process_id,
            "process_name": process_name,
            "task_list": [],
            "linked_systems": [],
        }
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to commit project mapping: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _get_project_mapping(tenant_id: str, project_id: str) -> Optional[Dict[str, Any]]:
    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)
        connection.commit()
        return _get_project_mapping_with_cursor(cursor, tenant_id, project_id)
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to get project mapping: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _get_project_mapping_with_cursor(cursor, tenant_id: str, project_id: str) -> Optional[Dict[str, Any]]:
    cursor.execute(
        f"""
        SELECT
            tenant_id,
            project_id,
            proc_def_id,
            process_name,
            task_id,
            task_name,
            linked_systems,
            task_payload,
            created_at,
            updated_at
        FROM {TABLE_NAME}
        WHERE tenant_id = %(tenant_id)s
          AND project_id = %(project_id)s
        ORDER BY proc_def_id ASC, id ASC
        """,
        {"tenant_id": tenant_id, "project_id": project_id},
    )
    rows = cursor.fetchall()
    if not rows:
        return None
    return _build_mapping_response(rows)


def _fetch_process_row(cursor, tenant_id: str, process_id: str) -> Optional[Dict[str, Any]]:
    cursor.execute(
        """
        SELECT id, name, definition
        FROM proc_def
        WHERE tenant_id = %(tenant_id)s
          AND id = %(process_id)s
          AND COALESCE(isdeleted, false) = false
        """,
        {"tenant_id": tenant_id, "process_id": process_id},
    )
    return cursor.fetchone()


def _build_mapping_response(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    first = rows[0]
    task_list = [_row_to_task(row) for row in rows]
    linked_systems = _dedupe_linked_systems(
        system
        for task in task_list
        for system in task.get("linked_systems", [])
    )
    process_names = _dedupe_text(row.get("process_name") for row in rows)
    process_ids = _dedupe_text(row.get("proc_def_id") for row in rows)

    response = {
        "tenant_id": first.get("tenant_id"),
        "project_id": first.get("project_id"),
        "process_id": process_ids[0] if len(process_ids) == 1 else process_ids,
        "process_name": process_names[0] if len(process_names) == 1 else process_names,
        "task_list": task_list,
        "linked_systems": linked_systems,
    }
    return response


def _row_to_task(row: Dict[str, Any]) -> Dict[str, Any]:
    linked_systems = _normalize_linked_systems(row.get("linked_systems"))
    task = {
        "task_id": row.get("task_id"),
        "task_name": row.get("task_name"),
        "process_id": row.get("proc_def_id"),
        "linked_systems": linked_systems,
    }
    task_payload = row.get("task_payload")
    if isinstance(task_payload, dict):
        for key in ("activity_id", "activityId", "type", "role"):
            if key in task_payload and key not in task:
                task[key] = task_payload[key]
    return task


def _definition_task_lookup(definition: Any) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}

    def visit(item: Dict[str, Any]) -> None:
        props = _activity_props(item)
        task_id = _text(props.get("id") or props.get("activity_id") or props.get("activityId"))
        if task_id:
            task_name = _text(props.get("name") or props.get("label") or props.get("activity_name") or task_id) or task_id
            lookup[task_id] = {
                "task_name": task_name,
                "linked_systems": _normalize_linked_systems(_system_items(props)),
            }
        for key in ("activities", "tasks", "children"):
            for child in _as_list(props.get(key)):
                if isinstance(child, dict):
                    visit(child)

    definition_dict = _as_dict(definition)

    for activity in _activities(definition_dict):
        if isinstance(activity, dict):
            visit(activity)
    return lookup


def _linked_system_models_to_dicts(systems: List[LinkedSystem]) -> List[Dict[str, Optional[str]]]:
    return _dedupe_linked_systems(
        {
            "system_id": system.system_id,
            "system_name": system.system_name,
        }
        for system in systems
    )


def _normalize_linked_systems(value: Any) -> List[Dict[str, Optional[str]]]:
    items = value if isinstance(value, list) else _as_list(value)
    normalized: List[Dict[str, Optional[str]]] = []
    for item in items:
        if isinstance(item, dict):
            system_id = _text(item.get("system_id") or item.get("systemId") or item.get("id")) or None
            system_name = _text(item.get("system_name") or item.get("systemName") or item.get("name")) or None
        else:
            system_id = None
            system_name = _text(item) or None
        if system_id or system_name:
            normalized.append({"system_id": system_id, "system_name": system_name})
    return _dedupe_linked_systems(normalized)


def _dedupe_linked_systems(systems: Any) -> List[Dict[str, Optional[str]]]:
    seen: set[Tuple[Optional[str], Optional[str]]] = set()
    result: List[Dict[str, Optional[str]]] = []
    for system in systems:
        if not isinstance(system, dict):
            system = {"system_name": _text(system) or None}
        system_id = _text(system.get("system_id") or system.get("systemId") or system.get("id")) or None
        system_name = _text(system.get("system_name") or system.get("systemName") or system.get("name")) or None
        if not system_id and not system_name:
            continue
        key = (system_id, system_name)
        if key in seen:
            continue
        seen.add(key)
        result.append({"system_id": system_id, "system_name": system_name})
    return result


def _dedupe_text(values: Any) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = _text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

