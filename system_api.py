import asyncio
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from psycopg2.extras import RealDictCursor

from database import db_config_var, subdomain_var


TABLE_NAME = "systems"
MAX_SEARCH_LIMIT = 200


class SystemCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    system_type: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    shortcut_link: Optional[str] = None
    is_active: Optional[int] = 1
    responsible_org_id: Optional[str] = None
    responsible_person: Optional[str] = None
    registration_status: Optional[str] = "active"
    created_by: Optional[str] = Field(default=None, validation_alias=AliasChoices("created_by", "createdBy"))
    created_by_display: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("created_by_display", "createdByDisplay"),
    )


class SystemUpdate(BaseModel):
    name: Optional[str] = None
    system_type: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    shortcut_link: Optional[str] = None
    is_active: Optional[int] = None
    responsible_org_id: Optional[str] = None
    responsible_person: Optional[str] = None
    registration_status: Optional[str] = None


class SystemDelete(BaseModel):
    deleted_by: Optional[str] = None


def add_routes_to_app(app: FastAPI):
    app.add_api_route("/systems", get_systems, methods=["GET"])
    app.add_api_route("/systems/deleted", get_deleted_systems, methods=["GET"])
    app.add_api_route("/systems/{system_id}", get_system, methods=["GET"])
    app.add_api_route("/systems", create_system, methods=["POST"])
    app.add_api_route("/systems/{system_id}", update_system, methods=["PUT"])
    app.add_api_route("/systems/{system_id}", delete_system, methods=["DELETE"])
    app.add_api_route("/systems/{system_id}/restore", restore_system, methods=["POST"])
    app.add_api_route("/systems/{system_id}/permanent", permanently_delete_system, methods=["DELETE"])
    app.add_api_route("/admin/systems/bulk-import", bulk_import_systems, methods=["POST"])


# ── Handlers ──────────────────────────────────────────────


async def get_systems(
    keyword: Optional[str] = Query(None),
    system_type: Optional[str] = Query(None),
    page: int = Query(0, ge=0),
    limit: int = Query(MAX_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT),
):
    tenant_id = _get_current_tenant_id()
    result = await asyncio.to_thread(_get_systems, tenant_id, keyword, system_type, page, limit)
    return result


async def get_deleted_systems(
    page: int = Query(0, ge=0),
    limit: int = Query(MAX_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT),
):
    tenant_id = _get_current_tenant_id()
    result = await asyncio.to_thread(_get_deleted_systems, tenant_id, page, limit)
    return result


async def get_system(system_id: str):
    tenant_id = _get_current_tenant_id()
    row = await asyncio.to_thread(_get_system_by_id, tenant_id, system_id)
    if not row:
        raise HTTPException(status_code=404, detail="System not found")
    return row


async def create_system(body: SystemCreate):
    tenant_id = _get_current_tenant_id()
    result = await asyncio.to_thread(_create_system, tenant_id, body)
    return result


async def update_system(system_id: str, body: SystemUpdate):
    tenant_id = _get_current_tenant_id()
    result = await asyncio.to_thread(_update_system, tenant_id, system_id, body)
    if not result:
        raise HTTPException(status_code=404, detail="System not found")
    return result


async def delete_system(system_id: str, body: Optional[SystemDelete] = Body(None)):
    tenant_id = _get_current_tenant_id()
    result = await asyncio.to_thread(_soft_delete_system, tenant_id, system_id, body.deleted_by if body else None)
    if not result:
        raise HTTPException(status_code=404, detail="System not found")
    return result


async def restore_system(system_id: str):
    tenant_id = _get_current_tenant_id()
    result = await asyncio.to_thread(_restore_system, tenant_id, system_id)
    if not result:
        raise HTTPException(status_code=404, detail="Deleted system not found")
    return result


async def permanently_delete_system(system_id: str):
    tenant_id = _get_current_tenant_id()
    result = await asyncio.to_thread(_permanently_delete_system, tenant_id, system_id)
    if not result:
        raise HTTPException(status_code=404, detail="Deleted system not found")
    return result


async def bulk_import_systems(systems: List[Dict[str, Any]]):
    """JSON 배열을 받아서 systems 테이블에 일괄 삽입. 기존 ID가 있으면 업데이트."""
    tenant_id = _get_current_tenant_id()
    result = await asyncio.to_thread(_bulk_import_systems, tenant_id, systems)
    return result


# ── Internal ──────────────────────────────────────────────


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
            id text PRIMARY KEY,
            tenant_id text NOT NULL,
            name text NOT NULL,
            system_type text,
            category text,
            description text,
            shortcut_link text,
            is_active integer NOT NULL DEFAULT 1,
            responsible_org_id text,
            responsible_person text,
            registration_status text NOT NULL DEFAULT 'active',
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            created_by text,
            created_by_display text,
            deleted_at timestamptz,
            deleted_by text
        )
        """
    )
    cursor.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {TABLE_NAME}_tenant_id_idx
            ON {TABLE_NAME} (tenant_id)
        """
    )
    cursor.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {TABLE_NAME}_tenant_name_idx
            ON {TABLE_NAME} (tenant_id, name)
        """
    )
    cursor.execute(
        f"""
        ALTER TABLE {TABLE_NAME}
        ADD COLUMN IF NOT EXISTS category text
        """
    )
    cursor.execute(
        f"""
        ALTER TABLE {TABLE_NAME}
        ADD COLUMN IF NOT EXISTS shortcut_link text
        """
    )
    cursor.execute(
        f"""
        ALTER TABLE {TABLE_NAME}
        ADD COLUMN IF NOT EXISTS deleted_at timestamptz
        """
    )
    cursor.execute(
        f"""
        ALTER TABLE {TABLE_NAME}
        ADD COLUMN IF NOT EXISTS deleted_by text
        """
    )
    cursor.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {TABLE_NAME}_tenant_deleted_at_idx
            ON {TABLE_NAME} (tenant_id, deleted_at)
        """
    )


def _serialize_row(row: dict) -> dict:
    out = dict(row)
    for key in ("created_at", "updated_at", "deleted_at"):
        val = out.get(key)
        if val and hasattr(val, "isoformat"):
            out[key] = val.isoformat()
    return out


def _get_first_present(item: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item.get(key)
    return None


def _get_systems(tenant_id: str, keyword: Optional[str], system_type: Optional[str], page: int, limit: int) -> list:
    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)
        connection.commit()

        conditions = ["tenant_id = %(tenant_id)s", "deleted_at IS NULL"]
        params: Dict[str, Any] = {"tenant_id": tenant_id, "limit": limit, "offset": page * limit}

        if keyword:
            kw = keyword.strip().lower()
            conditions.append(
                "(lower(name) LIKE %(kw)s OR lower(coalesce(id,'')) LIKE %(kw)s "
                "OR lower(coalesce(description,'')) LIKE %(kw)s "
                "OR lower(coalesce(category,'')) LIKE %(kw)s "
                "OR lower(coalesce(shortcut_link,'')) LIKE %(kw)s)"
            )
            params["kw"] = f"%{kw}%"

        if system_type:
            conditions.append("system_type = %(system_type)s")
            params["system_type"] = system_type

        where = " AND ".join(conditions)

        cursor.execute(
            f"""
            SELECT * FROM {TABLE_NAME}
            WHERE {where}
            ORDER BY name ASC
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            params,
        )
        rows = [_serialize_row(row) for row in cursor.fetchall()]
        return rows
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to get systems: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _get_deleted_systems(tenant_id: str, page: int, limit: int) -> list:
    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)
        connection.commit()

        cursor.execute(
            f"""
            SELECT * FROM {TABLE_NAME}
            WHERE tenant_id = %(tenant_id)s AND deleted_at IS NOT NULL
            ORDER BY deleted_at DESC, name ASC
            LIMIT %(limit)s OFFSET %(offset)s
            """,
            {"tenant_id": tenant_id, "limit": limit, "offset": page * limit},
        )
        rows = [_serialize_row(row) for row in cursor.fetchall()]
        return rows
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to get deleted systems: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _get_system_by_id(tenant_id: str, system_id: str) -> Optional[dict]:
    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)
        connection.commit()

        cursor.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE tenant_id = %(tenant_id)s AND id = %(id)s",
            {"tenant_id": tenant_id, "id": system_id},
        )
        row = cursor.fetchone()
        return _serialize_row(row) if row else None
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to get system: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _create_system(tenant_id: str, body: SystemCreate) -> dict:
    now = datetime.utcnow().isoformat()
    # ID 생성: SVSIN + timestamp 기반
    system_id = f"SVSIN{datetime.utcnow().strftime('%y%m%d%H%M%S')}"

    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)

        cursor.execute(
            f"""
            INSERT INTO {TABLE_NAME}
                (id, tenant_id, name, system_type, category, description, shortcut_link, is_active,
                 responsible_org_id, responsible_person, registration_status,
                 created_at, updated_at, created_by, created_by_display)
            VALUES
                (%(id)s, %(tenant_id)s, %(name)s, %(system_type)s, %(category)s, %(description)s, %(shortcut_link)s, %(is_active)s,
                 %(responsible_org_id)s, %(responsible_person)s, %(registration_status)s,
                 %(created_at)s, %(updated_at)s, %(created_by)s, %(created_by_display)s)
            RETURNING *
            """,
            {
                "id": system_id,
                "tenant_id": tenant_id,
                "name": body.name,
                "system_type": body.system_type,
                "category": body.category,
                "description": body.description,
                "shortcut_link": body.shortcut_link,
                "is_active": body.is_active if body.is_active is not None else 1,
                "responsible_org_id": body.responsible_org_id,
                "responsible_person": body.responsible_person,
                "registration_status": body.registration_status or "active",
                "created_at": now,
                "updated_at": now,
                "created_by": body.created_by,
                "created_by_display": body.created_by_display,
            },
        )
        row = cursor.fetchone()
        connection.commit()
        return _serialize_row(row)
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create system: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _update_system(tenant_id: str, system_id: str, body: SystemUpdate) -> Optional[dict]:
    updates = {k: v for k, v in body.dict(exclude_unset=True).items()}
    if not updates:
        return _get_system_by_id(tenant_id, system_id)

    updates["updated_at"] = datetime.utcnow().isoformat()

    set_clauses = [f"{key} = %({key})s" for key in updates]
    set_sql = ", ".join(set_clauses)

    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)

        params = {**updates, "tenant_id": tenant_id, "id": system_id}
        cursor.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET {set_sql}
            WHERE tenant_id = %(tenant_id)s AND id = %(id)s
            RETURNING *
            """,
            params,
        )
        row = cursor.fetchone()
        connection.commit()
        return _serialize_row(row) if row else None
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update system: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _soft_delete_system(tenant_id: str, system_id: str, deleted_by: Optional[str]) -> Optional[dict]:
    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)

        cursor.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET deleted_at = now(),
                deleted_by = %(deleted_by)s,
                updated_at = now()
            WHERE tenant_id = %(tenant_id)s
              AND id = %(id)s
              AND deleted_at IS NULL
            RETURNING *
            """,
            {"tenant_id": tenant_id, "id": system_id, "deleted_by": deleted_by},
        )
        row = cursor.fetchone()
        connection.commit()
        return _serialize_row(row) if row else None
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete system: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _restore_system(tenant_id: str, system_id: str) -> Optional[dict]:
    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)

        cursor.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET deleted_at = NULL,
                deleted_by = NULL,
                updated_at = now()
            WHERE tenant_id = %(tenant_id)s
              AND id = %(id)s
              AND deleted_at IS NOT NULL
            RETURNING *
            """,
            {"tenant_id": tenant_id, "id": system_id},
        )
        row = cursor.fetchone()
        connection.commit()
        return _serialize_row(row) if row else None
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to restore system: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _permanently_delete_system(tenant_id: str, system_id: str) -> Optional[dict]:
    connection = None
    cursor = None
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)

        cursor.execute(
            f"""
            DELETE FROM {TABLE_NAME}
            WHERE tenant_id = %(tenant_id)s
              AND id = %(id)s
              AND deleted_at IS NOT NULL
            RETURNING *
            """,
            {"tenant_id": tenant_id, "id": system_id},
        )
        row = cursor.fetchone()
        connection.commit()
        return _serialize_row(row) if row else None
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to permanently delete system: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def _bulk_import_systems(tenant_id: str, systems: List[Dict[str, Any]]) -> dict:
    if not systems:
        return {"imported": 0, "skipped": 0}

    connection = None
    cursor = None
    imported = 0
    skipped = 0
    try:
        connection = _connect_db()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        _ensure_table(cursor)
        connection.commit()

        for item in systems:
            sys_id = item.get("id")
            name = item.get("name")
            if not sys_id or not name:
                skipped += 1
                continue

            cursor.execute(
                f"""
                INSERT INTO {TABLE_NAME}
                    (id, tenant_id, name, system_type, category, description, shortcut_link, is_active,
                     responsible_org_id, responsible_person, registration_status,
                     created_at, updated_at, created_by, created_by_display)
                VALUES
                    (%(id)s, %(tenant_id)s, %(name)s, %(system_type)s, %(category)s, %(description)s, %(shortcut_link)s, %(is_active)s,
                     %(responsible_org_id)s, %(responsible_person)s, %(registration_status)s,
                     %(created_at)s, %(updated_at)s, %(created_by)s, %(created_by_display)s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    system_type = EXCLUDED.system_type,
                    category = EXCLUDED.category,
                    description = EXCLUDED.description,
                    shortcut_link = EXCLUDED.shortcut_link,
                    is_active = EXCLUDED.is_active,
                    responsible_org_id = EXCLUDED.responsible_org_id,
                    responsible_person = EXCLUDED.responsible_person,
                    registration_status = EXCLUDED.registration_status,
                    updated_at = EXCLUDED.updated_at,
                    created_by = EXCLUDED.created_by,
                    created_by_display = EXCLUDED.created_by_display
                """,
                {
                    "id": sys_id,
                    "tenant_id": tenant_id,
                    "name": name,
                    "system_type": item.get("system_type"),
                    "category": _get_first_present(item, "category", "division", "구분"),
                    "description": item.get("description"),
                    "shortcut_link": _get_first_present(item, "shortcut_link", "shortcutLink", "바로가기링크"),
                    "is_active": item.get("is_active", 1),
                    "responsible_org_id": item.get("responsible_org_id"),
                    "responsible_person": item.get("responsible_person"),
                    "registration_status": item.get("registration_status", "active"),
                    "created_at": item.get("created_at", datetime.utcnow().isoformat()),
                    "updated_at": item.get("updated_at", datetime.utcnow().isoformat()),
                    "created_by": item.get("created_by"),
                    "created_by_display": item.get("created_by_display"),
                },
            )
            imported += 1

        connection.commit()
        return {"imported": imported, "skipped": skipped}
    except HTTPException:
        if connection:
            connection.rollback()
        raise
    except Exception as exc:
        if connection:
            connection.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to bulk import systems: {exc}") from exc
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

