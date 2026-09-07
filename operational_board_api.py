"""
Tab B: Operational Board API
- 4.1 Process Bottleneck List (고비용 프로세스 Top 10)
- 4.2 Update Recency / Zombie Processes (좀비 프로세스 관리)
- 4.3 Action Required (지연 및 미결 업무 관리)
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from fastapi import FastAPI, HTTPException, Query, Request
from psycopg2.extras import RealDictCursor

from database import db_config_var, subdomain_var


FTE_WORKING_DAYS_PER_MONTH = 22  # FTE = SUM(activity.duration days) / 22
MAX_BOTTLENECK_LIMIT = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_conn():
    db_config = db_config_var.get()
    return psycopg2.connect(**db_config)


def _get_tenant_id() -> str:
    return subdomain_var.get()



# Domain lookup CTE — used as fallback when vw_dashboard_proc_domain does not exist.
_DOMAIN_CTE = """
    proc_domain AS (
        SELECT
            sub->>'proc_def_id' AS proc_def_id,
            major->>'domain'    AS domain,
            c.tenant_id
        FROM configuration c,
            jsonb_array_elements(c.value->'mega_proc_list') AS mega,
            jsonb_array_elements(mega->'major_proc_list')   AS major,
            jsonb_array_elements(major->'sub_proc_list')    AS sub
        WHERE c.key = 'proc_map'
          AND sub->>'proc_def_id' IS NOT NULL
    )
"""


# ---------------------------------------------------------------------------
# 4.1  Process Bottleneck List
# ---------------------------------------------------------------------------

async def get_bottleneck_list(
    limit: int = Query(10, ge=1, le=MAX_BOTTLENECK_LIMIT),
    sort: str = Query("fte_desc", regex="^(fte_desc|fte_asc|oss_desc|name_asc)$"),
):
    tenant_id = _get_tenant_id()

    sort_map = {
        "fte_desc": "total_fte DESC",
        "fte_asc": "total_fte ASC",
        "oss_desc": "oss_count DESC",
        "name_asc": "proc_def_name ASC",
    }
    order_clause = sort_map.get(sort, "total_fte DESC")

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"""
                WITH {_DOMAIN_CTE},
                fte_source AS (
                    -- FTE = SUM(activity.duration days) / 22 working-days per month
                    SELECT
                        pd.id AS proc_def_id,
                        ROUND(
                            SUM(COALESCE((act->>'duration')::numeric, 0))
                            / %(fte_divisor)s, 4
                        ) AS total_fte,
                        COUNT(*) AS task_count
                    FROM proc_def pd
                    CROSS JOIN jsonb_array_elements(
                        CASE WHEN pd.definition->'activities' IS NOT NULL
                             THEN pd.definition->'activities'
                             ELSE '[]'::jsonb END
                    ) AS act
                    WHERE pd.isdeleted = false
                      AND pd.tenant_id = %(tenant_id)s
                    GROUP BY pd.id
                ),
                oss_count AS (
                    SELECT
                        pd.id AS proc_def_id,
                        COUNT(DISTINCT act->>'tool') AS oss_count
                    FROM proc_def pd
                    CROSS JOIN jsonb_array_elements(
                        CASE WHEN pd.definition->'activities' IS NOT NULL
                             THEN pd.definition->'activities'
                             ELSE '[]'::jsonb END
                    ) AS act
                    WHERE pd.isdeleted = false
                      AND pd.tenant_id = %(tenant_id)s
                      AND act->>'tool' IS NOT NULL
                      AND btrim(act->>'tool') <> ''
                    GROUP BY pd.id
                ),
                owner_dept AS (
                    SELECT
                        pd.id AS proc_def_id,
                        (
                            SELECT string_agg(DISTINCT r->>'name', ', ')
                            FROM jsonb_array_elements(
                                CASE WHEN pd.definition->'roles' IS NOT NULL
                                     THEN pd.definition->'roles'
                                     ELSE '[]'::jsonb END
                            ) AS r
                            WHERE r->>'name' IS NOT NULL
                        ) AS owner_roles
                    FROM proc_def pd
                    WHERE pd.isdeleted = false AND pd.tenant_id = %(tenant_id)s
                )
                SELECT
                    ROW_NUMBER() OVER (ORDER BY {order_clause}) AS rank,
                    pd.id           AS proc_def_id,
                    pd.name         AS proc_def_name,
                    COALESCE(dm.domain, '미분류')  AS domain,
                    COALESCE(fs.total_fte, 0)      AS total_fte,
                    COALESCE(fs.task_count, 0)     AS task_count,
                    COALESCE(od.owner_roles, '-')  AS owner_department,
                    COALESCE(oc.oss_count, 0)      AS oss_count,
                    pd.tenant_id
                FROM proc_def pd
                LEFT JOIN fte_source fs ON fs.proc_def_id = pd.id
                LEFT JOIN oss_count oc  ON oc.proc_def_id = pd.id
                LEFT JOIN owner_dept od ON od.proc_def_id = pd.id
                LEFT JOIN proc_domain dm
                    ON dm.proc_def_id = pd.id AND dm.tenant_id = pd.tenant_id
                WHERE pd.isdeleted = false
                  AND pd.tenant_id = %(tenant_id)s
                ORDER BY {order_clause}
                LIMIT %(limit)s
            """

            cur.execute(query, {
                "tenant_id": tenant_id,
                "limit": limit,
                "fte_divisor": FTE_WORKING_DAYS_PER_MONTH,
            })
            rows = cur.fetchall()

            # total count
            cur.execute(
                "SELECT COUNT(*) FROM proc_def WHERE isdeleted = false AND tenant_id = %s",
                (tenant_id,),
            )
            total_count = cur.fetchone()["count"]

    finally:
        conn.close()

    return {
        "data": [dict(r) for r in rows],
        "total_count": total_count,
    }


# ---------------------------------------------------------------------------
# 4.2  Update Recency / Zombie Processes
# ---------------------------------------------------------------------------

async def get_zombie_processes(
    filter: str = Query("all", regex="^(all|3month|6month)$"),
):
    tenant_id = _get_tenant_id()

    # filter 기준: all = 90일+, 3month = 90~180일, 6month = 180일+
    if filter == "3month":
        interval_clause = "pd.saved_at < (now() - INTERVAL '90 days') AND pd.saved_at >= (now() - INTERVAL '180 days')"
    elif filter == "6month":
        interval_clause = "pd.saved_at < (now() - INTERVAL '180 days')"
    else:  # all
        interval_clause = "pd.saved_at < (now() - INTERVAL '90 days')"

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"""
                WITH {_DOMAIN_CTE},
                latest_state AS (
                    SELECT
                        pah.proc_def_id,
                        pah.to_state AS current_status,
                        ROW_NUMBER() OVER (
                            PARTITION BY pah.proc_def_id ORDER BY pah.created_at DESC
                        ) AS rn
                    FROM proc_def_approval_history pah
                    WHERE pah.tenant_id = %(tenant_id)s
                ),
                owner_info AS (
                    SELECT
                        pd.id AS proc_def_id,
                        jsonb_agg(
                            jsonb_build_object(
                                'role', r->>'name',
                                'default', r->>'default',
                                'endpoint', r->>'endpoint'
                            )
                        ) AS roles_detail
                    FROM proc_def pd
                    CROSS JOIN jsonb_array_elements(
                        CASE WHEN pd.definition->'roles' IS NOT NULL
                             THEN pd.definition->'roles'
                             ELSE '[]'::jsonb END
                    ) AS r
                    WHERE pd.isdeleted = false AND pd.tenant_id = %(tenant_id)s
                    GROUP BY pd.id
                )
                SELECT
                    pd.id                                           AS proc_def_id,
                    pd.name                                         AS proc_def_name,
                    COALESCE(dm.domain, '미분류')                    AS domain,
                    COALESCE(ls.current_status, 'draft')            AS current_status,
                    COALESCE(oi.roles_detail, '[]'::jsonb)          AS owner_roles,
                    pd.saved_at                                     AS last_modified_at,
                    EXTRACT(DAY FROM (now() - pd.saved_at))::int    AS days_since_update,
                    pd.tenant_id
                FROM proc_def pd
                LEFT JOIN latest_state ls ON ls.proc_def_id = pd.id AND ls.rn = 1
                LEFT JOIN owner_info oi   ON oi.proc_def_id = pd.id
                LEFT JOIN proc_domain dm
                    ON dm.proc_def_id = pd.id AND dm.tenant_id = pd.tenant_id
                WHERE pd.isdeleted = false
                  AND pd.tenant_id = %(tenant_id)s
                  AND COALESCE(ls.current_status, 'draft') IN ('draft', 'in_review')
                  AND {interval_clause}
                ORDER BY pd.saved_at ASC
            """
            cur.execute(query, {"tenant_id": tenant_id})
            rows = cur.fetchall()

            for row in rows:
                if isinstance(row.get("owner_roles"), str):
                    row["owner_roles"] = json.loads(row["owner_roles"])
    finally:
        conn.close()

    return {
        "data": [dict(r) for r in rows],
        "total_count": len(rows),
        "filter_applied": filter,
    }


# ---------------------------------------------------------------------------
# 4.2 Action: 갱신 요청 알림 발송
# ---------------------------------------------------------------------------

async def request_process_update(request: Request):
    """프로세스 현행화 요청 알림을 Owner에게 발송한다."""
    body = await request.json()
    proc_def_id = body.get("proc_def_id")
    if not proc_def_id:
        raise HTTPException(status_code=400, detail="proc_def_id is required")

    requester_id = body.get("requester_id", "")
    requester_name = body.get("requester_name", "시스템")
    message = body.get("message", "프로세스 현행화 요망")
    tenant_id = _get_tenant_id()

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # proc_def 존재 여부 및 roles 조회
            cur.execute(
                "SELECT id, name, definition->'roles' AS roles "
                "FROM proc_def WHERE id = %s AND tenant_id = %s AND isdeleted = false",
                (proc_def_id, tenant_id),
            )
            proc = cur.fetchone()
            if not proc:
                raise HTTPException(status_code=404, detail="Process definition not found")

            # recipient_snapshot 구성 (roles에서 이메일 추출)
            recipient_snapshot = _build_recipient_snapshot(cur, proc.get("roles"), tenant_id)
            if not recipient_snapshot:
                raise HTTPException(
                    status_code=400,
                    detail="No recipients found for this process definition",
                )

            payload = {
                "event_type": "update_request",
                "proc_def_id": proc_def_id,
                "proc_def_name": proc["name"],
                "requester_id": requester_id,
                "requester_name": requester_name,
                "message": message,
                "requested_at": datetime.utcnow().isoformat(),
            }

            dedupe_key = f"update_request:{proc_def_id}:{datetime.utcnow().strftime('%Y%m%d')}"

            # governance_notification_outbox에 직접 enqueue
            cur.execute(
                """
                INSERT INTO governance_notification_outbox
                    (id, tenant_id, event_type, status, payload, recipient_snapshot, dedupe_key)
                VALUES
                    (gen_random_uuid(), %s, 'update_request', 'PENDING', %s, %s, %s)
                ON CONFLICT (dedupe_key) DO NOTHING
                RETURNING id
                """,
                (
                    tenant_id,
                    json.dumps(payload, ensure_ascii=False),
                    json.dumps(recipient_snapshot, ensure_ascii=False),
                    dedupe_key,
                ),
            )
            result = cur.fetchone()
            conn.commit()

    finally:
        conn.close()

    if result:
        return {
            "success": True,
            "notification_id": str(result["id"]),
            "recipients_count": len(recipient_snapshot),
            "message": f"갱신 요청 알림이 {len(recipient_snapshot)}명에게 발송 대기됩니다.",
        }
    else:
        return {
            "success": False,
            "message": "이미 오늘 해당 프로세스에 대한 갱신 요청이 발송되었습니다.",
        }


def _build_recipient_snapshot(cur, roles_jsonb, tenant_id: str) -> List[Dict[str, str]]:
    """proc_def roles에서 수신자 목록을 구성한다."""
    recipients = []
    if not roles_jsonb:
        return recipients

    roles = roles_jsonb if isinstance(roles_jsonb, list) else json.loads(roles_jsonb)

    for role in roles:
        role_name = role.get("name", "")

        for field in ("default", "endpoint"):
            value = (role.get(field) or "").strip()
            if not value:
                continue

            if "@" in value:
                # 이메일 주소
                if not any(r["addr"] == value for r in recipients):
                    recipients.append({"name": role_name or value, "addr": value, "role": role_name})
            else:
                # user_id → 이메일 조회
                cur.execute(
                    "SELECT email, username FROM users WHERE id::text = %s AND tenant_id = %s LIMIT 1",
                    (value, tenant_id),
                )
                user = cur.fetchone()
                if user and user["email"] and not any(r["addr"] == user["email"] for r in recipients):
                    recipients.append({
                        "name": user.get("username") or role_name or user["email"],
                        "addr": user["email"],
                        "role": role_name,
                    })

    return recipients


# ---------------------------------------------------------------------------
# 4.3  Action Required
# ---------------------------------------------------------------------------

async def get_action_required(
    user_id: Optional[str] = Query(None, description="현재 사용자 ID (조직 필터용)"),
):
    tenant_id = _get_tenant_id()
    conn = _get_conn()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 4.3-a: 7일 이상 지연된 검토 건
            cur.execute(
                f"""
                WITH {_DOMAIN_CTE},
                latest_approval AS (
                    SELECT
                        pah.proc_def_id,
                        pah.id          AS approval_id,
                        pah.from_state,
                        pah.to_state,
                        pah.action,
                        pah.actor_id,
                        pah.actor_name,
                        pah.created_at  AS submitted_at,
                        pah.tenant_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY pah.proc_def_id ORDER BY pah.created_at DESC
                        ) AS rn
                    FROM proc_def_approval_history pah
                    WHERE pah.tenant_id = %(tenant_id)s
                )
                SELECT
                    la.proc_def_id,
                    pd.name                                          AS proc_def_name,
                    COALESCE(dm.domain, '미분류')                     AS domain,
                    la.from_state,
                    la.to_state                                      AS current_status,
                    la.action,
                    la.actor_name,
                    la.submitted_at,
                    EXTRACT(DAY FROM (now() - la.submitted_at))::int AS days_delayed,
                    la.tenant_id
                FROM latest_approval la
                JOIN proc_def pd ON pd.id = la.proc_def_id AND pd.tenant_id = la.tenant_id
                LEFT JOIN proc_domain dm
                    ON dm.proc_def_id = pd.id AND dm.tenant_id = pd.tenant_id
                WHERE la.rn = 1
                  AND la.to_state = 'in_review'
                  AND la.submitted_at < (now() - INTERVAL '7 days')
                  AND pd.isdeleted = false
                ORDER BY la.submitted_at ASC
                """,
                {"tenant_id": tenant_id},
            )
            delayed_reviews = [dict(r) for r in cur.fetchall()]

            # 4.3-b: Re-open / Reject 후 미처리 건
            cur.execute(
                f"""
                WITH {_DOMAIN_CTE},
                ranked_history AS (
                    SELECT
                        pah.proc_def_id,
                        pah.id          AS approval_id,
                        pah.from_state,
                        pah.to_state,
                        pah.action,
                        pah.actor_id,
                        pah.actor_name,
                        pah.created_at  AS requested_at,
                        pah.tenant_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY pah.proc_def_id ORDER BY pah.created_at DESC
                        ) AS rn
                    FROM proc_def_approval_history pah
                    WHERE pah.tenant_id = %(tenant_id)s
                )
                SELECT
                    rh.proc_def_id,
                    pd.name                                           AS proc_def_name,
                    COALESCE(dm.domain, '미분류')                      AS domain,
                    rh.from_state,
                    rh.to_state,
                    rh.action,
                    rh.actor_name                                     AS requester_name,
                    rh.requested_at,
                    EXTRACT(DAY FROM (now() - rh.requested_at))::int  AS days_pending,
                    rh.tenant_id
                FROM ranked_history rh
                JOIN proc_def pd ON pd.id = rh.proc_def_id AND pd.tenant_id = rh.tenant_id
                LEFT JOIN proc_domain dm
                    ON dm.proc_def_id = pd.id AND dm.tenant_id = pd.tenant_id
                WHERE rh.rn = 1
                  AND (
                      rh.action ILIKE '%%re-open%%'
                      OR rh.action ILIKE '%%reopen%%'
                      OR rh.action ILIKE '%%reject%%'
                      OR (rh.from_state IN ('published', 'in_review') AND rh.to_state = 'draft')
                  )
                  AND pd.isdeleted = false
                ORDER BY rh.requested_at ASC
                """,
                {"tenant_id": tenant_id},
            )
            pending_reopens = [dict(r) for r in cur.fetchall()]

    finally:
        conn.close()

    return {
        "delayed_reviews": delayed_reviews,
        "pending_reopens": pending_reopens,
        "summary": {
            "delayed_review_count": len(delayed_reviews),
            "pending_reopen_count": len(pending_reopens),
        },
    }


# ---------------------------------------------------------------------------
# Debug: 데이터 존재 여부 진단
# ---------------------------------------------------------------------------

async def debug_check_data():
    """Tab B 데이터 존재 여부를 진단합니다."""
    tenant_id = _get_tenant_id()
    conn = _get_conn()
    result = {}
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1) proc_def 존재 여부
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM proc_def WHERE tenant_id = %s AND isdeleted = false",
                (tenant_id,),
            )
            result["proc_def_count"] = cur.fetchone()["cnt"]

            # 2) proc_def 중 id LIKE 'proc-%' (더미 데이터)
            cur.execute(
                "SELECT id, name, saved_at, "
                "  definition->'activities' IS NOT NULL AS has_activities, "
                "  jsonb_array_length(CASE WHEN definition->'activities' IS NOT NULL THEN definition->'activities' ELSE '[]'::jsonb END) AS activity_count "
                "FROM proc_def WHERE tenant_id = %s AND isdeleted = false AND id LIKE 'proc-%%' "
                "ORDER BY id LIMIT 20",
                (tenant_id,),
            )
            result["dummy_proc_defs"] = [dict(r) for r in cur.fetchall()]

            # 3) FTE 계산 테스트
            cur.execute("""
                SELECT pd.id,
                    SUM(COALESCE((act->>'duration')::numeric, 0)) AS total_duration_days,
                    ROUND(SUM(COALESCE((act->>'duration')::numeric, 0)) / 22, 4) AS fte
                FROM proc_def pd
                CROSS JOIN jsonb_array_elements(
                    CASE WHEN pd.definition->'activities' IS NOT NULL
                         THEN pd.definition->'activities'
                         ELSE '[]'::jsonb END
                ) AS act
                WHERE pd.tenant_id = %s AND pd.isdeleted = false AND pd.id LIKE 'proc-%%'
                GROUP BY pd.id
                ORDER BY fte DESC
                LIMIT 10
            """, (tenant_id,))
            result["fte_test"] = [dict(r) for r in cur.fetchall()]

            # 4) approval_history 존재 여부
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM proc_def_approval_history WHERE tenant_id = %s AND proc_def_id LIKE 'proc-%%'",
                (tenant_id,),
            )
            result["approval_history_count"] = cur.fetchone()["cnt"]

            # 5) zombie 후보
            cur.execute("""
                SELECT pd.id, pd.name, pd.saved_at,
                    EXTRACT(DAY FROM (now() - pd.saved_at))::int AS days_ago
                FROM proc_def pd
                WHERE pd.tenant_id = %s AND pd.isdeleted = false
                  AND pd.saved_at < (now() - INTERVAL '90 days')
                ORDER BY pd.saved_at ASC
            """, (tenant_id,))
            result["zombie_candidates"] = [dict(r) for r in cur.fetchall()]

            # 6) 현재 tenant_id
            result["tenant_id"] = tenant_id

    finally:
        conn.close()

    return result


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def add_routes_to_app(app: FastAPI):
    app.add_api_route(
        "/operational-board/bottleneck",
        get_bottleneck_list,
        methods=["GET"],
        tags=["Operational Board"],
    )
    app.add_api_route(
        "/operational-board/zombie-processes",
        get_zombie_processes,
        methods=["GET"],
        tags=["Operational Board"],
    )
    app.add_api_route(
        "/operational-board/request-update",
        request_process_update,
        methods=["POST"],
        tags=["Operational Board"],
    )
    app.add_api_route(
        "/operational-board/action-required",
        get_action_required,
        methods=["GET"],
        tags=["Operational Board"],
    )
    app.add_api_route(
        "/operational-board/debug",
        debug_check_data,
        methods=["GET"],
        tags=["Operational Board"],
    )

