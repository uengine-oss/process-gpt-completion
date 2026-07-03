"""POST /validate-and-improve — 임시저장(draft) 프로세스를 실제 실행 엔진으로 검증하고
LLM 으로 자동 개선한다.

흐름(pdf2bpmn 의 ProcessValidator 를 그대로 재사용):
  1) 프론트가 생성된 정의를 proc_def 에 is_draft=true 로 저장(draft) → 이 엔드포인트 호출.
  2) 여기서 draft proc_def 를 id 로 로드 → ProcessValidator.validate_and_repair 실행.
     - 정적 검사 + 실제 엔진(/initiate·/complete) 실행 트레이스 + LLM 보정 루프.
     - 매 개선마다 save_definition 으로 **proc_def.definition(draft) 을 UPDATE**.
  3) 최종 리포트(passed/iterations/repaired/remaining_defects/final_definition) 반환.

deepagent 는 이 경로에 전혀 관여하지 않는다(정책: deepagent DB write 금지).
DB write 는 프론트(draft 저장/최종 승격)와 completion(검증 중 definition UPDATE)만 한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import HTTPException

from database import supabase_client_var, subdomain_var, fetch_process_definition
from llm_factory import openai_compatible_client_config, get_llm_model
from process_validator import ProcessValidator

logger = logging.getLogger(__name__)


def _engine_base_url() -> str:
    """검증 시 driving 할 실행 엔진(자기 자신)의 베이스 URL."""
    return (
        os.getenv("COMPLETION_VALIDATION_ENGINE_URL")
        or os.getenv("COMPLETION_ENGINE_URL")
        or "http://localhost:8000"
    ).rstrip("/")


def _make_llm_call():
    """ProcessValidator 가 기대하는 async llm_call(messages, max_tokens) -> dict 를 만든다."""
    from openai import AsyncOpenAI

    cfg = openai_compatible_client_config()
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["openai_base_url"])
    model = get_llm_model()

    async def llm_call(messages: list, max_tokens: int = 4000) -> dict:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=int(max_tokens or 4000),
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            return {}
        try:
            return json.loads(content)
        except Exception:
            # 코드펜스 등 잡음 제거 후 재시도
            s = content
            if "```" in s:
                s = s.split("```", 2)[1] if s.count("```") >= 2 else s
                s = s.replace("json", "", 1).strip("` \n")
            try:
                return json.loads(s)
            except Exception:
                return {}

    return llm_call


def _make_save_definition(supabase, tenant_id: str):
    """검증 루프가 개선할 때마다 draft proc_def.definition 을 UPDATE 한다."""

    async def save_definition(proc_def_id: str, definition: dict) -> None:
        def _do():
            return (
                supabase.table("proc_def")
                .update({"definition": definition})
                .eq("id", str(proc_def_id).lower())
                .eq("tenant_id", tenant_id)
                .execute()
            )

        await asyncio.to_thread(_do)

    return save_definition


def _make_fetch_instance_state(supabase, tenant_id: str):
    async def fetch_instance_state(proc_inst_id: str) -> dict:
        def _do():
            return (
                supabase.table("bpm_proc_inst")
                .select("status, current_activity_ids")
                .eq("proc_inst_id", proc_inst_id)
                .eq("tenant_id", tenant_id)
                .execute()
            )

        resp = await asyncio.to_thread(_do)
        rows = getattr(resp, "data", None) or []
        if not rows:
            return {"status": "RUNNING", "current_activity_ids": []}
        row = rows[0]
        return {
            "status": row.get("status") or "RUNNING",
            "current_activity_ids": row.get("current_activity_ids") or [],
        }

    return fetch_instance_state


async def _attribute_instances_to_user(supabase, tenant_id: str, inst_ids: list, user_uid: str) -> None:
    """검증으로 생성된 실행 인스턴스를 요청 사용자에게 귀속한다(participants 에 uid 추가).

    완료 인스턴스 목록은 participants 에 사용자 uid 가 있어야 노출되므로, 검증이 실제로
    프로세스를 end 까지 실행한 증거(완료 인스턴스)를 사용자가 좌측 패널에서 확인할 수 있게 한다.
    """
    if not (inst_ids and user_uid):
        return

    def _do():
        for pid in inst_ids:
            try:
                cur = (
                    supabase.table("bpm_proc_inst")
                    .select("participants")
                    .eq("proc_inst_id", pid)
                    .eq("tenant_id", tenant_id)
                    .limit(1)
                    .execute()
                )
                rows = getattr(cur, "data", None) or []
                if not rows:
                    continue
                parts = rows[0].get("participants") or []
                if not isinstance(parts, list):
                    parts = []
                if user_uid not in parts:
                    parts.append(user_uid)
                supabase.table("bpm_proc_inst").update({"participants": parts}).eq(
                    "proc_inst_id", pid
                ).eq("tenant_id", tenant_id).execute()
            except Exception:
                continue

    await asyncio.to_thread(_do)


def _make_cleanup_instance(supabase, tenant_id: str):
    async def cleanup_instance(proc_inst_id: str) -> None:
        def _do():
            try:
                supabase.table("todolist").delete().eq("proc_inst_id", proc_inst_id).eq(
                    "tenant_id", tenant_id
                ).execute()
            except Exception:
                pass
            try:
                supabase.table("bpm_proc_inst").delete().eq(
                    "proc_inst_id", proc_inst_id
                ).eq("tenant_id", tenant_id).execute()
            except Exception:
                pass

        await asyncio.to_thread(_do)

    return cleanup_instance


async def validate_and_improve(input: dict) -> dict:
    """draft proc_def 를 실행 엔진으로 검증 + 자동 개선하고 최종 정의/리포트를 반환한다.

    body: { "input": {
        "process_definition_id": "<draft proc_def id>",   # 필수
        "process_name": "<선택>",
        "forms": { "<activity_id>": {...} },               # 선택(프론트가 전달)
        "max_iters": 5,                                    # 선택
        "email": "<검증 actor>"                            # 선택
    }}
    """
    payload = input.get("input") if isinstance(input.get("input"), dict) else input
    proc_def_id = (payload or {}).get("process_definition_id")
    if not proc_def_id:
        raise HTTPException(status_code=400, detail="process_definition_id is required")

    supabase = supabase_client_var.get()
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase client not configured")
    tenant_id = subdomain_var.get() or "localhost"

    # draft proc_def 로드 (proc_def.definition = 런타임 정의 = proc_json)
    proc_json = await asyncio.to_thread(fetch_process_definition, proc_def_id, tenant_id)
    if not proc_json or not isinstance(proc_json, dict):
        raise HTTPException(
            status_code=404,
            detail=f"draft process definition not found: {proc_def_id}",
        )

    process_name = (
        (payload or {}).get("process_name")
        or proc_json.get("processDefinitionName")
        or proc_def_id
    )
    forms = (payload or {}).get("forms") or {}
    max_iters = int((payload or {}).get("max_iters") or 5)
    actor_email = (payload or {}).get("email")
    user_uid = (payload or {}).get("user_uid")

    # 검증 인스턴스 정리 정책: 기본 KEEP(완료 인스턴스를 좌측 패널에서 확인 가능).
    # 운영에서 정리하려면 COMPLETION_VALIDATION_KEEP_INSTANCES=0.
    keep_instances = os.getenv("COMPLETION_VALIDATION_KEEP_INSTANCES", "1") != "0"

    validator = ProcessValidator(
        llm_call=_make_llm_call(),
        save_definition=_make_save_definition(supabase, tenant_id),
        engine_base_url=_engine_base_url(),
        tenant_id=tenant_id,
        fetch_instance_state=_make_fetch_instance_state(supabase, tenant_id),
        cleanup_instance=None if keep_instances else _make_cleanup_instance(supabase, tenant_id),
        max_iters=max_iters,
        actor_email=actor_email,
        logger=logger,
    )

    try:
        report = await validator.validate_and_repair(
            proc_def_id=proc_def_id,
            process_name=process_name,
            proc_json=proc_json,
            forms=forms,
        )
        # 검증 인스턴스를 요청 사용자에게 귀속(완료 목록 노출). KEEP 모드일 때만.
        if keep_instances and user_uid:
            try:
                inst_ids = list(getattr(validator, "_test_proc_inst_ids", []) or [])
                await _attribute_instances_to_user(supabase, tenant_id, inst_ids, user_uid)
                report["test_instance_ids"] = inst_ids
            except Exception as _ae:
                logger.warning("[validate-and-improve] 인스턴스 귀속 실패(무시): %s", _ae)
    except Exception as e:
        logger.exception("[validate-and-improve] 실패: %s", e)
        # 검증 실패해도 흐름은 끊지 않는다 — 현재 정의를 그대로 final 로 반환.
        return {
            "passed": None,
            "repaired": False,
            "iterations": 0,
            "remaining_defects": [{"type": "validation_error", "detail": str(e)}],
            "final_definition": proc_json,
            "error": str(e),
        }

    # final_definition 이 없으면(스킵 등) 현재 정의 유지
    if not report.get("final_definition"):
        report["final_definition"] = proc_json
    return report


def add_routes_to_app(app):
    app.add_api_route("/validate-and-improve", validate_and_improve, methods=["POST"])
