"""regression/versions.py — 병합 요청의 두 버전 정의를 꺼낸다.

스킬은 깃 브랜치로 버전을 가르지만 프로세스·DMN 은 그렇지 않다. 화면이 병합 요청을 만들 때
branch_name 을 `v<버전>` 으로 넣고(`DmnChat.vue`, `ProcessDefinitionVersionDialog.vue`),
버전 스냅샷은 `proc_def_version.arcv_id`(= `<id>_<버전>`) 에 있다. 그 규칙을 되짚는다.

이 데이터(`proc_def`, `proc_def_version`)는 이 서비스의 것이므로 조회도 여기서 한다.
"""

from __future__ import annotations

import hashlib
import json

from database import supabase_client_var


def _version_of(ref: str) -> str:
    version = (ref or "").strip()
    return version[1:] if version.startswith("v") else version


def _supabase():
    supabase = supabase_client_var.get()
    if supabase is None:
        raise RuntimeError("Supabase client is not configured for this request")
    return supabase


def load_process_definition(tenant_id: str, resource_id: str, ref: str) -> tuple[dict, str]:
    """프로세스의 특정 버전 정의(flattened JSON)와 digest.

    digest 는 기준선 재사용 판단에 쓰이므로 같은 버전이면 반드시 같아야 한다 — 키 순서에
    흔들리지 않도록 사전 순으로 직렬화해 해싱한다.
    """
    sb = _supabase()
    definition = None
    version = _version_of(ref)
    if version:
        rows = (
            sb.table("proc_def_version")
            .select("definition")
            .eq("arcv_id", f"{resource_id}_{version}")
            .limit(1)
            .execute()
        )
        if rows.data:
            definition = rows.data[0].get("definition")

    if definition is None:
        # 스냅샷이 없으면 현재 정의를 쓴다 — 첫 버전이라 아직 아카이브가 없는 경우가 있다.
        rows = (
            sb.table("proc_def")
            .select("definition")
            .eq("tenant_id", tenant_id)
            .eq("id", resource_id)
            .limit(1)
            .execute()
        )
        if not rows.data:
            raise ValueError(f"'{resource_id}' 프로세스 정의를 찾을 수 없습니다.")
        definition = rows.data[0].get("definition")

    if not isinstance(definition, dict) or not definition:
        raise ValueError(f"'{resource_id}' 의 '{ref}' 버전 정의가 비어 있습니다.")

    digest = hashlib.sha256(
        json.dumps(definition, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return definition, digest


def load_dmn_xml(tenant_id: str, resource_id: str, ref: str) -> tuple[str, str]:
    """의사결정의 특정 버전 DMN XML 과 digest.

    DMN 은 `proc_def.bpmn`(스냅샷은 `proc_def_version.snapshot`)에 XML 로 들어 있다.
    """
    sb = _supabase()
    dmn_xml = None
    version = _version_of(ref)
    if version:
        rows = (
            sb.table("proc_def_version")
            .select("snapshot")
            .eq("arcv_id", f"{resource_id}_{version}")
            .limit(1)
            .execute()
        )
        if rows.data:
            dmn_xml = rows.data[0].get("snapshot")

    if not dmn_xml:
        rows = (
            sb.table("proc_def")
            .select("bpmn")
            .eq("tenant_id", tenant_id)
            .eq("id", resource_id)
            .limit(1)
            .execute()
        )
        if not rows.data:
            raise ValueError(f"'{resource_id}' 의사결정을 찾을 수 없습니다.")
        dmn_xml = rows.data[0].get("bpmn")

    if not dmn_xml or not str(dmn_xml).strip():
        raise ValueError(f"'{resource_id}' 의 '{ref}' 버전 DMN 이 비어 있습니다.")

    digest = hashlib.sha256(str(dmn_xml).encode("utf-8")).hexdigest()
    return str(dmn_xml), digest
