"""
프로세스(Process) 속성값 CRUD API

엔드포인트
- GET    /api/v1/processes               목록 조회 (검색 ?search= / 필터 ?domain=&mega=&major=&owner= / 페이징 ?page=&size=)
- POST   /api/v1/processes               신규 생성
- GET    /api/v1/processes/{uuid}  단건 상세 조회
- GET    /api/v1/processes/{uuid}/bpmn  저장된 As-Is BPMN XML 원문 조회 (기존 호환 경로)
- GET    /api/v1/processes/{uuid}/asis-bpmn  저장된 As-Is BPMN XML 원문 조회
- GET    /api/v1/processes/{uuid}/tobe-bpmn  저장된 To-Be BPMN XML 원문 조회
- GET    /api/v1/processes/{uuid}/piflags  BPMN XML에 등록된 PI 플래그 조회
- PUT    /api/v1/processes/{uuid}  정보 수정 (부분 병합)
- PATCH  /api/v1/processes/{uuid}  정보 수정 (부분 병합, PUT 과 동일)
- DELETE /api/v1/processes/{uuid}  소프트 삭제 + proc_map 분류 제거
- PUT    /api/v1/processes/{uuid}/activities/{activity_id}  태스크(액티비티) 속성값 부분 수정
- PATCH  /api/v1/processes/{uuid}/activities/{activity_id}  태스크(액티비티) 속성값 부분 수정 (PUT 과 동일)

데이터 모델
- 프로세스 본체는 ``proc_def`` 테이블(id, name, tenant_id, isdeleted, saved_at, definition[jsonb]).
  description / owner / activities / roles 등은 definition(jsonb) 안에 보관한다.
- domain / mega / major 계층은 ``configuration`` 테이블의 key='proc_map' 값
  (mega_proc_list → major_proc_list(name, domain) → sub_proc_list(proc_def_id, name))에 있다.
- 따라서 한 "프로세스" 응답은 proc_def + proc_map 분류 정보를 평탄화(join)한 결과다.
  domain/mega/major 를 수정/생성하면 proc_map 의 위치를 동기화한다(없는 mega/major 노드는 생성).

주의(설계 제약)
- domain 은 major 노드의 속성(같은 major 의 모든 프로세스가 공유)이다. 형제 프로세스가
  쓰는 major 의 domain 을 다른 값으로 바꾸려 하면 422 로 거부한다(무단 변경 방지).
- mega+major 가 함께 확정돼야 분류가 가능하다. domain 만/일부만 보내 분류가 불가능하면 422.
- proc_def 와 configuration(proc_map) 는 별도 테이블이라 쓰기가 단일 트랜잭션이 아니다.
  생성/수정은 proc_map 변경을 메모리에서 먼저 검증(422 시 무쓰기)한 뒤 쓰기를 수행해 부분반영을
  최소화하지만, 인프라 장애 시 두 테이블이 어긋날 잔여 가능성은 남는다(필요 시 RPC/트랜잭션으로 보강).
"""

from __future__ import annotations

import uuid
import json
import copy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Body, Response
from pydantic import BaseModel, ConfigDict

from database import supabase_client_var, subdomain_var

try:
    from defusedxml import ElementTree as SafeET
except ImportError:  # pragma: no cover - requirements include defusedxml.
    from xml.etree import ElementTree as SafeET


PROC_MAP_KEY = "proc_map"
# 검색(LIKE) / 평탄화 대상 텍스트 필드
_SEARCHABLE_FIELDS = ("id", "name", "description", "owner", "domain", "mega", "major")
_MAX_PAGE_SIZE = 1000
_NO_STORE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


# --------------------------------------------------------------------------- #
# Pydantic request models
# --------------------------------------------------------------------------- #
class ProcessCreate(BaseModel):
    """프로세스 신규 생성 요청 본문."""

    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    owner: Optional[str] = None
    domain: Optional[str] = None
    mega: Optional[str] = None
    major: Optional[str] = None
    # activities / roles / gateways / sequences 등 추가 definition 속성 일체
    definition: Optional[Dict[str, Any]] = None


class ProcessUpdate(BaseModel):
    """프로세스 수정 요청 본문(모든 필드 선택적, 제공된 값만 병합)."""

    model_config = ConfigDict(extra="ignore")

    # id 는 경로로 식별하며 변경 불가. 본문에 다른 id 를 보내면 422 로 거부(무시 방지).
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    owner: Optional[str] = None
    domain: Optional[str] = None
    mega: Optional[str] = None
    major: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None


class ActivityUpdate(BaseModel):
    """태스크(액티비티) 속성값 부분 수정 요청.

    - 명시적으로 보낸(제공된) 속성만 대상 태스크에 병합한다(부분 수정).
    - 아래 선언 필드 외에 systemName / assignee / laneName / pythonCode 등
      임의 속성도 허용(extra="allow")하며, 보낸 값을 그대로 병합한다.
    - id 는 경로로 식별하며 변경 불가. 본문에 다른 id 를 보내면 422 로 거부한다.
    """

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    role: Optional[str] = None
    instruction: Optional[str] = None
    duration: Optional[int] = None
    tool: Optional[str] = None
    properties: Optional[Any] = None
    inputData: Optional[List[Any]] = None
    outputData: Optional[List[Any]] = None
    checkpoints: Optional[List[Any]] = None
    attachedEvents: Optional[List[Any]] = None


# --------------------------------------------------------------------------- #
# 작은 유틸
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> Dict[str, Any]:
    """jsonb 컬럼 값이 dict / JSON 문자열 / None 어느 쪽이든 dict 로 정규화."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _norm_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _clean_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def _prevent_cache(response: Response) -> None:
    """프로세스 변경 직후의 GET이 중간 캐시에서 재사용되지 않게 한다."""
    for name, value in _NO_STORE_HEADERS.items():
        response.headers[name] = value


# --------------------------------------------------------------------------- #
# Supabase I/O (테스트에서 monkeypatch 가능하도록 모듈 레벨 함수로 분리)
# --------------------------------------------------------------------------- #
def _get_supabase():
    supabase = supabase_client_var.get()
    if supabase is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase client is not configured for this request",
        )
    return supabase


def _tenant_id() -> str:
    return subdomain_var.get()


def _fetch_proc_def_rows(include_deleted: bool = False) -> List[Dict[str, Any]]:
    """현재 테넌트의 proc_def 전체 행. include_deleted=False 면 isdeleted(true) 제외."""
    supabase = _get_supabase()
    tenant_id = _tenant_id()
    response = (
        supabase.table("proc_def")
        .select("id,uuid,name,definition,bpmn,isdeleted,saved_at")
        .eq("tenant_id", tenant_id)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    if include_deleted:
        return rows
    # isdeleted 가 None/False 인 경우 모두 "삭제되지 않음"으로 취급
    return [row for row in rows if not row.get("isdeleted")]


def _fetch_proc_def_row(proc_id: str) -> Optional[Dict[str, Any]]:
    supabase = _get_supabase()
    tenant_id = _tenant_id()
    response = (
        supabase.table("proc_def")
        .select("id,uuid,name,definition,bpmn,isdeleted,saved_at")
        .eq("id", _norm_id(proc_id))
        .eq("tenant_id", tenant_id)
        .execute()
    )
    data = getattr(response, "data", None) or []
    return data[0] if data else None


def _fetch_proc_def_row_by_uuid(process_uuid: UUID) -> Optional[Dict[str, Any]]:
    """uuid 로 XML 컬럼 포함 행을 단일 조회한다.

    과거의 uuid→id 2단 조회는 id 재조회 시 _norm_id 소문자화·중복 id 행 때문에
    SQL 로는 보이는 행이 404 가 되는 사례가 있어 uuid 단일 조회로 교체했다.
    """
    supabase = _get_supabase()
    tenant_id = _tenant_id()

    def _select(columns: str):
        return (
            supabase.table("proc_def")
            .select(columns)
            .eq("uuid", str(process_uuid))
            .eq("tenant_id", tenant_id)
            .execute()
        )

    try:
        response = _select("id,name,definition,bpmn,tobe,isdeleted,saved_at")
    except Exception:
        # tobe 분리 컬럼(20260812) 미적용 DB 폴백 — To-Be XML 은 definition 인라인에서 읽는다
        response = _select("id,name,definition,bpmn,isdeleted,saved_at")
    data = getattr(response, "data", None) or []
    return data[0] if data else None


def _fetch_process_detail_row_by_uuid(process_uuid: UUID) -> Optional[Dict[str, Any]]:
    """상세 조회용 proc_def 행을 UUID 기준으로 조회한다."""
    supabase = _get_supabase()
    tenant_id = _tenant_id()
    response = (
        supabase.table("proc_def")
        .select("id,uuid,name,definition,bpmn,isdeleted,saved_at")
        .eq("uuid", str(process_uuid))
        .eq("tenant_id", tenant_id)
        .execute()
    )
    data = getattr(response, "data", None) or []
    return data[0] if data else None


def _insert_proc_def(row: Dict[str, Any]):
    supabase = _get_supabase()
    return supabase.table("proc_def").insert(row).execute()


def _update_proc_def_row(proc_id: str, patch: Dict[str, Any]):
    supabase = _get_supabase()
    tenant_id = _tenant_id()
    return (
        supabase.table("proc_def")
        .update(patch)
        .eq("id", _norm_id(proc_id))
        .eq("tenant_id", tenant_id)
        .execute()
    )


def _soft_delete_proc_def(proc_id: str):
    return _update_proc_def_row(
        proc_id, {"isdeleted": True, "saved_at": _now_iso()}
    )


def _load_proc_map() -> Dict[str, Any]:
    """configuration(key='proc_map') 값. 없으면 빈 구조 반환."""
    supabase = _get_supabase()
    tenant_id = _tenant_id()
    response = (
        supabase.table("configuration")
        .select("*")
        .eq("key", PROC_MAP_KEY)
        .eq("tenant_id", tenant_id)
        .execute()
    )
    data = getattr(response, "data", None) or []
    value = _as_dict(data[0].get("value")) if data else {}
    if not isinstance(value.get("mega_proc_list"), list):
        value["mega_proc_list"] = []
    return value


def _save_proc_map(value: Dict[str, Any]):
    supabase = _get_supabase()
    tenant_id = _tenant_id()
    payload = {"key": PROC_MAP_KEY, "value": value, "tenant_id": tenant_id}
    return supabase.table("configuration").upsert(
        payload, on_conflict="key,tenant_id"
    ).execute()


# --------------------------------------------------------------------------- #
# proc_map (계층) 순수 변환 로직
# --------------------------------------------------------------------------- #
def _build_proc_map_index(proc_map: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """proc_def_id(소문자) -> {mega, major, domain, sub_name} 인덱스."""
    index: Dict[str, Dict[str, Any]] = {}
    if not isinstance(proc_map, dict):
        return index
    for mega in proc_map.get("mega_proc_list") or []:
        if not isinstance(mega, dict):
            continue
        mega_name = mega.get("name")
        for major in mega.get("major_proc_list") or []:
            if not isinstance(major, dict):
                continue
            major_name = major.get("name")
            domain = major.get("domain")
            for sub in major.get("sub_proc_list") or []:
                if not isinstance(sub, dict):
                    continue
                pid = _norm_id(sub.get("proc_def_id"))
                if not pid:
                    continue
                index[pid] = {
                    "mega": mega_name,
                    "major": major_name,
                    "domain": domain,
                    "sub_name": sub.get("name"),
                }
    return index


def _remove_from_proc_map(
    proc_map: Dict[str, Any], proc_def_id: str
) -> Optional[Dict[str, Any]]:
    """proc_def_id 의 sub 항목을 모두 제거한다. **제거로 인해 비게 된** major/mega 만 정리하고,
    원래부터 비어 있던(다른 위치의) 카테고리 노드는 보존한다. 제거된 첫 분류 정보를 반환(없으면 None)."""
    pid = _norm_id(proc_def_id)
    old: Optional[Dict[str, Any]] = None
    surviving_megas: List[Any] = []
    for mega in proc_map.get("mega_proc_list") or []:
        if not isinstance(mega, dict):
            surviving_megas.append(mega)
            continue
        surviving_majors: List[Any] = []
        emptied_a_major = False
        for major in mega.get("major_proc_list") or []:
            if not isinstance(major, dict):
                surviving_majors.append(major)
                continue
            subs = major.get("sub_proc_list") or []
            kept_subs = [
                s for s in subs
                if not (isinstance(s, dict) and _norm_id(s.get("proc_def_id")) == pid)
            ]
            removed_here = len(kept_subs) != len(subs)
            if removed_here and old is None:
                for s in subs:
                    if isinstance(s, dict) and _norm_id(s.get("proc_def_id")) == pid:
                        old = {
                            "mega": mega.get("name"),
                            "major": major.get("name"),
                            "domain": major.get("domain"),
                            "sub_name": s.get("name"),
                        }
                        break
            if removed_here:
                major["sub_proc_list"] = kept_subs
                if not kept_subs:  # 우리가 비운 major 만 제거
                    emptied_a_major = True
                    continue
            surviving_majors.append(major)
        mega["major_proc_list"] = surviving_majors
        # 우리가 major 를 비워서 mega 까지 비게 된 경우에만 mega 제거(원래 빈 mega 는 보존)
        if emptied_a_major and not surviving_majors:
            continue
        surviving_megas.append(mega)
    proc_map["mega_proc_list"] = surviving_megas
    return old


def _find_or_create(
    items: List[Any], key: str, value: Any, default: Dict[str, Any]
) -> Tuple[Dict[str, Any], bool]:
    """items 에서 key==value 인 dict 를 찾고 없으면 default 추가. (노드, 새로생성여부) 반환."""
    for item in items:
        if isinstance(item, dict) and item.get(key) == value:
            return item, False
    items.append(default)
    return default, True


def _apply_classification(
    proc_map: Dict[str, Any],
    proc_def_id: str,
    *,
    name: Optional[str] = None,
    mega: Optional[str] = None,
    major: Optional[str] = None,
    domain: Optional[str] = None,
    classification_requested: bool = True,
) -> Dict[str, Any]:
    """proc_def 를 계층(proc_map) 내에서 이동/배치한다.

    - ``name`` 은 proc_def.name(권위 있는 값)을 그대로 sub 이름에 기록한다
      (과거 stale sub_name 승계 금지).
    - ``mega``/``major`` 가 None 이면 기존 배치 값을 승계한다(부분 수정 지원).
    - ``domain`` 은 major 노드의 속성(해당 major 의 모든 프로세스가 공유)이므로:
        * 새로 만들어지는 major 노드: 명시 domain → 없으면 기존 domain 승계(같은 위치 재생성 시 보존).
        * 이미 존재하는(형제 프로세스가 공유하는) major 노드: 명시 domain 이
          기존 domain 과 충돌하면 422(형제 도메인 무단 변경 방지). 미명시면 손대지 않는다.
    - 유효 mega/major 가 모두 확정되지 않는데 분류가 요청되면 422.
    """
    pid = _norm_id(proc_def_id)
    old = _remove_from_proc_map(proc_map, pid)

    eff_mega = mega or (old.get("mega") if old else None)
    eff_major = major or (old.get("major") if old else None)
    domain_provided = domain is not None
    # 새 major 노드에 기록할 domain: 명시값 우선, 없으면 기존(같은 위치 재생성 시 보존)
    new_major_domain = domain if domain_provided else (old.get("domain") if old else None)

    if not (eff_mega and eff_major):
        if classification_requested:
            raise HTTPException(
                status_code=422,
                detail="mega and major are both required to classify a process",
            )
        return proc_map  # 방어적: 분류 요청이 아니면 미분류 유지

    proc_map.setdefault("mega_proc_list", [])
    mega_node, _ = _find_or_create(
        proc_map["mega_proc_list"], "name", eff_mega,
        {"name": eff_mega, "major_proc_list": []},
    )
    if not isinstance(mega_node.get("major_proc_list"), list):
        mega_node["major_proc_list"] = []
    major_node, major_created = _find_or_create(
        mega_node["major_proc_list"], "name", eff_major,
        {"name": eff_major, "sub_proc_list": []},
    )

    if major_created:
        if new_major_domain is not None:
            major_node["domain"] = new_major_domain
    elif domain_provided:
        existing_domain = major_node.get("domain")
        if existing_domain in (None, domain):
            major_node["domain"] = domain
        else:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"domain '{domain}' conflicts with existing domain "
                    f"'{existing_domain}' for major '{eff_major}'. "
                    "domain is shared by all processes under a major."
                ),
            )

    if not isinstance(major_node.get("sub_proc_list"), list):
        major_node["sub_proc_list"] = []
    major_node["sub_proc_list"].append({"proc_def_id": pid, "name": name})
    return proc_map


# --------------------------------------------------------------------------- #
# 평탄화 / 검색 / 필터 / 페이징 (순수)
# --------------------------------------------------------------------------- #
def _flatten_process(
    row: Dict[str, Any],
    classification: Optional[Dict[str, Any]],
    include_definition: bool = False,
) -> Dict[str, Any]:
    definition = _as_dict(row.get("definition"))
    activities = definition.get("activities")
    flat: Dict[str, Any] = {
        "id": row.get("id"),
        "uuid": row.get("uuid"),
        "name": row.get("name"),
        "description": definition.get("description"),
        "owner": definition.get("owner"),
        "domain": classification.get("domain") if classification else None,
        "mega": classification.get("mega") if classification else None,
        "major": classification.get("major") if classification else None,
        "isdeleted": bool(row.get("isdeleted")),
        "saved_at": row.get("saved_at"),
        "activity_count": len(activities) if isinstance(activities, list) else 0,
    }
    if include_definition:
        flat["definition"] = definition
    return flat


def _matches_search(item: Dict[str, Any], search: Optional[str]) -> bool:
    if not search:
        return True
    needle = search.strip().lower()
    if not needle:
        return True
    for key in _SEARCHABLE_FIELDS:
        value = item.get(key)
        if value and needle in str(value).lower():
            return True
    return False


def _matches_filters(item: Dict[str, Any], filters: Dict[str, Optional[str]]) -> bool:
    for key, expected in filters.items():
        if expected is None:
            continue
        actual = item.get(key)
        if actual is None:
            return False
        if str(actual).strip().lower() != str(expected).strip().lower():
            return False
    return True


def _paginate(
    items: List[Dict[str, Any]], page: int, size: Optional[int]
) -> Tuple[List[Dict[str, Any]], int, int]:
    """(page_items, page, size) 반환. size 가 None 이면 전체 반환."""
    total = len(items)
    if size is None:
        return items, 1, total
    start = (page - 1) * size
    return items[start:start + size], page, size


# --------------------------------------------------------------------------- #
# 서비스 레이어
# --------------------------------------------------------------------------- #
def _list_processes(
    *,
    search: Optional[str] = None,
    domain: Optional[str] = None,
    mega: Optional[str] = None,
    major: Optional[str] = None,
    owner: Optional[str] = None,
    page: int = 1,
    size: Optional[int] = None,
    include_deleted: bool = False,
) -> Dict[str, Any]:
    rows = _fetch_proc_def_rows(include_deleted=include_deleted)
    index = _build_proc_map_index(_load_proc_map())

    items = [
        _flatten_process(row, index.get(_norm_id(row.get("id"))))
        for row in rows
    ]
    # 빈 문자열 필터(?domain=&mega=)는 UI 기본값이므로 무시(None 으로 정규화)
    filters = {
        "domain": _clean_str(domain),
        "mega": _clean_str(mega),
        "major": _clean_str(major),
        "owner": _clean_str(owner),
    }
    items = [
        item
        for item in items
        if _matches_search(item, search) and _matches_filters(item, filters)
    ]
    items.sort(key=lambda it: ((it.get("name") or "").casefold(), it.get("id") or ""))

    total = len(items)
    page_items, page_out, size_out = _paginate(items, page, size)
    return {
        "items": page_items,
        "total": total,
        "page": page_out,
        "size": size_out,
    }


def _get_process(proc_id: str, *, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
    row = _fetch_proc_def_row(proc_id)
    if row is None:
        return None
    if not include_deleted and row.get("isdeleted"):
        return None
    index = _build_proc_map_index(_load_proc_map())
    classification = index.get(_norm_id(row.get("id")))
    return _flatten_process(row, classification, include_definition=True)


def _get_process_by_uuid(
    process_uuid: UUID, *, include_deleted: bool = False
) -> Optional[Dict[str, Any]]:
    row = _fetch_process_detail_row_by_uuid(process_uuid)
    if row is None or (not include_deleted and row.get("isdeleted")):
        return None
    index = _build_proc_map_index(_load_proc_map())
    classification = index.get(_norm_id(row.get("id")))
    return _flatten_process(row, classification, include_definition=True)


def _resolve_active_proc_id_by_uuid(process_uuid: UUID) -> Optional[str]:
    """외부 UUID를 내부 논리 ID로 변환한다. 삭제된 프로세스는 조회 대상이 아니다."""
    row = _fetch_process_detail_row_by_uuid(process_uuid)
    if row is None or row.get("isdeleted"):
        return None
    return _norm_id(row.get("id")) or None


def _get_process_bpmn(process_uuid: UUID, *, include_deleted: bool = False) -> Optional[str]:
    """proc_def.bpmn에 저장된 BPMN XML 원문을 변환 없이 반환한다."""
    row = _fetch_proc_def_row_by_uuid(process_uuid)
    if row is None or (not include_deleted and row.get("isdeleted")):
        return None

    bpmn = row.get("bpmn")
    if not isinstance(bpmn, str) or not bpmn.strip():
        return None
    return bpmn


def _extract_tobe_xml(row: Dict[str, Any]) -> Optional[str]:
    """proc_def.tobe 분리 컬럼과 미마이그레이션 definition 인라인 형태를 모두 지원한다.

    저장 형태(프론트 splitDefinitionForStorage 기준):
    - tobe 컬럼: {"tobe_bpmn": <캔버스 XML>, "blueprint_xml": <블루프린트 XML>, ...}
    - 미마이그레이션 행: definition["tobe_bpmn"] / definition["tobe"]["blueprint_xml"]
    """
    tobe_col = row.get("tobe")
    if isinstance(tobe_col, str) and tobe_col.strip():
        return tobe_col
    tobe = _as_dict(tobe_col)
    definition = _as_dict(row.get("definition"))
    candidates = (
        tobe.get("tobe_bpmn"),
        tobe.get("blueprint_xml"),
        definition.get("tobe_bpmn"),
        _as_dict(definition.get("tobe")).get("blueprint_xml"),
    )
    for xml in candidates:
        if isinstance(xml, str) and xml.strip():
            return xml
    return None


def _get_process_tobe_bpmn(
    process_uuid: UUID, *, include_deleted: bool = False
) -> Optional[str]:
    """proc_def.tobe에 저장된 To-Be BPMN XML 원문을 변환 없이 반환한다."""
    row = _fetch_proc_def_row_by_uuid(process_uuid)
    if row is None or (not include_deleted and row.get("isdeleted")):
        return None

    return _extract_tobe_xml(row)


def _xml_local_name(tag: Any) -> str:
    """ElementTree의 namespace 태그와 prefix 태그를 모두 처리한다."""
    return str(tag or "").rsplit("}", 1)[-1].split(":")[-1]


def _json_objects_from_legacy_text(raw: str) -> List[Dict[str, Any]]:
    """텍스트 안에 섞인 JSON object를 중괄호 짝으로 찾아 파싱한다."""
    objects: List[Dict[str, Any]] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    parsed = json.loads(raw[start:index + 1])
                except (TypeError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    objects.append(parsed)
                start = None
    return objects


def _parse_property_json(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw.strip())
    except (TypeError, ValueError):
        return _json_objects_from_legacy_text(raw)
    return [parsed] if isinstance(parsed, dict) else []


def _extract_piflags_from_bpmn(bpmn: str) -> List[Dict[str, Any]]:
    """uengine:Properties의 JSON/레거시 textContent에서 PI Flag comments를 추출한다."""
    try:
        root = SafeET.fromstring(bpmn)
    except Exception:
        return []

    flags: List[Dict[str, Any]] = []
    for properties in root.iter():
        if _xml_local_name(properties.tag).lower() != "properties":
            continue

        json_sources: List[Any] = [properties.attrib.get("json"), properties.text]
        json_sources.extend(
            child.text
            for child in list(properties)
            if _xml_local_name(child.tag).lower() == "json"
        )
        for raw in json_sources:
            for payload in _parse_property_json(raw):
                comments = payload.get("comments")
                if not isinstance(comments, list):
                    continue
                flags.extend(comment for comment in comments if isinstance(comment, dict))
    return flags


def _get_process_piflags(
    process_uuid: UUID, *, include_deleted: bool = False
) -> Optional[Dict[str, Any]]:
    row = _fetch_proc_def_row_by_uuid(process_uuid)
    if row is None or (not include_deleted and row.get("isdeleted")):
        return None

    bpmn = row.get("bpmn")
    flags = _extract_piflags_from_bpmn(bpmn) if isinstance(bpmn, str) else []
    return {
        "uuid": str(process_uuid),
        "flags": flags,
        "total": len(flags),
    }


def _build_definition(
    base: Optional[Dict[str, Any]],
    *,
    description: Optional[str],
    owner: Optional[str],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    definition = copy.deepcopy(base) if isinstance(base, dict) else {}
    if extra:
        definition.update(extra)
    if description is not None:
        definition["description"] = description
    if owner is not None:
        definition["owner"] = owner
    return definition


def _create_process(payload: ProcessCreate) -> Dict[str, Any]:
    proc_id = _norm_id(payload.id) if payload.id else f"proc-{uuid.uuid4().hex[:12]}"

    existing = _fetch_proc_def_row(proc_id)
    if existing is not None and not existing.get("isdeleted"):
        raise HTTPException(
            status_code=409, detail=f"Process with id '{proc_id}' already exists"
        )

    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")

    definition = _build_definition(
        payload.definition,
        description=payload.description,
        owner=payload.owner,
        extra=None,
    )

    # 분류(proc_map) 변경을 먼저 메모리에서 계산/검증한다 → 422 시 어떤 쓰기도 발생하지 않음.
    mega_c = _clean_str(payload.mega)
    major_c = _clean_str(payload.major)
    domain_c = _clean_str(payload.domain)
    classify = any((mega_c, major_c, domain_c))
    proc_map: Optional[Dict[str, Any]] = None
    if classify:
        proc_map = _load_proc_map()
        _apply_classification(
            proc_map, proc_id,
            name=name, mega=mega_c, major=major_c, domain=domain_c,
            classification_requested=True,
        )

    now = _now_iso()
    if existing is None:
        _insert_proc_def({
            "id": proc_id,
            "name": name,
            "tenant_id": _tenant_id(),
            "isdeleted": False,
            "saved_at": now,
            "definition": definition,
        })
    else:
        # 소프트 삭제됐던 id 재사용: 새 값으로 덮어쓰며 되살린다.
        _update_proc_def_row(proc_id, {
            "name": name,
            "isdeleted": False,
            "saved_at": now,
            "definition": definition,
        })

    if classify and proc_map is not None:
        _save_proc_map(proc_map)

    result = _get_process(proc_id)
    if result is None:  # 방어적: 방금 만든 행은 조회되어야 함
        raise HTTPException(status_code=500, detail="Failed to load created process")
    return result


def _update_process(proc_id: str, payload: ProcessUpdate) -> Optional[Dict[str, Any]]:
    row = _fetch_proc_def_row(proc_id)
    if row is None or row.get("isdeleted"):
        return None

    fields = payload.model_dump(exclude_unset=True)

    # id 는 불변. 본문에 경로와 다른 id 가 오면 422(조용한 무시 방지).
    if "id" in fields and payload.id is not None and _norm_id(payload.id) != _norm_id(proc_id):
        raise HTTPException(status_code=422, detail="process id is immutable")

    patch: Dict[str, Any] = {}
    new_name: Optional[str] = None
    if "name" in fields and payload.name is not None:
        new_name = payload.name.strip()
        if not new_name:
            raise HTTPException(status_code=422, detail="name must not be empty")
        patch["name"] = new_name

    touches_definition = any(k in fields for k in ("description", "owner", "definition"))
    if touches_definition:
        patch["definition"] = _build_definition(
            _as_dict(row.get("definition")),
            description=payload.description if "description" in fields else None,
            owner=payload.owner if "owner" in fields else None,
            extra=payload.definition if "definition" in fields else None,
        )

    # 계층(분류) 변경을 먼저 메모리에서 계산/검증한다 → 422 시 proc_def 쓰기 전에 중단.
    # mega/major/domain 중 하나라도 오면 재배치, 이름만 바뀌고 이미 분류돼 있으면 sub name 동기화.
    mega_c = _clean_str(payload.mega) if "mega" in fields else None
    major_c = _clean_str(payload.major) if "major" in fields else None
    domain_c = _clean_str(payload.domain) if "domain" in fields else None
    need_classify = any((mega_c, major_c, domain_c))

    proc_map = _load_proc_map()
    currently_classified = _norm_id(proc_id) in _build_proc_map_index(proc_map)
    do_map = need_classify or (new_name is not None and currently_classified)
    if do_map:
        # sub 이름은 항상 proc_def 의 권위 있는 이름을 사용한다.
        authoritative_name = new_name if new_name is not None else row.get("name")
        _apply_classification(
            proc_map, proc_id,
            name=authoritative_name,
            mega=mega_c, major=major_c, domain=domain_c,
            classification_requested=need_classify,
        )

    # 검증을 모두 통과한 뒤에 실제 쓰기를 수행한다.
    now = _now_iso()
    if patch:
        patch["saved_at"] = now
        _update_proc_def_row(proc_id, patch)
    elif do_map:
        # 분류만 바뀐 경우에도 proc_def 의 최종수정시각을 갱신(다운스트림 정렬/필터 일관성)
        _update_proc_def_row(proc_id, {"saved_at": now})
    if do_map:
        _save_proc_map(proc_map)

    return _get_process(proc_id)


# --------------------------------------------------------------------------- #
# 태스크(액티비티) 속성값 수정
# --------------------------------------------------------------------------- #
# 태스크(액티비티)가 담기는 definition 하위 컬렉션(탐색 우선순위 순).
_ACTIVITY_COLLECTIONS = ("activities", "subProcesses")


def _norm_activity_id(value: Any) -> str:
    # 액티비티 id 는 대소문자를 구분한다(proc_def id 와 달리 소문자화하지 않음).
    return str(value or "").strip()


def _find_activity_in_definition(
    definition: Dict[str, Any], activity_id: str
) -> Tuple[Optional[str], Optional[int], Optional[Dict[str, Any]]]:
    """definition 의 activities / subProcesses 에서 id 가 일치하는 항목을 찾는다.

    (컬렉션명, 인덱스, 항목) 을 반환하고, 없으면 (None, None, None) 을 반환한다.
    """
    target = _norm_activity_id(activity_id)
    if not target:
        return None, None, None
    for collection in _ACTIVITY_COLLECTIONS:
        items = definition.get(collection)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if isinstance(item, dict) and _norm_activity_id(item.get("id")) == target:
                return collection, index, item
    return None, None, None


def _update_activity(
    proc_id: str, activity_id: str, fields: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """프로세스 definition 내 단일 태스크(액티비티)의 속성값을 부분 병합한다.

    - 프로세스가 없거나 삭제됨 → None(라우트에서 404).
    - 액티비티를 찾지 못함 → 404.
    - 본문 id 가 경로 activity_id 와 다르면 422(조용한 무시 방지).
    - fields 에 제공된 키만 덮어쓰고 나머지 속성/컬렉션/다른 액티비티는 보존한다.
    """
    row = _fetch_proc_def_row(proc_id)
    if row is None or row.get("isdeleted"):
        return None

    definition = _as_dict(row.get("definition"))
    collection, index, activity = _find_activity_in_definition(definition, activity_id)
    if activity is None:
        raise HTTPException(
            status_code=404,
            detail=f"Activity '{activity_id}' not found in process '{proc_id}'",
        )

    patch = dict(fields)
    # id 는 불변. 경로와 다른 id 가 오면 422, 같은 id 를 함께 보내는 것은 허용(무시).
    if "id" in patch:
        if patch["id"] is not None and _norm_activity_id(patch["id"]) != _norm_activity_id(activity_id):
            raise HTTPException(status_code=422, detail="activity id is immutable")
        patch.pop("id")

    if "name" in patch:
        if not isinstance(patch["name"], str) or not patch["name"].strip():
            raise HTTPException(
                status_code=422, detail="activity name must not be empty"
            )
        patch["name"] = patch["name"].strip()

    updated_activity = {**activity, **patch}
    if patch:  # 실제 변경이 있을 때만 쓰기(빈 본문은 멱등 no-op)
        definition[collection][index] = updated_activity
        _update_proc_def_row(
            proc_id, {"definition": definition, "saved_at": _now_iso()}
        )

    return {
        "activity_id": _norm_activity_id(activity_id),
        "collection": collection,
        "activity": updated_activity,
    }


def _delete_process(proc_id: str) -> bool:
    row = _fetch_proc_def_row(proc_id)
    if row is None or row.get("isdeleted"):
        return False
    _soft_delete_proc_def(proc_id)
    proc_map = _load_proc_map()
    _remove_from_proc_map(proc_map, proc_id)
    _save_proc_map(proc_map)
    return True


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def add_routes_to_app(app: FastAPI):

    @app.get("/api/v1/processes")
    async def list_processes(
        response: Response,
        search: Optional[str] = Query(None, description="이름/설명/담당자 등 LIKE 검색"),
        domain: Optional[str] = Query(None, description="도메인 필터(정확히 일치)"),
        mega: Optional[str] = Query(None, description="메가 프로세스 필터(정확히 일치)"),
        major: Optional[str] = Query(None, description="메이저 프로세스 필터(정확히 일치)"),
        owner: Optional[str] = Query(None, description="담당자 필터(정확히 일치)"),
        page: int = Query(1, ge=1, description="페이지 번호(1-base)"),
        size: Optional[int] = Query(
            None, ge=1, le=_MAX_PAGE_SIZE, description="페이지 크기(미지정 시 전체)"
        ),
        include_deleted: bool = Query(False, description="삭제된 항목 포함 여부"),
    ):
        """프로세스 목록 조회. 검색/필터/페이징은 복합 적용 가능."""
        _prevent_cache(response)
        try:
            return _list_processes(
                search=search,
                domain=domain,
                mega=mega,
                major=major,
                owner=owner,
                page=page,
                size=size,
                include_deleted=include_deleted,
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 방어적
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/v1/processes", status_code=201)
    async def create_process(payload: ProcessCreate, response: Response):
        """프로세스 신규 생성."""
        _prevent_cache(response)
        try:
            return _create_process(payload)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 방어적
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/v1/processes/{uuid}")
    async def get_process(
        uuid: UUID,
        response: Response,
        include_deleted: bool = Query(False, description="삭제된 항목 포함 여부"),
    ):
        """프로세스 단건 상세 조회(모든 속성값 + definition 포함)."""
        _prevent_cache(response)
        try:
            result = _get_process_by_uuid(uuid, include_deleted=include_deleted)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 방어적
            raise HTTPException(status_code=500, detail=str(exc))
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Process '{uuid}' not found"
            )
        return result

    @app.get("/api/v1/processes/{uuid}/bpmn", include_in_schema=False)
    @app.get("/api/v1/processes/{uuid}/asis-bpmn")
    async def get_process_bpmn(
        uuid: UUID,
        include_deleted: bool = Query(False, description="삭제된 항목 포함 여부"),
    ):
        """저장된 As-Is BPMN XML을 원문 그대로 반환한다."""
        try:
            bpmn = _get_process_bpmn(uuid, include_deleted=include_deleted)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 방어적
            raise HTTPException(status_code=500, detail=str(exc))
        if bpmn is None:
            raise HTTPException(
                status_code=404, detail=f"BPMN XML for process '{uuid}' not found"
            )
        return Response(
            content=bpmn,
            media_type="application/xml",
            headers=_NO_STORE_HEADERS,
        )

    @app.get("/api/v1/processes/{uuid}/tobe-bpmn")
    async def get_process_tobe_bpmn(
        uuid: UUID,
        include_deleted: bool = Query(False, description="삭제된 항목 포함 여부"),
    ):
        """저장된 To-Be BPMN XML을 원문 그대로 반환한다."""
        try:
            tobe = _get_process_tobe_bpmn(
                uuid, include_deleted=include_deleted
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 방어적
            raise HTTPException(status_code=500, detail=str(exc))
        if tobe is None:
            raise HTTPException(
                status_code=404,
                detail=f"To-Be BPMN XML for process '{uuid}' not found",
            )
        return Response(
            content=tobe,
            media_type="application/xml",
            headers=_NO_STORE_HEADERS,
        )

    @app.get("/api/v1/processes/{uuid}/piflag", include_in_schema=False)
    @app.get("/api/v1/processes/{uuid}/piflags")
    async def get_process_piflags(
        uuid: UUID,
        response: Response,
        include_deleted: bool = Query(False, description="삭제된 항목 포함 여부"),
    ):
        """BPMN XML에 등록된 PI 플래그(comments)를 반환한다."""
        _prevent_cache(response)
        try:
            result = _get_process_piflags(
                uuid, include_deleted=include_deleted
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 방어적
            raise HTTPException(status_code=500, detail=str(exc))
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Process '{uuid}' not found"
            )
        return result

    @app.api_route("/api/v1/processes/{uuid}", methods=["PUT", "PATCH"])
    async def update_process(uuid: UUID, payload: ProcessUpdate = Body(...)):
        """프로세스 정보 수정(제공된 속성만 병합). PUT/PATCH 동일 동작."""
        try:
            proc_id = _resolve_active_proc_id_by_uuid(uuid)
            result = _update_process(proc_id, payload) if proc_id else None
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 방어적
            raise HTTPException(status_code=500, detail=str(exc))
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Process '{uuid}' not found"
            )
        return result

    @app.api_route(
        "/api/v1/processes/{uuid}/activities/{activity_id}",
        methods=["PUT", "PATCH"],
    )
    async def update_activity(
        uuid: UUID,
        activity_id: str,
        payload: ActivityUpdate = Body(...),
    ):
        """프로세스 내 단일 태스크(액티비티)의 속성값 수정(제공된 속성만 병합).

        definition.activities 에서 먼저 찾고, 없으면 subProcesses 에서 찾는다.
        PUT/PATCH 동일 동작(부분 병합). 응답에는 갱신된 액티비티 전체가 포함된다.
        """
        try:
            proc_id = _resolve_active_proc_id_by_uuid(uuid)
            result = (
                _update_activity(
                    proc_id, activity_id, payload.model_dump(exclude_unset=True)
                )
                if proc_id
                else None
            )
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 방어적
            raise HTTPException(status_code=500, detail=str(exc))
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Process '{uuid}' not found"
            )
        result["uuid"] = str(uuid)
        return result

    @app.delete("/api/v1/processes/{uuid}")
    async def delete_process(uuid: UUID):
        """프로세스 소프트 삭제(isdeleted=true) + proc_map 분류 제거."""
        try:
            proc_id = _resolve_active_proc_id_by_uuid(uuid)
            deleted = _delete_process(proc_id) if proc_id else False
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - 방어적
            raise HTTPException(status_code=500, detail=str(exc))
        if not deleted:
            raise HTTPException(
                status_code=404, detail=f"Process '{uuid}' not found"
            )
        return {"success": True, "uuid": str(uuid)}

