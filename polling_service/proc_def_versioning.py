from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union


def _version_as_float(v: Any) -> float:
    try:
        return float(str(v))
    except Exception:
        return 0.0


def fetch_proc_def_row(supabase, def_id: str, tenant_id: str) -> Optional[dict]:
    """
    proc_def 테이블에서 row 전체를 조회합니다. (definition/bpmn/prod_version 등 포함)
    - supabase: supabase python client
    """
    response = (
        supabase.table("proc_def")
        .select("*")
        .eq("id", def_id.lower())
        .eq("tenant_id", tenant_id)
        .execute()
    )
    if response.data and len(response.data) > 0:
        return response.data[0]
    return None


def fetch_process_definition_versions_by_tag(
    supabase,
    def_id: str,
    version_tag: str,
    tenant_id: str,
) -> List[dict]:
    """
    proc_def_version에서 proc_def_id + version_tag 기준으로 버전 목록을 조회합니다.
    """
    response = (
        supabase.table("proc_def_version")
        .select("*")
        .eq("proc_def_id", def_id.lower())
        .eq("tenant_id", tenant_id)
        .eq("version_tag", version_tag)
        .execute()
    )
    return response.data or []


def fetch_latest_process_definition_version_by_tag(
    supabase,
    def_id: str,
    version_tag: str,
    tenant_id: str,
) -> Optional[dict]:
    """
    version_tag(major/minor) 기준으로 최신 버전 row 1건을 선택합니다.
    - DB 정렬이 문자열 정렬일 수 있으므로, 조회 후 Python에서 float 변환 정렬로 보정합니다.
    """
    rows = fetch_process_definition_versions_by_tag(supabase, def_id, version_tag, tenant_id)
    if not rows:
        return None
    rows.sort(key=lambda r: _version_as_float(r.get("version")), reverse=True)
    return rows[0]


def fetch_process_definition_by_version_ts_style(
    *,
    supabase,
    def_id: str,
    tenant_id: str,
    version_tag: Optional[str] = None,
    version: Optional[Union[str, int]] = None,
    arcv_id: Optional[str] = None,
    fetch_arcv_rows: Optional[Callable[[str], List[dict]]] = None,
) -> Any:
    """
    TS 방식(prod_version 우선 → 최신 major → 최신 minor → proc_def)으로 실행 정의를 선택합니다.

    호환/명시 우선순위:
    - arcv_id가 주어지면: 해당 proc_def_arcv 버전 우선
    - version_tag + version 이 주어지면: proc_def_version(tag/version) 정확 매칭 우선

    기본(명시 버전이 없을 때):
    1) proc_def.prod_version 우선 (arcv_id로 저장된 경우가 있어 proc_def_arcv 먼저 시도)
    2) 최신 major (proc_def_version.version_tag='major')
    3) 최신 minor (proc_def_version.version_tag='minor')
    4) proc_def.definition
    """
    if not def_id:
        return None

    tag = (version_tag or "").lower()

    proc_def_row = fetch_proc_def_row(supabase, def_id, tenant_id)
    proc_def_definition = (proc_def_row or {}).get("definition")

    def inject_meta(defn: Any, v: Any, vt: Any) -> Any:
        if isinstance(defn, dict):
            if v is not None:
                defn.setdefault("version", v)
            if vt is not None:
                defn.setdefault("version_tag", vt)
        return defn

    # 0) arcv_id 우선
    if arcv_id and fetch_arcv_rows is not None:
        try:
            rows = fetch_arcv_rows(str(arcv_id)) or []
            if rows:
                row = rows[0]
                definition = row.get("definition", None)
                return inject_meta(definition, row.get("version"), row.get("version_tag") or "major")
        except Exception:
            pass

    # 0.5) 명시 버전 우선 (major/minor)
    if tag in ("major", "minor") and version is not None:
        try:
            resp = (
                supabase.table("proc_def_version")
                .select("*")
                .eq("proc_def_id", def_id.lower())
                .eq("tenant_id", tenant_id)
                .eq("version_tag", tag)
                .eq("version", str(version))
                .execute()
            )
            if resp.data and len(resp.data) > 0:
                row = resp.data[0]
                definition = row.get("definition", None)
                return inject_meta(definition, row.get("version"), row.get("version_tag") or tag)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 명시적 버전 핀(arcv_id / version_tag+version)이 없을 때의 기본 동작
    #
    # [중요] 과거에는 prod_version → 최신 major → 최신 minor → proc_def.definition
    # 순으로 선택했는데, 이 경우 "현재 작업본(proc_def.definition)보다 오래된
    # 발행 버전"이 우선 선택되어, 이미 삭제된 옛 액티비티가 예정업무(todolist)에
    # 되살아나는 회귀가 발생했다. (커밋 686c942 "fix minor major version")
    #
    # 명시적 핀이 없다면 실행/조회 대상은 '현재 최신 작업본'이어야 하므로
    # proc_def.definition 을 최우선으로 반환한다. (686c942 이전 동작 복원)
    # 특정 발행 버전으로 고정 실행하려면 호출부에서 arcv_id 또는
    # version_tag+version 을 명시적으로 넘겨야 한다.
    # ------------------------------------------------------------------
    if proc_def_definition:
        return inject_meta(proc_def_definition, version, version_tag or "minor")

    # 안전망: proc_def.definition 이 비어 있는 예외적 상황에서만
    # 발행 버전(prod_version → 최신 major → 최신 minor)으로 폴백한다.
    prod_version = None
    if isinstance(proc_def_row, dict):
        prod_version = proc_def_row.get("prod_version") or proc_def_row.get("prodVersion")

    if prod_version:
        # (a) prod_version을 arcv_id로 간주하고 proc_def_arcv 먼저 시도
        if fetch_arcv_rows is not None:
            try:
                rows = fetch_arcv_rows(str(prod_version)) or []
                if rows:
                    row = rows[0]
                    definition = row.get("definition", None)
                    return inject_meta(definition, row.get("version"), row.get("version_tag") or "major")
            except Exception:
                pass

        # (b) prod_version이 proc_def_version.version일 수도 있어 매칭 시도 (tag는 제한하지 않음)
        try:
            resp = (
                supabase.table("proc_def_version")
                .select("*")
                .eq("proc_def_id", def_id.lower())
                .eq("tenant_id", tenant_id)
                .eq("version", str(prod_version))
                .execute()
            )
            if resp.data and len(resp.data) > 0:
                row = resp.data[0]
                definition = row.get("definition", None)
                return inject_meta(definition, row.get("version"), row.get("version_tag") or "major")
        except Exception:
            pass

    latest_major = fetch_latest_process_definition_version_by_tag(supabase, def_id, "major", tenant_id)
    if latest_major:
        return inject_meta(
            latest_major.get("definition", None),
            latest_major.get("version"),
            latest_major.get("version_tag") or "major",
        )

    latest_minor = fetch_latest_process_definition_version_by_tag(supabase, def_id, "minor", tenant_id)
    if latest_minor:
        return inject_meta(
            latest_minor.get("definition", None),
            latest_minor.get("version"),
            latest_minor.get("version_tag") or "minor",
        )

    return inject_meta(proc_def_definition, version, version_tag or "minor")

