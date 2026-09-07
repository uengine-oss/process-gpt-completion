"""
Dashboard API — View 1 / 2 / 3 데이터 엔드포인트
Grafana 대시보드 SQL을 그대로 사용하여 실 데이터 제공
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import psycopg2
from fastapi import FastAPI, Query
from psycopg2.extras import RealDictCursor

try:
    from defusedxml import ElementTree as SafeET
except ImportError:  # pragma: no cover - requirements include defusedxml.
    from xml.etree import ElementTree as SafeET

from database import db_config_var, subdomain_var


def _get_conn():
    return psycopg2.connect(**db_config_var.get())


def _get_tenant_id() -> str:
    return subdomain_var.get()


def _domain_filter(domains: Optional[List[str]]) -> str:
    """domains 파라미터를 SQL IN 절로 변환"""
    if not domains:
        return "1=1"
    quoted = ", ".join(f"'{d}'" for d in domains)
    return f"COALESCE(dm.domain, '미분류') IN ({quoted})"


def _domain_filter_col(domains: Optional[List[str]], col: str = "domain") -> str:
    if not domains:
        return "1=1"
    quoted = ", ".join(f"'{d}'" for d in domains)
    return f"{col} IN ({quoted})"


# ─── raw_targets CTE (KPI 목표 파싱, Grafana view1과 동일) ──────────────
_RAW_TARGETS_CTE = """
raw_targets AS (
  SELECT
    c.tenant_id,
    COALESCE(NULLIF(elem->>'org_name', ''), NULLIF(elem->>'division', ''),
             NULLIF(elem->>'name', ''), NULLIF(elem->>'org_id', ''), '미지정') AS division,
    COALESCE(
      CASE
        WHEN btrim(COALESCE(elem->>'target_value', elem->>'target', elem->>'value', ''))
             ~ '^[0-9]+([.][0-9]+)?$'
        THEN btrim(COALESCE(elem->>'target_value', elem->>'target', elem->>'value', ''))::numeric
        ELSE NULL
      END,
      CASE
        WHEN jsonb_typeof(elem->'process_ids') = 'array'
        THEN jsonb_array_length(elem->'process_ids')::numeric
        ELSE 0::numeric
      END
    ) AS target_count,
    pid.process_id
  FROM configuration c
  CROSS JOIN LATERAL jsonb_array_elements(
    CASE
      WHEN jsonb_typeof(c.value) = 'array' THEN c.value
      WHEN jsonb_typeof(c.value->'items') = 'array' THEN c.value->'items'
      WHEN jsonb_typeof(c.value->'targets') = 'array' THEN c.value->'targets'
      ELSE '[]'::jsonb
    END
  ) AS elem
  LEFT JOIN LATERAL jsonb_array_elements_text(
    CASE
      WHEN jsonb_typeof(elem->'process_ids') = 'array' THEN elem->'process_ids'
      ELSE '[]'::jsonb
    END
  ) AS pid(process_id) ON TRUE
  WHERE c.key = 'kpi_targets'
    AND c.tenant_id = %(tenant_id)s
)
"""


ANNUAL_TARGET_COUNT = 70
MAX_CALL_ACTIVITY_DEPTH = 5
MULTI_VALUE_SPLIT_RE = re.compile(r"\s*(?:,|，|/|;|\||\n|ㆍ|·|&|＆)\s*")
DECISION_KEYWORDS = ("판단", "분석", "검토", "승인", "확인", "심사", "검증")
AUTO_TASK_TYPES = {"service", "send", "receive", "script", "businessrule"}
BPMN_TASK_TAGS = {
    "task",
    "userTask",
    "manualTask",
    "serviceTask",
    "scriptTask",
    "businessRuleTask",
    "sendTask",
    "receiveTask",
}
BPMN_TASK_TYPE_MAP = {
    "userTask": "user",
    "manualTask": "manual",
    "serviceTask": "service",
    "scriptTask": "script",
    "businessRuleTask": "businessrule",
    "sendTask": "send",
    "receiveTask": "receive",
    "task": "other",
}

# TM Forum 기반 Task 자동화 점수 매핑 (5점 만점 환산)
_AUTOMATION_SCORE_MAP: Dict[str, int] = {
    "manual": 1,
    "send": 1,
    "receive": 1,
    "user": 2,
    "service": 3,
    "script": 3,
    "businessrule": 3,
    "other": 1,
}
_AUTOMATION_MAX_SCORE = 3  # SCORE_MAP 최대값


def _calc_automation_score(task_counts: Dict[str, int]) -> float:
    """TM Forum 기준 가중 평균 자동화 점수 (5점 만점 환산)"""
    total_score = sum(
        count * _AUTOMATION_SCORE_MAP.get(task_type, 1)
        for task_type, count in task_counts.items()
    )
    total_tasks = sum(task_counts.values())
    if total_tasks == 0:
        return 0.0
    avg_level = total_score / total_tasks
    return round(avg_level / _AUTOMATION_MAX_SCORE * 5, 1)


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_proc_id(value: Any) -> str:
    text = _text(value)
    if text.lower().endswith(".bpmn"):
        text = text[:-5]
    return text


def _extract_extension_props(item: Dict[str, Any]) -> Dict[str, Any]:
    candidates: List[Any] = []
    ext = item.get("extensionElements") or item.get("bpmn:extensionElements")
    if isinstance(ext, dict):
        candidates.extend(_as_list(ext.get("values")))
        candidates.extend(_as_list(ext.get("$children")))
    candidates.extend(_as_list(item.get("extensionValues")))

    props: Dict[str, Any] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        raw_json = candidate.get("json") or candidate.get("body")
        if raw_json:
            parsed = _as_dict(raw_json)
            if parsed:
                props.update(parsed)
        nested = candidate.get("properties")
        if isinstance(nested, dict):
            props.update(nested)
    for key in ("uengineProps", "uengineProperties", "properties"):
        value = item.get(key)
        if isinstance(value, dict):
            props.update(value)
        elif isinstance(value, str) and value.strip():
            # uengine 은 properties 를 직렬화된 JSON 문자열로 저장하기도 함
            # (relatedProjects 등 일부 속성은 이 블롭 안에만 존재) → 파싱해서 병합.
            # 빈 값은 건너뛰어 top-level(systems 등) 값을 덮어쓰지 않음.
            for prop_key, prop_value in _as_dict(value).items():
                if prop_value not in (None, "", [], {}):
                    props[prop_key] = prop_value
    return props


def _activity_props(activity: Dict[str, Any]) -> Dict[str, Any]:
    props = _extract_extension_props(activity)
    merged = dict(activity)
    merged.update(props)
    return merged


def _definition_array(definition: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    value = definition.get(key)
    if value is None:
        value = definition.get(f"bpmn:{key}")
    return [item for item in _as_list(value) if isinstance(item, dict)]


def _activities(definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = _definition_array(definition, "activities")
    items.extend(_definition_array(definition, "tasks"))
    return items


def _gateways(definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _definition_array(definition, "gateways")


def _sequences(definition: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _definition_array(definition, "sequences")


def _classify_task_type(activity: Dict[str, Any]) -> str:
    raw_type = _text(activity.get("type") or activity.get("$type") or activity.get("_type")).lower()
    if "manual" in raw_type:
        return "manual"
    if "service" in raw_type:
        return "service"
    if "script" in raw_type:
        return "script"
    if "send" in raw_type:
        return "send"
    if "receive" in raw_type:
        return "receive"
    if "businessrule" in raw_type or "business_rule" in raw_type or "business rule" in raw_type:
        return "businessrule"
    if "user" in raw_type:
        return "user"
    return "other"


def _is_call_activity(activity: Dict[str, Any]) -> bool:
    raw_type = _text(activity.get("type") or activity.get("$type") or activity.get("_type")).lower()
    return "callactivity" in raw_type or "call_activity" in raw_type or "call activity" in raw_type


def _called_process_id(activity: Dict[str, Any]) -> str:
    props = _activity_props(activity)
    for key in (
        "definitionId",
        "calledElement",
        "called_element",
        "processDefinitionId",
        "proc_def_id",
        "process",
        "subprocess_id",
        "subProcessId",
    ):
        value = _normalize_proc_id(props.get(key))
        if value:
            return value
    return ""


def _xml_local_name(tag: Any) -> str:
    text = str(tag)
    if text.startswith("{"):
        return text.split("}", 1)[1]
    if ":" in text:
        return text.split(":", 1)[1]
    return text


BPMN_MODEL_NS = "http://www.omg.org/spec/BPMN/20100524/MODEL"


def _xml_namespace(tag: Any) -> str:
    text = str(tag)
    if text.startswith("{"):
        return text[1:].split("}", 1)[0]
    return ""


def _iter_bpmn_model_elements(root: Any):
    """BPMN 모델 요소만 순회한다.

    extensionElements 하위(zeebe/camunda/uengine 확장 메타데이터)와 BPMN 외
    네임스페이스 요소를 제외해, Camunda export 의 <zeebe:userTask/> 마커가
    id/이름 없는 유령 태스크로 집계되는 것을 막는다.
    """
    stack = [root]
    while stack:
        elem = stack.pop()
        if _xml_namespace(elem.tag) in ("", BPMN_MODEL_NS):
            yield elem
        for child in reversed(list(elem)):
            if _xml_local_name(child.tag) == "extensionElements":
                continue
            stack.append(child)


def _parse_bpmn_xml(bpmn: Any) -> Optional[Any]:
    if not isinstance(bpmn, str) or not bpmn.strip():
        return None
    try:
        return SafeET.fromstring(bpmn.encode("utf-8"))
    except Exception:
        return None


def _parse_xml_json_props(elem: Any) -> Dict[str, Any]:
    candidates: List[str] = []
    for candidate in elem.iter():
        if _xml_local_name(candidate.tag) == "json":
            text = _text(candidate.text)
            if text:
                candidates.append(text)
        json_attr = _text(candidate.attrib.get("json"))
        if json_attr:
            candidates.append(json_attr)
    props = _text(elem.attrib.get("properties"))
    if props:
        candidates.append(props)

    fallback: Dict[str, Any] = {}
    for raw in candidates:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            if parsed:
                return parsed
            fallback = parsed
    return fallback


def _bpmn_element_props(elem: Any) -> Dict[str, Any]:
    props = dict(elem.attrib)
    props.update(_parse_xml_json_props(elem))
    local_name = _xml_local_name(elem.tag)
    props["id"] = props.get("id") or elem.attrib.get("id")
    props["name"] = props.get("name") or elem.attrib.get("name")
    props["bpmn_tag"] = local_name
    props.setdefault("type", local_name)
    return props


def _bpmn_lane_roles(root: Any) -> Dict[str, List[str]]:
    lane_roles_by_node: Dict[str, List[str]] = {}
    for lane in root.iter():
        if _xml_local_name(lane.tag) != "lane":
            continue
        props = dict(lane.attrib)
        props.update(_parse_xml_json_props(lane))
        roles = _split_multi_value(
            props.get("role")
            or props.get("roles")
            or props.get("laneOrganization")
            or props.get("laneAssignee")
            or props.get("assignee")
            or props.get("name")
        )
        if not roles:
            continue
        for ref_elem in list(lane):
            if _xml_local_name(ref_elem.tag) != "flowNodeRef":
                continue
            node_id = _text(ref_elem.text)
            if node_id:
                lane_roles_by_node[node_id] = roles
    return lane_roles_by_node


def _classify_bpmn_task_type(local_name: str, props: Dict[str, Any]) -> str:
    if local_name != "task":
        return BPMN_TASK_TYPE_MAP.get(local_name, "other")
    return _classify_task_type(props)


def _collect_bpmn_tasks(
    proc_row: Dict[str, Any],
    proc_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    depth: int = 0,
    stack: Optional[Set[str]] = None,
    root_row: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    stack = set(stack or set())
    root_row = root_row or proc_row
    root = _parse_bpmn_xml(proc_row.get("bpmn"))
    if root is None:
        return []

    proc_id = _normalize_proc_id(proc_row.get("id"))
    if proc_id:
        stack.add(proc_id.lower())
    lane_roles_by_node = _bpmn_lane_roles(root)
    records: List[Dict[str, Any]] = []
    for elem in _iter_bpmn_model_elements(root):
        local_name = _xml_local_name(elem.tag)
        if local_name == "callActivity" and proc_lookup and depth < MAX_CALL_ACTIVITY_DEPTH:
            props = _bpmn_element_props(elem)
            child_id = _called_process_id(props)
            child_row = _lookup_proc(proc_lookup, child_id) if child_id else None
            child_key = _normalize_proc_id(child_id).lower()
            if child_row and child_key and child_key not in stack and _is_call_activity_sub_module(child_row):
                records.extend(_collect_bpmn_tasks(child_row, proc_lookup, depth + 1, stack, root_row))
            continue

        if local_name not in BPMN_TASK_TAGS:
            continue
        props = _bpmn_element_props(elem)
        activity_id = props.get("id") or props.get("activity_id")
        roles = _split_multi_value(props.get("role") or props.get("roles") or props.get("department"))
        if not roles:
            roles = lane_roles_by_node.get(_text(activity_id), [])
        records.append(
            {
                "root_proc_id": _normalize_proc_id(root_row.get("id")),
                "root_proc_name": root_row.get("name") or root_row.get("id"),
                "root_domain": root_row.get("domain") or "미분류",
                "source_proc_id": proc_id,
                "activity_id": activity_id,
                "name": props.get("name") or props.get("label") or props.get("id") or "이름 없음",
                "type": props.get("type") or local_name,
                "classified_type": _classify_bpmn_task_type(local_name, props),
                "roles": roles,
                "systems": _system_items(props),
                "projects": _project_items(props),
            }
        )
    return records


def _split_multi_value(value: Any) -> List[str]:
    pieces: List[str] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            text = _text(item.get("name") or item.get("role_name") or item.get("department") or item.get("team"))
            if text:
                pieces.append(text)
            continue
        text = _text(item)
        if not text:
            continue
        pieces.extend([part for part in MULTI_VALUE_SPLIT_RE.split(text) if part])
    seen: Set[str] = set()
    result: List[str] = []
    for piece in pieces:
        normalized = piece.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _system_items(activity: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    props = _activity_props(activity)
    raw_systems = props.get("systems") or props.get("system") or props.get("system_name")
    items: List[Dict[str, Optional[str]]] = []

    for item in _as_list(raw_systems):
        if isinstance(item, dict):
            system_id = _text(item.get("id") or item.get("system_id")) or None
            name = _text(item.get("name") or item.get("system_name")) or None
        else:
            system_id = None
            name = _text(item) or None
        if system_id or name:
            items.append({"id": system_id, "name": name})

    if not items:
        tool = _text(props.get("tool"))
        if tool and not tool.lower().startswith("formhandler:") and not tool.lower().startswith("formhander:"):
            items.append({"id": None, "name": tool})

    seen: Set[Tuple[Optional[str], Optional[str]]] = set()
    deduped: List[Dict[str, Optional[str]]] = []
    for item in items:
        key = (item.get("id"), item.get("name"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _project_items(activity: Dict[str, Any]) -> List[Dict[str, Optional[str]]]:
    props = _activity_props(activity)
    raw_projects = props.get("relatedProjects") or props.get("projects") or props.get("related_projects")
    items: List[Dict[str, Optional[str]]] = []
    for item in _as_list(raw_projects):
        if isinstance(item, dict):
            project_id = _text(item.get("id") or item.get("project_id")) or None
            name = _text(item.get("name") or item.get("title") or item.get("project_name")) or project_id or None
        else:
            project_id = None
            name = _text(item) or None
        if project_id or name:
            items.append({"id": project_id, "name": name})

    seen: Set[Tuple[Optional[str], Optional[str]]] = set()
    deduped: List[Dict[str, Optional[str]]] = []
    for item in items:
        key = (item.get("id"), item.get("name"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _make_proc_lookup(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        proc_id = _normalize_proc_id(row.get("id"))
        if not proc_id:
            continue
        lookup[proc_id] = row
        lookup[proc_id.lower()] = row
    return lookup


def _lookup_proc(proc_lookup: Dict[str, Dict[str, Any]], proc_id: str) -> Optional[Dict[str, Any]]:
    normalized = _normalize_proc_id(proc_id)
    return proc_lookup.get(normalized) or proc_lookup.get(normalized.lower())


def _collect_activities(
    proc_row: Dict[str, Any],
    proc_lookup: Dict[str, Dict[str, Any]],
    depth: int = 0,
    stack: Optional[Set[str]] = None,
    root_row: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    stack = set(stack or set())
    root_row = root_row or proc_row
    proc_id = _normalize_proc_id(proc_row.get("id"))
    definition = _as_dict(proc_row.get("definition"))
    records: List[Dict[str, Any]] = []

    if proc_id:
        stack.add(proc_id.lower())

    def add_activity(activity: Dict[str, Any], source_proc: Dict[str, Any]):
        props = _activity_props(activity)
        record = {
            "root_proc_id": _normalize_proc_id(root_row.get("id")),
            "root_proc_name": root_row.get("name") or root_row.get("id"),
            "root_domain": root_row.get("domain") or "미분류",
            "source_proc_id": _normalize_proc_id(source_proc.get("id")),
            "activity_id": props.get("id") or props.get("activity_id"),
            "name": props.get("name") or props.get("label") or props.get("id") or "이름 없음",
            "type": props.get("type") or props.get("$type") or props.get("_type"),
            "classified_type": _classify_task_type(props),
            "roles": _split_multi_value(props.get("role") or props.get("roles") or props.get("department")),
            "systems": _system_items(props),
            "projects": _project_items(props),
        }
        records.append(record)

    for activity in _activities(definition):
        if not isinstance(activity, dict):
            continue
        props = _activity_props(activity)
        nested_inline = [
            child
            for key in ("activities", "tasks", "children")
            for child in _as_list(props.get(key))
            if isinstance(child, dict)
        ]
        if nested_inline:
            for child in nested_inline:
                add_activity(child, proc_row)
            continue

        if _is_call_activity(props) and depth < MAX_CALL_ACTIVITY_DEPTH:
            child_id = _called_process_id(props)
            child_row = _lookup_proc(proc_lookup, child_id) if child_id else None
            child_key = _normalize_proc_id(child_id).lower()
            if child_row and child_key and child_key not in stack:
                records.extend(_collect_activities(child_row, proc_lookup, depth + 1, stack, root_row))
                continue

        add_activity(props, proc_row)

    return records


def _structure_metrics(
    proc_row: Dict[str, Any],
    proc_lookup: Dict[str, Dict[str, Any]],
    depth: int = 0,
    stack: Optional[Set[str]] = None,
) -> Counter:
    stack = set(stack or set())
    proc_id = _normalize_proc_id(proc_row.get("id"))
    if proc_id:
        stack.add(proc_id.lower())
    definition = _as_dict(proc_row.get("definition"))
    activities = [_activity_props(activity) for activity in _activities(definition)]
    activity_by_id = {_text(activity.get("id") or activity.get("activity_id")): activity for activity in activities}
    sequences = _sequences(definition)
    gateways = _gateways(definition)
    metrics: Counter = Counter()

    outgoing_by_source: Counter = Counter()
    for seq in sequences:
        source = _text(seq.get("source") or seq.get("sourceRef"))
        target = _text(seq.get("target") or seq.get("targetRef"))
        if source:
            outgoing_by_source[source] += 1
        source_roles = _split_multi_value(activity_by_id.get(source, {}).get("role"))
        target_roles = _split_multi_value(activity_by_id.get(target, {}).get("role"))
        if source_roles and target_roles and set(source_roles) != set(target_roles):
            metrics["handoff"] += 1

    seen_pairs: Set[Tuple[str, str]] = set()
    sequence_pairs = {
        (_text(seq.get("source") or seq.get("sourceRef")), _text(seq.get("target") or seq.get("targetRef")))
        for seq in sequences
    }
    for source, target in sequence_pairs:
        if not source or not target or source == target:
            continue
        pair = tuple(sorted((source, target)))
        if pair in seen_pairs:
            continue
        if (target, source) in sequence_pairs:
            seen_pairs.add(pair)
            metrics["loop"] += 1

    for gateway in gateways:
        gateway_type = _text(gateway.get("type") or gateway.get("$type")).lower()
        gateway_id = _text(gateway.get("id") or gateway.get("gateway_id"))
        outgoing_count = outgoing_by_source.get(gateway_id, 0)
        if "exclusive" in gateway_type or "xor" in gateway_type:
            metrics["xor"] += max(outgoing_count, 1)

    if depth < MAX_CALL_ACTIVITY_DEPTH:
        for activity in activities:
            if not _is_call_activity(activity):
                continue
            child_id = _called_process_id(activity)
            child_row = _lookup_proc(proc_lookup, child_id) if child_id else None
            child_key = _normalize_proc_id(child_id).lower()
            if child_row and child_key and child_key not in stack:
                metrics.update(_structure_metrics(child_row, proc_lookup, depth + 1, stack))

    return metrics


def _fetch_system_rows(cur, tenant_id: str) -> List[Dict[str, Any]]:
    cur.execute("SELECT to_regclass('public.systems') AS table_name")
    exists = cur.fetchone()
    if not exists or not exists.get("table_name"):
        return []
    cur.execute(
        """
        SELECT id, name, system_type, category
        FROM systems
        WHERE tenant_id = %(tenant_id)s
          AND COALESCE(is_active, 1) <> 0
        ORDER BY name
        """,
        {"tenant_id": tenant_id},
    )
    return [dict(r) for r in cur.fetchall()]


def _fetch_proc_map(cur, tenant_id: str) -> Dict[str, Any]:
    cur.execute(
        """
        SELECT value
        FROM configuration
        WHERE tenant_id = %(tenant_id)s
          AND key = 'proc_map'
        LIMIT 1
        """,
        {"tenant_id": tenant_id},
    )
    row = cur.fetchone()
    if not row:
        return {}
    return _as_dict(row.get("value"))


def _proc_map_subprocess_index(proc_map: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for mega in proc_map.get("mega_proc_list") or []:
        if not isinstance(mega, dict):
            continue
        for major in mega.get("major_proc_list") or []:
            if not isinstance(major, dict):
                continue
            domain = _text(major.get("domain") or major.get("domain_id")) or "미분류"
            for sub in major.get("sub_proc_list") or []:
                if not isinstance(sub, dict):
                    continue
                proc_id = _normalize_proc_id(sub.get("proc_def_id") or sub.get("id"))
                if not proc_id:
                    continue
                key = proc_id.lower()
                if key in index:
                    continue
                index[key] = {
                    "proc_def_id": proc_id,
                    "domain": domain,
                    "mega_id": _text(mega.get("id")),
                    "mega_name": _text(mega.get("name")),
                    "major_id": _text(major.get("id")),
                    "major_name": _text(major.get("name")),
                }
    return index


def _is_call_activity_sub_module(proc_row: Optional[Dict[str, Any]]) -> bool:
    if not proc_row:
        return False

    if _text(proc_row.get("type")) == "call-activity-sub":
        return True

    raw_definition = proc_row.get("definition")
    if isinstance(raw_definition, dict):
        return _text(raw_definition.get("type")) == "call-activity-sub"

    if isinstance(raw_definition, str) and raw_definition.strip():
        try:
            parsed = json.loads(raw_definition)
        except json.JSONDecodeError:
            return "call-activity-sub" in raw_definition
        if isinstance(parsed, dict):
            return _text(parsed.get("type")) == "call-activity-sub"

    return False


# ═══════════════════════════════════════════════════════════════════════
# View 1: Executive Summary
# ═══════════════════════════════════════════════════════════════════════

async def get_executive_summary(domains: Optional[List[str]] = Query(None)):
    """전체 진행 현황 + 본부별 KPI + Pipeline Funnel + Weekly Velocity + Churn 통합"""
    tenant_id = _get_tenant_id()
    domain_cond = _domain_filter_col(domains)

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # ── 1) 전체 진행 현황 ──
            cur.execute(f"""
                WITH state_counts AS (
                  SELECT
                    COUNT(*) AS total_count,
                    COUNT(*) FILTER (WHERE current_status = 'published') AS published_count,
                    COUNT(*) FILTER (WHERE current_status = 'draft') AS draft_count,
                    COUNT(*) FILTER (WHERE current_status = 'in_review') AS review_count,
                    COUNT(*) FILTER (WHERE current_status IN ('public_feedback', 'final_edit')) AS approved_count
                  FROM vw_dashboard_proc_state
                  WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                )
                SELECT
                  total_count, published_count, draft_count, review_count, approved_count,
                  %(annual_target)s AS target_count,
                  ROUND(published_count::numeric / NULLIF(total_count, 0) * 100, 1) AS progress_pct,
                  ROUND(published_count::numeric / NULLIF(%(annual_target)s, 0) * 100, 1) AS target_pct
                FROM state_counts
            """, {"tenant_id": tenant_id, "annual_target": ANNUAL_TARGET_COUNT})
            overview = dict(cur.fetchone() or {})

            # ── 2) 본부별 KPI 달성 현황 ──
            cur.execute(f"""
                WITH {_RAW_TARGETS_CTE},
                division_stage AS (
                  SELECT
                    rt.division,
                    MAX(rt.target_count) AS target_count,
                    COUNT(DISTINCT rt.process_id) FILTER (WHERE ps.current_status = 'draft') AS draft_count,
                    COUNT(DISTINCT rt.process_id) FILTER (WHERE ps.current_status = 'in_review') AS review_count,
                    COUNT(DISTINCT rt.process_id) FILTER (WHERE ps.current_status IN ('public_feedback', 'final_edit')) AS approved_count,
                    COUNT(DISTINCT rt.process_id) FILTER (WHERE ps.current_status = 'published') AS published_count
                  FROM raw_targets rt
                  LEFT JOIN vw_dashboard_proc_state ps
                    ON ps.tenant_id = rt.tenant_id AND ps.proc_def_id = rt.process_id
                    AND {_domain_filter_col(domains, 'ps.domain')}
                  GROUP BY rt.division
                )
                SELECT
                  division, target_count, draft_count, review_count, approved_count, published_count,
                  ROUND(published_count::numeric / NULLIF(target_count, 0) * 100, 1) AS completion_pct,
                  CASE
                    WHEN published_count::numeric / NULLIF(target_count, 0) >= 0.70 THEN 'G'
                    WHEN published_count::numeric / NULLIF(target_count, 0) >= 0.40 THEN 'A'
                    ELSE 'R'
                  END AS rag
                FROM division_stage
                ORDER BY completion_pct DESC NULLS LAST, division
                LIMIT 10
            """, {"tenant_id": tenant_id})
            kpi_divisions = [dict(r) for r in cur.fetchall()]

            # ── 3) Pipeline Funnel ──
            cur.execute(f"""
                SELECT current_status AS stage, COUNT(*) AS cnt
                FROM vw_dashboard_proc_state
                WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                GROUP BY current_status
                ORDER BY CASE current_status
                  WHEN 'draft' THEN 1 WHEN 'in_review' THEN 2
                  WHEN 'public_feedback' THEN 3 WHEN 'final_edit' THEN 4
                  WHEN 'published' THEN 5 ELSE 6
                END
            """, {"tenant_id": tenant_id})
            funnel = [dict(r) for r in cur.fetchall()]

            # ── 4) Weekly Velocity ──
            cur.execute(f"""
                WITH weekly AS (
                  SELECT
                    DATE_TRUNC('week', pah.created_at) AS week_start,
                    TO_CHAR(DATE_TRUNC('week', pah.created_at), 'YYYY-"W"WW') AS week_label,
                    COUNT(DISTINCT pah.proc_def_id) FILTER (WHERE pah.to_state = 'published') AS published_count
                  FROM proc_def_approval_history pah
                  JOIN proc_def pd ON pd.id = pah.proc_def_id AND pd.tenant_id = pah.tenant_id
                  LEFT JOIN vw_dashboard_proc_domain dm ON dm.proc_def_id = pd.id AND dm.tenant_id = pd.tenant_id
                  WHERE pah.tenant_id = %(tenant_id)s AND pd.isdeleted = false
                    AND {_domain_filter(domains)}
                  GROUP BY DATE_TRUNC('week', pah.created_at)
                )
                SELECT
                  week_label,
                  published_count AS actual,
                  CEIL(%(annual_target)s::numeric / NULLIF((SELECT COUNT(*) FROM weekly), 0)) AS target
                FROM weekly
                ORDER BY week_start
            """, {"tenant_id": tenant_id, "annual_target": ANNUAL_TARGET_COUNT})
            velocity = [dict(r) for r in cur.fetchall()]

            # ── 5) 단계 역행(Churn) 현황 ──
            cur.execute(f"""
                WITH history AS (
                  SELECT
                    pah.proc_def_id,
                    pd.name AS proc_def_name,
                    COALESCE(dm.domain, '미분류') AS domain,
                    pah.from_state, pah.to_state, pah.created_at
                  FROM proc_def_approval_history pah
                  JOIN proc_def pd ON pd.id = pah.proc_def_id AND pd.tenant_id = pah.tenant_id
                  LEFT JOIN vw_dashboard_proc_domain dm ON dm.proc_def_id = pd.id AND dm.tenant_id = pd.tenant_id
                  WHERE pah.tenant_id = %(tenant_id)s AND pd.isdeleted = false
                    AND {_domain_filter(domains)}
                ),
                aggregated AS (
                  SELECT
                    proc_def_id,
                    MAX(proc_def_name) AS proc_def_name,
                    MAX(domain) AS domain,
                    COUNT(*) FILTER (WHERE from_state IS NOT NULL AND to_state IS NOT NULL
                                     AND from_state <> to_state) AS total_churn,
                    COUNT(*) FILTER (WHERE from_state IN ('public_feedback', 'final_edit', 'published')
                                     AND to_state IN ('in_review', 'draft')) AS approved_to_review,
                    COUNT(*) FILTER (WHERE from_state = 'in_review' AND to_state = 'draft') AS review_to_draft
                  FROM history GROUP BY proc_def_id
                ),
                current_state AS (
                  SELECT proc_def_id, to_state AS current_state
                  FROM (
                    SELECT pah.proc_def_id, pah.to_state,
                           ROW_NUMBER() OVER (PARTITION BY pah.proc_def_id ORDER BY pah.created_at DESC) AS rn
                    FROM proc_def_approval_history pah
                    WHERE pah.tenant_id = %(tenant_id)s
                  ) ranked WHERE rn = 1
                ),
                churn_history AS (
                  SELECT proc_def_id, from_state, to_state,
                         CASE WHEN (from_state IN ('public_feedback', 'final_edit', 'published')
                                    AND to_state IN ('in_review', 'draft'))
                                OR (from_state = 'in_review' AND to_state = 'draft')
                              THEN 'rev' ELSE 'fwd' END AS dir,
                         created_at
                  FROM history
                  WHERE from_state IS NOT NULL AND to_state IS NOT NULL
                  ORDER BY created_at
                )
                SELECT
                  a.proc_def_id, a.proc_def_name, a.domain,
                  a.total_churn, a.approved_to_review, a.review_to_draft,
                  c.current_state,
                  (SELECT MAX(h.created_at)::date::text
                   FROM churn_history h
                   WHERE h.proc_def_id = a.proc_def_id AND h.dir = 'rev') AS last_revert,
                  (SELECT json_agg(json_build_object(
                     'from', h.from_state, 'to', h.to_state, 'dir', h.dir
                   ) ORDER BY h.created_at)
                   FROM churn_history h
                   WHERE h.proc_def_id = a.proc_def_id) AS history
                FROM aggregated a
                JOIN current_state c ON c.proc_def_id = a.proc_def_id
                WHERE (a.approved_to_review + a.review_to_draft) > 0
                ORDER BY a.total_churn DESC, a.approved_to_review DESC
                LIMIT 6
            """, {"tenant_id": tenant_id})
            churn = [dict(r) for r in cur.fetchall()]

        return {
            "overview": overview,
            "kpi_divisions": kpi_divisions,
            "funnel": funnel,
            "velocity": velocity,
            "churn": churn,
        }
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
# View 2: Process Analytics
# ═══════════════════════════════════════════════════════════════════════

async def get_process_analytics(domains: Optional[List[str]] = Query(None)):
    tenant_id = _get_tenant_id()
    domain_cond = _domain_filter_col(domains)

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            proc_map_index = _proc_map_subprocess_index(_fetch_proc_map(cur, tenant_id))

            cur.execute(
                f"""
                SELECT
                  pd.id,
                  pd.name,
                  pd.definition,
                  pd.bpmn,
                  pd.type,
                  COALESCE(dm.domain, '미분류') AS domain
                FROM proc_def pd
                LEFT JOIN vw_dashboard_proc_domain dm
                  ON dm.proc_def_id = pd.id AND dm.tenant_id = pd.tenant_id
                WHERE pd.tenant_id = %(tenant_id)s
                  AND pd.isdeleted = false
                """,
                {"tenant_id": tenant_id},
            )
            all_processes = [dict(r) for r in cur.fetchall()]
            for row in all_processes:
                location = proc_map_index.get(_normalize_proc_id(row.get("id")).lower())
                if location:
                    row["domain"] = location.get("domain") or row.get("domain") or "미분류"

            proc_lookup = _make_proc_lookup(all_processes)
            root_processes = [
                row for row in all_processes
                if _normalize_proc_id(row.get("id")).lower() in proc_map_index
                and not _is_call_activity_sub_module(row)
                and (not domains or row.get("domain") in domains)
            ]

            system_rows = _fetch_system_rows(cur, tenant_id)

            all_records: List[Dict[str, Any]] = []
            records_by_proc: Dict[str, List[Dict[str, Any]]] = {}
            for proc in root_processes:
                records = _collect_bpmn_tasks(proc, proc_lookup)
                proc_id = _normalize_proc_id(proc.get("id"))
                records_by_proc[proc_id] = records
                all_records.extend(records)

            # ── 1) 시스템 활용 맵: 시스템 관리 전체 시스템 + 프로세스 task 매핑 ──
            system_entries: Dict[str, Dict[str, Any]] = {}
            system_id_index: Dict[str, str] = {}
            system_name_index: Dict[str, str] = {}

            def register_system(system_id: Optional[str], name: Optional[str], system_type: Optional[str] = None, category: Optional[str] = None) -> str:
                clean_id = _text(system_id) or None
                clean_name = _text(name) or clean_id or "미지정"
                key = f"id:{clean_id}" if clean_id else f"name:{clean_name.lower()}"
                if key not in system_entries:
                    system_entries[key] = {
                        "system_id": clean_id,
                        "tool_name": clean_name,
                        "system_type": system_type,
                        "category": category,
                        "task_count": 0,
                        "process_ids": set(),
                        "connected_processes": {},
                    }
                if clean_id:
                    system_id_index[clean_id] = key
                if clean_name:
                    system_name_index[clean_name.lower()] = key
                return key

            for row in system_rows:
                register_system(row.get("id"), row.get("name"), row.get("system_type"), row.get("category"))

            for record in all_records:
                matched_keys: Set[str] = set()
                for system in record.get("systems", []):
                    system_id = _text(system.get("id")) if isinstance(system, dict) else ""
                    system_name = _text(system.get("name")) if isinstance(system, dict) else _text(system)
                    key = system_id_index.get(system_id) if system_id else None
                    if not key and system_name:
                        key = system_name_index.get(system_name.lower())
                    if not key:
                        key = register_system(system_id or None, system_name or None)
                    matched_keys.add(key)
                for key in matched_keys:
                    entry = system_entries[key]
                    entry["task_count"] += 1
                    proc_id = record["root_proc_id"]
                    entry["process_ids"].add(proc_id)
                    entry["connected_processes"][proc_id] = {
                        "proc_def_id": proc_id,
                        "proc_def_name": record["root_proc_name"],
                        "domain": record["root_domain"],
                    }

            system_map = []
            for entry in system_entries.values():
                connected = sorted(entry["connected_processes"].values(), key=lambda item: item["proc_def_name"])
                system_map.append({
                    "system_id": entry["system_id"],
                    "tool_name": entry["tool_name"],
                    "system_type": entry["system_type"],
                    "category": entry["category"],
                    "task_count": entry["task_count"],
                    "process_count": len(entry["process_ids"]),
                    "connected_processes": connected,
                })
            system_map.sort(key=lambda item: (-item["task_count"], item["tool_name"]))

            # ── 2) R&R 히트맵: multi role/부서 split 후 unique 합산 ──
            heat_counter: Counter = Counter()
            for record in all_records:
                for role in record.get("roles") or ["미지정"]:
                    heat_counter[(role, record["root_domain"])] += 1
            heatmap = [
                {"role_name": role, "domain": domain, "task_count": count}
                for (role, domain), count in heat_counter.items()
            ]
            heatmap.sort(key=lambda item: (item["domain"], item["role_name"]))

            # ── 3) Task 유형 분포 ──
            type_counter = Counter(record["classified_type"] for record in all_records)
            task_types = [
                {"task_type": task_type, "total_count": count}
                for task_type, count in type_counter.most_common()
            ]

            # ── 4) 프로세스별 Task 유형 비율 + 자동화 점수 ──
            task_ratio: List[Dict[str, Any]] = []
            automation_domain_counts: Dict[str, Counter] = defaultdict(Counter)
            for proc in root_processes:
                proc_id = _normalize_proc_id(proc.get("id"))
                records = records_by_proc.get(proc_id, [])
                counter = Counter(record["classified_type"] for record in records)
                total = sum(counter.values())
                if total:
                    task_dict = dict(counter)
                    task_ratio.append({
                        "name": proc.get("name") or proc_id,
                        "proc_def_id": proc_id,
                        "domain": proc.get("domain") or "미분류",
                        "tasks": task_dict,
                        "automation_score": _calc_automation_score(task_dict),
                    })
                    domain_key = proc.get("domain") or "미분류"
                    for task_type, count in counter.items():
                        automation_domain_counts[domain_key][task_type] += count
            task_ratio.sort(key=lambda item: (-sum(item["tasks"].values()), item["name"]))

            total_tasks = sum(type_counter.values())
            overall_score = _calc_automation_score(dict(type_counter))
            automation_by_domain = [
                {
                    "domain": domain,
                    "automation_score": _calc_automation_score(dict(counts)),
                    "total_count": sum(counts.values()),
                }
                for domain, counts in automation_domain_counts.items()
            ]
            automation_by_domain.sort(key=lambda item: (-item["automation_score"], item["domain"]))
            automation_summary = {
                "overall": overall_score,
                "total_count": total_tasks,
                "by_domain": automation_by_domain,
            }

            # ── 5) 과제 연결 맵: 과제(id 있으면 id, 없으면 name) 기준 그룹핑 ──
            project_entries: Dict[str, Dict[str, Any]] = {}
            for record in all_records:
                for project in record.get("projects", []):
                    project_id = _text(project.get("id")) or None
                    project_name = _text(project.get("name")) or project_id or "미지정"
                    key = f"id:{project_id}" if project_id else f"name:{project_name.lower()}"
                    entry = project_entries.setdefault(key, {
                        "project_name": project_name,
                        "task_count": 0,
                        "process_ids": set(),
                        "connected_processes": {},
                    })
                    entry["task_count"] += 1
                    proc_id = record["root_proc_id"]
                    entry["process_ids"].add(proc_id)
                    entry["connected_processes"][proc_id] = {
                        "proc_def_id": proc_id,
                        "proc_def_name": record["root_proc_name"],
                        "domain": record["root_domain"],
                    }
            project_map = []
            for entry in project_entries.values():
                project_map.append({
                    "project_name": entry["project_name"],
                    "task_count": entry["task_count"],
                    "process_count": len(entry["process_ids"]),
                    "connected_processes": sorted(entry["connected_processes"].values(), key=lambda item: item["proc_def_name"]),
                })
            project_map.sort(key=lambda item: (-item["task_count"], item["project_name"]))

            # ── 6) Top N 분석 ──
            top_buckets = {key: [] for key in ("handoff", "xor", "manual", "decision", "loop")}
            for proc in root_processes:
                proc_id = _normalize_proc_id(proc.get("id"))
                proc_name = proc.get("name") or proc_id
                domain = proc.get("domain") or "미분류"
                records = records_by_proc.get(proc_id, [])
                structure = _structure_metrics(proc, proc_lookup)
                manual_count = sum(1 for record in records if record["classified_type"] == "manual")
                decision_count = sum(
                    1
                    for record in records
                    if any(keyword in _text(record.get("name")) for keyword in DECISION_KEYWORDS)
                    or record["classified_type"] == "businessrule"
                )
                values = {
                    "handoff": structure["handoff"],
                    "xor": structure["xor"],
                    "manual": manual_count,
                    "decision": decision_count,
                    "loop": structure["loop"],
                }
                for key, count in values.items():
                    if count > 0:
                        top_buckets[key].append({"process": proc_name, "proc_def_id": proc_id, "count": count, "domain": domain})

            top_n = {
                key: sorted(items, key=lambda item: (-item["count"], item["process"]))[:5]
                for key, items in top_buckets.items()
            }

        return {
            "system_map": system_map,
            "heatmap": heatmap,
            "task_types": task_types,
            "task_ratio": task_ratio,
            "top_n": top_n,
            "automation_score": automation_summary,
            "project_map": project_map,
        }
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
# View 3: Governance & Quality
# ═══════════════════════════════════════════════════════════════════════

async def get_governance_quality(domains: Optional[List[str]] = Query(None)):
    tenant_id = _get_tenant_id()
    domain_cond = _domain_filter_col(domains)

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # ── 1) 프로세스 자산 현황 ──
            cur.execute(f"""
                WITH latest_status AS (
                  SELECT proc_def_id, to_state AS current_status,
                         ROW_NUMBER() OVER (PARTITION BY proc_def_id ORDER BY created_at DESC) AS rn
                  FROM proc_def_approval_history
                  WHERE tenant_id = %(tenant_id)s
                )
                SELECT
                  COALESCE(dm.domain, '미분류') AS domain,
                  COUNT(*) FILTER (WHERE pd.isdeleted = false
                    AND COALESCE(ls.current_status, 'draft') = 'published') AS active_count,
                  COUNT(*) FILTER (WHERE pd.isdeleted = false
                    AND COALESCE(ls.current_status, 'draft') <> 'published') AS draft_count,
                  COUNT(*) FILTER (WHERE pd.isdeleted = true) AS deprecated_count,
                  COUNT(*) AS total_count
                FROM proc_def pd
                LEFT JOIN latest_status ls ON ls.proc_def_id = pd.id AND ls.rn = 1
                LEFT JOIN vw_dashboard_proc_domain dm ON dm.proc_def_id = pd.id AND dm.tenant_id = pd.tenant_id
                WHERE pd.tenant_id = %(tenant_id)s
                  AND {_domain_filter(domains)}
                GROUP BY COALESCE(dm.domain, '미분류')
                ORDER BY total_count DESC, domain
            """, {"tenant_id": tenant_id})
            asset_status = [dict(r) for r in cur.fetchall()]

            # ── 2) 버전 변경 빈도 Top 5 ──
            cur.execute(f"""
                SELECT proc_def_name, change_count
                FROM vw_dashboard_version_frequency
                WHERE tenant_id = %(tenant_id)s
                  AND change_month >= DATE_TRUNC('month', NOW() - INTERVAL '1 month')
                  AND {domain_cond}
                ORDER BY change_count DESC, proc_def_name
                LIMIT 5
            """, {"tenant_id": tenant_id})
            version_top = [dict(r) for r in cur.fetchall()]

            # ── 3) DQ Score ──
            cur.execute(f"""
                SELECT * FROM (
                  SELECT 'description' AS field_key, '프로세스 설명' AS label,
                    ROUND(AVG(has_description)::numeric * 100, 1) AS score,
                    CASE WHEN ROUND(AVG(has_description)::numeric * 100, 1) >= 80 THEN 'up'
                         WHEN ROUND(AVG(has_description)::numeric * 100, 1) >= 60 THEN 'flat'
                         ELSE 'down' END AS trend, 1 AS sort_order
                  FROM vw_dashboard_quality_detail WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                  UNION ALL
                  SELECT 'roles', 'Owner 지정',
                    ROUND(AVG(has_roles)::numeric * 100, 1),
                    CASE WHEN ROUND(AVG(has_roles)::numeric * 100, 1) >= 80 THEN 'up'
                         WHEN ROUND(AVG(has_roles)::numeric * 100, 1) >= 60 THEN 'flat'
                         ELSE 'down' END, 2
                  FROM vw_dashboard_quality_detail WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                  UNION ALL
                  SELECT 'activity_desc', 'KPI 연계',
                    ROUND(AVG(activities_have_description)::numeric * 100, 1),
                    CASE WHEN ROUND(AVG(activities_have_description)::numeric * 100, 1) >= 80 THEN 'up'
                         WHEN ROUND(AVG(activities_have_description)::numeric * 100, 1) >= 60 THEN 'flat'
                         ELSE 'down' END, 3
                  FROM vw_dashboard_quality_detail WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                  UNION ALL
                  SELECT 'activity_role', '시스템 매핑',
                    ROUND(AVG(activities_have_role)::numeric * 100, 1),
                    CASE WHEN ROUND(AVG(activities_have_role)::numeric * 100, 1) >= 80 THEN 'up'
                         WHEN ROUND(AVG(activities_have_role)::numeric * 100, 1) >= 60 THEN 'flat'
                         ELSE 'down' END, 4
                  FROM vw_dashboard_quality_detail WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                  UNION ALL
                  SELECT 'end_event', 'SLA 입력',
                    ROUND(AVG(has_end_event)::numeric * 100, 1),
                    CASE WHEN ROUND(AVG(has_end_event)::numeric * 100, 1) >= 80 THEN 'up'
                         WHEN ROUND(AVG(has_end_event)::numeric * 100, 1) >= 60 THEN 'flat'
                         ELSE 'down' END, 5
                  FROM vw_dashboard_quality_detail WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                  UNION ALL
                  SELECT 'overall', '전체 DQ 평균 점수',
                    ROUND(AVG((has_description + has_name + has_roles + activities_have_description
                              + activities_have_role + has_end_event)::numeric / 6 * 100), 1),
                    'flat', 99
                  FROM vw_dashboard_quality_detail WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                ) ranked ORDER BY sort_order
            """, {"tenant_id": tenant_id})
            dq_scores = [dict(r) for r in cur.fetchall()]

            # ── 4) 문법 오류 프로세스 목록 ──
            cur.execute(f"""
                WITH quality AS (
                  SELECT proc_def_id, proc_def_name, domain,
                         has_description, has_roles,
                         activities_have_description, activities_have_role, has_end_event
                  FROM vw_dashboard_quality_detail
                  WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                ),
                error_flags AS (
                  SELECT proc_def_id,
                    MAX(CASE WHEN error_type = 'missing_start_event' THEN 1 ELSE 0 END) AS missing_start_event,
                    MAX(CASE WHEN error_type = 'missing_end_event' THEN 1 ELSE 0 END) AS missing_end_event
                  FROM vw_dashboard_error_processes
                  WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                  GROUP BY proc_def_id
                ),
                status_info AS (
                  SELECT proc_def_id,
                    COALESCE(last_status_change_at::date, saved_at::date, CURRENT_DATE) AS last_checked_at
                  FROM vw_dashboard_proc_state
                  WHERE tenant_id = %(tenant_id)s AND {domain_cond}
                ),
                scored AS (
                  SELECT q.proc_def_name, q.domain,
                    COALESCE(s.last_checked_at, CURRENT_DATE) AS last_checked_at,
                    COALESCE(e.missing_start_event, 0) AS missing_start_event,
                    COALESCE(e.missing_end_event, 0) AS missing_end_event,
                    (1 - q.has_description) AS missing_description,
                    (1 - q.has_roles) AS missing_roles,
                    (1 - q.activities_have_description) AS missing_activity_description,
                    (1 - q.activities_have_role) AS missing_activity_role
                  FROM quality q
                  LEFT JOIN error_flags e ON e.proc_def_id = q.proc_def_id
                  LEFT JOIN status_info s ON s.proc_def_id = q.proc_def_id
                )
                SELECT proc_def_name,
                  (missing_start_event + missing_end_event + missing_description
                   + missing_roles + missing_activity_description + missing_activity_role) AS error_count,
                  CASE
                    WHEN missing_end_event = 1 OR missing_start_event = 1 THEN '이벤트 누락'
                    WHEN missing_activity_role = 1 THEN '역할 누락'
                    WHEN missing_activity_description = 1 THEN '미완성 흐름'
                    WHEN missing_roles = 1 THEN '역할 정의 누락'
                    WHEN missing_description = 1 THEN '설명 누락'
                    ELSE '-'
                  END AS primary_error_type,
                  TO_CHAR(last_checked_at, 'YYYY-MM-DD') AS last_checked_at,
                  CASE
                    WHEN (missing_start_event + missing_end_event + missing_description
                         + missing_roles + missing_activity_description + missing_activity_role) > 0
                    THEN '수정 필요' ELSE '정상'
                  END AS status
                FROM scored
                ORDER BY error_count DESC, last_checked_at DESC, proc_def_name
                LIMIT 10
            """, {"tenant_id": tenant_id})
            grammar_errors = [dict(r) for r in cur.fetchall()]

        return {
            "asset_status": asset_status,
            "version_top": version_top,
            "dq_scores": dq_scores,
            "grammar_errors": grammar_errors,
        }
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
# Route Registration
# ═══════════════════════════════════════════════════════════════════════

def add_routes_to_app(app: FastAPI):
    app.add_api_route("/dashboard/executive-summary", get_executive_summary, methods=["GET"])
    app.add_api_route("/dashboard/process-analytics", get_process_analytics, methods=["GET"])
    app.add_api_route("/dashboard/governance-quality", get_governance_quality, methods=["GET"])

