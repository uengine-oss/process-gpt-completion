from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple


def read_variables_data(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, list):
        result: Dict[str, Any] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = item.get("key") or item.get("name")
            if key:
                result[str(key)] = item.get("value")
        return result
    return {}


def merge_variables_data(raw: Any, patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = read_variables_data(raw)
    merged.update(patch or {})
    return merged


def collect_mapping_contexts(activity: Any) -> List[Dict[str, Any]]:
    props = _parse_properties(getattr(activity, "properties", None))
    contexts: List[Dict[str, Any]] = []

    def push(mapping_context: Any) -> None:
        if isinstance(mapping_context, str):
            mapping_context = _parse_json(mapping_context)
        if isinstance(mapping_context, dict) and isinstance(mapping_context.get("mappingElements"), list):
            contexts.append(mapping_context)

    push(_get_path(props, "eventSynchronization.mappingContext"))
    for sync in props.get("eventSynchronizations") or []:
        if isinstance(sync, dict):
            push(sync.get("mappingContext"))
    push(props.get("mapperIn"))
    push(_get_path(props, "outputMapping.mappingContext"))

    return contexts


def evaluate_mapping_context(
    mapping_context: Dict[str, Any],
    source_context: Dict[str, Any],
    default_form_id: Optional[str] = None,
) -> Dict[str, Any]:
    result = {
        "variables_data": {},
        "form_values": {},
        "role_bindings": [],
        "mapped": {},
        "trace": [],
        "errors": [],
    }

    for element in mapping_context.get("mappingElements") or []:
        target = _normalize_path(_get_path(element, "argument.text") or _get_path(element, "transformerMapping.linkedArgumentName"))
        if not target:
            _append_trace(result, "", None, None, "skipped", "missing target")
            continue

        value = None
        source = _get_path(element, "variable.name")
        if _is_call_activity_target_path(source) and not _is_call_activity_target_path(target):
            target, source = source, target
        transformer_mapping = element.get("transformerMapping") if isinstance(element, dict) else None
        if transformer_mapping:
            value, source, error = _evaluate_transformer(transformer_mapping, source_context)
            if error:
                _append_trace(result, target, source, None, "error", error)
                result["errors"].append({
                    "code": "UNSUPPORTED_TRANSFORMER",
                    "target": target,
                    "transformerType": source,
                    "message": error,
                })
                continue
        else:
            value, source = _resolve_source(source, source_context)
            if value is None:
                _append_trace(result, target, source, None, "skipped", "source not found")
                continue

        scope = _apply_target(result, target, value, default_form_id)
        _append_trace(result, target, source, value, "mapped", None, scope)

    return result


def apply_mapping_result_to_instance(process_instance: Any, mapping_result: Dict[str, Any]) -> None:
    mapped_variables = mapping_result.get("variables_data") or {}
    if mapped_variables:
        process_instance.variables_data = merge_variables_data(getattr(process_instance, "variables_data", None), mapped_variables)

    mapped_role_bindings = mapping_result.get("role_bindings") or []
    if mapped_role_bindings:
        process_instance.role_bindings = merge_role_bindings(
            getattr(process_instance, "role_bindings", None),
            mapped_role_bindings,
        )


def merge_role_bindings(raw: Any, patch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    if isinstance(raw, str) and raw.strip():
        parsed = _parse_json(raw)
        raw = parsed if isinstance(parsed, list) else []
    if isinstance(raw, list):
        merged = [dict(item) for item in raw if isinstance(item, dict)]

    for item in patch or []:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        role_name = item["name"]
        existing = next((binding for binding in merged if binding.get("name") == role_name), None)
        if existing:
            existing.update(item)
        else:
            merged.append(dict(item))
    return merged


def _normalize_role_binding_value(field: str, value: Any) -> Any:
    if field != "endpoint":
        return value
    if isinstance(value, list):
        return value[0] if value else ""
    if isinstance(value, dict):
        return value.get("id") or value.get("email") or value.get("endpoint") or value.get("username") or ""
    return value


def merge_form_values(form_values: Dict[str, Any], mapping_result: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(form_values or {})
    for form_id, values in (mapping_result.get("form_values") or {}).items():
        if not isinstance(values, dict):
            continue
        current = merged.get(form_id)
        if not isinstance(current, dict):
            current = {}
        current.update(values)
        merged[form_id] = current
    return merged


def _parse_json(raw: str) -> Any:
    try:
        return json_loads(raw)
    except Exception:
        return {}


def _parse_properties(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        parsed = _parse_json(raw)
        return parsed if isinstance(parsed, dict) else {}
    return {}


def json_loads(raw: str) -> Any:
    import json

    return json.loads(raw)


def _is_call_activity_target_path(path: Any) -> bool:
    normalized = _normalize_path(path)
    return normalized.startswith('callActivity.')

def _get_path(source: Any, path: str) -> Any:
    cursor = source
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _normalize_path(path: Any) -> str:
    return str(path or "").strip()


def _split_path(path: str) -> List[str]:
    path = _normalize_path(path)
    if path.startswith("[variables]."):
        path = "variables." + path[len("[variables].") :]
    if path.startswith("[instance]."):
        path = "instance." + path[len("[instance].") :]
    if path.startswith("[activities]."):
        path = "forms.byActivity." + path[len("[activities].") :]
    return [part for part in path.split(".") if part]


def _read_path(source: Any, path: str) -> Any:
    cursor = source
    for part in _split_path(path):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _write_path(target: Dict[str, Any], path: str, value: Any) -> None:
    parts = _split_path(path)
    if not parts:
        return
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def _resolve_source(path: Any, context: Dict[str, Any]) -> Tuple[Any, str]:
    normalized = _normalize_path(path)
    if not normalized:
        return None, ""

    explicit_roots = [
        ("forms.", context.get("forms")),
        ("payload.", context.get("payload")),
        ("workitem.", context.get("workitem")),
        ("instance.", context.get("instance")),
        ("variables.", _get_path(context, "instance.variablesData")),
        ("Variables.", _get_path(context, "instance.variablesData")),
    ]
    for prefix, root in explicit_roots:
        if normalized.startswith(prefix):
            remainder = normalized[len(prefix) :]
            value = _read_path(root, remainder)
            if value is None and prefix == "forms." and isinstance(root, dict):
                value = _read_path(root.get("byId"), remainder)
            return value, normalized

    for form_root in ("parentForm.", "childForm."):
        if normalized.startswith(form_root):
            remainder = normalized[len(form_root) :]
            return _read_path(_get_path(context, "forms.byId"), remainder), normalized

    if normalized.startswith("[variables]."):
        return _read_path(_get_path(context, "instance.variablesData"), normalized[len("[variables].") :]), normalized
    if normalized.startswith("[Variables]."):
        return _read_path(_get_path(context, "instance.variablesData"), normalized[len("[Variables].") :]), normalized
    if normalized.startswith("[instance]."):
        return _read_path(context.get("instance"), normalized[len("[instance].") :]), normalized
    if normalized.startswith("[activities]."):
        return _read_path(_get_path(context, "forms.byActivity"), normalized[len("[activities].") :]), normalized

    candidates = [
        ("forms.current", _get_path(context, "forms.current")),
        ("forms.byId", _get_path(context, "forms.byId")),
        ("forms.byActivity", _get_path(context, "forms.byActivity")),
        ("payload.response", _get_path(context, "payload.response")),
        ("payload.request", _get_path(context, "payload.request")),
        ("workitem", context.get("workitem")),
        ("instance.variablesData", _get_path(context, "instance.variablesData")),
        ("instance", context.get("instance")),
    ]
    for label, root in candidates:
        value = _read_path(root, normalized)
        if value is not None:
            return value, f"{label}.{normalized}"
    return None, normalized


def _evaluate_transformer(transformer_mapping: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Any, str, Optional[str]]:
    transformer = transformer_mapping.get("transformer") or {}
    transformer_type = transformer.get("_type") or ""
    sources = transformer.get("argumentSourceMap") or {}
    values = []
    for key in sources:
        source = sources.get(key)
        if isinstance(source, dict) and source.get("transformer"):
            value, _source, error = _evaluate_transformer(source, context)
            if error:
                return None, transformer_type, error
            values.append(value)
        else:
            values.append(_resolve_source(source, context)[0])

    if transformer_type.endswith("ConcatTransformer"):
        return "".join("" if value is None else str(value) for value in values), transformer_type, None
    if transformer_type.endswith("SumTransformer"):
        return sum(_to_number(value) for value in values), transformer_type, None
    if transformer_type.endswith("AbsTransformer"):
        return abs(_to_number(values[0] if values else 0)), transformer_type, None
    if transformer_type.endswith("CeilTransformer"):
        return math.ceil(_to_number(values[0] if values else 0)), transformer_type, None
    if transformer_type.endswith("FloorTransformer"):
        return math.floor(_to_number(values[0] if values else 0)), transformer_type, None
    if transformer_type.endswith("RoundTransformer"):
        return round(_to_number(values[0] if values else 0)), transformer_type, None
    if transformer_type.endswith("MaxTransformer"):
        return max((_to_number(value) for value in values), default=0), transformer_type, None
    if transformer_type.endswith("MinTransformer"):
        return min((_to_number(value) for value in values), default=0), transformer_type, None
    if transformer_type.endswith("ReplaceTransformer"):
        text = "" if not values else str(values[0] or "")
        old = str(transformer.get("oldString") or "")
        new = str(transformer.get("newString") or "")
        if not old:
            return text, transformer_type, None
        if transformer.get("isRegularExp") in (True, "true", "True"):
            return re.sub(old, new, text), transformer_type, None
        return text.replace(old, new), transformer_type, None
    if transformer_type.endswith("DirectValueTransformer") or transformer_type.endswith("DirectSqlExpressionTransformer"):
        return transformer.get("value"), transformer_type, None

    return None, transformer_type, "Unsupported mapper transformer."


def _to_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value or "0").replace(",", ""))
    except Exception:
        return 0.0


def _apply_target(result: Dict[str, Any], target: str, value: Any, default_form_id: Optional[str]) -> str:
    role_binding_target = _parse_role_binding_target(target)
    if role_binding_target:
        role_name, field = role_binding_target
        value = _normalize_role_binding_value(field, value)
        binding = {"name": role_name, field: value}
        result["role_bindings"].append(binding)
        result["mapped"][f"roleBindings.{role_name}.{field}"] = value
        return "role_bindings"

    if target.startswith("forms.current.") and default_form_id:
        field = target[len("forms.current.") :]
        result["form_values"].setdefault(default_form_id, {})
        _write_path(result["form_values"][default_form_id], field, value)
        return f"forms.{default_form_id}"

    if target.startswith("forms."):
        parts = target.split(".")
        if len(parts) >= 3:
            form_id = parts[1]
            field = ".".join(parts[2:])
            result["form_values"].setdefault(form_id, {})
            _write_path(result["form_values"][form_id], field, value)
            return f"forms.{form_id}"

    for form_root in ("parentForm.", "childForm."):
        if target.startswith(form_root):
            parts = target[len(form_root) :].split(".")
            if len(parts) >= 2:
                form_id = parts[0]
                field = ".".join(parts[1:])
                result["form_values"].setdefault(form_id, {})
                _write_path(result["form_values"][form_id], field, value)
                result["mapped"][f"{form_root}{form_id}.{field}"] = value
                return f"{form_root}{form_id}"

    if target.startswith("callActivity.variables."):
        field = target[len("callActivity.variables.") :]
        _write_path(result["variables_data"], field, value)
        result["mapped"][field] = value
        return "variables_data"

    if target.startswith("variables.") or target.startswith("Variables.") or target.startswith("[variables].") or target.startswith("[Variables]."):
        field = target.replace("[variables].", "").replace("[Variables].", "").replace("variables.", "", 1).replace("Variables.", "", 1)
        _write_path(result["variables_data"], field, value)
        result["mapped"][field] = value
        return "variables_data"

    if target.startswith("__mapped."):
        field = target[len("__mapped.") :]
        _write_path(result["mapped"], field, value)
        return "__mapped"

    _write_path(result["variables_data"], target, value)
    result["mapped"][target] = value
    return "variables_data"


def _parse_role_binding_target(target: str) -> Optional[Tuple[str, str]]:
    for prefix in ("callActivity.lane.", "lane.", "instance.lane.", "laneBindings.", "instance.laneBindings.", "roleBindings.", "instance.roleBindings."):
        if target.startswith(prefix):
            parts = target[len(prefix) :].split(".")
            if len(parts) == 2 and parts[0] and parts[1] in ("endpoint", "resourceName"):
                return parts[0], parts[1]
    return None


def _append_trace(
    result: Dict[str, Any],
    target: str,
    source: Optional[str],
    value: Any,
    status: str,
    reason: Optional[str] = None,
    scope: Optional[str] = None,
) -> None:
    item = {
        "target": target,
        "source": source,
        "status": status,
    }
    if scope:
        item["scope"] = scope
    if reason:
        item["reason"] = reason
    if value is not None:
        item["valuePreview"] = str(value)[:120]
    result["trace"].append(item)






