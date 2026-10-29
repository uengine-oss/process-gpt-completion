"""Deterministic evaluation of simple numeric branch conditions.

Generated definitions write branch rules as text on the sequence, for example
"견적 총액 3,000만 원 미만" / "견적 총액 3,000만 원 이상". Every such rule used to go to the LLM
judge. The judge (gpt-4.1, temperature 0) got the plain comparison 6,531,250 < 30,000,000 wrong in
two of three runs and marked both branches of the exclusive gateway false. With no true branch and
no default flow the engine routed nowhere: the instance stayed RUNNING with no current activity.

A comparison of one field against one number does not need a model. When a condition is exactly
"<field> <amount> 미만|이하|초과|이상" (or "<field> < <amount>" etc.) and the field resolves to a
single number in the instance's outputs, it is compared here. Anything else — compound rules,
negations, words, unresolved or ambiguous fields, non-numeric values — returns None and goes to the
judge as before.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

_BIG_UNITS = {"만": 10 ** 4, "억": 10 ** 8, "조": 10 ** 12}
_SMALL_UNITS = {"천": 10 ** 3, "백": 10 ** 2, "십": 10}

# Words a sentence may end with after the comparator: "…미만인 경우", "…이상일 때", "…초과 시", "…이면".
_TRAILER = r"(?:\s*(?:인|일)?\s*(?:경우|때|시|건)(?:에|에는|에만)?|\s*(?:이면|면|임|입니다))?\s*[.。]?"
# Currency / count units written after an amount.
_UNIT = r"(?:\s*(?:원|KRW|krw|won|%|퍼센트|개|건|명|일|EA|ea))?"
# Digits, separators and Korean number units; validated by parse_amount.
_AMOUNT = r"[₩\\]?\s*(?:\d[\d,\.]*|[조억만천백십])(?:\s*(?:\d[\d,\.]*|[조억만천백십]))*"

_KOREAN_OPS = {"미만": "<", "이하": "<=", "초과": ">", "이상": ">="}
_SYMBOL_OPS = {"<": "<", "<=": "<=", "=<": "<=", "≤": "<=", ">": ">", ">=": ">=", "=>": ">=", "≥": ">="}

_KOREAN_RULE = re.compile(
    rf"^(?P<subject>.+?)\s*(?P<amount>{_AMOUNT}){_UNIT}\s*(?P<op>미만|이하|초과|이상){_TRAILER}$"
)
_SYMBOL_RULE = re.compile(
    rf"^(?P<subject>.+?)\s*(?P<op><=|>=|=<|=>|≤|≥|<|>)\s*(?P<amount>{_AMOUNT}){_UNIT}{_TRAILER}$"
)
# More than one rule, negation, ranges: leave to the judge.
_COMPOUND = re.compile(
    r"(그리고|또는|혹은|이거나|이고\s|및|아닌|않|없|사이|부터|까지|~|&&|\|\||\band\b|\bor\b|\bnot\b)",
    re.IGNORECASE,
)
_PARTICLE = re.compile(r"(이|가|은|는|의)$")


def parse_amount(text: Any) -> Optional[float]:
    """'3,000만 원' → 30000000, '1억 5천만' → 150000000, '6531250' → 6531250. None if not a number."""
    if isinstance(text, bool) or text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    if not isinstance(text, str):
        return None
    s = text.strip()
    s = re.sub(r"^[₩\\]", "", s)
    s = re.sub(r"\s*(원|KRW|krw|won|%|퍼센트|개|건|명|일|EA|ea)$", "", s)
    s = re.sub(r"[\s,]", "", s)
    if not s:
        return None
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)

    tokens = re.findall(r"\d+(?:\.\d+)?|[조억만천백십]|.", s)
    total = 0.0
    section = 0.0
    number: Optional[float] = None
    seen_unit = False
    for tok in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?", tok):
            if number is not None:
                return None
            number = float(tok)
        elif tok in _SMALL_UNITS:
            section += (number if number is not None else 1.0) * _SMALL_UNITS[tok]
            number = None
            seen_unit = True
        elif tok in _BIG_UNITS:
            section += number if number is not None else 0.0
            if section == 0:
                section = 1.0
            total += section * _BIG_UNITS[tok]
            section = 0.0
            number = None
            seen_unit = True
        else:
            return None
    if not seen_unit:
        return None
    return total + section + (number or 0.0)


def parse_threshold_condition(text: Any) -> Optional[Tuple[str, str, float]]:
    """'견적 총액 3,000만 원 미만' → ('견적 총액', '<', 30000000.0). None unless it is one plain comparison."""
    if not isinstance(text, str):
        return None
    s = re.sub(r"\s+", " ", text).strip()
    if not s or _COMPOUND.search(s):
        return None
    for pattern, ops in ((_KOREAN_RULE, _KOREAN_OPS), (_SYMBOL_RULE, _SYMBOL_OPS)):
        m = pattern.match(s)
        if not m:
            continue
        subject = m.group("subject").strip()
        op = ops.get(m.group("op"))
        amount = parse_amount(m.group("amount"))
        if not subject or op is None or amount is None:
            return None
        # "3,000만 원 미만" has no subject: the regex would take "3," as one — not a field name.
        if not re.search(r"[A-Za-z가-힣]", subject):
            return None
        if re.search(r"(미만|이하|초과|이상|[<>≤≥])", subject):
            return None
        return subject, op, amount
    return None


def _norm(value: Any) -> str:
    return re.sub(r"[\s_\-·.()\[\]{}:]", "", str(value or "")).lower()


def field_names_by_form(ui_definitions: Optional[Iterable[Any]]) -> Dict[str, Dict[str, str]]:
    """{form_id: {field_key: display text}} from form definitions (fields_json)."""
    names: Dict[str, Dict[str, str]] = {}
    for ui in ui_definitions or []:
        try:
            form_id = getattr(ui, "id", None) or (ui.get("id") if isinstance(ui, dict) else None)
            fields = getattr(ui, "fields_json", None)
            if fields is None and isinstance(ui, dict):
                fields = ui.get("fields_json")
        except Exception:
            continue
        if not isinstance(fields, list):
            continue
        mapping = names.setdefault(str(form_id or ""), {})
        for item in fields:
            if isinstance(item, dict) and isinstance(item.get("key"), str) and isinstance(item.get("text"), str):
                mapping[item["key"]] = item["text"]
    return names


def _unwrap(value: Any) -> Any:
    # apply_field_name_annotation_recursively wraps values as {"name": ..., "value": ...}
    if isinstance(value, dict) and "value" in value and "name" in value:
        return value.get("value")
    return value


def _candidates(data: Any) -> List[Tuple[Optional[str], str, Any]]:
    """(form_id, field_key, value) for {form_id: {field: value}} and flat {field: value} outputs."""
    found: List[Tuple[Optional[str], str, Any]] = []
    if not isinstance(data, dict):
        return found
    for key, value in data.items():
        if isinstance(value, dict) and not ("value" in value and "name" in value):
            for field, inner in value.items():
                found.append((str(key), str(field), _unwrap(inner)))
        else:
            found.append((None, str(key), _unwrap(value)))
    return found


def resolve_numeric_field(
    subject: str,
    data: Any,
    names: Dict[str, Dict[str, str]],
) -> Optional[Tuple[str, float]]:
    """Find the one number the subject names, by field key or display text. None when absent or ambiguous."""
    wanted = {_norm(subject)}
    stripped = _PARTICLE.sub("", subject.strip())
    if stripped and stripped != subject.strip():
        wanted.add(_norm(stripped))
    wanted.discard("")
    if not wanted:
        return None

    any_names: Dict[str, str] = {}
    for mapping in names.values():
        for k, v in mapping.items():
            any_names.setdefault(k, v)

    def _label(form_id: Optional[str], field: str) -> str:
        if form_id is not None and field in names.get(form_id, {}):
            return names[form_id][field]
        return any_names.get(field, field)

    def _match(exact: bool) -> List[Tuple[str, Any]]:
        hits: List[Tuple[str, Any]] = []
        for form_id, field, value in _candidates(data):
            label = _label(form_id, field)
            keys = {_norm(field), _norm(label)}
            if exact:
                ok = bool(keys & wanted)
            else:
                ok = any(
                    len(w) >= 2 and len(k) >= 2 and (k.endswith(w) or w.endswith(k))
                    for w in wanted for k in keys
                )
            if ok:
                hits.append((label, value))
        return hits

    hits = _match(True) or _match(False)
    if not hits:
        return None
    numbers = []
    for label, value in hits:
        number = parse_amount(value)
        if number is None:
            # The field exists but holds text ("확인 불가") — not ours to decide.
            return None
        numbers.append((label, number))
    distinct = {n for _, n in numbers}
    if len(distinct) != 1:
        return None
    return numbers[0]


def _compare(left: float, op: str, right: float) -> bool:
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise ValueError(op)


def _fmt(number: float) -> str:
    return f"{int(number):,}" if float(number).is_integer() else f"{number:,}"


def evaluate_threshold_condition(
    condition_text: Any,
    primary_data: Optional[dict],
    contexts: Iterable[Any],
    ui_definitions: Optional[Iterable[Any]],
) -> Optional[Tuple[bool, str]]:
    """(met, reason) for a plain numeric comparison whose field is found; None to let the judge decide.

    primary_data is the gateway's conditionData: when the gateway names its data, only that data is
    used (the judge is told the same). Otherwise the contexts are tried in order.
    """
    rule = parse_threshold_condition(condition_text)
    if rule is None:
        return None
    subject, op, threshold = rule
    names = field_names_by_form(ui_definitions)

    sources: List[Any] = [primary_data] if isinstance(primary_data, dict) and primary_data else list(contexts)
    for source in sources:
        resolved = resolve_numeric_field(subject, source, names)
        if resolved is None:
            continue
        label, value = resolved
        met = _compare(value, op, threshold)
        reason = (
            f"{label} {_fmt(value)} {op} {_fmt(threshold)} → {'참' if met else '거짓'} "
            f"(조건 「{str(condition_text).strip()}」 을 규칙으로 판정)"
        )
        return met, reason
    return None
