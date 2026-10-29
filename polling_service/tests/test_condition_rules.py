"""Numeric branch conditions are compared by rule, not by the LLM judge.

Regression: 「한빛오피스 대량 주문 견적 승인」 gateway 「1차 승인 여부 결정(3,000만 원 미만)」 —
the judge marked "견적 총액 3,000만 원 미만" false for a total of 6,531,250 (and "…이상" false too),
so the exclusive gateway had no branch and the instance stopped after the agent step.
"""
from __future__ import annotations

import types

import pytest

from condition_rules import (
    evaluate_threshold_condition,
    parse_amount,
    parse_threshold_condition,
    resolve_numeric_field,
)

FORM = "df017161-1553-4173-a7b2-bb339e2d1859_calculate_quote_form"
UI = [
    types.SimpleNamespace(
        id=FORM,
        fields_json=[
            {"key": "subtotal", "text": "제품별 합계"},
            {"key": "total_price", "text": "견적 총액"},
            {"key": "discount_rate", "text": "수량 할인율"},
            {"key": "quote_memo", "text": "비고"},
        ],
    )
]
OUTPUTS = {
    "df017161-1553-4173-a7b2-bb339e2d1859_receive_request_form": {"company_name": "한빛물산"},
    FORM: {"subtotal": 6875000, "total_price": 6531250, "discount_rate": 5, "quote_memo": "부가세 별도"},
}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3,000만 원", 30_000_000),
        ("3000만원", 30_000_000),
        ("3천만 원", 30_000_000),
        ("1억 5천만 원", 150_000_000),
        ("1억5,000만", 150_000_000),
        ("1.5억", 150_000_000),
        ("30,000,000원", 30_000_000),
        ("₩30,000,000", 30_000_000),
        ("천만", 10_000_000),
        ("백만원", 1_000_000),
        ("6,531,250원", 6_531_250),
        (6531250, 6_531_250),
        (5.5, 5.5),
    ],
)
def test_parse_amount_reads_korean_units(text, expected):
    assert parse_amount(text) == expected


@pytest.mark.parametrize("text", ["약 650만 원", "확인 불가", "", None, True, "12a", ["0:1"]])
def test_parse_amount_rejects_non_numbers(text):
    assert parse_amount(text) is None


@pytest.mark.parametrize(
    "text, expected",
    [
        ("견적 총액 3,000만 원 미만", ("견적 총액", "<", 30_000_000)),
        ("견적 총액 3,000만 원 이상", ("견적 총액", ">=", 30_000_000)),
        ("견적 총액이 3,000만 원 이상인 경우", ("견적 총액이", ">=", 30_000_000)),
        ("총액 3천만원 초과 시", ("총액", ">", 30_000_000)),
        ("수량 20개 이하일 때", ("수량", "<=", 20)),
        ("할인율 10% 이상이면", ("할인율", ">=", 10)),
        ("total_price < 30000000", ("total_price", "<", 30_000_000)),
        ("견적 총액 >= 3,000만 원", ("견적 총액", ">=", 30_000_000)),
    ],
)
def test_parse_threshold_condition_plain_comparisons(text, expected):
    assert parse_threshold_condition(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "3,000만 원 미만",  # no subject
        "견적 총액 3,000만 원 이상이 아닌 경우",  # negation
        "총액이 1억 이상이고 VIP 고객",  # compound
        "견적 총액 1천만~3천만",  # range
        "견적 총액이 3,000만 원 이상 또는 특별 할인",
        "반려(1차 또는 2차)",
        "최종 승인",
        "x = 5",
        None,
    ],
)
def test_parse_threshold_condition_leaves_the_rest_to_the_judge(text):
    assert parse_threshold_condition(text) is None


def test_resolve_numeric_field_by_display_text_key_and_suffix():
    from condition_rules import field_names_by_form

    names = field_names_by_form(UI)
    assert resolve_numeric_field("견적 총액", OUTPUTS, names) == ("견적 총액", 6531250)
    assert resolve_numeric_field("견적 총액이", OUTPUTS, names) == ("견적 총액", 6531250)
    assert resolve_numeric_field("total_price", OUTPUTS, names) == ("견적 총액", 6531250)
    assert resolve_numeric_field("총액", OUTPUTS, names) == ("견적 총액", 6531250)
    assert resolve_numeric_field("금액", OUTPUTS, names) is None
    # 숫자가 아닌 칸은 판정하지 않는다
    assert resolve_numeric_field("비고", OUTPUTS, names) is None


def test_resolve_numeric_field_ambiguous_values_are_not_guessed():
    from condition_rules import field_names_by_form

    ui = UI + [types.SimpleNamespace(id="other_form", fields_json=[{"key": "total", "text": "견적 총액"}])]
    data = {**OUTPUTS, "other_form": {"total": 99_000_000}}
    assert resolve_numeric_field("견적 총액", data, field_names_by_form(ui)) is None


def test_evaluate_threshold_condition_hanbit_quote_gateway():
    below = evaluate_threshold_condition("견적 총액 3,000만 원 미만", None, [OUTPUTS, {}], UI)
    above = evaluate_threshold_condition("견적 총액 3,000만 원 이상", None, [OUTPUTS, {}], UI)
    assert below is not None and below[0] is True
    assert above is not None and above[0] is False
    assert "6,531,250" in below[1] and "30,000,000" in below[1]


def test_evaluate_threshold_condition_boundary():
    data = {FORM: {"total_price": 30_000_000}}
    assert evaluate_threshold_condition("견적 총액 3,000만 원 미만", None, [data], UI)[0] is False
    assert evaluate_threshold_condition("견적 총액 3,000만 원 이상", None, [data], UI)[0] is True
    assert evaluate_threshold_condition("견적 총액 3,000만 원 이하", None, [data], UI)[0] is True
    assert evaluate_threshold_condition("견적 총액 3,000만 원 초과", None, [data], UI)[0] is False


def test_evaluate_threshold_condition_reads_string_amounts_and_later_contexts():
    previous = {FORM: {"total_price": "4,500만 원"}}
    assert evaluate_threshold_condition("견적 총액 3,000만 원 이상", None, [{}, previous], UI)[0] is True


def test_evaluate_threshold_condition_gateway_primary_data_only():
    # 갈림길이 conditionData 로 자료를 지정했으면 그 자료만 본다(판정 모델에게 주는 규칙과 같다).
    primary = {"approval_form": {"decision": "approve"}}
    assert evaluate_threshold_condition("견적 총액 3,000만 원 미만", primary, [OUTPUTS], UI) is None
    primary = {FORM: {"total_price": 45_000_000}}
    assert evaluate_threshold_condition("견적 총액 3,000만 원 미만", primary, [OUTPUTS], UI)[0] is False


def test_evaluate_threshold_condition_unknown_field_goes_to_judge():
    assert evaluate_threshold_condition("승인 금액 3,000만 원 미만", None, [OUTPUTS], UI) is None
    assert evaluate_threshold_condition("승인인 경우", None, [OUTPUTS], UI) is None
