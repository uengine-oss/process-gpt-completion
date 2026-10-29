"""inputData 에 폼 필드 참조가 아닌 항목(데이터 이름)이 있으면 인스턴스의 완료 산출물을 모두 넘긴다.

회귀(튜토리얼 VM, 인스턴스 4b98b714): 딥 에이전트가 생성한 「한빛오피스 대량 주문 견적 승인 프로세스」의
견적 산출 단계 inputData 가 ``["견적요청서", "재고확인결과"]`` 였다. get_field_value 는 ``split('.')`` 로
폼 id 와 필드를 나누다 실패해 None 을 돌려주고, 입력이 비자 바로 앞(재고 확인) 산출물만 넘어갔다.
견적 산출 에이전트는 두 단계 앞 요청서의 제품·수량을 못 받아 가격표를 찾고도 0원 견적을 냈다.

`<form_id>.<field>` 참조는 지금처럼 고른 칸만 넘긴다.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import database


PROC = "90216c1b-2e2b-4131-bbd4-49e71f12697f"
REQ = f"{PROC}_collect_request_form"
STOCK = f"{PROC}_check_stock_form"
QUOTE = f"{PROC}_calculate_quote_form"
INST = f"{PROC}.4b98b714"

REQUEST_OUTPUT = {REQ: {
    "customer_company": "한빛물산", "contact_person": "김민지 과장",
    "products_quantities": "메쉬 의자 에어 HB-C01 25개, 3단 이동 서랍 HB-S30 25개",
    "delivery_location": "서울 본사", "desired_date": "2026-11-20", "notes": "",
}}
STOCK_OUTPUT = {STOCK: {"product_name": "메쉬 의자 에어 HB-C01", "available_qty": "0", "can_meet_deadline": False}}


def _wi(activity_id, status, output, end, tool=None, wid=None):
    return SimpleNamespace(
        id=wid or f"wi-{activity_id}", activity_id=activity_id, status=status, output=output,
        end_date=end, start_date=None, execution_scope=0,
        tool=tool if tool is not None else f"formHandler:{PROC}_{activity_id}_form",
    )


class _Definition:
    processDefinitionId = PROC

    def __init__(self, input_data):
        self.activities = [
            SimpleNamespace(id="collect_request", tool=f"formHandler:{REQ}", inputData=[]),
            SimpleNamespace(id="check_stock", tool=f"formHandler:{STOCK}", inputData=["견적요청서"]),
            SimpleNamespace(id="calculate_quote", tool=f"formHandler:{QUOTE}", inputData=list(input_data)),
        ]

    def find_activity_by_id(self, activity_id):
        return next((a for a in self.activities if a.id == activity_id), None)

    def is_starting_activity(self, activity_id):
        return activity_id == "collect_request"

    def find_immediate_prev_activities(self, activity_id):
        order = [a.id for a in self.activities]
        i = order.index(activity_id)
        return [self.activities[i - 1]] if i > 0 else []


@pytest.fixture
def instance(monkeypatch):
    """인스턴스 4b98b714 를 줄인 꼴 — 요청서·재고 확인이 끝났고 견적 산출이 진행 중."""
    items = [
        _wi("collect_request", "DONE", REQUEST_OUTPUT, "2026-10-30 02:20:00+09:00"),
        _wi("check_stock", "DONE", STOCK_OUTPUT, "2026-10-30 02:21:00+09:00"),
        _wi("calculate_quote", "IN_PROGRESS", {}, None),
        _wi("sales_mgr_approval", "TODO", {}, None),
    ]

    def by_activity(inst, act, tenant=None, *a, **k):
        if inst != INST:
            return None
        return next((w for w in items if w.activity_id.lower() == str(act).lower()), None)

    monkeypatch.setattr(database, "fetch_workitem_by_proc_inst_and_activity", by_activity)
    monkeypatch.setattr(database, "fetch_workitems_by_proc_inst_id",
                        lambda inst, *a, **k: list(items) if inst == INST else [])
    monkeypatch.setattr(database, "fetch_workitems_by_root_proc_inst_id",
                        lambda inst, *a, **k: list(items) if inst == INST else [])
    monkeypatch.setattr(database, "fetch_process_instance",
                        lambda *a, **k: SimpleNamespace(root_proc_inst_id=INST, execution_scope=0))
    return items


def _input_for(definition, activity_id="calculate_quote"):
    workitem = {"id": f"wi-{activity_id}", "activity_id": activity_id, "proc_inst_id": INST, "tenant_id": "localhost"}
    return database.get_input_data(workitem, definition)


def test_named_inputs_get_every_completed_output_of_the_instance(instance):
    data = _input_for(_Definition(["견적요청서", "재고확인결과"]))
    # 두 단계 앞 요청서(제품·수량·설치 지역)까지 넘어간다
    assert data[REQ] == REQUEST_OUTPUT[REQ]
    assert data[STOCK] == STOCK_OUTPUT[STOCK]
    assert set(data) == {REQ, STOCK}


def test_dotted_refs_still_pass_only_the_chosen_fields(instance):
    data = _input_for(_Definition([f"{REQ}.products_quantities", f"{REQ}.delivery_location"]))
    assert data == {REQ: {
        "products_quantities": REQUEST_OUTPUT[REQ]["products_quantities"],
        "delivery_location": "서울 본사",
    }}


def test_mixed_refs_keep_the_chosen_fields_and_add_the_other_forms(instance):
    data = _input_for(_Definition([f"{REQ}.products_quantities", "재고확인결과"]))
    assert data[REQ] == {"products_quantities": REQUEST_OUTPUT[REQ]["products_quantities"]}
    assert data[STOCK] == STOCK_OUTPUT[STOCK]


def test_unfinished_and_current_workitems_are_not_passed(instance):
    instance[2].output = {QUOTE: {"total_price": 0}}       # 진행 중인 자기 자신(초안 저장 등)
    instance[3].output = {f"{PROC}_sales_mgr_approval_form": {"approval_decision": True}}  # 아직 TODO
    data = _input_for(_Definition(["견적요청서"]))
    assert set(data) == {REQ, STOCK}


def test_flat_agent_output_is_kept_under_its_form_id(instance):
    instance[1].output = {"product_name": "메쉬 의자 에어 HB-C01", "available_qty": "0"}
    data = _input_for(_Definition(["재고확인결과"]))
    assert data[STOCK] == {"product_name": "메쉬 의자 에어 HB-C01", "available_qty": "0"}
    assert "product_name" not in data


def test_later_completion_of_the_same_form_wins(instance):
    instance.append(_wi("check_stock", "DONE", {STOCK: {"available_qty": "30"}}, "2026-10-30 03:00:00+09:00",
                        wid="wi-check_stock-2"))
    data = _input_for(_Definition(["재고확인결과"]))
    assert data[STOCK] == {"available_qty": "30"}


def test_empty_input_data_still_falls_back_to_the_previous_step(instance):
    data = _input_for(_Definition([]))
    assert data == STOCK_OUTPUT


def test_is_form_field_ref():
    assert database._is_form_field_ref(f"{REQ}.notes")
    for entry in ("견적요청서", "", ".notes", f"{REQ}.", None, {"name": "견적요청서"}):
        assert not database._is_form_field_ref(entry)
