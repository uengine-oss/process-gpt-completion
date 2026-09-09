"""에이전트가 평면으로 저장한 output 도 inputData 로 넘어가야 한다.

회귀: 워크아이템 output 은 사람이 폼으로 제출하면 {form_id: {필드: 값}} 이지만
에이전트가 수행한 태스크는 폼 id 로 감싸지 않고 평면으로 저장한다. get_field_value 는
중첩만 보고 있어서, 에이전트 태스크의 산출물을 참조하는 후속 태스크의 inputData 가
통째로 비어서 전달됐다.

운영(uengine) 영업 제안 프로세스에서 관찰된 모습 — task8 이 선언한 7개 참조 중
에이전트가 수행한 task5 의 4개가 전부 빠진 채 에이전트에게 전달됐다:

    [InputData]
    {"..._task1_form": {"customer_company": ..., "customer_email": ...},
     "..._task6_form": {"approval_date": "2026-09-09"}}

그 결과 에이전트가 payment_due_date 를 채우지 못해 태스크가 반복 실패했다.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import database


PROC = "sales_process"
FORM = f"{PROC}_task5_form"


class _Definition:
    processDefinitionId = PROC

    def __init__(self):
        self.activities = [
            SimpleNamespace(id="task5", tool=f"formHandler:{FORM}"),
            SimpleNamespace(id="task8", tool=f"formHandler:{PROC}_task8_form"),
        ]


@pytest.fixture
def stub_db(monkeypatch):
    """워크아이템 한 건만 돌려주는 최소 스텁."""

    def _install(output):
        workitem = SimpleNamespace(activity_id="task5", output=output, execution_scope=0)

        monkeypatch.setattr(
            database, "fetch_workitem_by_proc_inst_and_activity",
            lambda inst, act, tenant, *a, **k: workitem if act == "task5" and inst == "inst-1" else None)
        monkeypatch.setattr(
            database, "fetch_workitems_by_proc_inst_id", lambda *a, **k: [workitem])
        monkeypatch.setattr(
            database, "fetch_process_instance",
            lambda *a, **k: SimpleNamespace(root_proc_inst_id="inst-1", execution_scope=0))
        monkeypatch.setattr(
            database, "fetch_workitems_by_root_proc_inst_id", lambda *a, **k: [workitem])
        return workitem

    return _install


def _get(field="payment_due_date"):
    return database.get_field_value(f"{FORM}.{field}", _Definition(), "inst-1", "uengine")


def test_nested_output_still_works(stub_db):
    stub_db({FORM: {"payment_due_date": "2026-09-30", "contract_no": "C-1"}})
    assert _get() == {FORM: {"payment_due_date": "2026-09-30"}}


def test_flat_agent_output_is_resolved(stub_db):
    """에이전트 산출물: form_id 중첩 없이 필드가 최상위에 있다."""
    stub_db({"text": "계약서를 생성했습니다", "payment_due_date": "2026-09-30", "contract_no": "C-1"})
    assert _get() == {FORM: {"payment_due_date": "2026-09-30"}}


def test_nested_wins_over_flat(stub_db):
    stub_db({"payment_due_date": "평면값", FORM: {"payment_due_date": "중첩값"}})
    assert _get() == {FORM: {"payment_due_date": "중첩값"}}


def test_flat_fallback_when_nested_lacks_the_field(stub_db):
    stub_db({"payment_due_date": "2026-09-30", FORM: {"contract_no": "C-1"}})
    assert _get() == {FORM: {"payment_due_date": "2026-09-30"}}


def test_missing_field_returns_none(stub_db):
    stub_db({"text": "설명만 있다", "contract_no": "C-1"})
    assert _get() is None


def test_empty_output_returns_none(stub_db):
    stub_db({})
    assert _get() is None


def test_other_forms_flat_field_is_not_borrowed(monkeypatch):
    """다른 액티비티의 평면 output 에서 같은 이름 필드를 집어오면 안 된다."""
    other = SimpleNamespace(activity_id="task2", output={"payment_due_date": "엉뚱한값"},
                            execution_scope=0)
    monkeypatch.setattr(database, "fetch_workitem_by_proc_inst_and_activity",
                        lambda *a, **k: None)
    monkeypatch.setattr(database, "fetch_workitems_by_proc_inst_id", lambda *a, **k: [other])
    monkeypatch.setattr(database, "fetch_process_instance",
                        lambda *a, **k: SimpleNamespace(root_proc_inst_id="inst-1", execution_scope=0))
    monkeypatch.setattr(database, "fetch_workitems_by_root_proc_inst_id", lambda *a, **k: [other])
    assert _get() is None


def test_grouping_shape_is_unchanged(stub_db):
    """group_fields_by_form 이 그대로 받아 쓸 수 있는 모양이어야 한다."""
    stub_db({"payment_due_date": "2026-09-30"})
    ref = f"{FORM}.payment_due_date"
    grouped = database.group_fields_by_form({ref: _get()})
    assert grouped == {FORM: {"payment_due_date": "2026-09-30"}}
