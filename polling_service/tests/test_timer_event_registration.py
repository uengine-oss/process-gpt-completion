"""타이머 중간 이벤트 등록 회귀 테스트.

운영에서 관찰된 실패: task6 -> event_payment_due 전이에서 파킹 워크아이템은 만들어지는데
크론이 등록되지 않아 인스턴스가 영구 정지했다. 원인 두 가지를 각각 덮는다.

1) 다음 액티비티 목록을 process_result(진입 시점 스냅샷)에서만 읽어, _process_next_activities 가
   갈아끼운 process_result_json 쪽 목록을 못 봤다.
2) 표현식 결정 LLM 이 expression 대신 dueDate('2026-10-10')로 답하면 그 값을 쓰는 곳이 없어
   'no expression' 으로 조용히 반환했다.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import workitem_processor as wp


class _Definition:
    def __init__(self, gateways):
        self._gateways = {g.id: g for g in gateways}

    def find_gateway_by_id(self, gateway_id):
        return self._gateways.get(gateway_id)


def _event(event_id="event_payment_due", event_type="intermediateThrowEvent", properties="{}"):
    return SimpleNamespace(id=event_id, name="수금 예정일 대기", type=event_type,
                           condition={}, properties=properties)


def _plain_gateway(gateway_id="gateway1"):
    return SimpleNamespace(id=gateway_id, name="분기", type="exclusiveGateway",
                           condition={}, properties="{}")


def _instance():
    return SimpleNamespace(proc_inst_id="proc.inst-1", proc_def_id="proc", tenant_id="uengine")


# --------------------------------------------------------------- dueDate → cron

@pytest.mark.parametrize("due,expected", [
    ("2026-10-10", "0 0 9 10 10 ? 2026"),
    ("2026-01-02T14:30", "0 30 14 2 1 ? 2026"),
    ("2027-12-31T00:00:00", "0 0 0 31 12 ? 2027"),
])
def test_due_date_to_cron(due, expected):
    assert wp._due_date_to_cron(due) == expected


@pytest.mark.parametrize("bad", [None, "", "   ", "not-a-date", "2026/10/10"])
def test_due_date_to_cron_rejects_garbage(bad):
    assert wp._due_date_to_cron(bad) is None


# ------------------------------------------------------- 표현식 우선순위 결정

def test_expression_wins_over_everything():
    event = {"expression": "0 0 9 1 1 ? 2027", "due_date": "2026-10-10",
             "properties": '{"expression": "0 0 8 2 2 ? 2028"}', "condition": {}}
    assert wp._resolve_timer_expression(event) == "0 0 9 1 1 ? 2027"


def test_falls_back_to_properties_expression():
    event = {"expression": None, "properties": '{"expression": "0 0 8 2 2 ? 2028"}',
             "condition": {}, "due_date": "2026-10-10"}
    assert wp._resolve_timer_expression(event) == "0 0 8 2 2 ? 2028"


def test_falls_back_to_condition_cron():
    event = {"expression": None, "properties": "{}",
             "condition": {"cron": "0 0 7 3 3 ? 2029"}, "due_date": "2026-10-10"}
    assert wp._resolve_timer_expression(event) == "0 0 7 3 3 ? 2029"


def test_falls_back_to_due_date():
    """운영에서 실제로 온 모양: expression 은 null, dueDate 만 채워져 있다."""
    event = {"expression": None, "properties": '{"expressionNL": "수금 예정일 당일"}',
             "condition": {}, "due_date": "2026-10-10"}
    assert wp._resolve_timer_expression(event) == "0 0 9 10 10 ? 2026"


def test_no_source_returns_none():
    assert wp._resolve_timer_expression(
        {"expression": None, "properties": "{}", "condition": {}, "due_date": None}) is None


def test_register_timer_event_skips_without_expression(capsys, monkeypatch):
    called = []
    monkeypatch.setattr(wp, "execute_rpc", lambda *a, **k: called.append(a))
    assert wp._register_timer_event(_instance(), {
        "event_id": "e1", "process_id": "proc.inst-1",
        "expression": None, "properties": "{}", "condition": {}, "due_date": None}) is None
    assert called == []


def test_register_timer_event_uses_due_date(monkeypatch):
    calls = []
    monkeypatch.setattr(wp, "execute_rpc", lambda name, params: calls.append((name, params)) or "ok")
    monkeypatch.setattr(wp, "fetch_workitem_by_proc_inst_and_activity",
                        lambda *a, **k: SimpleNamespace(id="wi-1"))
    wp._register_timer_event(_instance(), {
        "event_id": "event_payment_due", "process_id": "proc.inst-1",
        "expression": None, "properties": "{}", "condition": {}, "due_date": "2026-10-10"})
    assert len(calls) == 1
    name, params = calls[0]
    assert name == "register_cron_intermidiated"
    assert params["p_cron_expr"] == "0 0 9 10 10 ? 2026"
    assert params["p_job_name"] == "proc.inst-1_event_payment_due"


# ------------------------------------------------- 이벤트 후보 수집(_register_event)

def _capture_registered(monkeypatch):
    registered = []
    monkeypatch.setattr(wp, "_register_single_event",
                        lambda inst, event, prj: registered.append(event))
    return registered


def test_finds_event_from_process_result(monkeypatch):
    registered = _capture_registered(monkeypatch)
    process_result = SimpleNamespace(nextActivities=[
        SimpleNamespace(nextActivityId="event_payment_due", expression=None, dueDate="2026-10-10")])
    wp._register_event(_instance(), process_result, {"nextActivities": []},
                       _Definition([_event()]))
    assert [e["event_id"] for e in registered] == ["event_payment_due"]
    assert registered[0]["due_date"] == "2026-10-10"


def test_finds_event_only_present_in_result_json(monkeypatch):
    """게이트웨이 확장으로 process_result_json 쪽에만 남은 이벤트도 잡아야 한다."""
    registered = _capture_registered(monkeypatch)
    process_result = SimpleNamespace(nextActivities=[])
    result_json = {"nextActivities": [
        {"nextActivityId": "event_payment_due", "expression": None, "dueDate": "2026-10-10"}]}
    wp._register_event(_instance(), process_result, result_json, _Definition([_event()]))
    assert [e["event_id"] for e in registered] == ["event_payment_due"]
    assert registered[0]["due_date"] == "2026-10-10"


def test_event_is_not_registered_twice(monkeypatch):
    registered = _capture_registered(monkeypatch)
    process_result = SimpleNamespace(nextActivities=[
        SimpleNamespace(nextActivityId="event_payment_due", expression=None, dueDate=None)])
    result_json = {"nextActivities": [{"nextActivityId": "event_payment_due", "expression": None}]}
    wp._register_event(_instance(), process_result, result_json, _Definition([_event()]))
    assert len(registered) == 1


def test_plain_gateway_is_not_registered(monkeypatch):
    """find_gateway_by_id 는 평범한 게이트웨이도 잡는다. 크론을 걸면 안 된다."""
    registered = _capture_registered(monkeypatch)
    process_result = SimpleNamespace(nextActivities=[
        SimpleNamespace(nextActivityId="gateway1", expression="0 0 9 1 1 ? 2027", dueDate=None)])
    wp._register_event(_instance(), process_result, {"nextActivities": []},
                       _Definition([_plain_gateway()]))
    assert registered == []


@pytest.mark.parametrize("event_type", ["intermediateThrowEvent", "intermediateCatchEvent",
                                        "timerIntermediateEvent"])
def test_intermediate_event_types_are_accepted(monkeypatch, event_type):
    registered = _capture_registered(monkeypatch)
    process_result = SimpleNamespace(nextActivities=[
        SimpleNamespace(nextActivityId="event_payment_due", expression=None, dueDate="2026-10-10")])
    wp._register_event(_instance(), process_result, {"nextActivities": []},
                       _Definition([_event(event_type=event_type)]))
    assert len(registered) == 1
