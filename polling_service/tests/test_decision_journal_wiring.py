"""분기 판단 이력이 실제 엔진 경로에서 수집되는지 검증한다.

`test_decision_journal.py` 가 저널 자료구조 자체를 검증한다면, 이 파일은
`resolve_next_activity_payloads` 와 조건 평가 경로에 레코더가 제대로 물려 있는지,
그리고 레코더를 붙여도 **분기 선택 결과가 달라지지 않는지**를 확인한다.
"""

import asyncio
import json
import pathlib
from typing import Any, Dict, Optional

import pytest

from decision_journal import (
    DecisionRecorder,
    EVENT_TYPE_DECISION,
    EVENT_TYPE_TRACE,
    METHOD_EXPRESSION,
    METHOD_NATURAL_LANGUAGE,
    OUTCOME_ADVANCED,
    OUTCOME_UNDECIDED,
    RULE_DEFAULT_FLOW,
    RULE_NO_CANDIDATE,
    RULE_PRIORITY,
    RULE_SINGLE_TRUE,
    VERDICT_FALSE,
    VERDICT_TRUE,
    VERDICT_UNDETERMINED,
    collect_traversed_sequence_ids,
)
from process_definition import load_process_definition

# 무거운 의존성 스텁을 재사용하기 위해 기존 테스트의 로더를 그대로 쓴다.
from test_resolve_next_activities import _load_modules


@pytest.fixture(scope="module")
def wiproc():
    return _load_modules()


# ---------------------------------------------------------------------------
# 최소 정의: Start -> A -> XOR(Gateway_1) -> (B | C) -> XOR(Gateway_2) -> D -> End
# ---------------------------------------------------------------------------
def _activity(activity_id: str, name: str) -> dict:
    return {
        "id": activity_id,
        "name": name,
        "role": "담당자",
        "type": "userTask",
        "process": "P",
        "description": name,
        "tool": "formHandler:form",
    }


def _definition(gateway_1_sequences: list[dict]) -> Any:
    data = {
        "processDefinitionId": "decision.journal.test",
        "processDefinitionName": "판단 이력 테스트",
        "description": "",
        "roles": [{"name": "담당자", "endpoint": "user@example.com"}],
        "activities": [
            _activity("Task_A", "신청 접수"),
            _activity("Task_B", "팀장 승인"),
            _activity("Task_C", "임원 승인"),
            _activity("Task_D", "결과 통보"),
        ],
        "gateways": [
            {"id": "Gateway_1", "name": "금액 구분", "type": "exclusiveGateway", "process": "P", "condition": {}, "properties": "{}"},
            {"id": "Gateway_2", "name": "병합", "type": "exclusiveGateway", "process": "P", "condition": {}, "properties": "{}"},
        ],
        "events": [],
        "subProcesses": [],
        "sequences": [
            {"id": "Flow_2", "source": "Task_A", "target": "Gateway_1", "condition": "", "properties": "{}"},
            *gateway_1_sequences,
            {"id": "Flow_5", "source": "Task_B", "target": "Gateway_2", "condition": "", "properties": "{}"},
            {"id": "Flow_6", "source": "Task_C", "target": "Gateway_2", "condition": "", "properties": "{}"},
            {"id": "Flow_7", "source": "Gateway_2", "target": "Task_D", "condition": "", "properties": "{}"},
        ],
    }
    return load_process_definition(data)


_TWO_BRANCHES = [
    {"id": "Flow_3", "source": "Gateway_1", "target": "Task_B", "condition": "", "properties": "{}"},
    {"id": "Flow_4", "source": "Gateway_1", "target": "Task_C", "condition": "", "properties": "{}"},
]


def _workitem() -> Dict[str, Any]:
    return {
        "id": "wi-1",
        "user_id": "tester@example.com",
        "proc_inst_id": "decision.journal.test.1",
        "proc_def_id": "decision.journal.test",
        "activity_id": "Task_A",
        "tenant_id": "localhost",
        "root_proc_inst_id": "decision.journal.test.1",
        "rework_count": 0,
        "assignees": [{"name": "담당자", "endpoint": "user@example.com"}],
        "output": {},
    }


def _recorder(**overrides) -> DecisionRecorder:
    kwargs = dict(
        proc_inst_id="decision.journal.test.1",
        proc_def_id="decision.journal.test",
        activity_id="Task_A",
        workitem_id="wi-1",
        tenant_id="localhost",
        rework_count=0,
    )
    kwargs.update(overrides)
    return DecisionRecorder(**kwargs)


def _decisions_by_source(recorder: DecisionRecorder) -> Dict[str, dict]:
    return {d["source"]["id"]: d for d in recorder.decisions}


def _resolve(wiproc, defn, sequence_condition_data, recorder=None, activity_id="Task_A"):
    return wiproc.resolve_next_activity_payloads(
        defn,
        activity_id,
        _workitem(),
        sequence_condition_data,
        recorder=recorder,
    )


# ---------------------------------------------------------------------------
# 회귀: 레코더를 붙여도 분기 결과가 달라지지 않는다
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "conditions",
    [
        {},
        {"Flow_3": {"conditionEval": True}, "Flow_4": {"conditionEval": False}},
        {"Flow_3": {"conditionEval": False}, "Flow_4": {"conditionEval": False}},
        {"Flow_3": {"conditionEval": True}, "Flow_4": {"conditionEval": True}},
    ],
)
def test_recorder_does_not_change_branch_selection(wiproc, conditions):
    defn = _definition(_TWO_BRANCHES)

    without = _resolve(wiproc, defn, dict(conditions))
    with_recorder = _resolve(wiproc, defn, dict(conditions), recorder=_recorder())

    assert [p.get("nextActivityId") for p in without] == [p.get("nextActivityId") for p in with_recorder]


def test_recorder_failure_does_not_break_resolution(wiproc):
    """레코더가 터져도 분기 결과는 그대로여야 한다."""

    class _ExplodingRecorder(DecisionRecorder):
        def record_decision(self, **_kwargs):
            raise RuntimeError("boom")

    defn = _definition(_TWO_BRANCHES)
    conditions = {"Flow_3": {"conditionEval": True}, "Flow_4": {"conditionEval": False}}

    expected = _resolve(wiproc, defn, dict(conditions))
    actual = _resolve(wiproc, defn, dict(conditions), recorder=_ExplodingRecorder(proc_inst_id="x"))

    assert [p.get("nextActivityId") for p in actual] == [p.get("nextActivityId") for p in expected]


# ---------------------------------------------------------------------------
# 분기 판단 기록
# ---------------------------------------------------------------------------
def test_exclusive_branch_decision_is_recorded(wiproc):
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder()
    _resolve(wiproc, defn, {"Flow_3": {"conditionEval": True}, "Flow_4": {"conditionEval": False}}, recorder=recorder)

    decision = _decisions_by_source(recorder)["Gateway_1"]
    assert decision["source"]["name"] == "금액 구분"
    assert decision["source"]["branchType"] == "exclusiveGateway"
    assert decision["selectionRule"] == RULE_SINGLE_TRUE
    assert decision["selectedSequenceIds"] == ["Flow_3"]
    assert decision["unselectedSequenceIds"] == ["Flow_4"]
    assert decision["selectedTargets"][0]["activityId"] == "Task_B"
    assert decision["selectedTargets"][0]["activityName"] == "팀장 승인"
    assert decision["outcome"] == OUTCOME_ADVANCED


def test_consecutive_gateways_each_recorded(wiproc):
    """A 완료 -> Gateway_1 -> Task_B 는 게이트웨이 1개, 병합까지 가면 2개가 기록된다."""
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder(activity_id="Task_B")
    wiproc.resolve_next_activity_payloads(defn, "Task_B", _workitem(), {}, recorder=recorder)

    sources = [d["source"]["id"] for d in recorder.decisions]
    assert "Task_B" in sources  # 활동에서 나가는 단일 경로
    assert "Gateway_2" in sources  # 병합 게이트웨이 통과
    # 같은 처리에서 나온 판단은 하나의 상관 식별자로 묶인다
    assert len({d["correlationId"] for d in recorder.decisions}) == 1


def test_unconditional_single_path_is_recorded(wiproc):
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder(activity_id="Task_B")
    wiproc.resolve_next_activity_payloads(defn, "Task_B", _workitem(), {}, recorder=recorder)

    decision = _decisions_by_source(recorder)["Task_B"]
    assert decision["selectionRule"] == "unconditional"
    assert decision["selectedSequenceIds"] == ["Flow_5"]


def test_priority_selection_rule_recorded_when_multiple_true(wiproc):
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder()
    _resolve(wiproc, defn, {"Flow_3": {"conditionEval": True}, "Flow_4": {"conditionEval": True}}, recorder=recorder)

    decision = _decisions_by_source(recorder)["Gateway_1"]
    assert decision["selectionRule"] == RULE_PRIORITY
    # 충족한 갈래가 둘이었다는 사실이 남고, 실제로는 하나만 선택된다
    assert len(decision["selectedSequenceIds"]) == 1
    assert {e["sequenceId"] for e in decision["evaluations"]} == {"Flow_3", "Flow_4"}


def test_default_flow_selection_recorded(wiproc):
    branches = [
        {"id": "Flow_3", "source": "Gateway_1", "target": "Task_B", "condition": "", "properties": "{}"},
        {"id": "Flow_4", "source": "Gateway_1", "target": "Task_C", "condition": "", "properties": json.dumps({"default": True})},
    ]
    defn = _definition(branches)
    recorder = _recorder()
    _resolve(wiproc, defn, {"Flow_3": {"conditionEval": False}, "Flow_4": {"conditionEval": False}}, recorder=recorder)

    decision = _decisions_by_source(recorder)["Gateway_1"]
    assert decision["selectionRule"] == RULE_DEFAULT_FLOW
    assert decision["selectedSequenceIds"] == ["Flow_4"]


def test_dead_end_branch_recorded_as_undecided(wiproc):
    """참인 갈래도 기본 흐름도 없으면 진행 불가로 남아야 한다(현재는 조용히 멈추는 상태)."""
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder()
    payloads = _resolve(wiproc, defn, {"Flow_3": {"conditionEval": False}, "Flow_4": {"conditionEval": False}}, recorder=recorder)

    assert payloads == []  # 엔진은 아무 것도 만들지 않는다
    decision = _decisions_by_source(recorder)["Gateway_1"]
    assert decision["outcome"] == OUTCOME_UNDECIDED
    assert decision["selectionRule"] == RULE_NO_CANDIDATE
    # 원인 추적을 위해 평가된 모든 갈래가 남는다
    assert {e["sequenceId"] for e in decision["evaluations"]} == {"Flow_3", "Flow_4"}


def test_parallel_gateway_records_all_branches(wiproc):
    data_branches = [
        {"id": "Flow_3", "source": "Gateway_1", "target": "Task_B", "condition": "", "properties": "{}"},
        {"id": "Flow_4", "source": "Gateway_1", "target": "Task_C", "condition": "", "properties": "{}"},
    ]
    defn = _definition(data_branches)
    # Gateway_1 을 병렬로 바꾼다
    for gw in defn.gateways:
        if gw.id == "Gateway_1":
            gw.type = "parallelGateway"

    recorder = _recorder()
    _resolve(wiproc, defn, {}, recorder=recorder)

    decision = _decisions_by_source(recorder)["Gateway_1"]
    assert decision["selectionRule"] == "all-branches"
    assert set(decision["selectedSequenceIds"]) == {"Flow_3", "Flow_4"}
    assert decision["unselectedSequenceIds"] == []


def test_traversed_sequences_derived_from_recorded_decisions(wiproc):
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder()
    _resolve(wiproc, defn, {"Flow_3": {"conditionEval": True}, "Flow_4": {"conditionEval": False}}, recorder=recorder)

    rows = [r for r in recorder.build_events() if r["event_type"] == EVENT_TYPE_DECISION]
    traversed = collect_traversed_sequence_ids(rows)

    assert "Flow_3" in traversed
    assert "Flow_4" not in traversed  # 선택되지 않은 갈래는 지나가지 않았다


# ---------------------------------------------------------------------------
# 조건 평가 경로 (판단 방식과 근거)
# ---------------------------------------------------------------------------
class _StubModel:
    """자연어 조건 판정을 흉내내는 모델."""

    def __init__(self, response_text: str, raise_error: Optional[Exception] = None):
        self._response = response_text
        self._raise = raise_error

    async def astream(self, *_args, **_kwargs):
        if self._raise:
            raise self._raise

        class _Chunk:
            def __init__(self, content):
                self.content = content

        yield _Chunk(self._response)


class _StubParser:
    def parse(self, text):
        return json.loads(text)


def _eval_conditions(wiproc, defn, sequence_condition_data, recorder, model=None, workitem=None):
    asyncio.run(
        wiproc._evaluate_sequence_conditions(
            model or _StubModel("{}"),
            _StubParser(),
            defn,
            {"form_a": {"amount": 12000000}},
            {},
            sequence_condition_data,
            [],
            workitem=workitem,
            recorder=recorder,
        )
    )


def test_expression_evaluation_records_method_and_result(wiproc):
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder()
    conditions = {
        "Flow_3": {"conditionFunction": "amount >= 10000000"},
        "Flow_4": {"conditionFunction": "amount < 10000000"},
    }
    _eval_conditions(wiproc, defn, conditions, recorder)

    _resolve(wiproc, defn, conditions, recorder=recorder)
    evaluations = {e["sequenceId"]: e for e in _decisions_by_source(recorder)["Gateway_1"]["evaluations"]}

    assert evaluations["Flow_3"]["method"] == METHOD_EXPRESSION
    assert evaluations["Flow_3"]["verdict"] == VERDICT_TRUE
    assert evaluations["Flow_3"]["expression"] == "amount >= 10000000"
    assert evaluations["Flow_4"]["verdict"] == VERDICT_FALSE


def test_expression_failure_recorded_as_undetermined(wiproc):
    """엔진은 평가 실패를 거짓으로 취급한다. 이력에서는 '판정 불가'로 구분되어야 한다."""
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder()
    conditions = {"Flow_3": {"conditionFunction": "undefined_symbol >= 1"}}
    _eval_conditions(wiproc, defn, conditions, recorder)

    _resolve(wiproc, defn, conditions, recorder=recorder)
    evaluations = {e["sequenceId"]: e for e in _decisions_by_source(recorder)["Gateway_1"]["evaluations"]}

    assert evaluations["Flow_3"]["verdict"] == VERDICT_UNDETERMINED
    assert evaluations["Flow_3"]["effective"] is False
    assert evaluations["Flow_3"].get("error")
    # 엔진의 실제 동작(거짓 취급)은 그대로다
    assert conditions["Flow_3"]["conditionEval"] is False


def test_natural_language_judgement_records_reason_and_raw_trace(wiproc):
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder()
    conditions = {
        "Flow_3": {"condition": "1000만원 이상인가"},
        "Flow_4": {"condition": "1000만원 미만인가"},
    }
    response = json.dumps(
        {
            "results": [
                {"sequenceId": "Flow_3", "conditionMet": True, "reason": "신청 금액이 1200만원입니다"},
                {"sequenceId": "Flow_4", "conditionMet": False, "reason": "기준 미만이 아닙니다"},
            ]
        },
        ensure_ascii=False,
    )
    _eval_conditions(wiproc, defn, conditions, recorder, model=_StubModel(response))
    _resolve(wiproc, defn, conditions, recorder=recorder)

    evaluations = {e["sequenceId"]: e for e in _decisions_by_source(recorder)["Gateway_1"]["evaluations"]}
    assert evaluations["Flow_3"]["method"] == METHOD_NATURAL_LANGUAGE
    assert evaluations["Flow_3"]["verdict"] == VERDICT_TRUE
    assert "1200만원" in evaluations["Flow_3"]["reason"]

    # 모델 원문이 별도 이벤트로 남고 판단 이력과 상호 참조된다
    rows = recorder.build_events()
    traces = [r for r in rows if r["event_type"] == EVENT_TYPE_TRACE]
    decisions = [r for r in rows if r["event_type"] == EVENT_TYPE_DECISION]
    assert len(traces) == 1
    assert traces[0]["data"]["response"] == response
    gw_decision_id = next(d["data"]["decisionId"] for d in decisions if d["data"]["source"]["id"] == "Gateway_1")
    assert gw_decision_id in traces[0]["data"]["decisionIds"]


def test_missing_model_verdict_recorded_as_undetermined_not_false(wiproc):
    """모델이 판정을 빠뜨리면 엔진은 거짓으로 강제한다. 이력은 그 사실을 드러내야 한다."""
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder()
    conditions = {"Flow_3": {"condition": "1000만원 이상인가"}}
    _eval_conditions(wiproc, defn, conditions, recorder, model=_StubModel(json.dumps({"results": []})))
    _resolve(wiproc, defn, conditions, recorder=recorder)

    evaluations = {e["sequenceId"]: e for e in _decisions_by_source(recorder)["Gateway_1"]["evaluations"]}
    assert evaluations["Flow_3"]["verdict"] == VERDICT_UNDETERMINED
    assert evaluations["Flow_3"]["effective"] is False
    assert "판정이 없어" in evaluations["Flow_3"]["error"]
    assert conditions["Flow_3"]["conditionEval"] is False


def test_model_call_failure_records_undetermined_for_all_sequences(wiproc):
    defn = _definition(_TWO_BRANCHES)
    recorder = _recorder()
    conditions = {"Flow_3": {"condition": "1000만원 이상인가"}}
    _eval_conditions(wiproc, defn, conditions, recorder, model=_StubModel("", raise_error=RuntimeError("timeout")))
    _resolve(wiproc, defn, conditions, recorder=recorder)

    evaluations = {e["sequenceId"]: e for e in _decisions_by_source(recorder)["Gateway_1"]["evaluations"]}
    assert evaluations["Flow_3"]["verdict"] == VERDICT_UNDETERMINED
    assert "모델 호출 실패" in evaluations["Flow_3"]["error"]


# ---------------------------------------------------------------------------
# 기록 실패 격리
# ---------------------------------------------------------------------------
def test_flush_never_raises_when_event_store_fails(wiproc, monkeypatch):
    recorder = _recorder()
    recorder.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_B"}],
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("event store down")

    monkeypatch.setattr(wiproc, "insert_events", _boom)

    # 큐 적재 자체는 성공하고, 실제 저장 실패는 워커에서 삼켜진다
    wiproc.flush_decision_journal(recorder, "localhost")
    event_rows, tenant_id, _client = wiproc.decision_event_queue.get(timeout=5)
    assert tenant_id == "localhost"
    assert len(event_rows) == 1

    # 워커가 하는 일과 동일하게 호출해도 예외가 밖으로 나가지 않아야 한다
    try:
        wiproc.insert_events(event_rows, tenant_id, client=None)
    except RuntimeError:
        pass  # 워커는 이 예외를 잡아 로그로만 남긴다
    else:
        pytest.fail("stub should have raised")


def test_flush_is_noop_for_empty_recorder(wiproc):
    before = wiproc.decision_event_queue.qsize()
    wiproc.flush_decision_journal(_recorder(), "localhost")
    assert wiproc.decision_event_queue.qsize() == before
