"""분기 판단 이력(Gateway Decision Journal) 단위 테스트.

`decision_journal` 은 표준 라이브러리에만 의존하므로 무거운 스텁 없이 바로 임포트한다.
"""

import json

import pytest

from decision_journal import (
    DecisionRecorder,
    EVENT_TYPE_DECISION,
    EVENT_TYPE_TRACE,
    METHOD_EXPRESSION,
    METHOD_NATURAL_LANGUAGE,
    OUTCOME_ADVANCED,
    OUTCOME_UNDECIDED,
    OUTCOME_WAITING,
    RULE_DEFAULT_FLOW,
    RULE_NO_CANDIDATE,
    RULE_SINGLE_TRUE,
    VERDICT_FALSE,
    VERDICT_TRUE,
    VERDICT_UNDETERMINED,
    collect_traversed_sequence_ids,
    normalize_verdict,
    truncate_payload,
)


def _recorder(**overrides):
    """결정적 id/시각을 쓰는 레코더."""
    counter = {"n": 0}

    def _ids():
        counter["n"] += 1
        return f"id-{counter['n']}"

    kwargs = dict(
        proc_inst_id="inst-1",
        root_proc_inst_id="inst-1",
        proc_def_id="def-1",
        proc_def_version="3",
        activity_id="Task_A",
        workitem_id="wi-1",
        tenant_id="acme",
        execution_scope="scope-1",
        rework_count=0,
        clock=lambda: "2026-08-06T00:00:00+00:00",
        id_factory=_ids,
    )
    kwargs.update(overrides)
    return DecisionRecorder(**kwargs)


def _decision_events(rows):
    return [r for r in rows if r["event_type"] == EVENT_TYPE_DECISION]


def _trace_events(rows):
    return [r for r in rows if r["event_type"] == EVENT_TYPE_TRACE]


# ---------------------------------------------------------------------------
# 판정 결과 정규화 — 참/거짓과 "판정 불가"의 구분
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (True, VERDICT_TRUE),
        (False, VERDICT_FALSE),
        ("yes", VERDICT_TRUE),
        ("NO", VERDICT_FALSE),
        (1, VERDICT_TRUE),
        (0, VERDICT_FALSE),
        (None, VERDICT_UNDETERMINED),
        ("", VERDICT_UNDETERMINED),
        ("maybe", VERDICT_UNDETERMINED),
    ],
)
def test_normalize_verdict_distinguishes_undetermined(value, expected):
    assert normalize_verdict(value) == expected


# ---------------------------------------------------------------------------
# 분기 판단 이력 기록
# ---------------------------------------------------------------------------
def test_exclusive_branch_records_selected_and_unselected():
    rec = _recorder()
    rec.record_sequence_evaluation("Flow_3", method=METHOD_EXPRESSION, verdict=VERDICT_TRUE, effective=True, expression="amount >= 100")
    rec.record_sequence_evaluation("Flow_4", method=METHOD_EXPRESSION, verdict=VERDICT_FALSE, effective=False, expression="amount < 100")
    rec.record_decision(
        source_id="Gateway_1",
        source_name="금액 구분",
        source_type="gateway",
        branch_type="exclusiveGateway",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3", "Flow_4"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_B", "activityName": "팀장 승인"}],
    )

    rows = _decision_events(rec.build_events())
    assert len(rows) == 1
    data = rows[0]["data"]

    assert data["source"] == {"id": "Gateway_1", "name": "금액 구분", "type": "gateway", "branchType": "exclusiveGateway"}
    assert data["selectedSequenceIds"] == ["Flow_3"]
    assert data["unselectedSequenceIds"] == ["Flow_4"]
    assert data["selectedTargets"] == [{"activityId": "Task_B", "activityName": "팀장 승인"}]
    assert data["outcome"] == OUTCOME_ADVANCED
    assert data["selectionRule"] == RULE_SINGLE_TRUE

    by_seq = {e["sequenceId"]: e for e in data["evaluations"]}
    assert by_seq["Flow_3"]["selected"] is True
    assert by_seq["Flow_4"]["selected"] is False
    assert by_seq["Flow_3"]["expression"] == "amount >= 100"


def test_decision_carries_instance_coordinates():
    rec = _recorder()
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_1"],
        selected_sequence_ids=["Flow_1"],
        selected_targets=[{"activityId": "Task_B"}],
    )
    data = _decision_events(rec.build_events())[0]["data"]

    assert data["procInstId"] == "inst-1"
    assert data["rootProcInstId"] == "inst-1"
    assert data["procDefId"] == "def-1"
    assert data["procDefVersion"] == "3"
    assert data["triggerActivityId"] == "Task_A"
    assert data["workitemId"] == "wi-1"
    assert data["executionScope"] == "scope-1"
    assert data["reworkCount"] == 0
    assert data["decidedBy"] == "system"


def test_consecutive_gateways_produce_one_event_each_sharing_correlation():
    rec = _recorder()
    for gw, seq, target in (("Gateway_1", "Flow_3", "Gateway_2"), ("Gateway_2", "Flow_7", "Task_D")):
        rec.record_decision(
            source_id=gw,
            selection_rule=RULE_SINGLE_TRUE,
            candidate_sequence_ids=[seq],
            selected_sequence_ids=[seq],
            selected_targets=[{"activityId": target}],
        )

    rows = _decision_events(rec.build_events())
    assert [r["data"]["source"]["id"] for r in rows] == ["Gateway_1", "Gateway_2"]
    # 같은 워크아이템 처리에서 나온 이벤트는 하나의 상관 식별자로 묶인다
    assert len({r["job_id"] for r in rows}) == 1
    assert len({r["data"]["correlationId"] for r in rows}) == 1


def test_unconditional_sequence_is_recorded_as_true_without_prior_evaluation():
    rec = _recorder()
    rec.record_decision(
        source_id="Task_A",
        source_type="activity",
        selection_rule="unconditional",
        candidate_sequence_ids=["Flow_2"],
        selected_sequence_ids=["Flow_2"],
        selected_targets=[{"activityId": "Task_B"}],
    )
    entry = _decision_events(rec.build_events())[0]["data"]["evaluations"][0]
    assert entry["method"] == "unconditional"
    assert entry["verdict"] == VERDICT_TRUE


# ---------------------------------------------------------------------------
# 판단 방식과 근거
# ---------------------------------------------------------------------------
def test_expression_failure_recorded_as_undetermined_not_false():
    """엔진은 평가 실패를 거짓으로 취급하지만, 이력에서는 반드시 구분되어야 한다."""
    rec = _recorder()
    rec.record_sequence_evaluation(
        "Flow_3",
        method=METHOD_EXPRESSION,
        verdict=VERDICT_UNDETERMINED,
        effective=False,
        expression="amount >= 100",
        error="NameError: amount",
    )
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_NO_CANDIDATE,
        candidate_sequence_ids=["Flow_3"],
        selected_sequence_ids=[],
        selected_targets=[],
    )
    entry = _decision_events(rec.build_events())[0]["data"]["evaluations"][0]

    assert entry["verdict"] == VERDICT_UNDETERMINED
    assert entry["effective"] is False
    assert "NameError" in entry["error"]


def test_natural_language_reason_is_recorded():
    rec = _recorder()
    rec.record_sequence_evaluation(
        "Flow_3",
        method=METHOD_NATURAL_LANGUAGE,
        verdict=VERDICT_TRUE,
        effective=True,
        expression="1000만원 이상인가",
        reason="신청 금액이 1200만원이므로 기준을 넘습니다",
    )
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_C"}],
    )
    entry = _decision_events(rec.build_events())[0]["data"]["evaluations"][0]

    assert entry["method"] == METHOD_NATURAL_LANGUAGE
    assert "1200만원" in entry["reason"]


def test_default_flow_selection_rule_is_recorded():
    rec = _recorder()
    rec.record_sequence_evaluation("Flow_3", method=METHOD_EXPRESSION, verdict=VERDICT_FALSE, effective=False)
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_DEFAULT_FLOW,
        candidate_sequence_ids=["Flow_3", "Flow_4"],
        selected_sequence_ids=["Flow_4"],
        selected_targets=[{"activityId": "Task_C"}],
    )
    assert _decision_events(rec.build_events())[0]["data"]["selectionRule"] == RULE_DEFAULT_FLOW


# ---------------------------------------------------------------------------
# 입력 스냅샷
# ---------------------------------------------------------------------------
def test_input_snapshot_is_recorded():
    rec = _recorder()
    rec.record_sequence_evaluation(
        "Flow_3",
        method=METHOD_EXPRESSION,
        verdict=VERDICT_TRUE,
        effective=True,
        input_snapshot={"form_a": {"amount": 12000000}},
    )
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_C"}],
    )
    entry = _decision_events(rec.build_events())[0]["data"]["evaluations"][0]
    assert entry["inputSnapshot"] == {"form_a": {"amount": 12000000}}
    assert "inputSnapshotTruncated" not in entry


def test_oversized_snapshot_is_truncated_but_record_survives():
    rec = _recorder(snapshot_limit=64)
    rec.record_sequence_evaluation(
        "Flow_3",
        method=METHOD_EXPRESSION,
        verdict=VERDICT_TRUE,
        effective=True,
        input_snapshot={"blob": "x" * 5000},
    )
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_C"}],
    )
    entry = _decision_events(rec.build_events())[0]["data"]["evaluations"][0]

    # 기록이 생략되지 않고, 잘렸다는 사실이 남는다
    assert entry["inputSnapshotTruncated"] is True
    assert entry["inputSnapshot"]["truncated"] is True
    assert entry["inputSnapshot"]["originalLength"] > 64
    assert len(entry["inputSnapshot"]["preview"]) == 64
    assert entry["verdict"] == VERDICT_TRUE


def test_truncate_payload_passes_through_small_values():
    value, truncated = truncate_payload({"a": 1}, 1000)
    assert value == {"a": 1}
    assert truncated is False


def test_events_carry_tenant_for_isolation():
    rec = _recorder(tenant_id="acme")
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_1"],
        selected_sequence_ids=["Flow_1"],
        selected_targets=[{"activityId": "Task_B"}],
    )
    assert all(row["tenant_id"] == "acme" for row in rec.build_events())


# ---------------------------------------------------------------------------
# 모델 원문 보존과 상호 참조
# ---------------------------------------------------------------------------
def test_llm_trace_is_separate_event_cross_referenced_with_decision():
    rec = _recorder()
    rec.record_llm_trace(prompt={"conditions": [{"sequenceId": "Flow_3"}]}, response='{"results": []}', sequence_ids=["Flow_3"])
    rec.record_sequence_evaluation("Flow_3", method=METHOD_NATURAL_LANGUAGE, verdict=VERDICT_TRUE, effective=True, reason="충족")
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_B"}],
    )

    rows = rec.build_events()
    decision = _decision_events(rows)[0]
    trace = _trace_events(rows)[0]

    assert trace["data"]["decisionIds"] == [decision["data"]["decisionId"]]
    assert decision["data"]["traceIds"] == [trace["data"]["traceId"]]
    assert trace["data"]["prompt"] == {"conditions": [{"sequenceId": "Flow_3"}]}
    assert trace["data"]["response"] == '{"results": []}'


def test_decision_is_self_contained_when_trace_is_dropped():
    """원문이 만료·삭제되어도 선택 결과와 근거는 판단 이력 본문만으로 읽혀야 한다."""
    rec = _recorder()
    rec.record_llm_trace(prompt={"x": 1}, response="{}", sequence_ids=["Flow_3"])
    rec.record_sequence_evaluation("Flow_3", method=METHOD_NATURAL_LANGUAGE, verdict=VERDICT_TRUE, effective=True, reason="금액 기준 충족")
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_B"}],
    )

    decision = _decision_events(rec.build_events())[0]["data"]
    assert decision["selectedSequenceIds"] == ["Flow_3"]
    assert decision["evaluations"][0]["reason"] == "금액 기준 충족"


def test_no_trace_event_when_only_expressions_evaluated():
    rec = _recorder()
    rec.record_sequence_evaluation("Flow_3", method=METHOD_EXPRESSION, verdict=VERDICT_TRUE, effective=True)
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_B"}],
    )
    assert _trace_events(rec.build_events()) == []


def test_oversized_trace_is_truncated():
    rec = _recorder(trace_limit=32)
    rec.record_llm_trace(prompt="p" * 500, response="r" * 500, sequence_ids=["Flow_3"])
    trace = _trace_events(rec.build_events())[0]["data"]
    assert trace["promptTruncated"] is True
    assert trace["responseTruncated"] is True


# ---------------------------------------------------------------------------
# 대기 / 진행 불가
# ---------------------------------------------------------------------------
def test_no_candidate_branch_is_recorded_as_undecided():
    rec = _recorder()
    rec.record_sequence_evaluation("Flow_3", method=METHOD_EXPRESSION, verdict=VERDICT_FALSE, effective=False)
    rec.record_sequence_evaluation("Flow_4", method=METHOD_EXPRESSION, verdict=VERDICT_FALSE, effective=False)
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_NO_CANDIDATE,
        candidate_sequence_ids=["Flow_3", "Flow_4"],
        selected_sequence_ids=[],
        selected_targets=[],
    )
    data = _decision_events(rec.build_events())[0]["data"]

    assert data["outcome"] == OUTCOME_UNDECIDED
    # 원인을 확인할 수 있도록 평가된 모든 갈래가 남는다
    assert {e["sequenceId"] for e in data["evaluations"]} == {"Flow_3", "Flow_4"}


def test_join_wait_marks_decision_as_waiting_with_reason():
    rec = _recorder()
    rec.record_decision(
        source_id="Gateway_2",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_7"],
        selected_sequence_ids=["Flow_7"],
        selected_targets=[{"activityId": "Task_D", "activityName": "결과 통보"}],
    )
    rec.mark_deferred(["Task_D"], waiting_for=[{"activity_id": "Task_C", "activity_name": "임원 승인", "status": "IN_PROGRESS"}])

    data = _decision_events(rec.build_events())[0]["data"]
    assert data["outcome"] == OUTCOME_WAITING
    assert data["deferredTargets"] == [{"activityId": "Task_D", "activityName": "결과 통보"}]
    assert data["waitingFor"][0]["activity_id"] == "Task_C"


def test_mark_deferred_leaves_unrelated_decisions_untouched():
    rec = _recorder()
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_B"}],
    )
    rec.mark_deferred(["Task_Z"], waiting_for=[])
    assert _decision_events(rec.build_events())[0]["data"]["outcome"] == OUTCOME_ADVANCED


# ---------------------------------------------------------------------------
# 지나간 경로 확정
# ---------------------------------------------------------------------------
def test_traversed_sequences_exclude_unselected_waiting_and_undecided():
    rec = _recorder()
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3", "Flow_4"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_B"}],
    )
    rec.record_decision(
        source_id="Gateway_2",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_7"],
        selected_sequence_ids=["Flow_7"],
        selected_targets=[{"activityId": "Task_D"}],
    )
    rec.record_decision(
        source_id="Gateway_3",
        selection_rule=RULE_NO_CANDIDATE,
        candidate_sequence_ids=["Flow_9"],
        selected_sequence_ids=[],
        selected_targets=[],
    )
    rec.mark_deferred(["Task_D"], waiting_for=[])

    traversed = collect_traversed_sequence_ids(_decision_events(rec.build_events()))

    assert traversed == ["Flow_3"]  # 미선택(Flow_4), 대기(Flow_7), 진행 불가(Flow_9) 모두 제외


def test_traversed_sequences_empty_for_instance_without_journal():
    assert collect_traversed_sequence_ids([]) == []


def test_empty_recorder_produces_no_events():
    rec = _recorder()
    assert rec.has_records() is False
    assert rec.build_events() == []


def test_event_rows_are_json_serializable():
    rec = _recorder()
    rec.record_sequence_evaluation("Flow_3", method=METHOD_EXPRESSION, verdict=VERDICT_TRUE, effective=True, input_snapshot={"a": 1})
    rec.record_llm_trace(prompt={"p": 1}, response="ok", sequence_ids=["Flow_3"])
    rec.record_decision(
        source_id="Gateway_1",
        selection_rule=RULE_SINGLE_TRUE,
        candidate_sequence_ids=["Flow_3"],
        selected_sequence_ids=["Flow_3"],
        selected_targets=[{"activityId": "Task_B"}],
    )
    for row in rec.build_events():
        json.dumps(row, ensure_ascii=False)
