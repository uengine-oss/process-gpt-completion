"""회귀 검증 실행기 — 프로세스 경로 재생과 의사결정 표 평가를 고정한다.

실행기가 이 서비스에 있는 이유는 **판정 규칙의 진실 원천이 여기**이기 때문이다. 프로세스
라우팅은 폴링 서비스가, DMN 표 해석은 이 서비스가 기준이다. 사본을 다른 서비스에 두면
엔진이 규칙을 고칠 때 회귀 테스트만 옛 규칙으로 통과시킨다.

여기서 지키는 것:

- 재생에 모델도 실행 엔진 인스턴스도 쓰지 않는다.
- 판정 근거가 없으면 추측하지 않고 판정 불가로 남긴다. 0점으로 매기면 "이 변경으로 깨졌다"
  로 읽혀 확인 못 한 것과 깨진 것이 섞인다.
- 실제로 결과가 달라지는 변경(분기 대상 변경, 임계값 이동)을 잡아낸다.
"""

from __future__ import annotations

import json

from regression import dmn_derive, dmn_replay, process_replay


# --------------------------------------------------------------------------- 프로세스
def _definition(deep_target="deep_review"):
    return {
        "processDefinitionId": "mini",
        "activities": [{"id": a, "type": "userTask"} for a in
                       ("apply", "deep_review", "simple_review", "register")],
        "gateways": [{"id": "gw_amount", "type": "exclusiveGateway"}],
        "events": [{"id": "start_event", "type": "startEvent"},
                   {"id": "end_event", "type": "endEvent"}],
        "sequences": [
            {"id": "s_start_apply", "source": "start_event", "target": "apply"},
            {"id": "s_apply_gw", "source": "apply", "target": "gw_amount"},
            {"id": "s_gw_deep", "source": "gw_amount", "target": deep_target},
            {"id": "s_gw_simple", "source": "gw_amount", "target": "simple_review"},
            {"id": "s_deep_register", "source": "deep_review", "target": "register"},
            {"id": "s_simple_register", "source": "simple_review", "target": "register"},
            {"id": "s_register_end", "source": "register", "target": "end_event"},
        ],
    }


def _process_case(selected="s_gw_deep", path=("apply", "deep_review", "register")):
    return {
        "eval_name": "고액 승인 경로",
        "prompt": json.dumps({
            "activity_inputs": {"apply": {"amount": 700000}},
            "gateway_decisions": {"gw_amount": {"selected": [selected]}},
        }, ensure_ascii=False),
        "assertions": ["실행 경로가 " + " → ".join(path) + " 다", "프로세스가 endEvent 까지 진행된다"],
        "checks": [{"type": "path_equals", "value": list(path)},
                   {"type": "reached_end", "value": True}],
    }


def test_replay_follows_the_recorded_branch():
    observed = process_replay.replay(_definition(), json.loads(_process_case()["prompt"]))

    assert observed["activity_order"] == ["apply", "deep_review", "register"]
    assert observed["reached_end"] is True
    assert observed["undecided"] is None


def test_gateways_and_events_are_not_part_of_the_path():
    observed = process_replay.replay(_definition(), json.loads(_process_case()["prompt"]))

    assert "gw_amount" not in observed["activity_order"]
    assert "start_event" not in observed["activity_order"]


def test_missing_decision_is_undecided_not_a_failure():
    """기록이 없으면 추측하지 않는다.

    경로만 보고 역산하면 두 분기가 같은 액티비티로 향할 때 구분할 수 없고, 그 상태로 기준을
    굳히면 깨진 변경을 "이상 없음" 으로 통과시킨다.
    """
    case = _process_case()
    observed = process_replay.replay(_definition(), {"gateway_decisions": {}})

    assert observed["undecided"]
    assert process_replay.grade(observed, case["assertions"], case["checks"]) is None


def test_parallel_gateway_is_not_replayed():
    """합류 규칙이 필요한 분기는 재생 대상이 아니다 — 엔진 규칙의 사본을 만들지 않는다."""
    definition = _definition()
    definition["gateways"][0]["type"] = "parallelGateway"

    observed = process_replay.replay(definition, json.loads(_process_case()["prompt"]))

    assert observed["undecided"]
    assert "실엔진 모드" in observed["undecided"]


def test_changed_branch_target_is_detected():
    case = _process_case()
    base = process_replay.replay(_definition(), json.loads(case["prompt"]))
    head = process_replay.replay(_definition("simple_review"), json.loads(case["prompt"]))

    base_g = process_replay.grade(base, case["assertions"], case["checks"])
    head_g = process_replay.grade(head, case["assertions"], case["checks"])
    assert base_g["assertions"][0]["passed"] is True
    assert head_g["assertions"][0]["passed"] is False


def test_removed_branch_is_reported_as_gone():
    definition = _definition()
    definition["sequences"] = [s for s in definition["sequences"] if s["id"] != "s_gw_deep"]

    observed = process_replay.replay(definition, json.loads(_process_case()["prompt"]))

    assert "사라졌습니다" in observed["undecided"]


def test_replay_case_reads_the_stored_prompt():
    observed, graded = process_replay.replay_case(_definition(), _process_case())

    assert observed["activity_order"] == ["apply", "deep_review", "register"]
    assert graded["passed"] == 2


# --------------------------------------------------------------------------- 의사결정
def _dmn(threshold=500000, high_outcome="정밀검토"):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" id="d1" name="금액 판정">
  <decision id="dec_amount" name="금액 판정">
    <decisionTable id="t1" hitPolicy="FIRST">
      <input id="i1" label="금액">
        <inputExpression id="ie1" typeRef="number"><text>amount</text></inputExpression>
      </input>
      <output id="o1" label="결론" typeRef="string" />
      <annotation name="note" />
      <rule id="r1">
        <inputEntry><text>&gt;= {threshold}</text></inputEntry>
        <outputEntry><text>"{high_outcome}"</text></outputEntry>
        <annotationEntry><text>고액 건</text></annotationEntry>
      </rule>
      <rule id="r2">
        <inputEntry><text>&lt;= {threshold - 1}</text></inputEntry>
        <outputEntry><text>"간이검토"</text></outputEntry>
        <annotationEntry><text>소액 건</text></annotationEntry>
      </rule>
    </decisionTable>
  </decision>
</definitions>"""


def test_table_is_read_from_dmn_xml():
    table = dmn_replay.parse_decision_table(_dmn())

    assert [i["item"] for i in table["inputs"]] == ["amount"]
    assert table["rules"][0]["outcome"] == "정밀검토"
    assert table["rules"][0]["conditions"][0] == {"key": "amount", "operator": "gte", "value": 500000}


def test_first_matching_rule_wins():
    table = dmn_replay.parse_decision_table(_dmn())

    assert dmn_replay.evaluate(table, {"amount": 700000})["outcome"] == "정밀검토"
    assert dmn_replay.evaluate(table, {"amount": 300000})["outcome"] == "간이검토"


def test_boundary_belongs_to_the_gte_rule():
    """경계 판정이 화면 매처와 갈리면 안 된다."""
    table = dmn_replay.parse_decision_table(_dmn())

    assert dmn_replay.evaluate(table, {"amount": 500000})["matched_rule_index"] == 0
    assert dmn_replay.evaluate(table, {"amount": 499999})["matched_rule_index"] == 1


def test_string_input_is_compared_as_number():
    """폼 값은 문자열로 저장된다 — 숫자 비교가 문자열이라고 실패하면 안 된다."""
    table = dmn_replay.parse_decision_table(_dmn())

    assert dmn_replay.evaluate(table, {"amount": "700000"})["outcome"] == "정밀검토"


def test_no_match_is_not_an_outcome():
    table = dmn_replay.parse_decision_table(_dmn())

    observed = dmn_replay.evaluate(table, {"amount": "금액아님"})

    assert observed["matched_rule_index"] == -1
    assert observed["outcome"] == ""


def test_cases_are_derived_without_a_model():
    cases = dmn_derive.derive_cases(_dmn())

    outcomes = {c["observed"]["outcome"] for c in cases}
    assert {"정밀검토", "간이검토"} <= outcomes


def test_derived_cases_include_a_boundary_probe():
    """임계값을 옮기는 변경은 경계 바로 옆 입력에서만 드러난다."""
    amounts = {c["inputs"].get("amount") for c in dmn_derive.derive_cases(_dmn())}

    assert 500000 in amounts
    assert 499999 in amounts


def test_eval_cases_carry_only_mechanical_checks():
    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(_dmn()))

    for c in cases:
        assert len(c["checks"]) == len(c["assertions"])
        assert {x["type"] for x in c["checks"]} == {
            dmn_replay.CHECK_OUTCOME_EQUALS, dmn_replay.CHECK_MATCHED_RULE
        }
        assert c["files"] == []


def test_moved_threshold_breaks_the_boundary_case():
    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(_dmn(threshold=500000)))
    boundary = next(c for c in cases if json.loads(c["prompt"])["inputs"].get("amount") == 500000)

    _, base_g = dmn_replay.evaluate_case(_dmn(threshold=500000), boundary)
    _, head_g = dmn_replay.evaluate_case(_dmn(threshold=600000), boundary)

    assert base_g["passed"] == 2
    assert head_g["passed"] < 2


def test_unreadable_head_is_undecided_not_a_failure():
    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(_dmn()))

    observed, graded = dmn_replay.evaluate_case("<broken/>", cases[0])

    assert graded is None
    assert observed["undecided"]


def test_grade_refuses_checks_it_cannot_judge():
    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(_dmn()))
    observed, _ = dmn_replay.evaluate_case(_dmn(), cases[0])

    graded = dmn_replay.grade(
        observed,
        cases[0]["assertions"] + ["설명이 그럴듯하다"],
        cases[0]["checks"] + [{"type": "contains", "value": "승인"}],
    )

    assert graded is None
