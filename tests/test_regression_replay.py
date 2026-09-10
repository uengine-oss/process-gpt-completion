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

    mechanical = {
        dmn_replay.CHECK_OUTCOME_EQUALS,
        dmn_replay.CHECK_MATCHED_RULE,
        dmn_replay.CHECK_MATCHED_RULE_ID,
    }
    for c in cases:
        assert len(c["checks"]) == len(c["assertions"])
        assert {x["type"] for x in c["checks"]} <= mechanical
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


# --------------------------------------------------------------------------- #
# 결정이 여럿인 DMN
# --------------------------------------------------------------------------- #
_MULTI_DMN = """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" id="d" name="혜택">
  <decision id="decision_base" name="기본 혜택 결정">
    <decisionTable id="t1" hitPolicy="FIRST">
      <input id="i1"><inputExpression typeRef="string"><text>고객등급</text></inputExpression></input>
      <output id="o1" typeRef="number" />
      <rule id="r1"><inputEntry id="r1i"><text>"VIP"</text></inputEntry><outputEntry id="r1o"><text>20</text></outputEntry></rule>
      <rule id="r2"><inputEntry id="r2i"><text>"GOLD"</text></inputEntry><outputEntry id="r2o"><text>10</text></outputEntry></rule>
    </decisionTable>
  </decision>
  <decision id="decision_discount" name="할인율 산정">
    <decisionTable id="t2" hitPolicy="FIRST">
      <input id="i2"><inputExpression typeRef="number"><text>주문금액</text></inputExpression></input>
      <output id="o2" typeRef="number" />
      <rule id="r3"><inputEntry id="r3i"><text>&gt;= 1000000</text></inputEntry><outputEntry id="r3o"><text>5</text></outputEntry></rule>
    </decisionTable>
  </decision>
</definitions>"""


def _multi_dmn_with_discount(threshold: int) -> str:
    return _MULTI_DMN.replace("&gt;= 1000000", f"&gt;= {threshold}")


def test_scenarios_cover_every_decision_not_just_the_first():
    """첫 결정만 덮으면 나머지 표는 규칙이 통째로 바뀌어도 검증이 아무 말도 못 한다."""
    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(_MULTI_DMN))

    decisions = {json.loads(c["prompt"]).get("decision_name") for c in cases}
    assert decisions == {"기본 혜택 결정", "할인율 산정"}
    # 결정이 여럿이면 이름이 겹치지 않게 결정 이름을 앞에 붙인다.
    assert any(c["eval_name"].startswith("할인율 산정 · ") for c in cases)
    assert len({c["eval_name"] for c in cases}) == len(cases)


def test_each_case_is_judged_against_its_own_decision():
    """시나리오는 자기 결정의 표로 채점돼야 한다 — 첫 표로 재면 전부 매칭 없음이 된다."""
    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(_MULTI_DMN))
    discount = next(c for c in cases if c["eval_name"].startswith("할인율 산정 · 규칙 1행"))

    observed, graded = dmn_replay.evaluate_case(_MULTI_DMN, discount)

    assert graded["passed"] == graded["total"] == 2
    assert observed["outcome"] == "5"


def test_a_change_in_the_second_decision_is_caught():
    """두 번째 표의 임계값을 옮기면 그 경계 시나리오가 깨져야 한다."""
    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(_MULTI_DMN))
    boundary = next(
        c for c in cases
        if c["eval_name"].startswith("할인율 산정 · ") and json.loads(c["prompt"])["inputs"].get("주문금액") == 1000000
    )

    _, base_g = dmn_replay.evaluate_case(_MULTI_DMN, boundary)
    _, head_g = dmn_replay.evaluate_case(_multi_dmn_with_discount(2000000), boundary)

    assert base_g["passed"] == 2
    assert head_g["passed"] < 2


def test_a_removed_decision_fails_instead_of_being_undecided():
    """결정이 통째로 사라진 것도 동작 변화다. '판단 불가' 로 접으면 병합해도 되는 줄 안다 —
    실제로 그 결정을 부르는 쪽은 아무 결론도 받지 못한다."""
    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(_MULTI_DMN))
    discount = next(c for c in cases if c["eval_name"].startswith("할인율 산정 · 규칙 1행"))

    without_discount = _MULTI_DMN[: _MULTI_DMN.index('  <decision id="decision_discount"')] + "</definitions>"
    observed, graded = dmn_replay.evaluate_case(without_discount, discount)

    assert graded is not None, "사라진 결정은 판단 불가가 아니라 실패로 잡혀야 한다"
    assert graded["passed"] == 0
    assert "할인율 산정" in observed["note"]


def test_old_scenarios_without_a_decision_still_run_against_the_first_table():
    """예전에 만들어 둔 시나리오에는 decision 정보가 없다 — 첫 표로 재던 그대로 돈다."""
    legacy = {
        "eval_name": "규칙 1행 — 20",
        "prompt": json.dumps({"inputs": {"고객등급": "VIP"}}, ensure_ascii=False),
        "assertions": ["결론이 '20' 이다"],
        "checks": [{"type": dmn_replay.CHECK_OUTCOME_EQUALS, "value": "20"}],
    }

    observed, graded = dmn_replay.evaluate_case(_MULTI_DMN, legacy)

    assert observed["outcome"] == "20"
    assert graded["passed"] == 1


# --------------------------------------------------------------------------- #
# 행이 밀리는 것은 동작 변화가 아니다
# --------------------------------------------------------------------------- #
_SHIPPING_DMN = """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="https://www.omg.org/spec/DMN/20191111/MODEL/" id="d" name="배송">
  <decision id="decision_shipping" name="배송비 결정">
    <decisionTable id="t1" hitPolicy="FIRST">
      <input id="i1"><inputExpression typeRef="string"><text>region</text></inputExpression></input>
      <input id="i2"><inputExpression typeRef="number"><text>orderAmount</text></inputExpression></input>
      <output id="o1" typeRef="number" />
      RULES
    </decisionTable>
  </decision>
</definitions>"""

_RULE_SEOUL_FREE = '<rule id="rule_1"><inputEntry id="a"><text>"SEOUL"</text></inputEntry><inputEntry id="b"><text>&gt;= 50000</text></inputEntry><outputEntry id="c"><text>0</text></outputEntry></rule>'
_RULE_JEJU = '<rule id="rule_3"><inputEntry id="g"><text>"JEJU"</text></inputEntry><inputEntry id="h"><text>-</text></inputEntry><outputEntry id="i"><text>7000</text></outputEntry></rule>'
_RULE_JEJU_BIG = '<rule id="rule_6"><inputEntry id="j"><text>"JEJU"</text></inputEntry><inputEntry id="k"><text>&gt;= 100000</text></inputEntry><outputEntry id="l"><text>4000</text></outputEntry></rule>'
_RULE_DEFAULT_FREE = '<rule id="rule_4"><inputEntry id="m"><text>-</text></inputEntry><inputEntry id="n"><text>&gt;= 50000</text></inputEntry><outputEntry id="o"><text>0</text></outputEntry></rule>'


def _shipping(rules: list[str]) -> str:
    return _SHIPPING_DMN.replace("RULES", "\n      ".join(rules))


def test_inserting_a_rule_above_does_not_break_the_rows_below():
    """제주도 조건을 표 중간에 끼워 넣으면 아래 행이 한 칸씩 밀린다.

    밀린 행은 **같은 규칙이 같은 결론을 내는데도** 예전에는 "규칙 4행이 아니라 5행이
    적용됐다" 는 이유로 깨진 것으로 잡혔다. 자리는 바뀌어도 규칙은 그대로다.
    """
    before = _shipping([_RULE_SEOUL_FREE, _RULE_JEJU, _RULE_DEFAULT_FREE])
    after = _shipping([_RULE_SEOUL_FREE, _RULE_JEJU, _RULE_JEJU_BIG, _RULE_DEFAULT_FREE])

    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(before))
    default_case = next(c for c in cases if json.loads(c["expected_output"]).get("matched_rule_id") == "rule_4")

    base_obs, base_g = dmn_replay.evaluate_case(before, default_case)
    head_obs, head_g = dmn_replay.evaluate_case(after, default_case)

    # 행 번호는 밀렸지만 같은 규칙이 같은 결론을 낸다.
    assert base_obs["matched_rule_index"] != head_obs["matched_rule_index"]
    assert base_obs["matched_rule_id"] == head_obs["matched_rule_id"] == "rule_4"
    assert base_g["passed"] == base_g["total"]
    assert head_g["passed"] == head_g["total"], "행이 밀린 것만으로 깨졌다고 하면 안 된다"


def test_a_different_rule_taking_over_is_still_caught():
    """자리가 아니라 정체를 보므로, 결론이 우연히 같아도 다른 규칙이 대신 맞으면 잡는다."""
    before = _shipping([_RULE_SEOUL_FREE, _RULE_DEFAULT_FREE])
    # 서울 무료배송 규칙을 지우면 같은 입력이 기본 규칙(rule_4)에 맞는다 — 결론은 0 으로 같다.
    after = _shipping([_RULE_DEFAULT_FREE])

    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(before))
    seoul_case = next(c for c in cases if json.loads(c["expected_output"]).get("matched_rule_id") == "rule_1")

    head_obs, head_g = dmn_replay.evaluate_case(after, seoul_case)

    assert head_obs["outcome"] == "0", "결론은 우연히 같다"
    assert head_g["passed"] < head_g["total"], "다른 규칙이 대신 맞은 것은 드러나야 한다"


def test_tables_without_rule_ids_fall_back_to_row_numbers():
    """id 가 없는 표(예전 편집기 산출물)에서는 예전처럼 행 번호로 건다."""
    no_ids = _SHIPPING_DMN.replace("RULES", (
        '<rule><inputEntry><text>"SEOUL"</text></inputEntry>'
        '<inputEntry><text>&gt;= 50000</text></inputEntry>'
        '<outputEntry><text>0</text></outputEntry></rule>'
    ))

    cases = dmn_derive.to_eval_cases(dmn_derive.derive_cases(no_ids))
    kinds = {c["type"] for case in cases for c in case["checks"]}

    assert dmn_replay.CHECK_MATCHED_RULE in kinds
    _, graded = dmn_replay.evaluate_case(no_ids, cases[0])
    assert graded["passed"] == graded["total"]
