"""프로세스 회귀 시나리오 파생 — 정의만 보고 만든 시나리오가 기준선으로 성립하는지 고정한다.

이 파생이 있는 이유는, 생성 시 실엔진 검증 게이트를 거치지 않고 만들어진 프로세스에는
병합 전에 비교할 시나리오가 아예 없기 때문이다. 병합 전 검증은 저장된 갈림길 판정으로
정의를 재생할 뿐이므로(`regression/process_replay.py`), 시나리오에 필요한 것은 실행
기록이 아니라 정의뿐이다.

여기서 지키는 것:

- 갈림길마다 분기를 하나씩 골라 훑되, 기대값은 **그 판정을 지금 정의에 재생한 결과**다.
  모델이 추론한 "올바른 순서" 를 기준으로 굳히면 아직 고치지 않은 결함이 정상으로 굳는다.
- 만들어진 시나리오는 그대로 재생·채점되어 통과해야 한다. 파생과 재생이 어긋나면 방금
  만든 기준선이 처음부터 깨진 것으로 나온다.
- 재생할 수 없는 갈림길(병렬·포함)은 조용히 넘기지 않고 사유를 남긴다.
- 모델도 실행 엔진 인스턴스도 쓰지 않는다.
"""

from __future__ import annotations

from regression import process_derive, process_replay


def _definition(*, deep_target: str = "deep_review") -> dict:
    """분기 하나짜리 최소 정의. 두 분기가 같은 액티비티로 합류한다."""
    return {
        "processDefinitionId": "mini",
        "activities": [{"id": a, "name": a, "type": "userTask"} for a in
                       ("apply", "deep_review", "simple_review", "register")],
        "gateways": [{"id": "gw_amount", "name": "금액 판정", "type": "exclusiveGateway"}],
        "events": [{"id": "start_event", "type": "startEvent"},
                   {"id": "end_event", "type": "endEvent"}],
        "sequences": [
            {"id": "s_start_apply", "source": "start_event", "target": "apply"},
            {"id": "s_apply_gw", "source": "apply", "target": "gw_amount"},
            {"id": "s_gw_deep", "source": "gw_amount", "target": deep_target,
             "condition": "금액이 50만원 이상인 경우"},
            {"id": "s_gw_simple", "source": "gw_amount", "target": "simple_review",
             "condition": "금액이 50만원 미만인 경우"},
            {"id": "s_deep_register", "source": "deep_review", "target": "register"},
            {"id": "s_simple_register", "source": "simple_review", "target": "register"},
            {"id": "s_register_end", "source": "register", "target": "end_event"},
        ],
    }


def test_every_branch_gets_its_own_scenario():
    derived, skipped = process_derive.derive_cases(_definition())

    assert [c["name"] for c in derived] == [
        "금액 판정: 금액이 50만원 이상인 경우",
        "금액 판정: 금액이 50만원 미만인 경우",
    ]
    assert [c["activity_order"] for c in derived] == [
        ["apply", "deep_review", "register"],
        ["apply", "simple_review", "register"],
    ]
    assert skipped == []


def test_expected_path_comes_from_replaying_the_definition():
    """기대값은 만든 판정을 지금 정의에 재생한 결과여야 한다."""
    definition = _definition()
    derived, _ = process_derive.derive_cases(definition)

    for case in derived:
        observed = process_replay.replay(
            definition, {"gateway_decisions": case["gateway_decisions"]}
        )
        assert observed["undecided"] is None
        assert observed["activity_order"] == case["activity_order"]
        assert observed["reached_end"] == case["reached_end"]


def test_derived_cases_pass_on_the_version_they_were_made_from():
    """방금 만든 기준선은 그 버전에서 전부 통과해야 한다."""
    definition = _definition()
    derived, _ = process_derive.derive_cases(definition)
    cases = process_derive.to_eval_cases(derived)

    assert len(cases) == 2
    for case in cases:
        _observed, graded = process_replay.replay_case(definition, case)
        assert graded is not None
        assert graded["passed"] == graded["total"] == 2


def test_a_changed_branch_target_breaks_the_derived_baseline():
    """분기 대상이 바뀌면 그 시나리오가 깨진 것으로 나와야 한다 — 회귀 테스트의 본분."""
    derived, _ = process_derive.derive_cases(_definition())
    cases = process_derive.to_eval_cases(derived)
    changed = _definition(deep_target="simple_review")

    graded = [process_replay.replay_case(changed, c)[1] for c in cases]
    assert graded[0]["passed"] < graded[0]["total"]   # 고액 경로가 달라졌다


def test_scenarios_that_reach_the_same_path_are_kept_once():
    """두 분기가 같은 경로로 수렴하면 같은 검증을 두 번 하지 않는다."""
    definition = _definition()
    for seq in definition["sequences"]:
        if seq["id"] == "s_gw_simple":
            seq["target"] = "deep_review"

    derived, _ = process_derive.derive_cases(definition)
    assert len(derived) == 1


def test_parallel_gateway_is_reported_not_guessed():
    """재생할 수 없는 갈림길은 시나리오를 만들지 않고 사유를 남긴다."""
    definition = _definition()
    definition["gateways"] = [{"id": "gw_amount", "name": "병렬 분기", "type": "parallelGateway"}]

    derived, skipped = process_derive.derive_cases(definition)
    assert derived == []
    assert any("병렬 분기" in reason for reason in skipped)


def test_loop_back_branch_does_not_run_forever():
    """되돌아가는 흐름이 있어도 파생이 끝난다(같은 자리를 무한히 돌지 않는다)."""
    definition = _definition()
    for seq in definition["sequences"]:
        if seq["id"] == "s_gw_simple":
            seq["target"] = "apply"       # 반려 → 재신청

    derived, _ = process_derive.derive_cases(definition)
    assert derived                        # 승인 경로는 남는다
    assert len(derived) <= process_derive.MAX_DERIVED_CASES


def test_derivation_records_the_decision_it_used():
    """경로만 남기면 두 분기가 같은 액티비티로 갈 때 어느 조건이었는지 되찾을 수 없다."""
    derived, _ = process_derive.derive_cases(_definition())

    decision = derived[0]["gateway_decisions"]["gw_amount"]
    assert decision["selected"] == ["s_gw_deep"]
    # 고르지 않은 분기도 함께 싣는다 — 분기가 사라진 변경을 화면이 읽을 수 있어야 한다.
    assert set(decision["sequences"]) == {"s_gw_deep", "s_gw_simple"}


def test_eval_cases_only_use_value_checks():
    """단언에 의미 판단형이 섞이면 채점에 모델이 끼어들어 판정이 실행마다 흔들린다."""
    derived, _ = process_derive.derive_cases(_definition())
    cases = process_derive.to_eval_cases(derived)

    kinds = {check["type"] for case in cases for check in case["checks"]}
    assert kinds == {process_derive.CHECK_PATH_EQUALS, process_derive.CHECK_REACHED_END}
    assert all(case["files"] == [] for case in cases)
