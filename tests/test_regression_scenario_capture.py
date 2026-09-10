"""생성 시 검증이 만든 시나리오가 버려지지 않고 남는지 고정한다.

병합 전 회귀 테스트는 "저장된 시나리오를 변경 전/후 두 벌로 돌려 비교" 하는 방식이라
비교에 쓸 시나리오가 미리 확보돼 있어야 성립한다. 그 시나리오는 이미 만들어지고 있다 —
`ProcessValidator._build_test_plan` 이 분기별 케이스를 모델로 만들고, `_run_trace` 가
실행 엔진으로 실제로 돌려 통과를 확인한다. 문제는 그 결과가 리포트에 담기지 않고
사라진다는 것이었다.

여기서 고정하는 세 가지:

1. 폴링 서비스가 라우팅에 쓴 분기 판정(conditionEval)이 워크아이템 행에 남는다.
   실행 경로만으로는 두 분기가 같은 활동으로 향할 때 어느 조건이 참이었는지 구분할 수
   없어(고액 → 정밀검토, 특수건 → 정밀검토), 회귀 재생이 이 기록에 의존한다.
2. 게이트웨이 분기가 없는 워크아이템에는 쓰기가 생기지 않는다. 대부분의 워크아이템이
   직선 흐름이므로, 여기서 매번 쓰면 기록 하나 남기려고 전체 쓰기가 두 배가 된다.
3. 통과한 케이스의 기준 경로는 모델이 추론한 기대 순서가 아니라 **실제로 실행된 경로**다.
   회귀 테스트가 답할 질문은 "지금 되던 게 깨지는가" 이므로 기준은 현재 실제 동작이어야
   한다. 모델의 추론값으로 굳히면 아직 고치지 않은 논리 결함까지 회귀 결함으로 다시 뜬다.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wiproc():
    """폴링 서비스의 workitem_processor.

    폴링 서비스는 `polling_service/` 를 루트로 도는 별도 프로세스라, 저장소 루트 기준으로
    import 하면 같은 이름의 다른 `database` 모듈이 잡힌다. 실행 때와 같은 경로로 맞춘다.
    """
    added = str(REPO_ROOT / "polling_service")
    sys.path.insert(0, added)
    injected = "supabase_config" not in sys.modules
    if injected:
        stub = types.ModuleType("supabase_config")
        stub.get_supabase_key = lambda *a, **k: "test-key"
        stub.get_supabase_url = lambda *a, **k: "http://localhost"
        sys.modules["supabase_config"] = stub
    try:
        import workitem_processor  # noqa: E402

        yield workitem_processor
    finally:
        # 같은 이름의 모듈이 저장소 루트에도 있어, 경로를 남겨 두면 뒤따르는 테스트가
        # 엉뚱한 `database` 를 잡는다.
        if added in sys.path:
            sys.path.remove(added)
        if injected:
            sys.modules.pop("supabase_config", None)


class _Seq:
    def __init__(self, id, source, target, condition=None, name=None):
        self.id = id
        self.source = source
        self.target = target
        self.condition = condition
        self.name = name
        self.properties = None


class _Definition:
    """게이트웨이 하나에서 두 갈래가 나가는 최소 정의."""

    def __init__(self, sequences, gateway_ids):
        self.sequences = sequences
        self._gateways = set(gateway_ids)

    def find_gateway_by_id(self, node_id):
        return node_id if node_id in self._gateways else None


def _two_branch_definition():
    return _Definition(
        sequences=[
            _Seq("seq_apply_to_gw", "apply", "gw_amount"),
            _Seq("seq_high", "gw_amount", "detail_review", condition="금액이 100만원을 넘는 경우"),
            _Seq("seq_special", "gw_amount", "detail_review", condition="특수 거래처인 경우"),
            _Seq("seq_normal", "gw_amount", "simple_review", condition="그 밖의 경우"),
        ],
        gateway_ids={"gw_amount"},
    )


# --------------------------------------------------------------------------- 1
def test_gateway_decisions_record_which_branch_was_taken(wiproc):
    """같은 활동으로 향하는 두 분기를 판정 기록이 구분한다.

    seq_high 와 seq_special 은 둘 다 detail_review 로 간다. 실행 경로에는 detail_review
    하나만 남으므로, 이 기록이 없으면 어느 조건이 참이었는지 되찾을 수 없다.
    """
    sequence_condition_data = {
        "seq_high": {"condition": "금액이 100만원을 넘는 경우", "conditionEval": True,
                     "conditionReason": "신청 금액 1,200,000원"},
        "seq_special": {"condition": "특수 거래처인 경우", "conditionEval": False},
        "seq_normal": {"condition": "그 밖의 경우", "conditionEval": False},
    }

    decisions = wiproc.build_gateway_decisions(_two_branch_definition(), sequence_condition_data)

    assert decisions["gw_amount"]["selected"] == ["seq_high"]
    seqs = decisions["gw_amount"]["sequences"]
    assert seqs["seq_high"]["eval"] is True
    assert seqs["seq_high"]["target"] == "detail_review"
    assert seqs["seq_high"]["reason"] == "신청 금액 1,200,000원"
    assert seqs["seq_special"]["eval"] is False
    # 같은 대상으로 가는 두 분기가 기록에서는 구분된다.
    assert seqs["seq_special"]["target"] == seqs["seq_high"]["target"]


def test_unevaluated_sequences_are_not_recorded_as_false(wiproc):
    """판정을 거치지 않은 시퀀스는 남기지 않는다.

    "판정 안 됨" 과 "거짓으로 판정됨" 을 섞으면 재생 시 둘을 구분할 수 없고, 판정되지 않은
    분기를 '가지 않은 길' 로 굳혀 버린다.
    """
    sequence_condition_data = {
        "seq_high": {"condition": "금액이 100만원을 넘는 경우", "conditionEval": True},
        "seq_special": {"condition": "특수 거래처인 경우"},   # 판정 전
    }

    decisions = wiproc.build_gateway_decisions(_two_branch_definition(), sequence_condition_data)

    assert set(decisions["gw_amount"]["sequences"]) == {"seq_high"}


def test_sequences_not_leaving_a_gateway_are_ignored(wiproc):
    """게이트웨이에서 나가지 않는 시퀀스는 분기 판정이 아니다."""
    sequence_condition_data = {
        "seq_apply_to_gw": {"conditionEval": True},
    }

    decisions = wiproc.build_gateway_decisions(_two_branch_definition(), sequence_condition_data)

    assert decisions == {}


# --------------------------------------------------------------------------- 2
def test_straight_flow_workitem_writes_nothing(wiproc, monkeypatch):
    """분기가 없으면 워크아이템에 쓰지 않는다 — 기록 때문에 쓰기가 늘면 안 된다."""
    writes = []
    monkeypatch.setattr(wiproc, "upsert_workitem", lambda data, tid: writes.append(data))

    definition = _Definition(
        sequences=[_Seq("seq_a_to_b", "task_a", "task_b")],
        gateway_ids=set(),
    )
    wiproc._persist_gateway_decisions(
        {"id": "todo-1"}, definition, {"seq_a_to_b": {"conditionEval": True}}, "tn"
    )

    assert writes == []


def test_branching_workitem_writes_decisions(wiproc, monkeypatch):
    writes = []
    monkeypatch.setattr(wiproc, "upsert_workitem", lambda data, tid: writes.append(data))

    wiproc._persist_gateway_decisions(
        {"id": "todo-1"},
        _two_branch_definition(),
        {"seq_high": {"conditionEval": True}, "seq_normal": {"conditionEval": False}},
        "tn",
    )

    assert len(writes) == 1
    assert writes[0]["id"] == "todo-1"
    assert writes[0]["gateway_decisions"]["gw_amount"]["selected"] == ["seq_high"]


def test_persist_failure_does_not_stop_the_process(wiproc, monkeypatch):
    """기록용 쓰기가 실패해도 업무 진행을 막지 않는다."""
    def _boom(data, tid):
        raise RuntimeError("db down")

    monkeypatch.setattr(wiproc, "upsert_workitem", _boom)

    # 예외가 밖으로 나오면 이 호출이 워크아이템 처리를 중단시킨다.
    wiproc._persist_gateway_decisions(
        {"id": "todo-1"},
        _two_branch_definition(),
        {"seq_high": {"conditionEval": True}},
        "tn",
    )


# --------------------------------------------------------------------------- 3
def test_passing_cases_use_the_actual_path_not_the_proposed_one():
    """기준 경로는 실제 실행 경로다.

    모델은 '의미상 올바른 순서' 를 추론해 기대 순서를 낸다. 생성 시 검증에서는 그게 맞지만
    (정의의 시퀀스가 틀렸을 수 있으니 독립된 기준이 필요하다), 회귀의 기준으로 삼으면
    현재 동작이 아닌 이상적 동작을 굳히게 된다.
    """
    from process_validator import ProcessValidator

    case = {
        "name": "고액 승인 경로",
        "activity_inputs": {"apply": {"amount": 1200000}},
        "expected_activity_order": ["apply", "detail_review", "register"],
    }
    trace = {
        "actual_order": ["apply", "detail_review", "extra_notice", "register"],
        "reached_end": True,
        "gateway_decisions": {"gw_amount": {"selected": ["seq_high"]}},
    }

    cases = ProcessValidator._passing_cases([(case, trace, [])])

    assert len(cases) == 1
    assert cases[0]["expected_activity_order"] == trace["actual_order"]
    assert cases[0]["proposed_activity_order"] == case["expected_activity_order"]
    assert cases[0]["gateway_decisions"] == trace["gateway_decisions"]
    assert cases[0]["activity_inputs"] == case["activity_inputs"]


def test_cases_with_defects_are_not_promoted():
    """결함이 남은 케이스는 기준이 되지 않는다 — 그 결함이 '정상 동작' 으로 굳는다."""
    from process_validator import ProcessValidator

    case = {"name": "반려 경로", "activity_inputs": {}, "expected_activity_order": ["a", "b"]}
    trace = {"actual_order": ["a"], "reached_end": False, "gateway_decisions": {}}
    defects = [{"severity": "critical", "type": "not_reached_end"}]

    assert ProcessValidator._passing_cases([(case, trace, defects)]) == []


def test_cases_without_a_path_are_not_promoted():
    """실행 경로가 비면 비교할 기준이 없다."""
    from process_validator import ProcessValidator

    case = {"name": "빈 실행", "activity_inputs": {}, "expected_activity_order": []}
    trace = {"actual_order": [], "reached_end": False, "gateway_decisions": {}}

    assert ProcessValidator._passing_cases([(case, trace, [])]) == []


# --------------------------------------------------------------------------- #
# 4) 케이스 이름은 번호가 아니다
# --------------------------------------------------------------------------- #
def test_numbered_case_names_are_replaced_with_the_path():
    """모델이 `1`, `2` 로 이름을 보내도 경로로 이름을 짓는다.

    통과한 케이스는 회귀 시나리오로 승격돼 병합 전 검증 목록에 이름만으로 나열된다.
    번호로는 어느 시나리오가 깨졌는지 알 수 없어 매번 케이스를 열어 봐야 한다.
    """
    from process_validator import ProcessValidator

    plan = ProcessValidator._normalize_test_plan({
        "cases": [
            {"name": "1", "activity_inputs": {}, "expected_activity_order": ["apply", "review"]},
            {"name": " 2. ", "activity_inputs": {}, "expected_activity_order": []},
            {"name": "고액 승인 경로", "activity_inputs": {}, "expected_activity_order": ["apply", "audit"]},
        ]
    })

    names = [c["name"] for c in plan["cases"]]
    assert names[0] == "apply → review 경로"
    # 경로조차 없으면 더 나은 이름을 지어낼 재료가 없다 — 그때만 번호로 남는다.
    assert names[1] == "케이스 2"
    assert names[2] == "고액 승인 경로"


def test_long_paths_are_shortened_in_the_name():
    from process_validator import ProcessValidator

    order = [f"a{i}" for i in range(20)]
    plan = ProcessValidator._normalize_test_plan({
        "cases": [{"name": "3", "activity_inputs": {}, "expected_activity_order": order}]
    })
    assert plan["cases"][0]["name"] == "a0 → … → a19 경로"
