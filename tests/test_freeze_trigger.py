"""고착화 트리거가 DONE 확정 시점에 걸리는지, 그리고 절대 업무를 막지 않는지 고정한다.

트리거를 잘못 두면 조용히 아무것도 고착화되지 않는다. 예전에는 사람의 제출
(`/complete`)에 걸려 있었는데, 자율 완료(COMPLETE) 모드 에이전트 액티비티는 그
경로를 아예 지나지 않아 아무리 반복돼도 고착화되지 않았다.
"""

import importlib

import pytest


@pytest.fixture
def generator():
    """현재 로드된 deterministic_generator 모듈.

    다른 테스트가 `sys.modules["deterministic_generator"]` 를 스텁으로 갈아끼우므로,
    수집 시점에 이름을 묶어 두면 엉뚱한 모듈 객체를 패치하게 된다. 매번 지금 것을 쓴다.
    """
    return importlib.import_module("deterministic_generator")


@pytest.fixture
def frozen(monkeypatch, generator):
    """try_freeze 호출을 기록하는 스텁."""
    calls = []
    monkeypatch.setattr(generator, "try_freeze", lambda workitem: calls.append(workitem))
    return calls


_ROW = {
    "id": "todo-1",
    "status": "DONE",
    "proc_def_id": "pd",
    "activity_id": "act",
    "tenant_id": "tn",
}


def test_done_transition_triggers_freezing(frozen, generator):
    generator.freeze_on_done(_ROW)
    assert len(frozen) == 1
    workitem = frozen[0]
    assert (workitem.proc_def_id, workitem.activity_id, workitem.tenant_id) == ("pd", "act", "tn")


@pytest.mark.parametrize("status", ["SUBMITTED", "IN_PROGRESS", "PENDING", "TODO", "", None])
def test_non_done_transitions_are_ignored(frozen, generator, status):
    """제출·진행 중에는 아직 "성공적으로 끝났다"는 증거가 아니다."""
    generator.freeze_on_done({**_ROW, "status": status})
    assert frozen == []


def test_missing_row_is_ignored(frozen, generator):
    generator.freeze_on_done(None)
    generator.freeze_on_done({})
    assert frozen == []


def test_freezing_failure_never_propagates(monkeypatch, generator):
    """고착화는 최적화다. 실패가 워크아이템 저장을 되돌려서는 안 된다."""

    def _boom(workitem):
        raise RuntimeError("MCP 서버 응답 없음")

    monkeypatch.setattr(generator, "try_freeze", _boom)
    generator.freeze_on_done(_ROW)  # 예외가 새어 나오면 이 줄에서 실패한다


def test_saved_row_supplies_fields_the_partial_payload_lacks(frozen, generator):
    """호출부는 `{"id": ..., "status": "DONE"}` 처럼 일부만 보내는 경우가 많다.

    저장 응답의 행에는 proc_def_id/activity_id가 채워져 있어 그것을 써야 한다.
    """
    generator.freeze_on_done(_ROW)
    assert frozen[0].proc_def_id == "pd"


def test_polling_save_hook_uses_the_saved_row():
    """`upsert_workitem` 훅이 응답 행을 우선 쓰는지 — 부분 payload로는 고착화가 안 된다."""
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent / "polling_service" / "database.py").read_text(
        encoding="utf-8-sig"
    )
    hook = source[source.index("def _try_freeze_on_done("):]
    hook = hook[: hook.index("\n\n\n")] if "\n\n\n" in hook else hook
    assert 'getattr(response, "data"' in hook
    assert "rows[0] if rows else workitem_data" in hook
