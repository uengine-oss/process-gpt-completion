"""completion과 polling_service의 공유 모듈 사본이 어긋나지 않는지 고정한다.

두 서비스는 빌드 컨텍스트가 갈려 있다. completion 이미지는 저장소 루트에서,
polling_service 이미지는 `cd polling_service && docker build .` 로 만들어지므로
polling 이미지에는 루트 모듈이 들어가지 않는다. 그래서 결정론적 코드 생성에 쓰는
모듈은 양쪽에 같은 내용으로 존재한다(`database.py`, `llm_factory.py`가 이미 그렇다).

사본이 어긋나면 조용히 틀린다. 폴링 서비스가 만든 코드와 completion이 만든 보상
코드가 서로 다른 분류 규칙을 쓰게 되어, 되돌릴 대상을 다르게 보기 때문이다. 진단이
어려운 종류의 버그라 파일 단위로 동일성을 못박는다.

한쪽을 고쳤다면 다른 쪽에도 그대로 복사해야 이 테스트가 통과한다.
"""

import hashlib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_POLLING = _ROOT / "polling_service"

# 두 서비스가 같은 내용으로 들고 있어야 하는 파일들.
SHARED_MODULES = (
    "work_history.py",
    "deterministic_signature.py",
    "deterministic_template.py",
    "deterministic_generator.py",
    "mcp_tool_index.py",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("name", SHARED_MODULES)
def test_shared_module_copies_are_identical(name):
    root, polling = _ROOT / name, _POLLING / name
    assert root.exists(), f"루트에 {name}이 없습니다"
    assert polling.exists(), (
        f"polling_service/{name}이 없습니다. 폴링 이미지에는 루트 모듈이 들어가지 "
        f"않으므로 사본이 필요합니다: cp {name} polling_service/{name}"
    )
    assert _digest(root) == _digest(polling), (
        f"{name}의 두 사본이 다릅니다. 한쪽만 고치면 두 서비스가 서로 다른 분류 "
        f"규칙으로 작업 이력을 읽습니다: cp {name} polling_service/{name}"
    )


def test_polling_database_has_the_helpers_the_generator_needs():
    """생성기는 `from database import ...` 로 저장소에 닿는다.

    폴링 서비스에서는 그 이름이 `polling_service/database.py` 로 풀린다. 헬퍼가 하나라도
    빠지면 폴링 서비스 기동 시점이 아니라 **DONE 전환이 처음 일어날 때** 터진다.
    """
    source = (_POLLING / "database.py").read_text(encoding="utf-8-sig")
    for helper in (
        "fetch_mcp_python_code",
        "upsert_mcp_python_code",
        "fetch_workitems_by_activity",
        "fetch_events_by_todo_id",
        "fetch_last_deactivated_at",
        "fetch_related_workitem_outputs",
        "fetch_tenant_mcp_config",
    ):
        assert f"def {helper}(" in source, f"polling_service/database.py에 {helper}가 없습니다"


def test_freeze_is_triggered_from_the_workitem_save_path():
    """트리거는 DONE 전환 자리에 있어야 한다.

    사람의 제출(`/complete`)에 걸면 자율 완료 모드 에이전트 액티비티가 통째로 빠진다.
    """
    database = (_POLLING / "database.py").read_text(encoding="utf-8-sig")
    assert "_try_freeze_on_done" in database
    assert "freeze_on_done" in database

    engine = (_ROOT / "process_engine.py").read_text()
    assert "try_freeze(" not in engine, (
        "제출 경로에 고착화 트리거가 다시 들어왔습니다 — DONE 전환 자리에 있어야 합니다"
    )
