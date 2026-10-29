"""에이전트 단계의 실행기(agent_orch) 기본값.

정의의 activity.orchestration 이 비어 있는 에이전트 단계(COMPLETE)는 예전에 늘
'crewai-deep-research' 로 만들어졌다. 공식 설치에는 그 값을 폴링하는 소비자가 없어 단계가
생기기만 하고 시작되지 않았다(튜토리얼 VM 실측 — 돌던 todolist 소비자는 deepagents 뿐).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def db_mod():
    return importlib.import_module("database")


def test_fallback_is_the_runtime_the_install_polls(db_mod, monkeypatch):
    monkeypatch.delenv("DEFAULT_AGENT_ORCH", raising=False)
    assert db_mod.default_agent_orch() == "deepagents"


@pytest.mark.parametrize("value", ["", "  ", "none", "None", "null"])
def test_blank_env_keeps_the_fallback(db_mod, monkeypatch, value):
    monkeypatch.setenv("DEFAULT_AGENT_ORCH", value)
    assert db_mod.default_agent_orch() == "deepagents"


def test_install_can_choose_another_runtime(db_mod, monkeypatch):
    monkeypatch.setenv("DEFAULT_AGENT_ORCH", " crewai-deep-research ")
    assert db_mod.default_agent_orch() == "crewai-deep-research"


@pytest.mark.parametrize("value", [None, "", "none", "None", "null", " "])
def test_empty_orchestration_values(db_mod, value):
    assert db_mod._is_empty_orch(value) is True


def test_explicit_orchestration_is_not_empty(db_mod):
    assert db_mod._is_empty_orch("deepagents") is False


def test_every_creation_path_uses_the_configurable_default():
    # upsert_next_workitems(이미 있는 행·새 다음 단계)와 upsert_todo_workitems 세 곳이 모두 같은
    # 기본값을 쓴다 — 한 곳만 남아도 그 경로로 만든 에이전트 단계는 다시 아무도 집지 않는다.
    source = (Path(__file__).resolve().parents[1] / "database.py").read_text(encoding="utf-8")
    assert source.count("= default_agent_orch()") == 3
    assert "= 'crewai-deep-research'" not in source
