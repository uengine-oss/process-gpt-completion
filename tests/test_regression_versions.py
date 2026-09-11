"""regression.versions — 버전 스냅샷을 테넌트 안에서만 찾는지.

`proc_def_version.arcv_id` 는 `<id>_<버전>` 이라 테넌트가 달라도 같은 값이 나온다.
테넌트 조건 없이 찾으면 다른 테넌트의 정의로 회귀 검증을 돌린다.
"""

from __future__ import annotations

import pytest

from regression import versions


class _Query:
    def __init__(self, rows: list[dict]):
        self._rows = rows
        self._filters: list[tuple[str, object]] = []

    def select(self, *_):
        return self

    def eq(self, column, value):
        self._filters.append((column, value))
        return self

    def limit(self, *_):
        return self

    def execute(self):
        matched = [r for r in self._rows if all(r.get(c) == v for c, v in self._filters)]
        return type("Result", (), {"data": matched})()


class _FakeSupabase:
    def __init__(self, tables: dict[str, list[dict]]):
        self._tables = tables

    def table(self, name):
        return _Query(self._tables.get(name, []))


@pytest.fixture
def two_tenants(monkeypatch):
    tables = {
        "proc_def_version": [
            {"tenant_id": "other", "arcv_id": "order_2", "definition": {"owner": "other"},
             "snapshot": "<dmn owner='other'/>"},
            {"tenant_id": "hoseo-osy", "arcv_id": "order_2", "definition": {"owner": "hoseo-osy"},
             "snapshot": "<dmn owner='hoseo-osy'/>"},
        ],
        "proc_def": [
            {"tenant_id": "hoseo-osy", "id": "order", "definition": {"owner": "current"},
             "bpmn": "<dmn owner='current'/>"},
        ],
    }
    monkeypatch.setattr(versions, "_supabase", lambda: _FakeSupabase(tables))


def test_process_snapshot_is_read_from_own_tenant(two_tenants):
    definition, _ = versions.load_process_definition("hoseo-osy", "order", "v2")
    assert definition == {"owner": "hoseo-osy"}


def test_dmn_snapshot_is_read_from_own_tenant(two_tenants):
    xml, _ = versions.load_dmn_xml("hoseo-osy", "order", "v2")
    assert "hoseo-osy" in xml


def test_other_tenant_snapshot_falls_back_to_own_current_definition(two_tenants):
    # 'v3' 스냅샷은 어느 테넌트에도 없고, 'v2' 는 이 테넌트에도 있다 — 없는 버전만 현재 정의로.
    definition, _ = versions.load_process_definition("hoseo-osy", "order", "v3")
    assert definition == {"owner": "current"}


def test_unknown_tenant_does_not_borrow_another_tenants_definition(two_tenants):
    with pytest.raises(ValueError):
        versions.load_process_definition("nobody", "order", "v2")
