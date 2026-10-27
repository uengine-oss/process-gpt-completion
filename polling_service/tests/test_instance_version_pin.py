"""An instance runs on the definition version it started on.

upsert_process_instance used to re-stamp proc_def_version with the latest version on every save,
so an instance started on v1 continued on v2 once v2 was saved. The stamp is now written once (at
start) and kept; a pinned version is loaded from proc_def_version while a newer version exists.
"""
from __future__ import annotations

import importlib

import pytest


class _Q:
    """Just enough of the supabase query builder: eq / gt / order / limit / select → execute()."""

    def __init__(self, rows):
        self._rows = list(rows)

    def select(self, *_a, **_k):
        return self

    def eq(self, k, v):
        self._rows = [r for r in self._rows if r.get(k) == v]
        return self

    def gt(self, k, v):
        self._rows = [r for r in self._rows if (r.get(k) or "") > v]
        return self

    def order(self, k, desc=False):
        self._rows.sort(key=lambda r: r.get(k) or "", reverse=desc)
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def execute(self):
        return type("R", (), {"data": self._rows})()


class _DB:
    def __init__(self, tables):
        self.tables = tables

    def table(self, name):
        return _Q(self.tables.get(name, []))


V1 = {"arcv_id": "p_1.0", "proc_def_id": "p", "version": "1.0", "version_tag": "major", "tenant_id": "t",
      "timeStamp": "2026-07-31T00:00:00", "is_draft": False, "definition": {"activities": [{"id": "review"}]}}
V2 = {"arcv_id": "p_2.0", "proc_def_id": "p", "version": "2.0", "version_tag": "major", "tenant_id": "t",
      "timeStamp": "2026-10-26T22:00:10", "is_draft": False, "definition": {"activities": [{"id": "skip"}]}}
DRAFT = {"arcv_id": "p_3.0", "proc_def_id": "p", "version": "3.0", "version_tag": "major", "tenant_id": "t",
         "timeStamp": "2026-10-27T00:00:00", "is_draft": True, "definition": {"activities": []}}


class _Inst:
    def __init__(self, inst_id, pinned=None):
        self.proc_inst_id = inst_id
        self.proc_def_version = pinned

    def get_def_id(self):
        return self.proc_inst_id.split(".")[0]


@pytest.fixture(params=["database"])
def db_mod(request):
    return importlib.import_module(request.param)


def test_running_instance_keeps_its_version(db_mod):
    db = _DB({"proc_def_version": [V1, V2], "bpm_proc_inst": [{"proc_inst_id": "p.1", "tenant_id": "t", "proc_def_version": "p_1.0"}]})
    # the model carries the stamp …
    assert db_mod._instance_version(_Inst("p.1", "p_1.0"), db, "t") == "p_1.0"
    # … or only the stored row does: still not re-stamped with v2
    assert db_mod._instance_version(_Inst("p.1"), db, "t") == "p_1.0"


def test_new_instance_gets_latest_released_version(db_mod):
    db = _DB({"proc_def_version": [V1, V2, DRAFT], "bpm_proc_inst": []})
    assert db_mod._instance_version(_Inst("p.2"), db, "t") == "p_2.0"


def test_pinned_older_version_is_loaded_newest_falls_back_to_current(db_mod):
    db = _DB({"proc_def_version": [V1, V2, DRAFT]})
    rows = db_mod._pinned_version_rows(db, "p", "p_1.0", "t")
    assert [r["arcv_id"] for r in rows] == ["p_1.0"]
    # the newest released version (only a draft after it) → the current definition decides
    assert db_mod._pinned_version_rows(db, "p", "p_2.0", "t") == []
    # a draft never pins
    assert db_mod._pinned_version_rows(db, "p", "p_3.0", "t") == []
    assert db_mod._pinned_version_rows(db, "p", None, "t") == []
