"""fetch_user_info: 같은 id 의 사용자(에이전트)가 테넌트마다 한 줄씩 있을 때 지금 테넌트의 줄을 고른다.

users 의 키는 (id, tenant_id) 다. 첫 줄을 그대로 쓰면 다른 테넌트의 이름(예: 한국어 에이전트 이름)이
영어 테넌트의 작업 항목에 적혔다.
"""

from __future__ import annotations

import importlib

import pytest

ROWS = [
    {"id": "a1", "tenant_id": "localhost", "username": "정산 검토 에이전트"},
    {"id": "a1", "tenant_id": "en-demo", "username": "Expense review agent"},
]


@pytest.mark.parametrize("module", ["database", "polling_service.database"])
def test_지금_테넌트의_줄을_고른다(module):
    db = importlib.import_module(module)
    token = db.subdomain_var.set("en-demo")
    try:
        assert db._in_tenant(ROWS)["username"] == "Expense review agent"
    finally:
        db.subdomain_var.reset(token)
    token = db.subdomain_var.set("localhost")
    try:
        assert db._in_tenant(ROWS)["username"] == "정산 검토 에이전트"
    finally:
        db.subdomain_var.reset(token)


@pytest.mark.parametrize("module", ["database", "polling_service.database"])
def test_지금_테넌트의_줄이_없으면_첫_줄(module):
    db = importlib.import_module(module)
    token = db.subdomain_var.set("other")
    try:
        assert db._in_tenant(ROWS)["tenant_id"] == "localhost"
    finally:
        db.subdomain_var.reset(token)
