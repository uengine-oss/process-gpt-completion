"""게이트웨이가 넘기는 헤더에서 테넌트를 정한다."""
import pytest
from starlette.datastructures import Headers

from tenant_resolver import resolve_tenant_id


@pytest.mark.parametrize("headers,expected", [
    ({"x-tenant-id": "ohsy"}, "ohsy"),
    ({"x-forwarded-host": "ohsy.process-gpt.io"}, "ohsy"),
    ({"x-forwarded-host": "ohsy.process-gpt.io", "x-tenant-id": "other"}, "ohsy"),
    ({"x-tenant-id": "localhost"}, "localhost"),
    ({}, "localhost"),
])
def test_tenant_from_gateway_headers(headers, expected):
    assert resolve_tenant_id(Headers(headers)) == expected


def test_default_tenant_wins():
    assert resolve_tenant_id(Headers({"x-tenant-id": "ohsy"}), "fixed") == "fixed"
