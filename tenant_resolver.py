from typing import Mapping


def resolve_tenant_id(headers: Mapping[str, str], default_tenant_id: str = "") -> str:
    """고정 테넌트가 없으면 게이트웨이가 넘긴 헤더에서 테넌트를 정한다."""
    if default_tenant_id:
        return default_tenant_id
    # 게이트웨이(Spring Cloud Gateway 5)는 신뢰 프록시 설정이 없으면 X-Forwarded-* 를 지운다.
    # 같은 필터가 붙이는 X-Tenant-Id 는 그대로 도착한다.
    host_name = headers.get("X-Forwarded-Host") or headers.get("X-Tenant-Id")
    if host_name is None or "localhost" in host_name:
        return "localhost"
    return host_name.split(".")[0]
