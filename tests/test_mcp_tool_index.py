# -*- coding: utf-8 -*-
"""원격 MCP 서버에 붙을 때 설정에 적힌 URL을 글자 그대로 쓰는지 고정한다.

fastmcp 2.x 는 "자동 리다이렉트를 피한다"며 경로 끝에 슬래시를 붙인다. 슬래시에
엄격한 서버가 있다 — `mcp.supabase.com` 은 `/mcp` 에 200, `/mcp/` 에 404 를 낸다.
404 는 MCP SDK 를 거치며 `Session terminated` 로 바뀌어 올라오므로, 원인이 URL이라는
사실이 어디에도 남지 않는다.

그 결과가 조용하고 나쁘다. 그 서버의 도구가 통째로 인덱스에서 빠지고, 도구→서버 표가
빈 채로 보상 코드에 박힌다 — 되돌릴 수 있었던 활동이 "서버를 찾지 못했다"로 실패한다.

completion 은 fastmcp 2.9.0, 폴링 서비스는 3.4.4 를 쓴다(3.x 는 이 동작을 없앴다).
같은 파일이 두 서비스에 사본으로 놓이므로 버전에 기대지 않고 고정한다.
"""

import mcp_tool_index as index


def test_the_configured_url_is_used_verbatim():
    """슬래시 하나가 붙으면 그 서버의 도구가 통째로 사라진다."""
    url = "https://mcp.supabase.com/mcp?project_ref=abc"
    transport = index._remote_transport(
        {"url": url, "type": "http", "headers": {"Authorization": "Bearer x"}}
    )
    assert transport is not None
    assert transport.url == url


def test_a_url_that_already_ends_in_a_slash_is_left_alone():
    url = "https://example.test/mcp/"
    assert index._remote_transport({"url": url, "type": "http"}).url == url


def test_sse_servers_get_the_sse_transport():
    from fastmcp.client.transports import SSETransport

    transport = index._remote_transport({"url": "https://example.test/sse", "type": "sse"})
    assert isinstance(transport, SSETransport)
    assert transport.url == "https://example.test/sse"


def test_stdio_servers_have_no_url_and_fall_back_to_the_config():
    """stdio 서버는 URL이 없다. 전송 객체를 지어내지 않고 설정을 그대로 넘긴다."""
    assert index._remote_transport({"command": "uvx", "args": ["some-server"]}) is None


def test_headers_survive():
    """헤더가 빠지면 401 이 나고, fastmcp 는 그걸 OAuth 흐름으로 오해한다."""
    transport = index._remote_transport(
        {"url": "https://example.test/mcp", "headers": {"Authorization": "Bearer x"}}
    )
    assert transport.headers.get("Authorization") == "Bearer x"
