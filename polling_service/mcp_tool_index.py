"""도구 이름 → MCP 서버 키 매핑.

생성된 코드가 `call_tool(server_key, tool_name, args)` 를 부르려면 그 도구를 어느
서버가 제공하는지 알아야 한다. 테넌트 MCP 구성에는 서버 목록만 있고 도구 목록은
없으므로, 서버에 실제로 붙어 물어본다.

순방향 생성기(`deterministic_generator`)와 보상 생성기(`compensation_handler`)가
모두 쓰므로 별도 모듈로 둔다. 두 생성기가 서로 다른 서비스에 놓이더라도 이 모듈은
양쪽에 같은 모양으로 따라간다.
"""

from typing import Any, Dict, Optional

import logging

from database import fetch_tenant_mcp_config

logger = logging.getLogger(__name__)

# 도구 목록 조회 상한(초). 상한이 없으면 응답하지 않는 MCP 서버 하나가 호출한 쪽을
# 영원히 붙잡는다. 매핑을 못 얻으면 그 도구는 고착화 대상에서 빠질 뿐, 업무 처리는
# 그대로 진행된다 — 기다릴 이유가 없다.
LIST_TOOLS_TIMEOUT_SECONDS = 15


def _remote_transport(server_cfg):
    """원격 MCP 서버의 전송 객체. 설정에 적힌 URL을 **글자 그대로** 쓴다.

    fastmcp 2.x 는 "자동 리다이렉트를 피한다"며 경로 끝에 슬래시를 붙인다
    (`client/transports.py`: `path + "/"`). 그런데 슬래시에 엄격한 서버가 있다 —
    `mcp.supabase.com` 은 `/mcp` 에 200, `/mcp/` 에 **404** 를 낸다. 404는 MCP SDK를
    거치며 `Session terminated` 로 바뀌어 올라오므로, 원인이 URL이라는 사실이 어디에도
    남지 않는다. 그 서버의 도구가 통째로 인덱스에서 빠지고, 그 도구를 되돌리는
    보상 코드는 "서버를 찾지 못했다"로 실패한다.

    fastmcp 3.x 는 이 동작을 없앴다("Some servers are strict about trailing slashes").
    폴링 서비스는 3.4.4 라 멀쩡하고 completion 은 2.9.0 이라 막혔다 — 같은 파일이
    두 서비스에 사본으로 놓이므로, 버전에 기대지 않고 여기서 URL을 고정한다.
    """
    config = dict(server_cfg or {})
    url = config.get("url")
    if not url:
        return None
    kind = str(config.get("transport") or config.get("type") or "http").lower()
    headers = config.get("headers") or {}
    from fastmcp.client.transports import SSETransport, StreamableHttpTransport

    transport_cls = SSETransport if kind == "sse" else StreamableHttpTransport
    transport = transport_cls(url=url, headers=headers)
    # 생성자가 경로를 바꿨으면 되돌린다. 3.x 에서는 애초에 바꾸지 않아 무해하다.
    if getattr(transport, "url", None) != url:
        transport.url = url
    return transport


def _client_config(server_key: str, server_cfg):
    """저장된 서버 설정을 fastmcp 가 읽는 모양으로 맞춘다.

    테넌트 MCP 설정은 Claude Desktop 관례대로 원격 서버를 `"type": "http"` 로 적는다.
    그런데 fastmcp 의 MCPConfig 는 그 자리를 `transport` 로 읽는다. 그대로 넘기면
    url·headers 를 가진 원격 서버가 stdio 서버로 오인되어 설정 전체가 거부되고,
    도구 목록이 통째로 비어 그 서버의 도구를 쓰는 활동은 영영 고착화되지 않는다.
    실패가 서버 단위로 갇혀 있어 조용히 지나가는 종류의 어긋남이라 여기서 맞춘다.
    """
    config = dict(server_cfg or {})
    config.pop("enabled", None)
    if "transport" not in config and "type" in config:
        config["transport"] = config.pop("type")
    return {"mcpServers": {server_key: config}}


def build_tool_index_from_tenant(
    tenant_id: str, timeout_s: Optional[int] = None
) -> Dict[str, str]:
    """tenant의 MCP 설정을 읽어 tool_name -> server_key 매핑을 구성한다.

    서버에 실제로 붙어 도구 목록을 물어본다. 어떤 실패도 밖으로 내보내지 않는다 —
    매핑이 비면 호출자가 그 도구를 포기하면 된다.

    붙지 못했을 때 이름으로 짐작해 채우지 않는다. 예전에는 `send_email_tool` 하나를
    gmail 서버에 매핑해 두었는데, 그것이 맞아떨어지는 활동은 우연히 굳고 나머지는
    전부 "서버를 찾지 못했다"로 막혔다. 어느 쪽이든 근거가 없다. 매핑이 비었다는
    사실을 그대로 남기는 편이, 왜 고착화되지 않는지 알아볼 수 있게 한다.
    """
    deadline = timeout_s if timeout_s is not None else LIST_TOOLS_TIMEOUT_SECONDS
    tool_to_server: Dict[str, str] = {}
    try:
        mcp = fetch_tenant_mcp_config(tenant_id) or {}
        servers = (mcp or {}).get("mcpServers", mcp)
        try:
            from fastmcp import Client as McpClient  # type: ignore
            import asyncio as _a

            async def _list_for_server(server_k: str, server_cfg: Any):
                """서버 하나의 도구 목록. 상한과 실패를 **서버 단위로** 가둔다.

                상한을 서버 전체 gather에 걸면 안 된다. 느린 서버 하나가 상한을
                넘기는 순간 gather가 취소되면서, 이미 성공했을 다른 서버의 도구까지
                통째로 사라진다(테넌트에 stdio 서버가 하나만 섞여 있어도 매핑이
                늘 비게 된다). 각자 자기 상한 안에서 성공하거나 자기만 실패한다.
                """
                # 원격 서버는 전송 객체를 직접 만들어 URL을 보존한다. stdio 서버는
                # URL이 없으므로 기존대로 설정을 그대로 넘긴다.
                target = _remote_transport(server_cfg) or _client_config(server_k, server_cfg)
                try:
                    async with _a.timeout(deadline):
                        client = McpClient(target)
                        async with client:
                            # ping 은 MCP 스펙의 선택 기능이다. 구현하지 않은 서버는
                            # "Method not found" 를 돌려주는데, 그것을 연결 실패로 읽으면
                            # 멀쩡히 도구를 제공하는 서버가 색인에서 통째로 빠진다.
                            # 살아 있는지는 바로 다음 줄의 list_tools 가 말해 준다.
                            try:
                                await client.ping()
                            except Exception:
                                pass
                            tools = await client.list_tools()
                            for t in tools:
                                tool_to_server[t.name] = server_k
                except Exception:
                    logger.warning(
                        "MCP 도구 목록 조회 실패 | tenant=%s server=%s", tenant_id, server_k,
                        exc_info=True,
                    )

            async def _run_all():
                await _a.gather(
                    *[_list_for_server(k, v) for k, v in servers.items()],
                    return_exceptions=True,
                )

            try:
                _a.get_running_loop()
            except RuntimeError:
                # 도는 루프가 없다 — 여기서 바로 돌린다.
                _a.run(_run_all())
            else:
                # 이미 도는 루프 안이면 별도 스레드에서 새 루프로 돌린다.
                import threading

                def runner():
                    try:
                        _a.run(_run_all())
                    except Exception:  # noqa: BLE001
                        logger.warning("MCP 도구 목록 조회 실패 | tenant=%s", tenant_id, exc_info=True)

                t = threading.Thread(target=runner, daemon=True)
                t.start()
                # 스레드 join에도 상한을 둔다. asyncio 상한만 두면 루프 바깥에서
                # 멈춘 경우(예: 동기 소켓 대기)를 잡지 못한다.
                t.join(timeout=deadline * len(servers or {1}) + 5)
                if t.is_alive():
                    logger.warning(
                        "MCP 도구 목록 조회가 상한을 넘겨 부분 결과로 진행 | tenant=%s", tenant_id
                    )
        except Exception:
            # fastmcp 를 못 쓰면 도구 목록을 얻을 길이 없다. 조용히 비워 두면 MCP 도구를
            # 쓰는 활동이 영영 고착화되지 않는데도 이유가 어디에도 남지 않는다.
            logger.warning(
                "MCP 도구 목록 조회 불가 — 이 테넌트의 MCP 도구는 고착화되지 않는다 | tenant=%s",
                tenant_id, exc_info=True,
            )
    except Exception:
        logger.warning("테넌트 MCP 구성 조회 실패 | tenant=%s", tenant_id, exc_info=True)
    return tool_to_server
