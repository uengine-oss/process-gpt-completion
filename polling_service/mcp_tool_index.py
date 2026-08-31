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


def build_tool_index_from_tenant(
    tenant_id: str, timeout_s: Optional[int] = None
) -> Dict[str, str]:
    """tenant의 MCP 설정을 읽어 tool_name -> server_key 매핑을 구성한다.

    fastmcp를 쓸 수 있으면 실제 서버에서 도구 목록을 조회하고, 불가하면 휴리스틱으로
    매핑한다. 어떤 실패도 밖으로 내보내지 않는다 — 매핑이 비면 호출자가 그 도구를
    포기하면 된다.
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
                config = {"mcpServers": {server_k: dict(server_cfg or {})}}
                config["mcpServers"][server_k].pop("enabled", None)
                try:
                    async with _a.timeout(deadline):
                        client = McpClient(config)
                        async with client:
                            await client.ping()
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
            server_keys = list(servers.keys())
            fallback_server = server_keys[0] if server_keys else "default"
            gmail_server = next((k for k in server_keys if "gmail" in k.lower()), fallback_server)
            tool_to_server.setdefault("send_email_tool", gmail_server)
    except Exception:
        logger.warning("테넌트 MCP 구성 조회 실패 | tenant=%s", tenant_id, exc_info=True)
    return tool_to_server
