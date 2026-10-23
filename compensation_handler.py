from typing import Dict, Any, List
import json
import sys
import os

from llm_factory import create_llm
from database import fetch_tenant_mcp_config, fetch_mcp_python_code, upsert_mcp_python_code, fetch_events_by_proc_inst_id_until_activity, upsert_workitem, fetch_user_info_by_uid

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

TEMPLATE = """# -*- coding: utf-8 -*-
import os
import sys
import json
import asyncio
from typing import Dict, Any, List
from string import Template
from fastmcp import Client

def render(tpl: str, inputs: Dict[str, Any]) -> str:
    return Template(tpl).substitute(inputs)

def load_mcp_config() -> dict:
    \"\"\"환경 변수에서 MCP 설정을 로드합니다.\"\"\"
    mcp_config_str = os.environ.get("MCP_CONFIG")
    if not mcp_config_str:
        raise RuntimeError("환경 변수 MCP_CONFIG가 설정되지 않았습니다.")
    return json.loads(mcp_config_str)

async def _client_from_server_key(server_key: str) -> Client:
    mcp_config = load_mcp_config()
    server_config = mcp_config["mcpServers"][server_key]
    # Client는 mcpServers 형식을 기대하므로 올바른 구조로 전달
    config = {{"mcpServers": {{server_key: server_config}}}}
    return Client(config)

async def call_tool(server_key: str, tool_name: str, args: Dict[str, Any], timeout_s: int = 60):
    client = await _client_from_server_key(server_key)
    async with client:
        await client.ping()
        res = await asyncio.wait_for(client.call_tool(tool_name, args), timeout=timeout_s)
        safe = json.loads(json.dumps(res.data, ensure_ascii=False, default=str))
        return {{"tool": tool_name, "data": safe, "server": server_key}}

async def run(inputs: Dict[str, Any], timeout_s: int = 60) -> List[Dict[str, Any]]:
    \"\"\"
    생성된 워크플로우를 입력 파라미터로 실행합니다.
    
    Args:
        inputs: 파라미터 딕셔너리 (예: {{"product_name": "iPhone", "stock_quantity": 100}})
        timeout_s: 각 툴 호출의 타임아웃 (초)
    
    Returns:
        각 툴 실행 결과 리스트
    
    Parameters:
{param_docs}
    \"\"\"
    results = []
{steps}
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: 입력 파라미터가 필요합니다.", file=sys.stderr)
        print("사용법: python {{sys.argv[0]}} <JSON_STRING_OR_FILE>", file=sys.stderr)
        print("예시: python {{sys.argv[0]}} '{{\\"param\\": \\"value\\"}}'", file=sys.stderr)
        sys.exit(1)
    
    arg = sys.argv[1]
    
    try:
        # 파일 경로인지 JSON 문자열인지 확인
        if os.path.exists(arg):
            with open(arg, 'r', encoding='utf-8') as f:
                inputs = json.load(f)
        else:
            inputs = json.loads(arg)
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON 파싱 실패: {{e}}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: 입력 처리 실패: {{e}}", file=sys.stderr)
        sys.exit(1)
    
    print(f"실행 시작 - 입력 파라미터: {{inputs}}", file=sys.stderr)
    
    try:
        results = asyncio.run(run(inputs))
        print(f"실행 완료 - 총 {{len(results)}}개의 툴 호출 완료", file=sys.stderr)
        payload = {{"ok": True, "results": results}}
        print(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        print(f"ERROR: 실행 중 오류 발생: {{e}}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
"""

def build_tool_index_from_tenant(tenant_id: str) -> Dict[str, str]:
    """tenant의 MCP 설정을 읽어 tool_name -> server_key 매핑을 구성합니다.
    fastmcp가 사용 가능하면 실제 서버에서 툴 목록을 조회하고, 불가하면 휴리스틱으로 매핑합니다.
    """
    tool_to_server: Dict[str, str] = {}
    try:
        mcp = fetch_tenant_mcp_config(tenant_id) or {}
        servers = (mcp or {}).get("mcpServers", mcp)
        # 런타임 전용 enabled 속성은 MCP 클라이언트 설정이 아니다 — 비활성 서버는 빼고 속성만 지운다(DeepAgents replay 와 같음).
        def _client_cfg(v: Dict[str, Any]) -> Dict[str, Any]:
            # fastmcp 2.x 는 정의되지 않은 키(type·is_default 등)를 거부한다 — 원격은 url·headers·transport, 로컬은 command 계열만.
            if v.get("url"):
                cfg = {"url": v["url"]}
                if v.get("headers"):
                    cfg["headers"] = v["headers"]
                if v.get("transport") or v.get("type"):
                    cfg["transport"] = v.get("transport") or v.get("type")
                return cfg
            return {kk: vv for kk, vv in v.items() if kk in ("command", "args", "env", "cwd")}

        servers = {k: _client_cfg(v) for k, v in servers.items() if isinstance(v, dict) and v.get("enabled", True) is not False}
        try:
            from fastmcp import Client as McpClient  # type: ignore
            import asyncio as _a

            async def _list_for_server(server_k: str, server_cfg: Any):
                config = {"mcpServers": {server_k: server_cfg}}
                client = McpClient(config)
                async with client:
                    await client.ping()
                    tools = await client.list_tools()
                    for t in tools:
                        tool_to_server[t.name] = server_k

            async def _run_all():
                await _a.gather(*[_list_for_server(k, v) for k, v in servers.items()], return_exceptions=True)

            try:
                loop = _a.get_event_loop()
                if loop.is_running():
                    # run in a new loop in thread if already running
                    import threading
                    result_holder: Dict[str, Exception | None] = {"err": None}

                    def runner():
                        try:
                            _a.run(_run_all())
                        except Exception as e:  # noqa: BLE001
                            result_holder["err"] = e

                    t = threading.Thread(target=runner, daemon=True)
                    t.start()
                    t.join()
                    if result_holder["err"]:
                        raise result_holder["err"]
                else:
                    loop.run_until_complete(_run_all())
            except Exception:
                pass
        except Exception:
            server_keys = list(servers.keys())
            fallback_server = server_keys[0] if server_keys else "default"
            gmail_server = next((k for k in server_keys if "gmail" in k.lower()), fallback_server)
            tool_to_server.setdefault("send_email_tool", gmail_server)
    except Exception:
        pass
    return tool_to_server


def generate_deterministic_compensation_code(tenant_id: str, query: str, event_logs: List[Dict[str, Any]]) -> str:
    """
    보상(역행) 처리용 결정론적 MCP 워크플로우 코드를 생성합니다.
    - 이벤트 로그 + 워크아이템 쿼리 → LLM으로 보상 지시문(compensation_handling) 생성
    - steps를 TEMPLATE로 컴파일하여 실행 가능한 스크립트 생성
    """
    user_input_query = query or ''

    # 1) 보상 지시문(결정적 파이썬 코드) 생성: TEMPLATE를 그대로 채워 반환 (구조 동일)
    tool_to_server = build_tool_index_from_tenant(tenant_id)
    tool_map_json = json.dumps(tool_to_server, ensure_ascii=False)

    prompt = (
        "Generate deterministic Python code that performs compensation (undo) from EVENT LOGS ONLY by filling the TEMPLATE.\n"
        "STRICT REQUIREMENTS:\n"
        "- Return ONLY Python code, no markdown.\n"
        "- Keep the TEMPLATE structure (imports, Client, call_tool, run, __main__) exactly.\n"
        "- Inside run(inputs, ...), FIRST reconstruct all necessary parameters by deterministically parsing inputs (which will contain ONLY event logs).\n"
        "- CRITICAL: Parse values dynamically from log['log_data'] - NEVER hardcode specific values.\n"
        "- Extract values using Python string parsing, regex, or JSON parsing from log_data.\n"
        "- For SQL UPDATE/DELETE/INSERT: parse the original query to extract all values and reverse them.\n"
        "- SQL reversal examples:\n"
        "  * 'UPDATE product SET stock_quantity = stock_quantity - 20 WHERE product_name = \"노트북\"'\n"
        "    → Parse: extract 20 (number), product_name = '노트북' from WHERE clause\n"
        "    → Reverse: 'UPDATE product SET stock_quantity = stock_quantity + 20 WHERE product_name = \"노트북\"'\n"
        "  * Use regex or string parsing like: re.search(r'SET stock_quantity = stock_quantity - (\\d+)', query) to extract numbers\n"
        "  * Parse WHERE clause to get conditions: re.search(r\"WHERE product_name = ['\"](.*?)['\"]\", query)\n"
        "- For email: extract recipient, subject from args and reverse the email action.\n"
        "- For file operations: extract filenames, paths from args and reverse the operation.\n"
        "- Example reconstruction code pattern (NO HARDCODED VALUES):\n"
        "  import re\n"
        "  event_logs = inputs['event_logs']\n"
        "  for log in event_logs:\n"
        "      log_data = log['log_data']\n"
        "      tool_name = log_data.get('tool_name')\n"
        "      args = log_data.get('args', {})\n"
        "      query = args.get('query', '')\n"
        "      if tool_name == 'execute_sql' and query:\n"
        "          # Parse query to extract values dynamically\n"
        "          # Example: match = re.search(r'- (\\d+)', query); quantity = int(match.group(1)) if match else 0\n"
        "          # Reverse the operation using extracted values\n"
        "- After reconstruction, implement reverse steps by calling call_tool(server_key, tool_name, args).\n"
        "- Replace {param_docs} with a concise list of parameters expected in inputs (e.g., event_log schema).\n"
        "- Replace {steps} with actual tool calls using call_tool(server_key, tool_name, args).\n"
        "- Use ONLY tools present in this tool_to_server mapping (tool_name -> server_key):\n" + tool_map_json + "\n\n"
        "TEMPLATE:\n" + TEMPLATE + "\n\n"
        "Context JSON (event_logs only):\n" + json.dumps({"event_logs": event_logs, "user_input_query": user_input_query}, ensure_ascii=False)
    )

    try:
        generator = create_llm(streaming=False, temperature=0)
        resp = generator.invoke(prompt)
        code_str = getattr(resp, 'content', None) or str(resp)
        if "async def run(" in code_str and "call_tool(" in code_str:
            return code_str
        # Fallback: return empty template if generation fails
        fallback_template = TEMPLATE.replace("{param_docs}", "        None")
        fallback_template = fallback_template.replace("{steps}", "")
        return fallback_template
    except Exception:
        # Fallback: return empty template if generation fails
        fallback_template = TEMPLATE.replace("{param_docs}", "        None")
        fallback_template = fallback_template.replace("{steps}", "")
        return fallback_template


async def generate_compensation(workitem, new_workitem):
    try:
        if workitem is None:
            raise Exception("Workitem is None")
        
        deterministic_code = fetch_mcp_python_code(workitem.proc_def_id, workitem.activity_id, workitem.tenant_id)
        if deterministic_code and deterministic_code.get("compensation") is not None:
            return
        
        # 현재 액티비티까지의 워크아이템 이벤트만 가져옴
        events = fetch_events_by_proc_inst_id_until_activity(
            workitem.proc_def_id,
            workitem.proc_inst_id,
            workitem.activity_id,
            workitem.tenant_id
        )
        
        if len(events) == 0:
            return
        
        # 보상 코드는 테넌트 MCP 도구로만 되돌린다 — 실행기 내장 도구(task·read_file 등)는 입력에서 뺀다.
        mcp_tools = set(build_tool_index_from_tenant(workitem.tenant_id).keys())
        event_logs: List[Dict[str, Any]] = []
        for event in events:
            # Only include finished tool usage actions (CrewAI Action · DeepAgents)
            if event.get("event_type") != "tool_usage_finished":
                continue
            if event.get("crew_type") not in ("action", "deepagents"):
                continue

            data_raw = event.get("data")
            if not data_raw:
                continue

            data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
            tool_name = (data or {}).get("tool_name")
            args = (data or {}).get("args") or {}

            if not tool_name:
                continue
            if event.get("crew_type") == "deepagents" and mcp_tools and tool_name not in mcp_tools:
                continue
            # Exclude specific tools
            if tool_name in ("mem0", "memento", "human_asked", "dmn_rule"):
                continue
            # Exclude execute_sql that only performs SELECT
            if tool_name == "execute_sql":
                query = args.get("query", "")
                if isinstance(query, str) and query.strip().upper().startswith("SELECT"):
                    continue
            event_logs.append({"timestamp": event.get("timestamp"), "log_data": data})
        
        if len(event_logs) == 0:
            return
        
        user_input_query = workitem.query
        if user_input_query is None:
            user_input_query = ''
        
        compensation_code = generate_deterministic_compensation_code(workitem.tenant_id, workitem.query or '', event_logs)
        if compensation_code is None:
            return
        else:
            if deterministic_code:
                deterministic_code["compensation"] = compensation_code
                upsert_mcp_python_code(deterministic_code)
            else:
                upsert_mcp_python_code({
                    "compensation": compensation_code,
                    "proc_def_id": workitem.proc_def_id,
                    "activity_id": workitem.activity_id,
                    "tenant_id": workitem.tenant_id
                })
        
            user_id = None
            user_name = None
            if workitem.assignees and len(workitem.assignees) > 0:
                assignee_id = workitem.assignees[0].get('endpoint')
                if isinstance(assignee_id, list):
                    user_list = []
                    for id in assignee_id:
                        user_info = fetch_user_info_by_uid(id)
                        if user_info:
                            user_list.append(user_info)
                    user_id = ','.join([user.get('id') for user in user_list])
                    user_name = ','.join([user.get('username') for user in user_list])
            else:
                user_id = workitem.user_id
                user_name = workitem.username

            upsert_workitem({
                "id": new_workitem.get('id'),
                "status": "IN_PROGRESS",
                "user_id": user_id,
                "username": user_name,
                # 원래 실행기를 유지한다 — DeepAgents 활동은 DeepAgents 가 undo → 순방향을 실행한다(deepagents_deterministic-replay-undo).
                "agent_orch": getattr(workitem, "agent_orch", None) or "crewai-action",
                "log": "Compensation Handling..."
            })

    except Exception as e:
        print(f"[ERROR] Failed to handle compensation: {str(e)}")
        raise Exception(f"Compensation handling failed: {str(e)}") from e


