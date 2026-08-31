"""보상(역방향) 결정론적 코드 생성.

순방향 고착화(`deterministic_generator`)와 같은 작업 이력을 읽는다. 다른 점은 그
이력을 재현하는 대신 거꾸로 되돌린다는 것뿐이다. 그래서 이력 해석은 같은
`work_history` 정규화를 쓰고, 실행 골격도 `deterministic_template`을 공유한다.

되돌릴 대상은 MCP 호출만이 아니다. 스킬의 셸 스크립트가 만든 파일, 스크립트가 건드린
외부 상태도 함께 남는다. 그래서 보상 코드에도 `run_shell`/`write_file` 원시 동작을
그대로 준다.
"""

from typing import Dict, Any, List
import json
import os

from llm_factory import create_llm
from database import fetch_mcp_python_code, upsert_mcp_python_code, fetch_events_by_proc_inst_id_until_activity, upsert_workitem, fetch_user_info_by_uid
from mcp_tool_index import build_tool_index_from_tenant
from deterministic_template import empty_script, skeleton
from work_history import normalize_events, summarize, to_log_entries

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

def _strip_code_fence(code: str) -> str:
    """LLM 응답의 마크다운 코드펜스를 제거한다.

    "코드만 돌려달라"고 해도 펜스를 붙여 오는 경우가 있다. 그대로 파일로 써서 실행하면
    첫 줄에서 문법 오류가 난다.
    """
    text = (code or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    lines = lines[1:]  # ``` 또는 ```python 여는 줄
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines).strip()


# 생성된 보상 코드가 갖춰야 할 최소 조건. 골격의 원시 동작 중 하나라도 쓰지 않으면
# 되돌리는 일을 실제로 하지 않는 껍데기다.
_UNDO_PRIMITIVES = ("call_tool(", "run_shell(", "write_file(", "edit_file(")


def _is_usable_compensation(code: str) -> bool:
    if "async def run(" not in code:
        return False
    if not any(primitive in code for primitive in _UNDO_PRIMITIVES):
        return False
    try:
        compile(code, "compensation.py", "exec")
    except SyntaxError:
        return False
    return True


def generate_deterministic_compensation_code(
    tenant_id: str,
    query: str,
    trace_entries: List[Dict[str, Any]],
    provenance: Dict[str, Any] | None = None,
) -> str:
    """작업 이력을 되돌리는 결정론적 파이썬 코드를 생성한다.

    입력은 MCP 호출 목록이 아니라 **정규화된 작업 이력**이다. 각 항목에는 종류(kind)가
    붙어 있어, 셸 실행은 셸로, 파일 생성은 파일 삭제·복원으로 되돌리도록 지시할 수
    있다. 맥락 항목(읽은 스킬·파일)은 되돌릴 대상은 아니지만 "왜 그렇게 했는지"를
    알려 주어 올바른 역순서를 잡는 데 쓰인다.
    """
    tool_to_server = build_tool_index_from_tenant(tenant_id)
    tool_map_json = json.dumps(tool_to_server, ensure_ascii=False)

    prompt = (
        "You are given the NORMALIZED WORK HISTORY of a completed workitem that must now be undone.\n"
        "Generate deterministic Python code that reverses it, by filling in the body of run() in the SKELETON.\n"
        "\n"
        "STRICT REQUIREMENTS:\n"
        "- Return ONLY Python code, no markdown fences.\n"
        "- Keep the SKELETON structure exactly (imports, helpers, run, __main__). Only replace the\n"
        "  Parameters docstring section and the body of run().\n"
        "- inputs contains ONLY {'event_logs': [...]} — the same work history shown below at undo time.\n"
        "  Parse values out of it at runtime with string/regex/JSON parsing. NEVER hardcode observed values.\n"
        "- Undo in REVERSE chronological order: the last effect is undone first.\n"
        "\n"
        "Each history entry has a 'kind'. Reverse it with the matching primitive:\n"
        "- kind 'mcp_call'   → await call_tool(server_key, tool_name, args)\n"
        "    * SQL is PostgreSQL. Single quotes are string literals, double quotes are identifiers.\n"
        "    * UPDATE t SET c = c - N WHERE k = 'v'  → UPDATE t SET c = c + N WHERE k = 'v'\n"
        "    * INSERT INTO t (...) VALUES (...)      → DELETE FROM t WHERE <the inserted key>\n"
        "    * DELETE FROM t WHERE ...               → cannot be reversed without the old row; skip it\n"
        "      and record the reason in results instead of failing.\n"
        "    * Non-SQL tools (email, messaging): send a correction/cancellation using the same args.\n"
        "- kind 'shell'      → await run_shell(command, cwd) with the command that reverses the original\n"
        "    (git revert/checkout, rm of a file the script created, restoring a backup, a compensating script).\n"
        "- kind 'file_write' → await write_file(path, previous_content) to restore, or run_shell('rm -f <path>')\n"
        "    when the file did not exist before. Use await read_file(path) first if you need current content.\n"
        "- kind 'skill_read' / 'file_read' / 'inspect' / 'plan' / 'delegate' → CONTEXT ONLY. Never 'undo' a read.\n"
        "    Use them to understand which procedure was followed and what the reverse should look like.\n"
        "\n"
        "- Use ONLY tools present in this tool_to_server mapping (tool_name -> server_key):\n" + tool_map_json + "\n"
        "- Append a dict to results for every step you take, including steps you deliberately skip.\n"
        "\n"
        "SKELETON:\n" + skeleton() + "\n\n"
        "WORK HISTORY SUMMARY (which tools/skills/shell scripts this activity used):\n"
        + json.dumps(provenance or {}, ensure_ascii=False)
        + "\n\nWORK HISTORY (chronological):\n"
        + json.dumps(
            {"event_logs": trace_entries, "user_input_query": query or ""}, ensure_ascii=False
        )
    )

    try:
        generator = create_llm(streaming=False, temperature=0)
        resp = generator.invoke(prompt)
        code_str = _strip_code_fence(getattr(resp, 'content', None) or str(resp))
        if _is_usable_compensation(code_str):
            return code_str
    except Exception:
        pass
    # 생성이 실패하면 아무 일도 하지 않는 골격을 남긴다. 실행해도 부수효과가 없으므로
    # 되돌리기는 건너뛰고 재실행은 에이전트가 맡는다 — 워크아이템은 멈추지 않는다.
    return empty_script("compensation.py (generation failed — no-op)")


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
        
        # 이벤트 봉투 해석은 런타임에 매이지 않는다. crewai-action은 crew_type "action"
        # 으로, DeepAgents는 "deepagents"로 발행하고 데이터 모양도 다르다. 특정 런타임의
        # 스키마로 걸러내면 살아있는 다른 런타임의 이력이 전부 버려진다.
        trace = normalize_events(events)
        if not any(action.has_effect for action in trace):
            # 되돌릴 부수효과가 없다. 조회와 읽기만 한 활동은 보상할 것이 없다.
            return

        trace_entries = to_log_entries(trace)
        provenance = summarize(trace)

        compensation_code = generate_deterministic_compensation_code(
            workitem.tenant_id, workitem.query or '', trace_entries, provenance
        )
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
        
            user_id = workitem.user_id
            user_name = workitem.username
            if workitem.assignees and len(workitem.assignees) > 0:
                assignee_id = workitem.assignees[0].get('endpoint')
                # endpoint는 문자열 하나이거나 목록일 수 있다. 목록만 처리하면 단일
                # 담당자 워크아이템의 user_id/username이 None으로 덮인다.
                endpoints = assignee_id if isinstance(assignee_id, list) else (
                    [assignee_id] if assignee_id else []
                )
                user_list = []
                for endpoint in endpoints:
                    user_info = fetch_user_info_by_uid(endpoint)
                    if user_info:
                        user_list.append(user_info)
                if user_list:
                    user_id = ','.join([user.get('id') for user in user_list])
                    user_name = ','.join([user.get('username') for user in user_list])

            upsert_workitem({
                "id": new_workitem.get('id'),
                "status": "IN_PROGRESS",
                "user_id": user_id,
                "username": user_name,
                # 실행 런타임을 바꾸지 않는다. 퇴역한 crewai-action으로 찍으면 어떤
                # 폴링 워커도 이 워크아이템을 가져가지 않아 영구히 멈춘다.
                "agent_orch": workitem.agent_orch,
                "log": "Compensation Handling..."
            })

    except Exception as e:
        print(f"[ERROR] Failed to handle compensation: {str(e)}")
        raise Exception(f"Compensation handling failed: {str(e)}") from e


