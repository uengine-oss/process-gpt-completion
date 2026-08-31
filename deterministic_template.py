"""생성 코드의 실행 골격.

순방향(고착화) 코드와 역방향(보상) 코드는 같은 골격 위에서 돈다. 골격이 갈리면
실행 런타임이 두 가지 실행 방식을 따로 지원해야 하므로 한 곳에 둔다.

골격이 제공하는 원시 동작은 작업 이력에서 관측되는 부수효과의 종류와 1:1로 맞춘다.

- ``call_tool``  MCP 도구 호출
- ``run_shell``  셸 명령 실행
- ``write_file`` 파일 생성·덮어쓰기
- ``edit_file``  파일 부분 치환
- ``remove_file`` 파일 삭제
- ``move_file`` / ``copy_file`` / ``make_dir`` 파일 이동·복사·디렉터리 생성
- ``read_file``  파일 읽기(보상 코드가 되돌릴 내용을 확인할 때 쓴다)

파일 조작을 ``write_file`` 하나로 뭉뚱그리면 삭제가 "빈 파일 만들기"로 재현된다.
관측된 조작마다 대응하는 동작이 있어야 한다.

MCP 도구만 호출할 수 있게 두면 스킬 절차를 셸 스크립트로 수행한 활동은 영영
고착화되지 않는다. 실제 작업 이력에는 셸과 파일 조작이 함께 남는다.
"""

TEMPLATE = '''# -*- coding: utf-8 -*-
# {header}
import os
import sys
import json
import asyncio
from typing import Dict, Any, List
from string import Template

# 셸·파일 작업의 기준 디렉터리. 실행 런타임이 워크스페이스 경로를 넣어 준다.
WORKDIR = os.environ.get("DETERMINISTIC_WORKDIR") or os.getcwd()


def render(tpl: str, inputs: Dict[str, Any]) -> str:
    return Template(tpl).substitute(inputs)


def load_mcp_config() -> dict:
    mcp_config_str = os.environ.get("MCP_CONFIG")
    if not mcp_config_str:
        raise RuntimeError("환경 변수 MCP_CONFIG가 설정되지 않았습니다.")
    return json.loads(mcp_config_str)


async def _client_from_server_key(server_key: str):
    from fastmcp import Client

    mcp_config = load_mcp_config()
    server_config = mcp_config["mcpServers"][server_key]
    return Client({{"mcpServers": {{server_key: server_config}}}})


async def call_tool(server_key: str, tool_name: str, args: Dict[str, Any], timeout_s: int = 60):
    client = await _client_from_server_key(server_key)
    async with client:
        await client.ping()
        res = await asyncio.wait_for(client.call_tool(tool_name, args), timeout=timeout_s)
        safe = json.loads(json.dumps(res.data, ensure_ascii=False, default=str))
        return {{"kind": "mcp_call", "tool": tool_name, "data": safe, "server": server_key}}


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(WORKDIR, path)


async def run_shell(command: str, cwd: str = "", timeout_s: int = 300):
    """셸 명령을 실행한다. 0이 아닌 종료 코드는 실패로 본다."""
    workdir = _resolve(cwd) if cwd else WORKDIR
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=workdir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    output = (stdout or b"").decode("utf-8", errors="replace")
    if proc.returncode:
        raise RuntimeError(
            "셸 명령 실패(exit={{0}}): {{1}}\\n{{2}}".format(proc.returncode, command, output[-2000:])
        )
    return {{"kind": "shell", "command": command, "cwd": workdir, "output": output[-4000:]}}


async def write_file(path: str, content: str):
    target = _resolve(path)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(content)
    return {{"kind": "file_write", "path": target, "bytes": len(content.encode("utf-8"))}}


async def edit_file(path: str, old_string: str, new_string: str):
    target = _resolve(path)
    with open(target, "r", encoding="utf-8") as handle:
        body = handle.read()
    if old_string not in body:
        raise RuntimeError("치환 대상을 찾지 못했습니다: {{0}}".format(target))
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(body.replace(old_string, new_string, 1))
    return {{"kind": "file_write", "path": target, "mode": "edit"}}


async def remove_file(path: str):
    target = _resolve(path)
    existed = os.path.exists(target)
    if existed:
        os.remove(target)
    return {{"kind": "file_write", "op": "delete", "path": target, "existed": existed}}


async def move_file(source: str, destination: str):
    src, dst = _resolve(source), _resolve(destination)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    os.replace(src, dst)
    return {{"kind": "file_write", "op": "move", "source": src, "destination": dst}}


async def copy_file(source: str, destination: str):
    import shutil

    src, dst = _resolve(source), _resolve(destination)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copyfile(src, dst)
    return {{"kind": "file_write", "op": "copy", "source": src, "destination": dst}}


async def make_dir(path: str):
    target = _resolve(path)
    os.makedirs(target, exist_ok=True)
    return {{"kind": "file_write", "op": "mkdir", "path": target}}


async def read_file(path: str):
    target = _resolve(path)
    with open(target, "r", encoding="utf-8") as handle:
        return {{"kind": "file_read", "path": target, "content": handle.read()}}


async def run(inputs: Dict[str, Any], timeout_s: int = 60) -> List[Dict[str, Any]]:
    """관측된 작업 이력을 입력 파라미터로 재현한다.

    Parameters:
{param_docs}
    """
    results = []
{steps}
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: 입력 파라미터가 필요합니다.", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]
    try:
        if os.path.exists(arg):
            with open(arg, "r", encoding="utf-8") as f:
                inputs = json.load(f)
        else:
            inputs = json.loads(arg)
    except Exception as e:
        print("ERROR: 입력 처리 실패: {{0}}".format(e), file=sys.stderr)
        sys.exit(1)

    try:
        results = asyncio.run(run(inputs))
        print(json.dumps({{"ok": True, "results": results}}, ensure_ascii=False))
    except Exception as e:
        print("ERROR: 실행 중 오류 발생: {{0}}".format(e), file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
'''


def empty_script(header: str = "generated (no steps)") -> str:
    """단계가 없는 골격. 생성 실패 시의 안전한 폴백."""
    return TEMPLATE.format(header=header, param_docs="        None", steps="    pass")


def skeleton() -> str:
    """LLM에게 보여줄 골격.

    `.format()` 템플릿을 그대로 보여주면 이중 중괄호가 그대로 코드에 베껴진다.
    한 번 렌더한 결과를 보여 주고 `run()` 본문만 채우게 한다.
    """
    return TEMPLATE.format(
        header="compensation.py (fill in the body of run())",
        param_docs="        <입력 파라미터 설명을 여기에 채운다>",
        steps="    # <되돌리기 단계를 여기에 채운다>",
    )
