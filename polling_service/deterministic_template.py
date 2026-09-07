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
- ``from_step``  앞 단계의 결과에서 값 이어받기

인자가 늘 밖에서 오는 것은 아니다. 앞 단계가 만들어 낸 값(찍은 시각, 조회 결과)이
뒤 단계의 인자로 흐르는 활동이 있다. 그런 자리는 ``from_step`` 으로 잇는다 — 값이
실행마다 달라도 **이번 실행의** 값으로 채워진다.

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


def render(tpl: str, inputs: Dict[str, Any], linked: Dict[str, Any] = None) -> str:
    """템플릿을 이번 실행의 값으로 채운다.

    ``linked`` 는 앞 단계에서 이어받은 값이다. 밖에서 받은 입력과 섞어 쓰되, 같은
    이름이 겹치지 않도록 생성 단계에서 이름을 갈라 둔다.
    """
    values = dict(inputs)
    values.update(linked or {{}})
    return Template(tpl).substitute(values)


def _descend(node, key):
    """앞 단계 결과 한 겹을 들어간다. 자리를 못 찾으면 실패한다.

    JSON 문자열로 온 결과는 풀어서 들어간다 — 러너와 도구에 따라 구조 그대로 오기도,
    문자열로 오기도 한다.

    비슷한 자리를 뒤져 대신 채우지 않는다. 이어받을 자리를 못 찾았다는 것은 앞 단계가
    지난번과 다른 것을 냈다는 뜻이고, 그때는 엉뚱한 값으로 도구를 부르느니 실패해서
    에이전트에게 넘기는 편이 낫다.
    """
    if isinstance(node, str):
        try:
            node = json.loads(node)
        except ValueError:
            raise RuntimeError("앞 단계 결과를 구조로 읽을 수 없습니다: {{0}}".format(key))
    if isinstance(node, dict):
        if key in node:
            return node[key]
    elif isinstance(node, list) and isinstance(key, int) and -len(node) <= key < len(node):
        return node[key]
    raise RuntimeError("앞 단계 결과에서 '{{0}}' 자리를 찾지 못했습니다.".format(key))


def from_step(results: List[Dict[str, Any]], index: int, root: str, path=(), line=None):
    """앞 단계의 결과에서 값 하나를 이어받는다.

    ``index`` 는 이 실행 안에서의 단계 번호이고, ``root`` 는 그 단계 결과에서 본문이
    실린 키다(셸은 ``output``, MCP 도구는 ``data``). ``line`` 은 본문이 여러 줄일 때
    고를 줄이다.

    자리는 생성 단계에서 표본 전부를 대조해 정한 것이다. 그 자리가 비어 있으면 값을
    지어내지 않고 실패한다.
    """
    if index < 0 or index >= len(results):
        raise RuntimeError("앞 단계 결과가 없습니다: results[{{0}}]".format(index))
    node = results[index]
    if root:
        node = _descend(node, root)
    for key in path:
        node = _descend(node, key)
    if line is not None:
        lines = str(node).splitlines()
        if line >= len(lines):
            raise RuntimeError("앞 단계 결과에 {{0}}번째 줄이 없습니다.".format(line))
        node = lines[line]
    if node is None or (isinstance(node, str) and not node.strip()):
        raise RuntimeError("앞 단계 결과에서 이어받을 값이 비어 있습니다.")
    return node.strip() if isinstance(node, str) else node


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
        try:
            await client.ping()
        except Exception:
            # ping 은 MCP 스펙의 선택 기능이다. 구현하지 않은 서버는 "Method not found" 를
            # 돌려주는데, 그것을 연결 실패로 읽으면 도구를 불러 보지도 못하고 실행이 통째로
            # 죽는다. 같은 서버를 에이전트는 잘 쓴다 — 에이전트 쪽 클라이언트는 ping 을
            # 하지 않기 때문이다. 살아 있는지는 바로 다음 줄의 실제 호출이 말해 준다.
            pass
        res = await asyncio.wait_for(client.call_tool(tool_name, args), timeout=timeout_s)
        safe = json.loads(json.dumps(res.data, ensure_ascii=False, default=str))
        # 인자를 함께 남긴다. 이 결과가 다음 재작업의 **이력**이 된다 — 무엇을 넣었는지가
        # 없으면 되돌릴 방법도 없어 재작업 전체가 에이전트에게 넘어간다.
        sent = json.loads(json.dumps(args, ensure_ascii=False, default=str))
        return {{
            "kind": "mcp_call", "tool": tool_name, "args": sent,
            "data": safe, "server": server_key,
        }}


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
    return {{
        "kind": "file_write", "op": "write", "path": target,
        "bytes": len(content.encode("utf-8")),
    }}


async def edit_file(path: str, old_string: str, new_string: str):
    target = _resolve(path)
    with open(target, "r", encoding="utf-8") as handle:
        body = handle.read()
    if old_string not in body:
        raise RuntimeError("치환 대상을 찾지 못했습니다: {{0}}".format(target))
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(body.replace(old_string, new_string, 1))
    return {{"kind": "file_write", "op": "edit", "path": target}}


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


# 되돌리기 실행기의 본문. 활동마다 다른 코드를 짓지 않는다 — 무엇을 되돌릴지는 관측에서
# 이미 정해져(`compensation.invert`) 단계 목록으로 넘어오므로, 저장되는 코드는 그 단계를
# 그대로 수행하는 한 벌이면 된다. 활동마다 코드를 지어내던 옛 방식은 그 지어내기가
# 곧 추측이었다.
_UNDO_BODY = """    servers = {servers}
    for step in inputs.get("undo_steps") or []:
        kind = str(step.get("kind") or "")
        if kind == "mcp_call":
            tool = str(step.get("tool") or "")
            server = servers.get(tool)
            if not server:
                raise RuntimeError("되돌릴 도구의 서버를 찾지 못했습니다: " + tool)
            outcome = await call_tool(server, tool, step.get("args") or {{}}, timeout_s=timeout_s)
        elif kind == "file_delete":
            outcome = await remove_file(step.get("path") or "")
        elif kind == "file_restore":
            outcome = await write_file(step.get("path") or "", step.get("content") or "")
        else:
            raise RuntimeError("알 수 없는 되돌리기 단계입니다: " + kind)
        # 무엇을 되돌렸는지 결과에 남긴다. 재작업 결과 카드에서 사람이 확인한다.
        outcome["undone"] = step.get("describes") or ""
        results.append(outcome)"""


def undo_script(tool_to_server: dict) -> str:
    """되돌리기 단계 목록을 수행하는 코드.

    `inputs["undo_steps"]` 로 단계를 받는다. 활동마다 내용이 다른 것은 단계 목록이지
    코드가 아니다.
    """
    import json as _json

    header = "compensation.py (auto-created from observed work history)"
    body = _UNDO_BODY.format(
        servers=_json.dumps(dict(tool_to_server or {}), ensure_ascii=False)
    )
    return TEMPLATE.format(
        header=header,
        param_docs='        - undo_steps (list): 되돌릴 단계 목록. 관측된 이력에서 만들어진다.',
        steps=body,
    )


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
