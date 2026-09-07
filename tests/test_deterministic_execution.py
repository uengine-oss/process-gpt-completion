"""생성된 코드가 실제로 실행되는지, 그리고 실행 결과가 런타임이 읽는 모양인지 고정한다.

실행 런타임(DeepAgents 등)은 생성 코드를 별도 프로세스로 돌리고 **마지막 출력 줄**을
JSON으로 읽어 워크아이템 결과에 반영한다. 그 계약이 깨지면 코드가 제 일을 다 하고도
결과가 사라진다. 여기서는 스텁이 아니라 진짜 파이썬 프로세스로 돌려 확인한다.
"""

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from deterministic_generator import compile_code
from deterministic_signature import identify_parameters
from work_history import to_action


def _sample(qty, name, path):
    """스킬 절차대로 셸을 돌리고 파일을 쓴 실행 한 건."""
    return [
        to_action("execute", {"command": f"printf '%s' {qty} > count.txt"}),
        to_action("write_file", {"file_path": path, "content": f"입고 {name}"}),
    ]


def _build_script():
    samples = [
        _sample(10, "노트북", "out-a.txt"),
        _sample(20, "마우스", "out-b.txt"),
        _sample(30, "키보드", "out-c.txt"),
    ]
    plan = identify_parameters(samples)
    return compile_code("todo-1", samples[0], {}, plan, {"sample_count": 3}), plan


def _run(script: str, inputs: dict, cwd: Path):
    path = cwd / "generated.py"
    path.write_text(script, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(path), json.dumps(inputs, ensure_ascii=False)],
        capture_output=True, text=True, cwd=cwd,
        env={"PATH": "/usr/bin:/bin", "DETERMINISTIC_WORKDIR": str(cwd), "LANG": "C.UTF-8"},
    )


def test_generated_script_is_valid_python():
    script, _plan = _build_script()
    ast.parse(script)


def test_generated_script_performs_shell_and_file_work(tmp_path):
    """LLM 없이, 관측된 셸·파일 작업을 새 입력으로 그대로 재현한다.

    파일명 `out-a.txt` / `out-b.txt` 에서 실제로 변한 것은 가운데 한 글자뿐이므로,
    파라미터도 그 토막만 받는다 — 호출자가 파일명 규칙을 알 필요가 없다.
    """
    script, plan = _build_script()
    inputs = {
        param["name"]: 99 if param["type"] == "integer" else "z"
        for param in plan.parameters
    }

    result = _run(script, inputs, tmp_path)
    assert result.returncode == 0, result.stderr

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert [step["kind"] for step in payload["results"]] == ["shell", "file_write"]

    # 실제로 세상이 바뀌었다. 파일명의 상수 부분(`out-`, `.txt`)은 코드에 굳어 있다.
    assert (tmp_path / "count.txt").read_text() == "99"
    assert (tmp_path / "out-z.txt").exists()


def test_result_envelope_is_the_last_stdout_line(tmp_path):
    """실행 도구는 마지막 줄만 읽는다. 그 앞에 무엇이 찍혀도 결과를 잃지 않아야 한다."""
    script, plan = _build_script()
    inputs = {p["name"]: (1 if p["type"] == "integer" else "x.txt") for p in plan.parameters}
    result = _run(script, inputs, tmp_path)
    last = result.stdout.strip().splitlines()[-1]
    assert json.loads(last)["ok"] is True


def test_failing_step_exits_nonzero_so_the_runtime_can_fall_back(tmp_path):
    """실패는 조용히 성공으로 보고되면 안 된다 — 런타임이 폴백을 결정할 근거다."""
    broken = [to_action("execute", {"command": "exit 3"})]
    plan = identify_parameters([broken, broken, broken])
    script = compile_code("todo-2", broken, {}, plan, {})
    result = _run(script, {}, tmp_path)
    assert result.returncode != 0
    assert "셸 명령 실패" in result.stderr


def test_missing_input_fails_loudly(tmp_path):
    """파라미터가 빠지면 조용히 `${name}` 을 그대로 실행하지 않는다."""
    script, plan = _build_script()
    assert plan.parameters  # 파라미터가 있어야 의미 있는 검사다
    result = _run(script, {}, tmp_path)
    assert result.returncode != 0


def test_compensation_script_can_undo_file_and_shell_work(tmp_path):
    """보상 코드도 같은 골격 위에서 돈다 — 셸·파일 되돌리기가 가능해야 한다."""
    from deterministic_template import TEMPLATE

    (tmp_path / "산출물.txt").write_text("입고 노트북", encoding="utf-8")
    (tmp_path / "count.txt").write_text("20", encoding="utf-8")

    steps = textwrap.dedent("""
        results.append(await run_shell("rm -f 산출물.txt", ""))
        results.append(await write_file("count.txt", "0"))
    """).strip("\n")
    script = TEMPLATE.format(
        header="compensation test",
        param_docs="        None",
        steps=textwrap.indent(steps, "    "),
    )
    result = _run(script, {"event_logs": []}, tmp_path)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "산출물.txt").exists()
    assert (tmp_path / "count.txt").read_text() == "0"
