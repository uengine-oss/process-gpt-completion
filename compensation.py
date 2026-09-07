# -*- coding: utf-8 -*-
"""관측된 작업 이력에서 되돌리기 단계를 만든다. LLM 없이.

순방향 고착화는 관측으로 코드를 만드는데 되돌리기는 LLM이 추측으로 지어 왔다. 그러면
재작업 경로 전체가 추측 위에 선다 — 무엇이 지워졌는지 아무도 모르는 채 다시 실행되고,
같은 내용이 두 번 반영되거나 지워지지 않은 채 남는다.

여기서는 되돌릴 수 있는 행위만 관측에서 정확히 뒤집고, 나머지는 **되돌릴 수 없다고
말한다**. 지어낸 되돌리기보다 "못 되돌린다"가 낫다 — 못 되돌린다고 하면 호출자가
재작업 전체를 에이전트에게 넘기지만, 지어낸 되돌리기는 절반만 되돌린 세상을 남긴다.

되돌릴 수 있는 것:

- ``INSERT`` 한 행 → 넣은 값 전부를 조건으로 하는 ``DELETE``
- 제자리 증감 ``UPDATE`` (``SET c = c + n``) → 부호를 뒤집은 같은 구문
- 이 실행이 **만든** 파일 → 삭제
- 이 실행이 **덮어쓴** 파일 중 덮어쓰기 전 내용을 읽어 둔 것 → 그 내용으로 복원

그 밖에는 전부 되돌릴 수 없다. 메일 발송, 값을 덮어쓴 ``UPDATE``, ``DELETE``, 셸 실행이
그렇다 — 이전 상태를 관측하지 않았으므로 무엇으로 되돌려야 할지 알 방법이 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from deterministic_signature import invert_sql, looks_like_sql
from work_history import FILE_READ, FILE_WRITE, MCP_CALL, SHELL, Action

# 되돌리기 한 단계의 종류. 실행 골격의 원시 동작과 1:1로 맞춘다.
UNDO_TOOL = "mcp_call"
UNDO_DELETE_FILE = "file_delete"
UNDO_RESTORE_FILE = "file_restore"


@dataclass(frozen=True)
class UndoStep:
    """되돌리기 한 단계. 실행기가 그대로 수행할 수 있는 모양이다."""

    kind: str
    # 무엇을 되돌리는지 사람이 읽을 설명. 결과에 남아 "무엇을 되돌렸는지"가 보인다.
    describes: str
    tool: str = ""
    args: dict[str, Any] | None = None
    path: str = ""
    content: str = ""

    def as_dict(self) -> dict[str, Any]:
        step: dict[str, Any] = {"kind": self.kind, "describes": self.describes}
        if self.kind == UNDO_TOOL:
            step["tool"] = self.tool
            step["args"] = dict(self.args or {})
        elif self.kind == UNDO_DELETE_FILE:
            step["path"] = self.path
        elif self.kind == UNDO_RESTORE_FILE:
            step["path"] = self.path
            step["content"] = self.content
        return step


def _sql_argument(action: Action) -> tuple[str, Any] | None:
    """이 호출의 인자 중 SQL인 것. 인자 이름은 테넌트 구성마다 다르므로 내용으로 찾는다."""
    for key in sorted(action.args):
        value = action.args[key]
        if looks_like_sql(value):
            return key, value
    return None


def _known_content_before(path: str, earlier: list[Action]) -> str | None:
    """덮어쓰기 전 그 파일의 내용을 이 실행에서 읽어 두었는가.

    읽어 두었다면 되돌리기는 삭제가 아니라 복원이다. 읽은 적이 없으면 그 파일이 전에
    있었는지조차 알 수 없다.
    """
    for action in reversed(earlier):
        if action.kind != FILE_READ or str(action.args.get("path") or "") != path:
            continue
        result = action.result
        if isinstance(result, dict):
            for key in ("content", "text", "data"):
                if isinstance(result.get(key), str):
                    return result[key]
        elif isinstance(result, str):
            return result
        return None
    return None


def _invert_one(action: Action, earlier: list[Action]) -> tuple[UndoStep | None, str]:
    """행위 하나를 되돌리는 단계. 되돌릴 수 없으면 (None, 사유)."""
    if action.kind == MCP_CALL:
        found = _sql_argument(action)
        if found is None:
            # 메일 발송처럼 바깥 세상에 나간 호출. 취소 방법은 도구마다 다르고,
            # 관측에는 그 방법이 없다.
            return None, f"도구 호출을 되돌릴 방법이 이력에 없습니다: {action.tool}"
        key, sql = found
        reversed_sql = invert_sql(sql)
        if reversed_sql is None:
            return None, f"되돌릴 수 없는 SQL입니다: {str(sql).strip()[:60]}"
        args = dict(action.args)
        args[key] = reversed_sql
        return UndoStep(
            kind=UNDO_TOOL,
            describes=f"{action.tool}: {reversed_sql[:80]}",
            tool=action.tool,
            args=args,
        ), ""

    if action.kind == FILE_WRITE:
        operation = str(action.args.get("op") or "write")
        path = str(action.args.get("path") or "")
        if operation != "write" or not path:
            # 삭제·이동·부분 수정은 이전 상태를 알아야 되돌릴 수 있다.
            return None, f"되돌릴 수 없는 파일 조작입니다: {operation} {path}".strip()
        previous = _known_content_before(path, earlier)
        if previous is not None:
            return UndoStep(
                kind=UNDO_RESTORE_FILE,
                describes=f"파일 복원: {path}",
                path=path,
                content=previous,
            ), ""
        return UndoStep(
            kind=UNDO_DELETE_FILE, describes=f"파일 삭제: {path}", path=path
        ), ""

    if action.kind == SHELL:
        return None, f"셸 실행을 되돌릴 방법이 이력에 없습니다: {str(action.args.get('command') or '')[:60]}"

    return None, f"되돌릴 수 없는 행위입니다: {action.kind}"


def invert(actions: Iterable[Action]) -> tuple[list[UndoStep], list[str]]:
    """작업 이력을 되돌리는 단계 목록과, 되돌릴 수 없는 것들의 사유.

    되돌리기는 **역순**이다. 마지막에 한 일을 먼저 되돌려야 중간 상태가 앞뒤로 어긋나지
    않는다.

    사유가 하나라도 있으면 호출자는 되돌리기를 시도하지 않아야 한다. 절반만 되돌린
    세상은 되돌리지 않은 세상보다 나쁘다 — 무엇이 남았는지 아무도 모르기 때문이다.
    """
    history = list(actions)
    effects = [action for action in history if action.has_effect]
    steps: list[UndoStep] = []
    reasons: list[str] = []
    for action in reversed(effects):
        earlier = history[: history.index(action)]
        step, reason = _invert_one(action, earlier)
        if step is None:
            reasons.append(reason)
            continue
        steps.append(step)
    return steps, reasons


def undo_plan(actions: Iterable[Action]) -> dict[str, Any] | None:
    """이 이력을 온전히 되돌릴 수 있으면 되돌리기 계획, 아니면 None.

    부분 성공을 돌려주지 않는다. 되돌릴 수 없는 행위가 하나라도 섞여 있으면 재작업
    전체를 에이전트가 맡아야 한다.
    """
    steps, reasons = invert(actions)
    if reasons or not steps:
        return None
    return {"undo_steps": [step.as_dict() for step in steps]}


__all__ = [
    "UNDO_DELETE_FILE",
    "UNDO_RESTORE_FILE",
    "UNDO_TOOL",
    "UndoStep",
    "invert",
    "undo_plan",
]
