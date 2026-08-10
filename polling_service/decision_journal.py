"""분기 판단 이력(Gateway Decision Journal).

프로세스 엔진이 "다음에 어느 갈래로 갈 것인가"를 판단할 때 만들어지는 정보를
재현 가능한 형태로 모아 이벤트 저장소에 남길 행(row)들을 구성한다.

이 모듈은 의도적으로 표준 라이브러리에만 의존한다. 판단 로직 자체를 바꾸지 않고
관측만 담당하며, 기록 실패가 프로세스 진행을 막지 않도록 예외를 밖으로 던지지
않는 것이 원칙이다(수집 단계에서의 예외는 삼키고 로그만 남긴다).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


# 이벤트 종류
EVENT_TYPE_DECISION = "gateway_decision"
EVENT_TYPE_TRACE = "gateway_decision_trace"

# 시퀀스 판정 방식
METHOD_EXPRESSION = "expression"
METHOD_NATURAL_LANGUAGE = "natural-language"
METHOD_UNCONDITIONAL = "unconditional"

# 시퀀스 판정 결과. 엔진이 실제로 사용한 값(effective)과 별개로,
# "참으로 판정했다"와 "판정하지 못했다"를 반드시 구분한다.
VERDICT_TRUE = "true"
VERDICT_FALSE = "false"
VERDICT_UNDETERMINED = "undetermined"

# 분기 선택 규칙
RULE_UNCONDITIONAL = "unconditional"
RULE_SINGLE_TRUE = "single-true"
RULE_PRIORITY = "priority"
RULE_DEFAULT_FLOW = "default-flow"
RULE_NO_CANDIDATE = "no-candidate"
RULE_ALL_BRANCHES = "all-branches"
RULE_FILTERED = "filtered"

# 판단 결과 구분
OUTCOME_ADVANCED = "advanced"
OUTCOME_WAITING = "waiting"
OUTCOME_UNDECIDED = "undecided"

DECIDED_BY_ENGINE = "system"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except Exception:
        return default


# 입력 스냅샷과 모델 원문의 저장 크기 상한(직렬화 후 문자 수).
SNAPSHOT_LIMIT = _env_int("DECISION_JOURNAL_SNAPSHOT_LIMIT", 8192)
TRACE_LIMIT = _env_int("DECISION_JOURNAL_TRACE_LIMIT", 65536)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def normalize_verdict(value: Any) -> str:
    """모델/평가기가 돌려준 값을 판정 결과로 정규화한다.

    None 은 "판정 불가"이며, 참/거짓과 반드시 구분된다.
    """
    if value is None:
        return VERDICT_UNDETERMINED
    if isinstance(value, bool):
        return VERDICT_TRUE if value else VERDICT_FALSE
    if isinstance(value, (int, float)):
        return VERDICT_TRUE if value else VERDICT_FALSE
    if isinstance(value, str):
        token = value.strip().lower()
        if token in ("true", "yes", "y", "1"):
            return VERDICT_TRUE
        if token in ("false", "no", "n", "0"):
            return VERDICT_FALSE
        if token in ("", "none", "null", "unknown", "undetermined"):
            return VERDICT_UNDETERMINED
    return VERDICT_UNDETERMINED


def truncate_payload(value: Any, limit: int) -> tuple[Any, bool]:
    """크기 상한을 넘는 값을 잘라내고, 잘렸다는 사실을 함께 돌려준다.

    상한 초과가 기록 생략으로 이어지지 않게 하는 것이 목적이다.
    """
    if value is None:
        return None, False
    try:
        serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        serialized = str(value)

    if len(serialized) <= limit:
        return value, False

    return {
        "truncated": True,
        "originalLength": len(serialized),
        "preview": serialized[:limit],
    }, True


class DecisionRecorder:
    """워크아이템 한 건을 처리하는 동안의 분기 판단을 모은다.

    수집만 하고 저장은 하지 않는다. `build_events()` 가 돌려주는 행들을
    호출자가 진행 경로 밖에서 저장한다.
    """

    def __init__(
        self,
        *,
        proc_inst_id: Optional[str] = None,
        root_proc_inst_id: Optional[str] = None,
        proc_def_id: Optional[str] = None,
        proc_def_version: Optional[str] = None,
        activity_id: Optional[str] = None,
        workitem_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        execution_scope: Optional[str] = None,
        rework_count: Optional[int] = None,
        correlation_id: Optional[str] = None,
        snapshot_limit: int = SNAPSHOT_LIMIT,
        trace_limit: int = TRACE_LIMIT,
        clock=None,
        id_factory=None,
    ) -> None:
        self.proc_inst_id = proc_inst_id
        self.root_proc_inst_id = root_proc_inst_id
        self.proc_def_id = proc_def_id
        self.proc_def_version = proc_def_version
        self.activity_id = activity_id
        self.workitem_id = workitem_id
        self.tenant_id = tenant_id
        self.execution_scope = execution_scope
        self.rework_count = rework_count
        self.snapshot_limit = snapshot_limit
        self.trace_limit = trace_limit

        self._clock = clock or _utcnow_iso
        self._id_factory = id_factory or _new_id
        self.correlation_id = correlation_id or self._id_factory()

        self._evaluations: dict[str, dict[str, Any]] = {}
        self._decisions: list[dict[str, Any]] = []
        self._traces: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 수집
    # ------------------------------------------------------------------
    def record_sequence_evaluation(
        self,
        sequence_id: Optional[str],
        *,
        method: str,
        verdict: str,
        effective: Optional[bool] = None,
        expression: Optional[str] = None,
        reason: Optional[str] = None,
        error: Optional[str] = None,
        input_snapshot: Any = None,
    ) -> None:
        """시퀀스 하나의 조건 평가 결과를 기록한다."""
        if not sequence_id:
            return
        try:
            entry: dict[str, Any] = {
                "sequenceId": sequence_id,
                "method": method,
                "verdict": verdict,
            }
            if effective is not None:
                entry["effective"] = bool(effective)
            if isinstance(expression, str) and expression.strip():
                entry["expression"] = expression.strip()
            if isinstance(reason, str) and reason.strip():
                entry["reason"] = reason.strip()
            if isinstance(error, str) and error.strip():
                entry["error"] = error.strip()
            if input_snapshot is not None:
                snapshot, truncated = truncate_payload(input_snapshot, self.snapshot_limit)
                entry["inputSnapshot"] = snapshot
                if truncated:
                    entry["inputSnapshotTruncated"] = True
            self._evaluations[str(sequence_id)] = entry
        except Exception as exc:  # pragma: no cover - 수집 실패가 진행을 막지 않게
            print(f"[WARN] decision journal: 시퀀스 평가 기록 실패 {sequence_id}: {exc}")

    def record_llm_trace(self, *, prompt: Any, response: Any, sequence_ids: Optional[list[str]] = None) -> Optional[str]:
        """자연어 조건 판정에 쓰인 모델 프롬프트·응답 원문을 기록한다."""
        try:
            trace_id = self._id_factory()
            prompt_payload, prompt_truncated = truncate_payload(prompt, self.trace_limit)
            response_payload, response_truncated = truncate_payload(response, self.trace_limit)
            self._traces.append(
                {
                    "traceId": trace_id,
                    "sequenceIds": [str(s) for s in (sequence_ids or [])],
                    "prompt": prompt_payload,
                    "promptTruncated": prompt_truncated,
                    "response": response_payload,
                    "responseTruncated": response_truncated,
                    "recordedAt": self._clock(),
                }
            )
            return trace_id
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] decision journal: 모델 원문 기록 실패: {exc}")
            return None

    def record_decision(
        self,
        *,
        source_id: Optional[str],
        source_name: Optional[str] = None,
        source_type: Optional[str] = None,
        branch_type: Optional[str] = None,
        selection_rule: str,
        candidate_sequence_ids: Optional[list[str]] = None,
        selected_sequence_ids: Optional[list[str]] = None,
        selected_targets: Optional[list[dict[str, Any]]] = None,
        input_snapshot: Any = None,
    ) -> Optional[str]:
        """판단 주체 노드 하나에 대한 분기 판단 결과를 기록한다."""
        if not source_id:
            return None
        try:
            candidates = [str(s) for s in (candidate_sequence_ids or [])]
            selected = [str(s) for s in (selected_sequence_ids or [])]
            unselected = [s for s in candidates if s not in set(selected)]

            evaluations = [self._evaluation_for(seq_id) for seq_id in candidates]
            for entry in evaluations:
                entry["selected"] = entry.get("sequenceId") in set(selected)

            decision_id = self._id_factory()
            decision: dict[str, Any] = {
                "decisionId": decision_id,
                "correlationId": self.correlation_id,
                "procInstId": self.proc_inst_id,
                "rootProcInstId": self.root_proc_inst_id,
                "executionScope": self.execution_scope,
                "procDefId": self.proc_def_id,
                "procDefVersion": self.proc_def_version,
                "triggerActivityId": self.activity_id,
                "workitemId": self.workitem_id,
                "reworkCount": self.rework_count,
                "source": {
                    "id": source_id,
                    "name": source_name,
                    "type": source_type,
                    "branchType": branch_type,
                },
                "selectionRule": selection_rule,
                "evaluations": evaluations,
                "selectedSequenceIds": selected,
                "unselectedSequenceIds": unselected,
                "selectedTargets": selected_targets or [],
                "outcome": OUTCOME_ADVANCED if selected_targets else OUTCOME_UNDECIDED,
                "decidedAt": self._clock(),
                "decidedBy": DECIDED_BY_ENGINE,
            }
            if input_snapshot is not None:
                snapshot, truncated = truncate_payload(input_snapshot, self.snapshot_limit)
                decision["inputSnapshot"] = snapshot
                if truncated:
                    decision["inputSnapshotTruncated"] = True

            self._decisions.append(decision)
            return decision_id
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] decision journal: 분기 판단 기록 실패 {source_id}: {exc}")
            return None

    def mark_deferred(self, deferred_activity_ids: list[str], *, waiting_for: Optional[list[dict[str, Any]]] = None) -> None:
        """병합 지점에서 진행이 보류된 대상을 대기로 표시한다."""
        try:
            deferred = {str(a) for a in (deferred_activity_ids or []) if a}
            if not deferred:
                return
            for decision in self._decisions:
                targets = decision.get("selectedTargets") or []
                hit = [t for t in targets if str(t.get("activityId")) in deferred]
                if not hit:
                    continue
                decision["outcome"] = OUTCOME_WAITING
                decision["deferredTargets"] = hit
                if waiting_for:
                    decision["waitingFor"] = waiting_for
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] decision journal: 대기 표시 실패: {exc}")

    # ------------------------------------------------------------------
    # 조회 / 구성
    # ------------------------------------------------------------------
    def has_records(self) -> bool:
        return bool(self._decisions or self._traces)

    @property
    def decisions(self) -> list[dict[str, Any]]:
        return self._decisions

    @property
    def traces(self) -> list[dict[str, Any]]:
        return self._traces

    def _evaluation_for(self, sequence_id: str) -> dict[str, Any]:
        recorded = self._evaluations.get(str(sequence_id))
        if recorded:
            return dict(recorded)
        # 조건이 지정되지 않아 평가 단계를 거치지 않은 시퀀스.
        return {
            "sequenceId": str(sequence_id),
            "method": METHOD_UNCONDITIONAL,
            "verdict": VERDICT_TRUE,
            "effective": True,
        }

    def build_events(self) -> list[dict[str, Any]]:
        """이벤트 저장소에 넣을 행 목록을 만든다.

        판단 이력과 모델 원문은 서로 다른 이벤트로 나뉘며, 어느 쪽에서 시작해도
        반대쪽을 찾을 수 있도록 상호 참조를 건다. 근거 문구는 원문이 아니라 판단
        이력 본문에 복사되어 있으므로, 원문이 만료돼도 판단 이력은 자립적으로 읽힌다.
        """
        if not self.has_records():
            return []

        try:
            sequence_to_decisions: dict[str, list[str]] = {}
            for decision in self._decisions:
                for entry in decision.get("evaluations") or []:
                    seq_id = str(entry.get("sequenceId"))
                    sequence_to_decisions.setdefault(seq_id, []).append(decision["decisionId"])

            for trace in self._traces:
                linked: list[str] = []
                for seq_id in trace.get("sequenceIds") or []:
                    for decision_id in sequence_to_decisions.get(str(seq_id), []):
                        if decision_id not in linked:
                            linked.append(decision_id)
                trace["decisionIds"] = linked
                trace["correlationId"] = self.correlation_id

            decision_to_traces: dict[str, list[str]] = {}
            for trace in self._traces:
                for decision_id in trace.get("decisionIds") or []:
                    decision_to_traces.setdefault(decision_id, []).append(trace["traceId"])
            for decision in self._decisions:
                trace_ids = decision_to_traces.get(decision["decisionId"])
                if trace_ids:
                    decision["traceIds"] = trace_ids

            rows: list[dict[str, Any]] = []
            for decision in self._decisions:
                rows.append(self._event_row(EVENT_TYPE_DECISION, decision))
            for trace in self._traces:
                rows.append(self._event_row(EVENT_TYPE_TRACE, trace))
            return rows
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] decision journal: 이벤트 구성 실패: {exc}")
            return []

    def _event_row(self, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": self._id_factory(),
            "job_id": self.correlation_id,
            "todo_id": self.workitem_id,
            "proc_inst_id": self.proc_inst_id,
            "event_type": event_type,
            "data": data,
            "tenant_id": self.tenant_id,
        }


def collect_traversed_sequence_ids(decision_events: list[dict[str, Any]]) -> list[str]:
    """판단 이력들로부터 실제로 지나간 시퀀스 집합을 확정한다.

    대기(waiting)와 진행 불가(undecided)는 아직 지나가지 않은 상태이므로 제외한다.
    """
    traversed: list[str] = []
    seen: set[str] = set()
    for event in decision_events or []:
        data = event.get("data") if isinstance(event, dict) else None
        if not isinstance(data, dict):
            continue
        if data.get("outcome") != OUTCOME_ADVANCED:
            continue
        for seq_id in data.get("selectedSequenceIds") or []:
            key = str(seq_id)
            if key not in seen:
                seen.add(key)
                traversed.append(key)
    return traversed
