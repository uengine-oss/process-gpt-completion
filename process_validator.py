"""생성된 BPMN 프로세스의 실행 검증 + 자동 개선 모듈.

pdf2bpmn 가 프로세스 생성을 마친 직후(PASS 2)에 호출된다.
생성된 프로세스를 process-gpt-completion 실행 엔진으로 start event 부터
end event 까지 실제로 실행시켜 보고:

  0. 테스트플랜 — LLM 이 폼 예시 입력값 + '의미상 올바른' 기대 실행 순서를 1회 생성
  ── Phase 1 (정적 게이트) ──
  1. 정적 검사  — start/end 존재, 전 노드 연결성(끊긴 흐름/고립 노드/게이트웨이 없는 분기) 검출
  2. 정적 결함이 있으면 → LLM 이 결함+현재 정의를 받아 교정 → 재저장 → 다시 1.
     (정적 검사를 통과할 때까지. 통과해야 Phase 2 로 넘어간다.)
  ── Phase 2 (실제 실행) ──
  3. 실제 실행  — 엔진을 driving 하여 start→end 실제 실행 트레이스 수집 (해피패스 1회)
  4. 비교       — 기대 vs 실제 → 결함 목록
  5. 실행 결함이 있으면 → LLM 이 교정 → 재저장 → 다시 3. (정적 결함 재발 시 Phase 1 로)
  6. 전 과정 합쳐 최대 N회. 성공하면 종료, 실패해도 결함 리포트를 첨부해 전달.

엔진 driving 방식 — 운영(프론트)과 동일한 실제 경로를 그대로 탄다:
  ① 시작 : POST /initiate   (실제 엔드포인트 — 프론트의 프로세스 시작과 동일)
  ② 제출 : POST /complete   (submit_workitem — 프론트의 폼 제출과 동일)
  ③ 다음 : bpm_proc_inst.current_activity_ids 를 DB 에서 직접 조회
  ④ 정리 : 검증용 인스턴스 row 를 DB 에서 직접 삭제
제출(/complete)하면 폴링 서비스가 알아서 다음 태스크를 찾아 current_activity_ids 에
반영한다. 검증기는 그 결과를 DB 에서 읽어 다음 활동을 제출할 뿐, 다음-태스크 탐색이나
실행 로직을 재구현하지 않는다(전부 폴링 서비스가 한다). ③④ 는 단순 DB read/delete 라
별도 엔드포인트 없이 pdf2bpmn 의 Supabase 클라이언트로 직접 처리한다.

핵심 설계:
- 기대 실행 순서는 proc_json 의 sequences 배열이 아니라 '액티비티의 의미'에서
  추론한다. (그래야 sequences 에 버그가 있어도 그 버그에 오염되지 않는다.)
  엔진은 buggy 한 sequences 대로 실행하므로, 의미상 기대값과 비교하면 결함이 드러난다.
- 분기 결정·다음 태스크 결정은 실제 엔진(폴링 서비스)이 한다. 검증은 그 결과만 본다.
- 엔진/폴링 서비스가 떠 있지 않으면 검증을 건너뛴다(프로세스 전달은 정상 진행).

전제: process-gpt-completion 의 API 서버(/initiate·/complete) + 폴링 서비스가
      떠 있고 pdf2bpmn 에서 HTTP 도달 가능해야 한다. COMPLETION_ENGINE_URL 로 설정.
"""

import asyncio
import copy
import json
import logging
import os
import time
from datetime import datetime, timezone

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None


CRITICAL = "critical"
WARNING = "warning"

# 검증기가 직접 제출하는(폼 입력이 필요한) 액티비티 타입.
# 게이트웨이/이벤트는 엔진이 자동 통과, serviceTask/scriptTask 는 폴링 서비스가 자동 실행.
_SUBMITTABLE_TYPES = {"userTask", "manualTask", "task", "user"}

# 정적 결함을 critical/warning 로 가중한 점수. 낮을수록 좋음.
_SCORE_CRITICAL = 100
_SCORE_WARNING = 1


class EngineUnreachable(Exception):
    """실행 엔진(/initiate·/complete API)에 접속할 수 없을 때. → 검증을 건너뛴다."""


def _defect(severity: str, dtype: str, detail: str, **extra) -> dict:
    d = {"severity": severity, "type": dtype, "detail": detail}
    d.update(extra)
    return d


def _score(defects) -> int:
    return sum(_SCORE_CRITICAL if d.get("severity") == CRITICAL else _SCORE_WARNING
               for d in (defects or []))


def _trunc(s, n: int = 240) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[:n] + "…"


class ProcessValidator:
    """생성된 프로세스를 실행 엔진으로 검증하고 결함을 자동 개선한다.

    의존성은 모두 생성자 인자로 주입받아, executor 내부 구현과 느슨하게 결합한다.

    Args:
        llm_call: async (messages: list[dict], max_tokens: int) -> dict|None
        save_definition: async (proc_def_id: str, definition: dict) -> bool
        engine_base_url: process-gpt-completion API 서버 base URL (/initiate·/complete 용)
        tenant_id: 멀티테넌트 식별자(X-Forwarded-Host 헤더로 전달)
        fetch_instance_state: async (proc_inst_id: str) -> { "status": str,
                              "current_activity_ids": [str] }. bpm_proc_inst 를 DB 에서
                              직접 읽어 폴링 서비스가 진행시킨 결과를 가져온다.
        cleanup_instance: async (proc_inst_id: str) -> None. 검증용 인스턴스/워크아이템
                          row 를 DB 에서 직접 삭제한다(없으면 정리를 건너뜀).
        max_iters: 검증-개선 루프 최대 반복 횟수
        actor_email: 검증 실행 시 /initiate·/complete 에 넘길 행위자 이메일.
                     (submit_workitem 이 user 정보를 요구한다. 미등록 이메일도
                      엔진이 graceful 처리하므로 흐름 검증엔 어떤 값이든 무방.)
        advance_timeout: 제출 후 폴링 서비스 진행을 기다리는 최대 시간(초)
        poll_interval: current_activity_ids 폴링 간격(초)
        report_path: 검증 흐름 상세 리포트(.md)를 저장할 경로. 테스트 시나리오·실제
                     실행 경로·결함·자동 개선 전후 비교를 모두 기록한다. (None 이면 미작성)
        logger: 로거 (없으면 모듈 로거)
        progress: async (message: str, pct: int, extra: dict|None) -> None  (선택)
    """

    def __init__(
        self,
        *,
        llm_call,
        save_definition,
        engine_base_url: str,
        tenant_id: str,
        fetch_instance_state=None,
        cleanup_instance=None,
        max_iters: int = 5,
        actor_email: str = None,
        advance_timeout: float = 70.0,
        poll_interval: float = 1.5,
        report_path: str = None,
        logger=None,
        progress=None,
    ):
        self._llm = llm_call
        self._save = save_definition
        self._fetch_state = fetch_instance_state
        self._cleanup = cleanup_instance
        self.engine_base_url = (engine_base_url or "").rstrip("/")
        self.tenant_id = tenant_id or "localhost"
        self.max_iters = max(1, int(max_iters or 1))
        self.actor_email = actor_email or "pdf2bpmn-validation@validation.local"
        self._advance_timeout = max(5.0, float(advance_timeout or 70.0))
        self._poll_interval = max(0.3, float(poll_interval or 1.5))
        self.log = logger or logging.getLogger(__name__)
        self._progress = progress
        # 검증 흐름을 사람이 읽을 수 있게 정리해 저장할 리포트 파일 경로 (없으면 미작성).
        self._report_path = report_path
        self._rep_lines: list = []          # 리포트 본문 누적
        self._test_proc_inst_ids: list = []  # 검증용으로 생성된 실행 인스턴스 id

    # ------------------------------------------------------------------ #
    # public
    # ------------------------------------------------------------------ #
    async def validate_and_repair(
        self,
        *,
        proc_def_id: str,
        process_name: str,
        proc_json: dict,
        forms: dict,
        extracted: dict = None,
    ) -> dict:
        """프로세스를 검증하고, 결함이 있으면 최대 N회 자동 개선한다.

        extracted: 원문 추출 컨텍스트. 현재는 사용하지 않으며 향후 확장용으로 받아 둔다
                   (액티비티 이름·설명·지시사항이 의미 추론에 충분하기 때문).

        Returns: 검증 리포트(dict). proc_json 이 수정됐다면 report["final_definition"]
                 에 최종 정의가 들어 있고, DB(proc_def)에도 이미 재저장된 상태다.
        """
        self._rep_lines = []
        self._test_proc_inst_ids = []
        report = {
            "proc_def_id": proc_def_id,
            "process_name": process_name,
            "passed": None,
            "skipped": False,
            "skip_reason": None,
            "iterations": 0,
            "repaired": False,
            "max_iters": self.max_iters,
            "final_score": None,
            "remaining_defects": [],
            "history": [],
            "trace": None,
            "final_definition": None,
        }

        if httpx is None:
            report["skipped"] = True
            report["skip_reason"] = "httpx 미설치 — 검증 불가"
            return report
        if not self.engine_base_url or not proc_def_id:
            report["skipped"] = True
            report["skip_reason"] = "engine_base_url/proc_def_id 누락"
            return report
        if self._fetch_state is None:
            report["skipped"] = True
            report["skip_reason"] = "인스턴스 상태 조회 함수(fetch_instance_state) 미주입 — 검증 불가"
            return report

        current = copy.deepcopy(proc_json)
        best_def = None
        best_score = None
        last_defects: list = []
        last_trace = None
        engine_ran = False
        phase_scores: list = []   # 비수렴 판정용 — 단계(static/runtime)가 바뀌면 초기화
        prev_phase = None

        # 테스트 플랜(폼 예시값 + 의미 기반 기대 실행순서)은 1회만 생성해 전 단계에서 재사용.
        # 기대 순서는 정적 교정 단계의 개선 LLM 에도 '올바른 흐름' 힌트로 쓰인다.
        await self._emit(
            f"[검증] {process_name} — 테스트 시나리오(입력값/기대경로) 생성 중",
            84, {"proc_def_id": proc_def_id},
        )
        test_plan = await self._build_test_plan(current, forms)

        # --- 상세 리포트: 헤더 + 테스트 시나리오 ---
        nmap0 = self._node_name_map(current)
        self._rep("# 프로세스 실행 검증 리포트")
        self._rep("")
        self._rep(f"- proc_def_id: `{proc_def_id}`")
        self._rep(f"- 프로세스명: {process_name}")
        self._rep(f"- 검증 시각: {datetime.now(timezone.utc).isoformat()}")
        self._rep(f"- 실행 엔진: {self.engine_base_url}  (운영과 동일한 실제 /initiate·/complete 호출)")
        self._rep(f"- 행위자(actor) 이메일: {self.actor_email}")
        self._rep(f"- 최대 반복: {self.max_iters}")
        self._rep("")
        _cases0 = test_plan.get("cases") or []
        self._rep(f"## 테스트 시나리오 — 분기별 {len(_cases0)}개 케이스 (LLM 이 '문서 의미' 기반으로 생성)")
        self._rep("각 케이스는 매 검증 회차마다 '새 인스턴스'로 처음부터 실제 실행된다. "
                  "모든 케이스가 통과해야 검증 통과로 본다.")
        for _ci, _case in enumerate(_cases0, 1):
            self._rep("")
            self._rep(f"### 케이스 {_ci}: {_case.get('name')}")
            _eo = [str(x) for x in (_case.get("expected_activity_order") or [])]
            self._rep("- 의미상 기대 실행 순서:")
            if _eo:
                for _j, _aid in enumerate(_eo, 1):
                    self._rep(f"  {_j}. {nmap0.get(_aid, _aid)}  (`{_aid}`)")
            else:
                self._rep("  (기대 순서 없음 — 연결성 위주로만 검증)")
            _ainp = _case.get("activity_inputs") or {}
            if _ainp:
                self._rep("- 액티비티별 예시 입력값:")
                for _aid, _vals in _ainp.items():
                    self._rep(f"  · {nmap0.get(str(_aid), _aid)} (`{_aid}`): "
                              f"{json.dumps(_vals, ensure_ascii=False)}")
        if test_plan.get("rationale"):
            self._rep("")
            self._rep("### 케이스 구성 근거")
            self._rep(f"  {_trunc(test_plan.get('rationale'), 500)}")
        self._rep("")

        for it in range(1, self.max_iters + 1):
            report["iterations"] = it

            static_defects = self._static_check(current)
            nmap = self._node_name_map(current)
            case_runs: list = []   # [(case, trace, case_defects)] — runtime 단계에서만 채워짐

            if static_defects:
                # ── Phase 1: 정적 검사 결함 → 엔진 실행 없이 LLM 으로 먼저 교정한다.
                #   정적 검사를 통과해야 비로소 Phase 2(엔진 실행)로 넘어간다.
                phase = "static"
                defects = list(static_defects)
                await self._emit(
                    f"[검증] {process_name} (#{it}) — 정적 검사 결함 {len(defects)}건, 구조 교정 진행",
                    84, {"proc_def_id": proc_def_id, "phase": "static", "defects": len(defects)},
                )
            else:
                # ── Phase 2: 정적 검사 통과 → 분기별 케이스를 '각각 새 인스턴스'로 실제 실행.
                phase = "runtime"
                cases = test_plan.get("cases") or [
                    {"name": "기본 경로", "activity_inputs": {}, "expected_activity_order": []}
                ]
                await self._emit(
                    f"[검증] {process_name} (#{it}) — 정적 통과, 분기 {len(cases)}개 케이스 실제 실행 중",
                    85, {"proc_def_id": proc_def_id, "phase": "runtime", "cases": len(cases)},
                )
                defects = []
                for case in cases:
                    try:
                        trace = await self._run_trace(
                            proc_def_id, current, case.get("activity_inputs") or {}
                        )
                    except EngineUnreachable as e:
                        report["skipped"] = True
                        report["skip_reason"] = f"실행 엔진 접속 실패: {e}"
                        self.log.warning(f"[VALIDATION] engine unreachable: {e}")
                        return self._finalize_skip(report, current)
                    last_trace = trace
                    # 첫 엔진 실행에서 진행이 전혀 없으면 폴링 서비스 미가동 가능성 → 건너뜀.
                    if not engine_ran and trace.get("no_progress"):
                        report["skipped"] = True
                        report["skip_reason"] = (
                            "엔진이 프로세스를 전혀 진행시키지 못함 — 폴링 서비스 미가동 가능성"
                        )
                        self.log.warning(
                            f"[VALIDATION] no progress for {proc_def_id} — skipping (polling down?)"
                        )
                        return self._finalize_skip(report, current)
                    engine_ran = True
                    cdefs = self._diff(case.get("expected_activity_order") or [], trace, current)
                    cdefs = [dict(d, case=case.get("name")) for d in cdefs]
                    case_runs.append((case, trace, cdefs))
                    defects.extend(cdefs)

            cur_score = _score(defects)
            last_defects = defects
            if phase != prev_phase:
                phase_scores = []          # 단계 전환 시 비수렴 카운터 초기화
                prev_phase = phase
            phase_scores.append(cur_score)
            report["history"].append({
                "iteration": it,
                "phase": phase,
                "score": cur_score,
                "defect_count": len(defects),
                "defects": defects,
                "cases": [
                    {"name": c.get("name"),
                     "proc_inst_id": (tr or {}).get("proc_inst_id"),
                     "actual_order": (tr or {}).get("actual_order"),
                     "reached_end": (tr or {}).get("reached_end"),
                     "defect_count": len(cd)}
                    for (c, tr, cd) in case_runs
                ],
            })

            # --- 상세 리포트: 이번 회차 ---
            self._rep(f"## Iteration {it} — 단계: {phase}")
            if phase == "static":
                self._rep("- 정적 검사 단계 (엔진 실행 없이 구조 결함만 점검)")
                self._rep(f"- 발견된 결함: {len(defects)}건 (score={cur_score})")
                for _i, _d in enumerate(defects, 1):
                    self._rep(
                        f"  {_i}. [{_d.get('severity')}] {_d.get('type')} — "
                        f"{_trunc(_d.get('detail'), 280)}"
                    )
            else:
                self._rep(f"- 분기 케이스 {len(case_runs)}개를 각각 '새 인스턴스'로 실행:")
                for (c, tr, cd) in case_runs:
                    pinst = str((tr or {}).get("proc_inst_id") or "")
                    if pinst:
                        self._test_proc_inst_ids.append(pinst)
                    _act = [str(a) for a in ((tr or {}).get("actual_order") or [])]
                    self._rep("")
                    self._rep(f"### 케이스: {c.get('name')}")
                    if pinst:
                        self._rep(f"- 실행 엔진 인스턴스 proc_inst_id: `{pinst}` (DB 보존)")
                    self._rep(
                        "- 실제 실행 경로: "
                        + (" → ".join(nmap.get(a, a) for a in _act) or "(진행 없음)")
                    )
                    self._rep(f"- endEvent 도달: {(tr or {}).get('reached_end')}")
                    if (tr or {}).get("errors"):
                        self._rep(f"- 엔진 호출 오류: {(tr or {}).get('errors')[:5]}")
                    self._rep(f"- 이 케이스 결함: {len(cd)}건")
                    for _i, _d in enumerate(cd, 1):
                        self._rep(
                            f"  {_i}. [{_d.get('severity')}] {_d.get('type')} — "
                            f"{_trunc(_d.get('detail'), 280)}"
                        )
                self._rep("")
                self._rep(f"- 회차 전체 결함 합계: {len(defects)}건 (score={cur_score})")
            self._rep("")

            # 검증된 버전 중 최적본 추적 (<= 로 동점이면 더 개선된 최신본을 택함)
            if best_score is None or cur_score <= best_score:
                best_score = cur_score
                best_def = copy.deepcopy(current)

            if not defects:
                # static 통과 + 모든 분기 케이스가 엔진 실행에서 무결.
                report["passed"] = True
                report["final_score"] = 0
                self._rep("- ✅ 모든 분기 케이스 통과 — start→end 정상 실행")
                self._rep("")
                await self._emit(
                    f"[검증] {process_name} — 검증 통과 (#{it}회차, 모든 분기 정상 실행)",
                    88, {"proc_def_id": proc_def_id, "passed": True},
                )
                break

            self.log.info(
                f"[VALIDATION] {proc_def_id} iter#{it}({phase}): "
                f"{len(defects)} defect(s), score={cur_score}"
            )

            # 마지막 회차면 개선하지 않는다(검증되지 않은 버전을 전달하지 않기 위해).
            if it >= self.max_iters:
                self._rep(f"- 최대 반복({self.max_iters}회) 도달 — 추가 개선 없이 종료")
                self._rep("")
                break

            # 비수렴 조기 종료: 같은 단계에서 3회 이상 진행했는데 점수가 안 나아지면 중단.
            if len(phase_scores) >= 3 and phase_scores[-1] >= phase_scores[-3]:
                self.log.info(f"[VALIDATION] {proc_def_id}: 비수렴({phase}) — 조기 종료")
                report["history"][-1]["note"] = f"비수렴({phase})으로 조기 종료"
                self._rep(f"- 비수렴({phase} 단계, 점수가 더 나아지지 않음) — 조기 종료")
                self._rep("")
                break

            # 자동 개선
            await self._emit(
                f"[검증] {process_name} (#{it}) — 결함 {len(defects)}건 발견, 프로세스 자동 교정 중",
                86, {"proc_def_id": proc_def_id, "defects": len(defects)},
            )
            improved, improve_meta = await self._improve(current, defects, test_plan)
            self._rep(f"### 자동 개선 (#{it})")
            self._rep(f"- LLM 에 전달한 결함 {len(defects)}건:")
            for _d in defects:
                _cs = f"[{_d.get('case')}] " if _d.get("case") else ""
                self._rep(
                    f"  · {_cs}[{_d.get('severity')}] {_d.get('type')}: "
                    f"{_trunc(_d.get('detail'), 220)}"
                )
            if not improved:
                report["history"][-1]["note"] = "개선안 생성 실패 — 중단"
                self._rep("- LLM 이 개선안을 생성하지 못함 → 중단")
                self._rep("")
                break
            report["history"][-1]["improvement"] = improve_meta
            self._rep(f"- LLM 응답: changed={improve_meta.get('changed')}")
            if improve_meta.get("explanation"):
                self._rep(f"- LLM 설명: {_trunc(improve_meta.get('explanation'), 500)}")
            # 원래(current) → 개선(improved) 무엇이 어떻게 바뀌었는지 기록
            self._rep_seq_diff(current, improved)
            ok, why = self._validate_definition_shape(improved)
            if not ok:
                self.log.warning(f"[VALIDATION] {proc_def_id}: 개선안 구조 불량({why}) — 중단")
                report["history"][-1]["note"] = f"개선안 구조 불량({why}) — 중단"
                self._rep(f"- 개선안 구조 불량({why}) → 적용하지 않고 중단")
                self._rep("")
                break
            # 파괴적 개선 방지 — 개선안이 정적 구조를 오히려 '악화'시키면 적용하지 않는다.
            worse_by = _score(self._static_check(improved)) - _score(self._static_check(current))
            if worse_by > 0:
                self.log.warning(
                    f"[VALIDATION] {proc_def_id}: 개선안이 구조를 악화(+{worse_by}) — 적용 안 함"
                )
                report["history"][-1]["note"] = "개선안이 구조를 악화 — 적용하지 않고 중단"
                self._rep(
                    f"- ⚠️ 개선안이 정적 구조를 오히려 악화시킴(결함 점수 +{worse_by}) "
                    f"→ 적용하지 않고 중단 (파괴적 개선 방지)"
                )
                self._rep("")
                break

            current = improved
            saved = await self._save(proc_def_id, current)
            report["repaired"] = True
            self._rep(f"- 개선본 proc_def 재저장: {'성공' if saved else '실패'}")
            self._rep("")
            if not saved:
                self.log.warning(f"[VALIDATION] {proc_def_id}: 개선본 저장 실패")
                report["history"][-1]["note"] = "개선본 저장 실패"
                break

        # ---- 마무리 -------------------------------------------------- #
        report["trace"] = last_trace
        if report["passed"]:
            # 모든 분기 케이스를 실제 엔진으로 통과한 '최종 버전'을 그대로 전달.
            report["final_definition"] = current
            report["remaining_defects"] = []
            report["note"] = "검증 통과 — 모든 분기 케이스를 실제 엔진으로 통과한 최종 버전을 전달"
            self._write_report(report)
            return report

        # 미통과: 모든 케이스를 통과한 버전을 만들지 못함 → 가장 양호한 버전을 경고와 함께 전달.
        report["passed"] = False
        report["final_score"] = best_score
        report["remaining_defects"] = last_defects
        if best_def is not None and best_def != current:
            await self._save(proc_def_id, best_def)
            report["repaired"] = True
            report["final_definition"] = best_def
        else:
            report["final_definition"] = current
        report["note"] = (
            "검증 미통과 — 모든 분기 케이스를 통과한 버전을 만들지 못했습니다. "
            "결함 리포트와 함께 가장 양호한 버전을 전달합니다."
        )
        await self._emit(
            f"[검증] {process_name} — 검증 미통과({len(last_defects)}건 잔여), 결함 리포트와 함께 전달",
            88, {"proc_def_id": proc_def_id, "passed": False, "remaining": len(last_defects)},
        )
        self._write_report(report)
        return report

    # ------------------------------------------------------------------ #
    # 1) 정적 검사
    # ------------------------------------------------------------------ #
    def _static_check(self, proc_json: dict) -> list:
        """엔진 실행 없이 그래프 구조를 검사한다 — start/end 존재, 전 노드 연결성."""
        defects: list = []
        events = [e for e in (proc_json.get("events") or []) if isinstance(e, dict)]
        activities = [a for a in (proc_json.get("activities") or []) if isinstance(a, dict)]
        gateways = [g for g in (proc_json.get("gateways") or []) if isinstance(g, dict)]
        subs = [s for s in (proc_json.get("subProcesses") or []) if isinstance(s, dict)]
        sequences = [s for s in (proc_json.get("sequences") or []) if isinstance(s, dict)]

        starts = [e for e in events if str(e.get("type") or "").lower() == "startevent"]
        ends = [e for e in events if str(e.get("type") or "").lower() == "endevent"]
        if not starts:
            defects.append(_defect(CRITICAL, "no_start_event", "프로세스에 startEvent 가 없습니다."))
        if not ends:
            defects.append(_defect(CRITICAL, "no_end_event", "프로세스에 endEvent 가 없습니다."))
        if not activities:
            defects.append(_defect(CRITICAL, "no_activity", "프로세스에 액티비티가 하나도 없습니다."))

        node_ids = set()
        for coll in (events, activities, gateways, subs):
            for n in coll:
                nid = str(n.get("id") or "").strip()
                if nid:
                    node_ids.add(nid)
        start_ids = {str(e.get("id")) for e in starts if e.get("id")}
        end_ids = {str(e.get("id")) for e in ends if e.get("id")}

        adj: dict = {}
        radj: dict = {}
        for s in sequences:
            src = str(s.get("source") or "").strip()
            tgt = str(s.get("target") or "").strip()
            sid = str(s.get("id") or "?")
            if not src or src not in node_ids:
                defects.append(_defect(
                    CRITICAL, "dangling_sequence",
                    f"시퀀스({sid})의 source '{src}' 가 존재하지 않는 노드를 가리킵니다."))
                continue
            if not tgt or tgt not in node_ids:
                defects.append(_defect(
                    CRITICAL, "dangling_sequence",
                    f"시퀀스({sid})의 target '{tgt}' 가 존재하지 않는 노드를 가리킵니다."))
                continue
            adj.setdefault(src, set()).add(tgt)
            radj.setdefault(tgt, set()).add(src)

        # start 에서 도달 불가능한 노드 → 흐름 단절(critical)
        if start_ids and node_ids:
            reachable = set()
            stack = list(start_ids)
            while stack:
                n = stack.pop()
                if n in reachable:
                    continue
                reachable.add(n)
                stack.extend(m for m in adj.get(n, ()) if m not in reachable)
            for nid in sorted(node_ids - reachable - start_ids):
                defects.append(_defect(
                    CRITICAL, "unreachable_node",
                    f"노드 '{nid}' 가 startEvent 에서 도달 불가능합니다(흐름 단절).", node=nid))

        # end 로 도달할 수 없는 노드 → dead-end(warning)
        if end_ids and node_ids:
            can_reach_end = set()
            stack = list(end_ids)
            while stack:
                n = stack.pop()
                if n in can_reach_end:
                    continue
                can_reach_end.add(n)
                stack.extend(m for m in radj.get(n, ()) if m not in can_reach_end)
            for nid in sorted(node_ids - can_reach_end - end_ids):
                defects.append(_defect(
                    WARNING, "dead_end_node",
                    f"노드 '{nid}' 에서 endEvent 로 도달할 수 없습니다.", node=nid))

        # endEvent 가 들어오는 시퀀스가 없으면 종료 불가
        for e in ends:
            eid = str(e.get("id") or "")
            if eid and eid not in radj:
                defects.append(_defect(
                    CRITICAL, "endevent_unwired",
                    f"endEvent '{eid}' 로 들어오는 시퀀스가 없습니다(프로세스가 끝나지 못함).", node=eid))

        # 게이트웨이가 아닌 노드가 후행 경로를 2개 이상 가지면 = 게이트웨이 없는 분기(fan-out).
        # 엔진은 비게이트웨이 노드의 모든 후행 시퀀스를 동시에 진행시키므로, 흐름이
        # 의도치 않게 여러 갈래로 갈라져 프로세스가 정상 종료하지 못한다.
        gw_ids = {str(g.get("id") or "") for g in gateways if g.get("id")}
        for nid, outs in adj.items():
            if nid in gw_ids or nid in start_ids:
                continue
            if len(outs) > 1:
                defects.append(_defect(
                    WARNING, "uncontrolled_split",
                    f"노드 '{nid}' 가 게이트웨이 없이 {len(outs)}개 경로로 분기합니다 "
                    f"(→ {sorted(outs)}). 분기는 반드시 게이트웨이를 거쳐야 합니다. "
                    f"이대로면 실행 시 여러 갈래로 동시에 진행되어 흐름이 깨집니다.",
                    node=nid))
        return defects

    # ------------------------------------------------------------------ #
    # 2) 테스트 플랜 (LLM)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_test_plan(result) -> dict:
        """LLM 응답을 { "cases": [ {name, activity_inputs, expected_activity_order} ], ... } 로 정규화."""
        out = {"cases": [], "rationale": ""}
        if isinstance(result, dict):
            out["rationale"] = str(result.get("rationale") or "")
            raw_cases = result.get("cases")
            # 단일 케이스 형태(activity_inputs 가 최상위)도 호환
            if not isinstance(raw_cases, list) and (
                "activity_inputs" in result or "expected_activity_order" in result
            ):
                raw_cases = [result]
            for i, c in enumerate(raw_cases or []):
                if not isinstance(c, dict):
                    continue
                ai = c.get("activity_inputs")
                eo = c.get("expected_activity_order")
                out["cases"].append({
                    "name": str(c.get("name") or f"케이스 {i + 1}"),
                    "activity_inputs": ai if isinstance(ai, dict) else {},
                    "expected_activity_order": (
                        [str(x) for x in eo] if isinstance(eo, list) else []
                    ),
                })
        if not out["cases"]:
            out["cases"] = [{"name": "기본 경로", "activity_inputs": {},
                             "expected_activity_order": []}]
        return out

    async def _build_test_plan(self, proc_json: dict, forms: dict) -> dict:
        """LLM 으로 '분기별 테스트 케이스'를 생성한다.

        게이트웨이 분기를 모두 커버하도록 여러 케이스를 만든다. 각 케이스는 특정
        경로를 타도록 폼 입력값을 정하고, 그 경로의 기대 액티비티 순서를 명시한다.
        반환: { "cases": [ {name, activity_inputs, expected_activity_order}, ... ], "rationale" }
        """
        activities = [a for a in (proc_json.get("activities") or []) if isinstance(a, dict)]
        gateways = [g for g in (proc_json.get("gateways") or []) if isinstance(g, dict)]
        events = [e for e in (proc_json.get("events") or []) if isinstance(e, dict)]
        sequences = [s for s in (proc_json.get("sequences") or []) if isinstance(s, dict)]

        act_payload = []
        for a in activities:
            aid = str(a.get("id") or "")
            form = (forms or {}).get(aid) or {}
            fields = []
            for f in (form.get("fields_json") or []):
                if isinstance(f, dict):
                    fields.append({
                        "key": f.get("key") or f.get("name"),
                        "label": f.get("text") or f.get("alias"),
                        "type": f.get("type") or "text",
                    })
            act_payload.append({
                "id": aid,
                "name": a.get("name"),
                "role": a.get("role"),
                "type": a.get("type") or "userTask",
                "description": _trunc(a.get("description"), 300),
                "instruction": _trunc(a.get("instruction"), 300),
                "form_id": (form.get("form_id") or ""),
                "form_fields": fields,
            })

        gw_payload = [{
            "id": g.get("id"), "name": g.get("name"), "type": g.get("type"),
            "condition": g.get("condition"),
        } for g in gateways]
        ev_payload = [{"id": e.get("id"), "type": e.get("type"), "name": e.get("name")}
                      for e in events]
        seq_payload = [{
            "id": s.get("id"), "source": s.get("source"),
            "target": s.get("target"), "condition": s.get("condition"),
        } for s in sequences]
        valid_ids = [a["id"] for a in act_payload]

        system = (
            "당신은 BPMN 프로세스 검증 전문가입니다. 생성된 프로세스를 실제 실행 엔진으로 "
            "테스트하기 위한 '분기별 테스트 케이스'를 JSON 으로 만듭니다.\n"
            "원칙:\n"
            "- 게이트웨이(분기)의 '모든 분기'가 최소 하나의 케이스에서 실행되도록 케이스를 구성한다.\n"
            "  (예: 승인/반려 분기가 있으면 '승인 경로' 케이스와 '반려 경로' 케이스를 각각 만든다.\n"
            "   분기가 여러 개면 각 분기를 한 번씩이라도 타도록 케이스를 충분히 만든다.)\n"
            "- 게이트웨이가 없으면 케이스는 1개(해피패스)면 충분하다.\n"
            "- 각 케이스의 '기대 실행 순서'는 sequences 배열이 아니라, 액티비티의 이름·설명·"
            "지시사항이 의미하는 '논리적으로 올바른 순서'로 추론한다. (sequences 에는 오류가 있을 수 있다.)\n"
            "- 각 케이스의 입력값은 그 케이스가 의도한 분기를 실제로 타도록 정한다 "
            "(게이트웨이 조건을 만족/불만족시키는 값)."
        )
        user = (
            "다음 BPMN 프로세스의 분기별 테스트 케이스를 만든다. JSON 으로만 응답하라.\n\n"
            f"[프로세스] {proc_json.get('processDefinitionName')}\n"
            f"설명: {_trunc(proc_json.get('description'), 300)}\n\n"
            f"[이벤트]\n{json.dumps(ev_payload, ensure_ascii=False)}\n\n"
            f"[액티비티]\n{json.dumps(act_payload, ensure_ascii=False)}\n\n"
            f"[게이트웨이]\n{json.dumps(gw_payload, ensure_ascii=False)}\n\n"
            f"[시퀀스(참고용 — 오류 가능)]\n{json.dumps(seq_payload, ensure_ascii=False)}\n\n"
            "다음 JSON 형식으로 응답하라:\n"
            "{\n"
            '  "cases": [\n'
            '    {\n'
            '      "name": "<이 케이스가 타는 경로 설명, 예: 심의 승인 경로>",\n'
            '      "activity_inputs": { "<activity_id>": { "<form_field_key>": <예시값>, ... }, ... },\n'
            '      "expected_activity_order": ["<activity_id>", ...]\n'
            '    }\n'
            "  ],\n"
            '  "rationale": "<케이스 구성 근거 간단히>"\n'
            "}\n\n"
            "규칙:\n"
            "- cases: 게이트웨이의 모든 분기가 최소 한 케이스에서 실행되도록 만든다. "
            "게이트웨이가 없으면 cases 는 1개.\n"
            "- activity_inputs: 각 액티비티의 form_fields 에 맞춰 현실적인 예시값. 필드 type 에 맞게. "
            "form_fields 가 없으면 빈 객체 {}. 그 케이스가 의도한 분기를 타도록 값을 정한다.\n"
            "- expected_activity_order: 그 케이스에서 startEvent 직후~endEvent 직전까지 실행될 "
            "액티비티 id 를 '의미상 올바른' 순서로. 게이트웨이/이벤트 id 는 넣지 않는다.\n"
            f"- activity_id 는 반드시 다음 중 하나여야 한다: {json.dumps(valid_ids, ensure_ascii=False)}\n"
        )
        try:
            result = await self._llm(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                8000,
            )
        except Exception as e:
            self.log.warning(f"[VALIDATION] test plan LLM 실패: {e}")
            result = None
        return self._normalize_test_plan(result)

    # ------------------------------------------------------------------ #
    # 3) 실제 실행 (엔진 driving)
    # ------------------------------------------------------------------ #
    async def _run_trace(self, proc_def_id: str, proc_json: dict,
                          activity_inputs: dict) -> dict:
        """엔진을 운영과 동일한 실제 경로로 driving 해 start→end 트레이스를 수집한다.

        한 번 호출 = 한 케이스(분기) 실행. 매 호출이 /initiate 로 '새 인스턴스'를
        만들어 처음부터 실행한다.

        ① 시작 : 실제 POST /initiate  (새 인스턴스)
        ② 제출 : 실제 POST /complete (submit_workitem — task_id 없이 activity_id 로 제출)
        ③ 다음 : bpm_proc_inst.current_activity_ids 를 DB 에서 직접 조회
        ④ 정리 : (옵션) 검증용 인스턴스 row 를 DB 에서 직접 삭제
        제출하면 폴링 서비스가 알아서 다음 태스크를 찾는다. 검증기는 그 결과
        (current_activity_ids)를 읽어 다음 활동을 제출할 뿐이다.

        activity_inputs: { activity_id: {field: value} } — 이 케이스용 폼 입력값.
        """
        activity_inputs = activity_inputs or {}
        form_id_by_act = self._form_id_by_activity(proc_json)
        node_type = self._build_node_type_map(proc_json)
        n_act = len([a for a in (proc_json.get("activities") or []) if isinstance(a, dict)])
        max_steps = n_act * 3 + 12
        base = self.engine_base_url
        headers = self._engine_headers()

        actual_order: list = []
        errors: list = []
        proc_inst_id = None
        reached_end = False
        last_status = "RUNNING"
        step = 0

        timeout = httpx.Timeout(40.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            # --- ① 프로세스 시작 (실제 /initiate — 프론트의 프로세스 시작과 동일) ---
            try:
                r = await client.post(
                    f"{base}/initiate",
                    json={"input": {"process_definition_id": proc_def_id,
                                    "email": self.actor_email}},
                )
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                raise EngineUnreachable(f"/initiate connect 실패: {e}")
            except httpx.HTTPError as e:
                raise EngineUnreachable(f"/initiate HTTP 오류: {e}")
            if r.status_code >= 400:
                # 시작 자체가 실패 = 검증 불가(결함이 아니라 사전조건 실패) → 건너뜀.
                raise EngineUnreachable(
                    f"/initiate {r.status_code} — 프로세스 시작 실패 "
                    f"(proc_def '{proc_def_id}' tenant 불일치/정의 오류 가능): {_trunc(r.text, 200)}"
                )

            # initiate_workitem 은 생성된 workitem row(dict)를 그대로 반환한다.
            init = r.json() if r.content else {}
            proc_inst_id = init.get("proc_inst_id")
            first_aid = init.get("activity_id")
            if not proc_inst_id or not first_aid:
                errors.append(f"/initiate 응답 형식 오류: {_trunc(init, 160)}")
                return self._trace_result(proc_inst_id, actual_order, False, last_status, errors, 0)

            submitted: set = set()
            queue: list = [str(first_aid)]

            try:
                while queue and step < max_steps:
                    aid = queue.pop(0)
                    if aid in submitted:
                        continue
                    submitted.add(aid)
                    step += 1
                    actual_order.append(aid)

                    # --- ② 폼 제출 (실제 /complete — task_id 없이 activity_id 로) ---
                    #   submit_workitem 이 (proc_inst_id, activity_id)로 워크아이템을 찾아
                    #   SUBMITTED 로 만든다 = 프론트의 폼 제출과 동일 경로.
                    fv = self._form_values_for(aid, activity_inputs, form_id_by_act)
                    await self._submit(client, base, proc_def_id, proc_inst_id, aid, fv, errors)

                    # --- ③ 폴링 서비스가 다음으로 진행할 때까지 대기 ---
                    #   (current_activity_ids 를 DB 에서 폴링. 다음 태스크 탐색은 폴링 서비스가 함.)
                    state = await self._wait_for_advance(proc_inst_id, aid, node_type, errors)
                    last_status = state.get("status") or last_status
                    if str(last_status).upper() == "COMPLETED":
                        reached_end = True
                        break
                    for next_aid in (state.get("current_activity_ids") or []):
                        next_aid = str(next_aid)
                        if next_aid in submitted or next_aid in queue:
                            continue
                        if not self._is_submittable(node_type, next_aid):
                            # 게이트웨이/이벤트/serviceTask 등 — 검증기가 제출하지 않는다.
                            # (serviceTask 는 폴링 서비스가 자동 실행하며, _wait_for_advance 가
                            #  그 다음의 실제 활동이 나타날 때까지 기다린다.)
                            continue
                        queue.append(next_aid)
            finally:
                # --- ④ 검증용 인스턴스 정리 (DB 직접 삭제) ---
                if proc_inst_id and self._cleanup is not None:
                    try:
                        await self._cleanup(proc_inst_id)
                    except Exception as e:
                        self.log.debug(f"[VALIDATION] cleanup 실패(무시): {e}")

        return self._trace_result(proc_inst_id, actual_order, reached_end, last_status, errors, step)

    async def _submit(self, client, base, proc_def_id, proc_inst_id, activity_id,
                      form_values, errors) -> bool:
        """실제 /complete(submit_workitem)로 제출한다 — 프론트 폼 제출과 동일 경로.

        task_id 대신 (process_instance_id, activity_id)로 제출한다. submit_workitem 이
        fetch_workitem_by_proc_inst_and_activity 로 해당 워크아이템을 찾아 SUBMITTED 로
        만들고, 그 뒤는 폴링 서비스가 처리한다.
        """
        try:
            r = await client.post(
                f"{base}/complete",
                json={"input": {
                    "process_definition_id": proc_def_id,
                    "process_instance_id": proc_inst_id,
                    "activity_id": activity_id,
                    "form_values": form_values,
                    "email": self.actor_email,
                }},
            )
            if r.status_code >= 400:
                errors.append(f"/complete[{activity_id}] {r.status_code}: {_trunc(r.text, 160)}")
                return False
            return True
        except httpx.HTTPError as e:
            errors.append(f"/complete[{activity_id}] 오류: {e}")
            return False

    async def _wait_for_advance(self, proc_inst_id, submitted_aid, node_type, errors) -> dict:
        """제출 후 폴링 서비스가 프로세스를 진행시킬 때까지 current_activity_ids 를 폴링한다.

        /complete 는 즉시 리턴(fire-and-forget)이고 폴링 서비스가 비동기(약 5초 주기)로
        진행하므로, bpm_proc_inst.current_activity_ids 에 '제출 가능한 다음 활동'이
        나타나거나(또는 프로세스 완료) 타임아웃까지 기다린다.
        serviceTask/scriptTask 는 폴링 서비스가 자동 실행하므로, 그 다음의 실제 활동이
        나타날 때까지 계속 기다린다.
        반환: { "status", "current_activity_ids" } (마지막 조회 결과)
        """
        deadline = time.monotonic() + self._advance_timeout
        last: dict = {"status": "RUNNING", "current_activity_ids": []}
        while time.monotonic() < deadline:
            await asyncio.sleep(self._poll_interval)
            try:
                state = await self._fetch_state(proc_inst_id)
            except Exception as e:
                errors.append(f"인스턴스 상태 조회 오류: {e}")
                continue
            if not isinstance(state, dict):
                continue
            last = state
            if str(state.get("status") or "").upper() == "COMPLETED":
                return state
            active = [str(x) for x in (state.get("current_activity_ids") or [])]
            # 제출한 활동이 소비되고, 제출 가능한 다음 활동이 나타났으면 = 진행됨.
            actionable = [a for a in active
                          if a != submitted_aid and self._is_submittable(node_type, a)]
            if actionable:
                return state
        return last

    @staticmethod
    def _trace_result(proc_inst_id, actual_order, reached_end, status, errors, steps) -> dict:
        return {
            "proc_inst_id": proc_inst_id,
            "actual_order": actual_order,
            "reached_end": bool(reached_end),
            "stuck_at": (actual_order[-1] if (actual_order and not reached_end) else None),
            "final_status": status,
            "errors": errors,
            "steps": steps,
            "no_progress": (len(actual_order) <= 1 and not reached_end),
        }

    @staticmethod
    def _build_node_type_map(proc_json: dict) -> dict:
        """proc_json 의 모든 노드(액티비티/게이트웨이/이벤트/서브프로세스) id → type 맵."""
        m: dict = {}
        for a in (proc_json.get("activities") or []):
            if isinstance(a, dict) and a.get("id"):
                m[str(a["id"])] = str(a.get("type") or "userTask")
        for g in (proc_json.get("gateways") or []):
            if isinstance(g, dict) and g.get("id"):
                m[str(g["id"])] = str(g.get("type") or "gateway")
        for e in (proc_json.get("events") or []):
            if isinstance(e, dict) and e.get("id"):
                m[str(e["id"])] = str(e.get("type") or "event")
        for s in (proc_json.get("subProcesses") or []):
            if isinstance(s, dict) and s.get("id"):
                m[str(s["id"])] = str(s.get("type") or "subProcess")
        return m

    @staticmethod
    def _is_submittable(node_type: dict, node_id: str) -> bool:
        """검증기가 폼을 채워 /complete 로 제출할 노드인지 판정.

        생성 프로세스의 액티비티는 전부 userTask 라, 맵에 없는 미지의 id 는
        액티비티로 보고 제출 대상으로 취급한다(게이트웨이/이벤트는 맵에 있어 걸러짐).
        """
        t = node_type.get(node_id)
        if t is None:
            return True
        return t in _SUBMITTABLE_TYPES

    def _form_id_by_activity(self, proc_json: dict) -> dict:
        res = {}
        for a in (proc_json.get("activities") or []):
            if not isinstance(a, dict):
                continue
            tool = str(a.get("tool") or "")
            if tool.startswith("formHandler:"):
                fid = tool.split(":", 1)[1].strip()
                if fid and fid.lower() not in ("defaultform", "defaultform"):
                    res[str(a.get("id"))] = fid
        return res

    @staticmethod
    def _form_values_for(activity_id, activity_inputs: dict, form_id_by_act: dict) -> dict:
        """엔진 제출용 form_values 를 만든다.

        게이트웨이 조건 평가가 form_id 로 묶인 값을 읽을 수도, 평탄한 값을 읽을 수도 있어
        둘 다 제공한다: {<field>: v, ..., <form_id>: {<field>: v, ...}}.
        """
        flat = activity_inputs.get(str(activity_id)) or {}
        if not isinstance(flat, dict):
            flat = {}
        out = dict(flat)
        fid = form_id_by_act.get(str(activity_id))
        if fid:
            out[fid] = dict(flat)
        return out

    def _engine_headers(self) -> dict:
        # process-gpt-completion 의 미들웨어는 X-Forwarded-Host 의 첫 라벨을 tenant 로 쓴다.
        if self.tenant_id and self.tenant_id != "localhost":
            return {"X-Forwarded-Host": f"{self.tenant_id}.pdf2bpmn-validation.internal"}
        return {}

    # ------------------------------------------------------------------ #
    # 4) 비교 (기대 vs 실제)
    # ------------------------------------------------------------------ #
    def _diff(self, expected_order: list, trace: dict, proc_json: dict) -> list:
        """한 케이스의 기대 순서(expected_order) vs 실제 트레이스를 비교해 결함을 낸다."""
        defects: list = []
        if not trace:
            return defects

        for e in (trace.get("errors") or [])[:5]:
            defects.append(_defect(WARNING, "engine_error", f"엔진 호출 오류: {e}"))

        if not trace.get("reached_end"):
            defects.append(_defect(
                CRITICAL, "not_reached_end",
                f"프로세스가 endEvent 까지 진행되지 못했습니다. "
                f"마지막 진행 액티비티: '{trace.get('stuck_at')}'. "
                f"실제 진행 경로: {trace.get('actual_order')}",
                actual=trace.get("actual_order")))

        act_ids = {str(a.get("id")) for a in (proc_json.get("activities") or [])
                   if isinstance(a, dict)}
        expected = [str(x) for x in (expected_order or [])
                    if str(x) in act_ids]
        actual = [str(x) for x in (trace.get("actual_order") or []) if str(x) in act_ids]

        if not expected or not actual:
            return defects
        if expected == actual:
            return defects

        # 첫 분기 지점 탐지
        diverged = False
        for i in range(min(len(expected), len(actual))):
            if expected[i] != actual[i]:
                prev = expected[i - 1] if i > 0 else "(시작)"
                defects.append(_defect(
                    CRITICAL, "order_mismatch",
                    f"실행 순서 불일치: '{prev}' 다음에는 '{expected[i]}' 가 실행되어야 하지만 "
                    f"실제로는 '{actual[i]}' 가 실행되었습니다. 시퀀스 연결이 잘못된 것으로 보입니다.",
                    expected=expected, actual=actual))
                diverged = True
                break

        if not diverged:
            # 한쪽이 다른 쪽의 prefix — 누락/추가
            missing = [x for x in expected if x not in actual]
            extra = [x for x in actual if x not in expected]
            if missing:
                defects.append(_defect(
                    CRITICAL, "missing_activity",
                    f"기대 경로의 액티비티가 실행되지 않았습니다: {missing}",
                    expected=expected, actual=actual))
            if extra:
                defects.append(_defect(
                    WARNING, "extra_activity",
                    f"기대하지 않은 액티비티가 실행되었습니다: {extra}",
                    expected=expected, actual=actual))
        return defects

    # ------------------------------------------------------------------ #
    # 5) 자동 개선 (LLM)
    # ------------------------------------------------------------------ #
    async def _improve(self, proc_json: dict, defects: list, test_plan: dict):
        """결함을 근거로 LLM 이 proc_json 을 교정한 새 정의를 만든다.

        Returns: (교정된_정의 | None, 메타정보 dict)
        """
        activities = [{
            "id": a.get("id"), "name": a.get("name"), "type": a.get("type"),
            "role": a.get("role"), "tool": a.get("tool"),
            "description": _trunc(a.get("description"), 200),
        } for a in (proc_json.get("activities") or []) if isinstance(a, dict)]
        gateways = [g for g in (proc_json.get("gateways") or []) if isinstance(g, dict)]
        events = [e for e in (proc_json.get("events") or []) if isinstance(e, dict)]
        sequences = [s for s in (proc_json.get("sequences") or []) if isinstance(s, dict)]
        roles = [r for r in (proc_json.get("roles") or []) if isinstance(r, dict)]

        # 분기별 기대 경로 — 모든 케이스가 올바르게 흐르도록 sequences 를 고쳐야 한다.
        cases = (test_plan or {}).get("cases") or []
        expected_paths = [
            {"case": c.get("name"), "expected_activity_order": c.get("expected_activity_order")}
            for c in cases if isinstance(c, dict)
        ]

        system = (
            "당신은 BPMN 프로세스 정의를 교정하는 전문가입니다. 실행 검증에서 발견된 결함을 "
            "근거로, 프로세스 정의(JSON)를 수정해 모든 분기 경로가 start→end 로 올바르게 "
            "흐르도록 고칩니다.\n"
            "수정 범위: sequences(흐름 연결), gateways(분기 조건), events, roles(역할), "
            "그리고 activities 의 일부 속성. JSON 으로만 응답합니다."
        )
        user = (
            "아래 프로세스에 실행 검증 결함이 있다. 결함을 해소하도록 정의를 교정하라. JSON 으로만 응답하라.\n\n"
            f"[발견된 결함]\n{json.dumps(defects, ensure_ascii=False, indent=1)}\n\n"
            f"[분기별 기대 경로 — 모든 경로가 실행 가능해야 함]\n"
            f"{json.dumps(expected_paths, ensure_ascii=False)}\n\n"
            f"[현재 activities]\n{json.dumps(activities, ensure_ascii=False)}\n\n"
            f"[현재 events]\n{json.dumps(events, ensure_ascii=False)}\n\n"
            f"[현재 gateways]\n{json.dumps(gateways, ensure_ascii=False)}\n\n"
            f"[현재 sequences]\n{json.dumps(sequences, ensure_ascii=False)}\n\n"
            f"[현재 roles]\n{json.dumps(roles, ensure_ascii=False)}\n\n"
            "다음 JSON 형식으로 응답하라(변경이 필요한 키만 넣어도 된다):\n"
            "{\n"
            '  "sequences": [ {"id","name","source","target","condition","properties"}, ... ],\n'
            '  "gateways":  [ ... ],   // 분기 조건을 고칠 때만\n'
            '  "events":    [ ... ],   // 이벤트를 고칠 때만\n'
            '  "roles":     [ ... ],   // 역할을 고칠 때만\n'
            '  "activities_patch": [ {"id":"<activity_id>", "<바꿀필드>":<값>}, ... ],\n'
            '  "changed": ["flow"|"gateway_condition"|"role"|"event"|"activity"],\n'
            '  "explanation": "<무엇을 왜 고쳤는지>"\n'
            "}\n\n"
            "규칙:\n"
            "- 핵심은 sequences 교정이다: source/target 연결을 고쳐, 위 '분기별 기대 경로'가 "
            "모두 실행 가능하도록 만든다. startEvent→첫 액티비티, 마지막 액티비티→endEvent 까지 "
            "모든 노드가 빠짐없이 연결되어야 한다.\n"
            "- 모든 액티비티·게이트웨이·이벤트가 startEvent 에서 도달 가능하고 endEvent 로 "
            "도달할 수 있어야 한다. 고립 노드/끊긴 흐름이 남으면 안 된다.\n"
            "- 액티비티/게이트웨이/이벤트의 id 는 절대 바꾸지 않는다(폼이 id 로 연결되어 있다).\n"
            "- 액티비티를 삭제하거나 새로 만들지 않는다. 순서만 시퀀스로 바로잡는다.\n"
            "- 게이트웨이 분기가 있으면 모든 분기 시퀀스를 유지하되, 조건이 틀렸으면 condition 을 고친다.\n"
            "- sequences 는 전체 목록을 빠짐없이 반환한다. 절대 일부만 반환하지 마라 — "
            "누락하면 그 흐름이 사라져 프로세스가 끊긴다.\n"
        )
        try:
            result = await self._llm(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                9000,
            )
        except Exception as e:
            self.log.warning(f"[VALIDATION] improve LLM 실패: {e}")
            return None, {}
        if not isinstance(result, dict):
            return None, {}
        meta = {
            "changed": result.get("changed"),
            "explanation": _trunc(result.get("explanation"), 600),
        }
        return self._merge_improvement(proc_json, result), meta

    @staticmethod
    def _merge_improvement(proc_json: dict, improvement: dict) -> dict:
        """LLM 개선안을 현재 정의에 병합한다. activities 는 교체하지 않고 patch 만 적용."""
        new = copy.deepcopy(proc_json)
        for key in ("sequences", "gateways", "events", "roles"):
            val = improvement.get(key)
            if isinstance(val, list) and val:
                new[key] = val
        patch = improvement.get("activities_patch")
        if isinstance(patch, list):
            acts_by_id = {str(a.get("id")): a for a in (new.get("activities") or [])
                          if isinstance(a, dict) and a.get("id")}
            for p in patch:
                if not isinstance(p, dict):
                    continue
                aid = str(p.get("id") or "")
                if aid in acts_by_id:
                    for k, v in p.items():
                        if k != "id":
                            acts_by_id[aid][k] = v
        return new

    @staticmethod
    def _validate_definition_shape(defn: dict):
        """개선안이 최소한의 구조를 갖췄는지 확인(완전한 검증 아님 — 다음 회차에서 재검사)."""
        if not isinstance(defn, dict):
            return False, "정의가 dict 가 아님"
        if not defn.get("activities"):
            return False, "activities 누락"
        seqs = defn.get("sequences")
        if not isinstance(seqs, list) or not seqs:
            return False, "sequences 누락/비어있음"
        events = [e for e in (defn.get("events") or []) if isinstance(e, dict)]
        has_start = any(str(e.get("type") or "").lower() == "startevent" for e in events)
        has_end = any(str(e.get("type") or "").lower() == "endevent" for e in events)
        if not has_start or not has_end:
            return False, "start/end 이벤트 누락"
        return True, ""

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _finalize_skip(self, report: dict, current: dict) -> dict:
        # 건너뛴 경우 proc_json 을 수정하지 않았으므로 final_definition 은 비워 둔다
        # (executor 가 굳이 동일 내용으로 재할당하지 않게 함).
        report["passed"] = None
        report["final_definition"] = None
        report["remaining_defects"] = []
        self._write_report(report)
        return report

    async def _emit(self, message: str, pct: int, extra: dict = None):
        if self._progress is None:
            return
        try:
            await self._progress(message, pct, extra or {})
        except Exception as e:
            self.log.debug(f"[VALIDATION] progress emit 실패(무시): {e}")

    # ------------------------------------------------------------------ #
    # 상세 리포트 — 검증이 '실제로 어떻게 동작했는지' 를 파일로 남긴다
    # ------------------------------------------------------------------ #
    def _rep(self, line: str = "") -> None:
        """리포트 본문 한 줄 누적."""
        try:
            self._rep_lines.append(str(line))
        except Exception:
            pass

    @staticmethod
    def _node_name_map(definition: dict) -> dict:
        """노드 id → 표시 이름 맵 (리포트 가독성용)."""
        m: dict = {}
        if not isinstance(definition, dict):
            return m
        for coll in ("activities", "gateways", "events", "subProcesses"):
            for n in (definition.get(coll) or []):
                if isinstance(n, dict) and n.get("id"):
                    m[str(n["id"])] = str(n.get("name") or n["id"])
        return m

    @staticmethod
    def _seq_pairs(definition: dict) -> dict:
        """sequences 를 (source, target) → condition 맵으로."""
        out: dict = {}
        for s in (definition.get("sequences") or []):
            if isinstance(s, dict):
                out[(str(s.get("source") or ""), str(s.get("target") or ""))] = \
                    str(s.get("condition") or "")
        return out

    def _rep_seq_diff(self, before: dict, after: dict) -> None:
        """before→after 흐름 변경을 '원래 어땠는데 어떻게 바뀜' 형태로 리포트에 기록."""
        names: dict = {}
        for d in (before, after):
            names.update(self._node_name_map(d))

        def nm(nid: str) -> str:
            return names.get(str(nid), str(nid))

        bp = self._seq_pairs(before)
        ap = self._seq_pairs(after)
        removed = [k for k in bp if k not in ap]
        added = [k for k in ap if k not in bp]
        cond_changed = [k for k in ap if k in bp and ap[k] != bp[k]]
        if not (removed or added or cond_changed):
            self._rep("- 시퀀스(흐름) 변경: 없음")
        else:
            self._rep("- 시퀀스(흐름) 변경 — 원래 → 변경:")
            for (s, t) in removed:
                self._rep(f"  · [흐름 제거] {nm(s)} → {nm(t)}")
            for (s, t) in added:
                self._rep(f"  · [흐름 추가] {nm(s)} → {nm(t)}")
            for (s, t) in cond_changed:
                self._rep(f"  · [조건 변경] {nm(s)} → {nm(t)}: "
                          f"'{bp[(s, t)]}' → '{ap[(s, t)]}'")
        b_gw = len(before.get("gateways") or [])
        a_gw = len(after.get("gateways") or [])
        if b_gw != a_gw:
            self._rep(f"- 게이트웨이 수: {b_gw}개 → {a_gw}개")
        self._rep(
            f"- 시퀀스 총 개수: {len(before.get('sequences') or [])}개 "
            f"→ {len(after.get('sequences') or [])}개"
        )

    def _write_report(self, report: dict) -> None:
        """누적된 리포트에 최종 결과를 덧붙여 파일(.md)로 저장한다."""
        if not self._report_path:
            return
        lines = list(self._rep_lines)
        lines.append("## 최종 결과")
        if report.get("skipped"):
            lines.append("- 상태: ⏭️ 검증 건너뜀 (실행 테스트 미수행)")
            lines.append(f"- 사유: {report.get('skip_reason')}")
        elif report.get("passed"):
            lines.append("- 상태: ✅ 검증 통과 — start→end 정상 실행 확인")
        else:
            lines.append("- 상태: ❌ 검증 미통과")
            rem = report.get("remaining_defects") or []
            lines.append(f"- 잔여 결함: {len(rem)}건")
            for d in rem:
                lines.append(
                    f"  · [{d.get('severity')}] {d.get('type')}: "
                    f"{_trunc(d.get('detail'), 240)}"
                )
        lines.append(f"- 반복 횟수: {report.get('iterations')}")
        lines.append(f"- 자동 개선(proc_json 수정) 적용: {report.get('repaired')}")
        if report.get("note"):
            lines.append(f"- 비고: {report.get('note')}")
        if self._test_proc_inst_ids:
            lines.append("")
            lines.append("## 검증용 실행 인스턴스 (DB 보존됨)")
            lines.append(
                "검증기가 실제 엔진(/initiate·/complete)으로 만든 실행 인스턴스다. "
                "PDF2BPMN_VALIDATION_CLEANUP=false 이므로 삭제하지 않고 남겨둔다. "
                "아래 쿼리로 폴링 서비스가 진행시킨 워크아이템 흐름을 직접 확인할 수 있다:"
            )
            for pid in self._test_proc_inst_ids:
                lines.append(f"- proc_inst_id `{pid}`")
                lines.append(
                    f"  - `SELECT activity_id, activity_name, status, start_date, end_date "
                    f"FROM todolist WHERE proc_inst_id = '{pid}' ORDER BY start_date;`"
                )
                lines.append(
                    f"  - `SELECT status, current_activity_ids FROM bpm_proc_inst "
                    f"WHERE proc_inst_id = '{pid}';`"
                )
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._report_path)), exist_ok=True)
            with open(self._report_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self.log.info(f"[VALIDATION] 검증 상세 리포트 저장: {self._report_path}")
        except Exception as e:
            self.log.warning(f"[VALIDATION] 리포트 파일 쓰기 실패: {e}")
