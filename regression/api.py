"""regression/api.py — 병합 전 회귀 검증의 실행 엔드포인트.

deepagents 가 회차를 만들고(`resource_eval_runs`) 시나리오를 꺼내 여기로 보내면, 이 서비스가
두 버전의 정의를 읽어 케이스를 재생·채점해 돌려준다. 회차 상태 기록과 base/head 비교는
호출자가 한다 — 여기는 "이 정의에서 이 시나리오는 어떤 결과인가" 만 답한다.

## 비용

모델을 부르지 않고 실행 엔진 인스턴스도 만들지 않는다. 응답에 `llm_calls`,
`engine_instances` 를 실어 호출자가 그대로 화면에 올릴 수 있게 한다 — 숫자가 보이지 않으면
아무도 새지 않는지 확인하지 않는다.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException, Request

from . import dmn_derive, dmn_replay, process_derive, process_replay, versions

logger = logging.getLogger(__name__)

BASE_VARIANT = "base"
HEAD_VARIANT = "head"


def _payload(body: dict) -> dict:
    inner = body.get("input")
    return inner if isinstance(inner, dict) else body


async def run_regression(request: Request) -> dict:
    """시나리오를 두 버전에 재생해 케이스별 결과를 돌려준다.

    body: { "input": {
        "resource_type": "bpmn" | "dmn",
        "resource_id": "...",
        "base_ref": "v1.0", "head_ref": "v1.1",
        "cases": [{eval_name, prompt, assertions, checks}],
        "skip_base": ["<eval_name>"]        # 기준선을 재사용하는 시나리오(선택)
    }}
    """
    body = await request.json()
    payload = _payload(body if isinstance(body, dict) else {})

    resource_type = str(payload.get("resource_type") or "").strip()
    resource_id = str(payload.get("resource_id") or "").strip()
    cases = payload.get("cases") or []
    if resource_type not in ("bpmn", "dmn"):
        raise HTTPException(status_code=400, detail="resource_type must be 'bpmn' or 'dmn'")
    if not resource_id:
        raise HTTPException(status_code=400, detail="resource_id is required")
    if not isinstance(cases, list) or not cases:
        raise HTTPException(status_code=400, detail="cases is required")

    from database import subdomain_var

    tenant_id = subdomain_var.get() or "localhost"
    skip_base = set(payload.get("skip_base") or [])

    def _load(ref: str):
        if resource_type == "dmn":
            return versions.load_dmn_xml(tenant_id, resource_id, ref)
        return versions.load_process_definition(tenant_id, resource_id, ref)

    definitions: dict = {}
    digests: dict = {}
    for variant, ref in ((BASE_VARIANT, payload.get("base_ref") or ""),
                         (HEAD_VARIANT, payload.get("head_ref") or "")):
        try:
            definitions[variant], digests[variant] = await asyncio.to_thread(_load, ref)
        except ValueError as e:
            return {"error": "version_unavailable", "variant": variant,
                    "message": f"{variant}({ref}) 를 가져오지 못했습니다: {e}"}

    evaluate = dmn_replay.evaluate_case if resource_type == "dmn" else process_replay.replay_case

    results: list[dict] = []
    undecided: list[dict] = []
    for case in cases:
        for variant in (BASE_VARIANT, HEAD_VARIANT):
            if variant == BASE_VARIANT and case.get("eval_name") in skip_base:
                continue
            observed, graded = evaluate(definitions[variant], case)
            if graded is None:
                undecided.append({
                    "eval_name": case.get("eval_name"),
                    "variant": variant,
                    # 판정하지 못한 이유를 그대로 돌려준다 — 조용히 두면 호출자가
                    # broken_count=0 을 "이상 없음" 으로 읽는다.
                    "reason": observed.get("undecided") or "이 시나리오는 재생으로 채점할 수 없습니다.",
                })
                continue
            results.append({
                "eval_name": case.get("eval_name"),
                "variant": variant,
                "passed": graded["passed"],
                "total": graded["total"],
                "pass_rate": graded["pass_rate"],
                "assertions": graded["assertions"],
            })

    logger.info(
        "[regression-run] %s/%s %s→%s 결과 %d건 판정불가 %d건",
        resource_type, resource_id, payload.get("base_ref"), payload.get("head_ref"),
        len(results), len(undecided),
    )
    return {
        "results": results,
        "undecided": undecided,
        "base_digest": digests[BASE_VARIANT],
        "head_digest": digests[HEAD_VARIANT],
        "mode": "dmn" if resource_type == "dmn" else "replay",
        "llm_calls": 0,
        "engine_instances": 0,
    }


async def derive_regression_scenarios(request: Request) -> dict:
    """변경 전 정의에서 회귀 시나리오를 파생한다(모델 호출 없음).

    의사결정은 규칙 표에서, 프로세스는 배타 게이트웨이 조합에서 만든다. 어느 쪽이든 기준은
    **변경 전(base) 버전**이다 — 변경 후를 보고 만들면 바뀐 동작을 기준선으로 굳혀 버려서
    그 검증이 아무것도 잡아내지 못한다.

    body: { "input": { "resource_type": "dmn" | "bpmn", "resource_id", "base_ref" } }
    `resource_type` 이 없으면 의사결정으로 본다(이 경로를 처음 쓴 쪽이 DMN 이라 호환).
    """
    body = await request.json()
    payload = _payload(body if isinstance(body, dict) else {})

    resource_id = str(payload.get("resource_id") or "").strip()
    if not resource_id:
        raise HTTPException(status_code=400, detail="resource_id is required")
    resource_type = str(payload.get("resource_type") or "dmn").strip()
    if resource_type not in ("bpmn", "dmn"):
        raise HTTPException(status_code=400, detail="resource_type must be 'bpmn' or 'dmn'")

    from database import subdomain_var

    tenant_id = subdomain_var.get() or "localhost"
    base_ref = str(payload.get("base_ref") or "")

    loader = versions.load_dmn_xml if resource_type == "dmn" else versions.load_process_definition
    try:
        definition, _digest = await asyncio.to_thread(loader, tenant_id, resource_id, base_ref)
    except ValueError as e:
        return {"error": "no_definition", "message": str(e)}

    if resource_type == "dmn":
        derived = await asyncio.to_thread(dmn_derive.derive_cases, definition)
        cases = dmn_derive.to_eval_cases(derived)
        skipped: list[str] = []
        empty_reason = (
            f"'{resource_id}' 의 의사결정 표에서 시나리오를 만들지 못했습니다. "
            "규칙 행이 없거나 DMN 을 읽을 수 없습니다."
        )
    else:
        derived, skipped = await asyncio.to_thread(process_derive.derive_cases, definition)
        cases = process_derive.to_eval_cases(derived)
        empty_reason = (
            f"'{resource_id}' 의 프로세스 정의에서 시나리오를 만들지 못했습니다."
            + ((" " + skipped[0]) if skipped else "")
        )

    if not cases:
        return {"error": "no_cases", "message": empty_reason}

    logger.info(
        "[regression-scenarios] %s/%s base=%s 시나리오 %d건 (건너뜀 %d건)",
        resource_type, resource_id, base_ref, len(cases), len(skipped),
    )
    # 만들지 못한 갈림길은 숨기지 않는다 — "시나리오 3건" 만 보이면 리뷰어가 커버되지 않은
    # 분기를 커버된 것으로 읽는다.
    return {"cases": cases, "skipped_reasons": skipped[:5], "llm_calls": 0}


def add_routes_to_app(app):
    app.add_api_route("/regression-run", run_regression, methods=["POST"])
    app.add_api_route("/regression-scenarios", derive_regression_scenarios, methods=["POST"])
