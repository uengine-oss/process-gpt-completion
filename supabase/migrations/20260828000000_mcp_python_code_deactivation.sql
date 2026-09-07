-- 고착화된 결정론적 코드의 비활성화 이력.
--
-- 재작업이 반복되면 고착화된 코드가 틀렸다는 증거로 보고 비활성화한다. 행을 지우지
-- 않고 남기는 이유는 (1) 언제 왜 풀렸는지 추적하기 위해서이고, (2) 표본 재축적의
-- 기준 시각이 필요하기 때문이다. 비활성 시각 이전의 실행을 다시 표본으로 세면
-- 방금 비활성화한 것과 동일한 코드를 곧바로 재생성해 무한 반복에 빠진다.

ALTER TABLE public.mcp_python_code
    ADD COLUMN IF NOT EXISTS deactivated_at timestamptz,
    ADD COLUMN IF NOT EXISTS deactivated_reason text;

-- 실행 경로는 활성 코드 한 건만 조회한다.
CREATE INDEX IF NOT EXISTS mcp_python_code_active_idx
    ON public.mcp_python_code (proc_def_id, activity_id, tenant_id, created_at DESC)
    WHERE deactivated_at IS NULL;

-- 표본 재축적 기준 시각 조회용.
CREATE INDEX IF NOT EXISTS mcp_python_code_deactivated_idx
    ON public.mcp_python_code (proc_def_id, activity_id, tenant_id, deactivated_at DESC)
    WHERE deactivated_at IS NOT NULL;

COMMENT ON COLUMN public.mcp_python_code.deactivated_at IS
    '고착화 해제 시각. NULL이면 활성. 표본 재축적의 기준점으로도 쓰인다.';
COMMENT ON COLUMN public.mcp_python_code.deactivated_reason IS
    '고착화 해제 사유. 현재는 재작업 반복(rework).';
