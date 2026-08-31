-- 생성된 결정론적 코드의 출처(작업 이력 요약).
--
-- 코드는 "무엇을 하는가"만 보여 준다. 그 코드가 왜 그렇게 생겼는지 — 어떤 스킬
-- 절차를 따랐고, 어떤 셸 스크립트를 돌렸으며, 어떤 파일을 참고했는지 — 는 이력에만
-- 남는다. 나중에 그 스킬이 바뀌었을 때 코드를 의심할 근거가 필요하므로 함께 저장한다.

ALTER TABLE public.mcp_python_code
    ADD COLUMN IF NOT EXISTS work_history jsonb;

COMMENT ON COLUMN public.mcp_python_code.work_history IS
    '고착화 근거가 된 작업 이력 요약: 표본 수, 행위 종류별 횟수, 사용 도구, 읽은 스킬/파일, 실행한 셸 명령, 실행 지문.';
