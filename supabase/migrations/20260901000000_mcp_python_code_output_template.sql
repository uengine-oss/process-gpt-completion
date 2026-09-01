-- 고착화된 활동의 폼 산출물 템플릿.
--
-- 에이전트는 도구를 부르고 끝나지 않는다. 마지막에 워크아이템의 폼 필드 값을 내놓고,
-- 다음 활동은 그 값을 입력으로 받는다. 고착화 코드가 도구 호출만 재현하면 폼이 빈 채로
-- 워크아이템이 완료되어 다음 단계가 굶는다.
--
-- 모양: {"form_id": "...", "fields": {"<필드>": {"const": <값>} | {"template": "..."} | null}}
--   const    - 표본 전체에서 값이 같았다. 그대로 쓴다.
--   template - 값의 차이가 파라미터로 전부 설명된다. `${이름}` 을 이번 입력으로 렌더한다.
--   null     - 설명되지 않는 차이가 남는다(에이전트가 매번 다르게 쓴 산문 등). 옛 표본의
--              사실을 굳히면 조용히 틀리므로, 실행기가 이번 실행의 사실로 채운다.

ALTER TABLE public.mcp_python_code
    ADD COLUMN IF NOT EXISTS output_template jsonb;

COMMENT ON COLUMN public.mcp_python_code.output_template IS
    '폼 산출물 템플릿. 필드마다 const(상수) / template(파라미터 렌더) / null(실행 요약으로 대체).';
