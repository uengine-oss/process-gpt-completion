-- 워크아이템이 완료될 때 게이트웨이가 어느 분기를 골랐는지 남긴다.
--
-- 폴링 서비스는 이미 이 판정을 하고 있다 — `_evaluate_sequence_conditions` 가 시퀀스마다
-- conditionEval(참·거짓)과 conditionReason(판정 이유)을 만들어 라우팅에 쓴다. 문제는 그
-- 값을 담은 sequence_condition_data 가 워크아이템 처리 때마다 새로 만들어져 쓰이고
-- 사라진다는 것이다. DB 에 남는 곳이 없다.
--
-- 남겨야 하는 이유는 두 가지다.
--
-- 1) 병합 전 회귀 테스트. 저장된 시나리오를 엔진 없이 재생하려면 "이 갈림길에서 어느
--    쪽으로 갔는가" 가 있어야 한다. 실행 경로(지나간 액티비티 목록)만 보고 역산하는 방식은
--    두 분기가 같은 액티비티로 향할 때("고액 → 정밀검토", "특수건 → 정밀검토") 구분이
--    불가능하다. 그 상태로 기준을 굳히면 나중에 한쪽 대상이 바뀌었을 때 깨진 변경을
--    "이상 없음" 으로 통과시킨다.
-- 2) 사람이 "이 건이 왜 이 길로 갔는지" 를 되짚을 근거. 자연어 조건 판정은 모델이 하므로
--    판정 이유가 남지 않으면 나중에 설명할 방법이 없다.
--
-- 모양: {"<gateway_id>": {"selected": ["<seq_id>"],
--                         "sequences": {"<seq_id>": {"target", "condition", "eval", "reason"}}}}
--
-- 게이트웨이에서 나가는 분기가 없는 워크아이템(직선 흐름)에는 쓰지 않는다 — 대부분의
-- 워크아이템이 여기 해당하므로 쓰기 비용이 늘지 않는다.

ALTER TABLE public.todolist
    ADD COLUMN IF NOT EXISTS gateway_decisions jsonb;

COMMENT ON COLUMN public.todolist.gateway_decisions IS
    '이 워크아이템 완료 시점의 게이트웨이 분기 판정. {gateway_id: {selected: [seq_id], sequences: {seq_id: {target, condition, eval, reason}}}}. 회귀 테스트 재생과 분기 근거 추적에 쓴다.';
