"""
알림을 누구에게 보낼 것인가.

`notifications.user_id` 에는 두 가지가 섞여 들어온다.
  - 채팅 알림 : 이메일        (예: someone@company.com)
  - 업무 알림 : 사용자 UUID   (예: efddd554-4e2a-...)

업무 알림은 `todolist.user_id` 를 그대로 옮겨 담기 때문이다. 그런데
`user_devices` 의 키는 이메일(`user_email`)이라, UUID 로는 절대 찾지 못한다.
그래서 **업무 알림 푸시는 한 번도 나간 적이 없었다.** 화면(포털)은 이메일과
UUID 를 모두 조회해 왔기 때문에 알림 목록에서는 정상으로 보였고, 안 오는 것은
푸시뿐이라 눈에 띄지 않았다.

또 한 업무에 담당자가 여럿이면 콤마로 이어 붙는다(`"uuid-a,uuid-b"`).
그 경우 모두에게 보내야 한다.

이 모듈은 데이터베이스도 Firebase 도 알지 못한다 — 조회 함수를 받아서 쓴다.
그래야 이 규칙만 따로 시험할 수 있다.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

UUID_PATTERN = re.compile(
    r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
)


def looks_like_uuid(value: Optional[str]) -> bool:
    return bool(UUID_PATTERN.match((value or '').strip()))


def split_recipients(user_id: Optional[str]) -> List[str]:
    """담당자가 여럿이면 콤마로 이어 붙는다. 하나씩 떼어 낸다."""
    return [part.strip() for part in (user_id or '').split(',') if part.strip()]


def resolve_user_emails(
    user_id: Optional[str],
    lookup_emails: Callable[[List[str]], Iterable[Dict[str, str]]],
    on_error: Optional[Callable[[Exception], None]] = None,
) -> List[str]:
    """
    수신자 칸을 실제 이메일 목록으로 바꾼다.

    Args:
        user_id: 알림의 수신자 칸. 이메일 · UUID · 콤마로 이은 여럿.
        lookup_emails: UUID 목록을 받아 `{id, email}` 들을 돌려주는 함수.
        on_error: 조회가 실패했을 때 알릴 곳(선택).

    Returns:
        중복을 없앤 이메일 목록. 못 찾은 값은 조용히 버린다 —
        하나 때문에 나머지 수신자까지 잃으면 안 된다.
    """
    emails: List[str] = []
    uuids: List[str] = []

    for value in split_recipients(user_id):
        if looks_like_uuid(value):
            uuids.append(value)
        else:
            # 이메일이거나, 봇 이름 같은 그 밖의 식별자. 그대로 시도한다.
            emails.append(value)

    if uuids:
        try:
            for row in (lookup_emails(uuids) or []):
                email = (row.get('email') or '').strip()
                if email:
                    emails.append(email)
        except Exception as e:  # noqa: BLE001 - 나머지 수신자에게는 보내야 한다
            if on_error:
                on_error(e)

    return dedupe(emails)


def dedupe(values: Iterable[str]) -> List[str]:
    """순서를 지키면서 중복만 없앤다."""
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def usable_tokens(rows: Iterable[Dict[str, str]]) -> List[str]:
    """
    실제로 보낼 수 있는 토큰만 남긴다.

    `user_devices` 에는 토큰 칸이 비어 있는 행이 실제로 있다(등록만 되고 토큰을
    받지 못한 경우). 빈 값으로 발송을 시도하면 Firebase 가 거절한다.
    """
    return dedupe((row.get('device_token') or '').strip() for row in (rows or []))


INSTANCE_SUFFIX = re.compile(
    r'[._][0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
)


def readable_instance_name(name: Optional[str]) -> str:
    """
    알림 본문에 쓸 건 이름을 사람이 읽을 수 있게 다듬는다.

    엔진이 만드는 인스턴스 이름은 `휴가 신청 프로세스_d0933d4d-ba58-...` 처럼
    끝에 인스턴스 식별자가 붙는다. 화면에서는 옆에 다른 정보가 있어 견딜 만하지만,
    알림은 두 줄이 전부다. 그 두 줄의 절반을 사람이 읽을 수 없는 문자열이
    차지하면 무슨 일로 온 알림인지 알 수 없다.

    꼬리의 식별자만 떼고 나머지는 그대로 둔다 — 이름 안에 뜻이 있을 수 있다.
    """
    value = (name or '').strip()
    if not value:
        return ''
    return INSTANCE_SUFFIX.sub('', value).rstrip(' _.-') or value


def readable_message(content: Optional[str]) -> str:
    """
    대화 알림에 쓸 한 줄.

    에이전트가 되물을 때는 본문이 JSON 으로 온다.

        {"user_request_type": "ask_user", "question": "어떤 목록을 ...", ...}

    그것을 그대로 알림에 실으면 잠금 화면에 중괄호와 따옴표 덩어리가 뜬다 —
    실제로 그렇게 보였다. 두 줄이 전부인 알림에서 그 두 줄이 기계어면
    무슨 일로 온 것인지 알 수 없다. 물음만 꺼내 쓴다.
    """
    value = (content or '').strip()
    if not value.startswith('{'):
        return value

    try:
        import json

        parsed = json.loads(value)
    except Exception:  # noqa: BLE001 - JSON 이 아니면 원문이 맞다
        return value

    if not isinstance(parsed, dict):
        return value

    for key in ('question', 'message', 'text', 'content'):
        picked = parsed.get(key)
        if isinstance(picked, str) and picked.strip():
            return picked.strip()

    # 아는 모양이 아니다. 기계어를 보여 주느니 아무 말도 하지 않는다 —
    # 부르는 쪽이 기본 문구로 채운다.
    return ''


def notification_text(title: Optional[str], description: Optional[str]) -> tuple:
    """
    알림의 제목과 본문.

    제목은 무슨 일인지(활동 이름 · 보낸 말), 본문은 어느 건인지(프로세스 이름 ·
    대화방 이름)다. 본문이 비거나 제목과 같으면 같은 말을 두 번 쓰지 않는다.
    """
    head = readable_message(title)
    body = readable_instance_name(description)
    if body == head:
        body = ''
    return head, body


# =============================================================================
# 어느 기기로 보낼 것인가
# =============================================================================

# 이 시간 안에 쓴 기기는 "지금 쓰고 있는 기기" 로 본다.
#
# 왜 10분인가
#   메신저들이 쓰는 값과 같다(Slack 의 자리 비움 판정이 10분이다). 너무 짧으면
#   잠깐 다른 창을 본 사이에 "안 쓰는 기기" 가 되어 알림이 여기저기로 흩어지고,
#   너무 길면 이미 덮어 둔 노트북으로만 가서 휴대폰에는 오지 않는다.
ACTIVE_WINDOW_SECONDS = 600


def _as_epoch(value) -> Optional[float]:
    """시각을 숫자로. 모양이 제각각이라(문자열·datetime) 한 곳에서 흡수한다."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    # Postgres 는 '2026-09-07T05:49:46.808+00:00' 처럼 준다. 'Z' 도 받아 준다.
    try:
        from datetime import datetime

        return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()
    except Exception:  # noqa: BLE001 - 못 읽으면 "모르는 시각" 으로 둔다
        return None


def target_devices(rows, now=None, window_seconds: int = ACTIVE_WINDOW_SECONDS) -> List[Dict]:
    """
    이 사람의 기기들 중 어디로 보낼지 고른다.

    규칙은 메신저들이 하는 것과 같다.
      - 지금 쓰고 있는 기기가 있으면 **그 기기들로만** 보낸다.
        노트북을 보고 있는데 휴대폰이 같이 울릴 이유가 없다.
      - 아무 기기도 쓰고 있지 않으면 **가진 기기 모두로** 보낸다.
        노트북을 덮어 두었는지, 꺼 두었는지 우리는 알 수 없다. 어느 것을 집어
        들든 보이게 하는 편이 안전하다 — 못 받는 것이 가장 나쁘다.

    `last_active_at` 이 없는 기기(옛날에 등록만 된 것)는 "쓰고 있지 않다" 로 본다.
    그래도 아무도 활동 중이 아니면 함께 받는다.
    """
    usable = [row for row in (rows or []) if (row.get('device_token') or '').strip()]
    if not usable:
        return []

    import time

    current = float(now if now is not None else time.time())
    active = []
    for row in usable:
        seen = _as_epoch(row.get('last_active_at'))
        if seen is not None and (current - seen) <= window_seconds:
            active.append(row)

    return active or usable


def tokens_of(rows) -> List[str]:
    """고른 기기들의 토큰. 같은 토큰이 두 줄에 있어도 한 번만 보낸다."""
    return dedupe((row.get('device_token') or '').strip() for row in (rows or []))
