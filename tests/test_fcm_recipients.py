"""
알림 수신자 해석 검증.

지키려는 것은 하나다: **업무 알림이 실제로 발송되는 것.**

notifications.user_id 에는 두 가지가 섞여 들어온다.
  - 채팅 알림 : 이메일
  - 업무 알림 : 사용자 UUID (todolist.user_id 를 그대로 옮겨 담는다)

user_devices 의 키는 이메일이라, UUID 를 그대로 조회하면 절대 찾지 못한다.
그래서 업무 알림 푸시는 한 번도 나간 적이 없었다. 이 시험이 그 회귀를 막는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fcm_service"))

from recipients import (  # noqa: E402
    looks_like_uuid,
    resolve_user_emails,
    split_recipients,
    usable_tokens,
)

USERS = {
    "efddd554-4e2a-4c0b-8068-9d0d83e0fe3b": "mobile@test.local",
    "ff863b27-51a4-4f58-8246-41e1761046ff": "second@test.local",
}


def lookup(uuids):
    return [{"id": u, "email": USERS[u]} for u in uuids if u in USERS]


def test_email_passes_through():
    """채팅 알림은 이미 이메일로 온다. 그대로 쓴다."""
    assert resolve_user_emails("mobile@test.local", lookup) == ["mobile@test.local"]


def test_uuid_is_resolved_to_email():
    """업무 알림은 UUID 로 온다. 바꾸지 않으면 기기를 못 찾는다."""
    assert resolve_user_emails("efddd554-4e2a-4c0b-8068-9d0d83e0fe3b", lookup) == ["mobile@test.local"]


def test_multiple_assignees_are_all_resolved():
    """한 업무에 담당자가 여럿이면 콤마로 이어 붙는다. 모두에게 보내야 한다."""
    both = "efddd554-4e2a-4c0b-8068-9d0d83e0fe3b,ff863b27-51a4-4f58-8246-41e1761046ff"
    assert resolve_user_emails(both, lookup) == ["mobile@test.local", "second@test.local"]


def test_mixed_email_and_uuid():
    mixed = "someone@company.com, efddd554-4e2a-4c0b-8068-9d0d83e0fe3b"
    assert resolve_user_emails(mixed, lookup) == ["someone@company.com", "mobile@test.local"]


def test_unknown_uuid_is_dropped_quietly():
    """모르는 UUID 때문에 나머지 수신자까지 잃지 않는다."""
    mixed = "00000000-0000-4000-8000-000000000000,mobile@test.local"
    assert resolve_user_emails(mixed, lookup) == ["mobile@test.local"]


def test_lookup_failure_still_sends_to_the_others():
    """이메일 변환이 실패해도 이미 이메일인 수신자에게는 보낸다."""
    seen = []

    def broken(_uuids):
        raise RuntimeError("db down")

    out = resolve_user_emails(
        "mobile@test.local,efddd554-4e2a-4c0b-8068-9d0d83e0fe3b",
        broken,
        on_error=seen.append,
    )
    assert out == ["mobile@test.local"]
    assert len(seen) == 1


def test_duplicates_are_collapsed():
    """같은 사람이 두 번 적혀도 알림은 한 번만."""
    dup = "mobile@test.local,efddd554-4e2a-4c0b-8068-9d0d83e0fe3b"
    assert resolve_user_emails(dup, lookup) == ["mobile@test.local"]


def test_blank_input():
    assert resolve_user_emails("", lookup) == []
    assert resolve_user_emails(None, lookup) == []


def test_bot_names_are_left_alone():
    """봇 이름이 담당자로 들어오는 업무가 실제로 있다. UUID 가 아니므로 그대로 둔다."""
    assert resolve_user_emails("approval_notification_bot", lookup) == ["approval_notification_bot"]


def test_looks_like_uuid():
    assert looks_like_uuid("efddd554-4e2a-4c0b-8068-9d0d83e0fe3b")
    assert not looks_like_uuid("mobile@test.local")
    assert not looks_like_uuid("")
    assert not looks_like_uuid(None)


def test_split_recipients_trims_and_drops_blanks():
    assert split_recipients(" a , ,b ") == ["a", "b"]


def test_usable_tokens_drops_blank_rows():
    """토큰 칸이 비어 있는 행이 실제로 있다. 빈 값으로 발송을 시도하지 않는다."""
    rows = [
        {"device_token": "token-1"},
        {"device_token": "   "},
        {"device_token": None},
        {"device_token": "token-1"},
        {"device_token": "token-2"},
    ]
    assert usable_tokens(rows) == ["token-1", "token-2"]


# ---------------------------------------------------------------------------
# 알림 문구
# ---------------------------------------------------------------------------

from recipients import notification_text, readable_instance_name  # noqa: E402


def test_instance_id_is_stripped_from_the_body():
    """알림은 두 줄이 전부다. 절반이 UUID 면 무슨 일인지 알 수 없다."""
    assert readable_instance_name(
        "휴가 신청 프로세스_d0933d4d-ba58-b525-5ec3-4e92ee3c56e6"
    ) == "휴가 신청 프로세스"


def test_dot_separated_instance_id_is_stripped():
    assert readable_instance_name(
        "leave_request_process.d0933d4d-ba58-b525-5ec3-4e92ee3c56e6"
    ) == "leave_request_process"


def test_name_without_id_is_left_alone():
    """이름 안에 뜻이 있을 수 있다. 꼬리의 식별자만 뗀다."""
    assert readable_instance_name("2026년 3분기 투자심의") == "2026년 3분기 투자심의"


def test_blank_name():
    assert readable_instance_name("") == ""
    assert readable_instance_name(None) == ""


def test_notification_text_splits_what_and_which():
    head, body = notification_text(
        "휴가 신청서 검토 요청", "휴가 신청 프로세스_d0933d4d-ba58-b525-5ec3-4e92ee3c56e6"
    )
    assert head == "휴가 신청서 검토 요청"
    assert body == "휴가 신청 프로세스"


def test_notification_text_does_not_repeat_itself():
    """제목과 본문이 같으면 같은 말을 두 번 보여 주지 않는다."""
    head, body = notification_text("승인 요청", "승인 요청")
    assert head == "승인 요청"
    assert body == ""
