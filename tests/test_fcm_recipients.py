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
    ACTIVE_WINDOW_SECONDS,
    looks_like_uuid,
    notification_text,
    resolve_user_emails,
    split_recipients,
    target_devices,
    tokens_of,
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


# =============================================================================
# 어느 기기로 보낼 것인가
#
# 지키려는 것은 하나다: **어느 기기를 집어 들든 알림이 거기 있는 것.**
#
# 예전에는 사람당 기기가 하나였다. 회사 PC 에서 웹을 켜면 휴대폰 토큰이 덮어써져
# 휴대폰 알림이 조용히 끊겼다. 끊긴 줄도 모른다.
# =============================================================================

NOW = 1_757_000_000.0


def _device(token, seconds_ago=None, kind="web"):
    return {
        "device_token": token,
        "device_type": kind,
        "last_active_at": None if seconds_ago is None else NOW - seconds_ago,
    }


def test_보고_있는_기기로만_보낸다():
    """노트북을 보고 있는데 휴대폰이 같이 울릴 이유가 없다."""
    devices = [_device("pc", 60), _device("phone", 4000, "android")]

    assert tokens_of(target_devices(devices, now=NOW)) == ["pc"]


def test_휴대폰을_보고_있으면_휴대폰으로():
    devices = [_device("pc", 4000), _device("phone", 30, "android")]

    assert tokens_of(target_devices(devices, now=NOW)) == ["phone"]


def test_아무_기기도_안_쓰면_모든_기기로_보낸다():
    """
    PC 를 마지막으로 썼더라도 지금 꺼져 있는지 우리는 알 수 없다.
    어느 것을 집어 들든 보이게 하는 편이 안전하다 — 못 받는 것이 가장 나쁘다.
    """
    devices = [_device("pc", 4000), _device("phone", 5000, "android")]

    assert tokens_of(target_devices(devices, now=NOW)) == ["pc", "phone"]


def test_둘_다_쓰고_있으면_둘_다():
    devices = [_device("pc", 10), _device("phone", 20, "android")]

    assert tokens_of(target_devices(devices, now=NOW)) == ["pc", "phone"]


def test_기준_시간_경계():
    """딱 경계에 걸친 기기는 아직 쓰고 있는 것으로 본다."""
    on_edge = [_device("pc", ACTIVE_WINDOW_SECONDS), _device("phone", 99999, "android")]
    just_over = [_device("pc", ACTIVE_WINDOW_SECONDS + 1), _device("phone", 99999, "android")]

    assert tokens_of(target_devices(on_edge, now=NOW)) == ["pc"]
    # 아무도 활동 중이 아니므로 모두에게
    assert tokens_of(target_devices(just_over, now=NOW)) == ["pc", "phone"]


def test_활동_기록이_없는_기기도_아무도_안_쓰면_받는다():
    """그 기기가 유일한 통로일 수 있다."""
    only_unknown = [_device("old", None)]
    assert tokens_of(target_devices(only_unknown, now=NOW)) == ["old"]

    with_active = [_device("old", None), _device("pc", 10)]
    assert tokens_of(target_devices(with_active, now=NOW)) == ["pc"]


def test_토큰_없는_줄에는_보내지_않는다():
    """빈 값으로 보내면 Firebase 가 거절한다."""
    devices = [_device("", 99999), _device(None, 99999), _device("phone", 4000, "android")]

    assert tokens_of(target_devices(devices, now=NOW)) == ["phone"]


# ---------------------------------------------------------------------------
# 푸시를 받을 수 없는 기기도 "쓰고 있는가" 판단에는 넣는다
#
# 포털은 브라우저용 푸시 토큰을 만들지 않는다 — 화면 위쪽 종 모양에 실시간으로
# 띄운다. 그래서 PC 앞에 앉아 있는 사람에게는 이미 보이고 있다.
# ---------------------------------------------------------------------------


def test_PC_웹을_보고_있으면_휴대폰을_울리지_않는다():
    """PC 화면에 이미 떠 있는데 주머니 속 휴대폰까지 울리면 방해다."""
    devices = [_device(None, 30, "web"), _device("phone", 4000, "android")]

    assert tokens_of(target_devices(devices, now=NOW)) == []


def test_PC_를_떠나면_휴대폰으로_간다():
    """PC 가 꺼져 있는지 우리는 알 수 없다. 어느 것을 집어 들든 보이게 한다."""
    devices = [_device(None, 4000, "web"), _device("phone", 5000, "android")]

    assert tokens_of(target_devices(devices, now=NOW)) == ["phone"]


def test_PC_와_휴대폰을_둘_다_쓰고_있으면_휴대폰으로도_간다():
    """휴대폰을 손에 들고 있다면 거기서 보는 것이 자연스럽다."""
    devices = [_device(None, 30, "web"), _device("phone", 30, "android")]

    assert tokens_of(target_devices(devices, now=NOW)) == ["phone"]


def test_마지막_사용_시각을_모르는_줄은_발송을_막지_않는다():
    """모르는 것을 '쓰고 있다' 로 세면 엉뚱하게 알림이 끊긴다."""
    devices = [_device(None, None, "web"), _device("phone", 4000, "android")]

    assert tokens_of(target_devices(devices, now=NOW)) == ["phone"]


def test_기기가_없으면_보낼_곳도_없다():
    assert target_devices([], now=NOW) == []
    assert target_devices(None, now=NOW) == []


def test_시각이_문자열로_와도_읽는다():
    """Postgres 는 '2026-09-07T05:49:46.808+00:00' 처럼 준다."""
    import datetime

    recent = datetime.datetime.fromtimestamp(NOW - 60, datetime.timezone.utc).isoformat()
    stale = datetime.datetime.fromtimestamp(NOW - 9999, datetime.timezone.utc).isoformat()
    devices = [
        {"device_token": "pc", "last_active_at": recent},
        {"device_token": "phone", "last_active_at": stale},
    ]

    assert tokens_of(target_devices(devices, now=NOW)) == ["pc"]


def test_읽을_수_없는_시각은_모르는_것으로_둔다():
    """모양이 이상하다고 그 기기를 잃으면 안 된다."""
    devices = [{"device_token": "odd", "last_active_at": "언젠가"}]

    assert tokens_of(target_devices(devices, now=NOW)) == ["odd"]


def test_같은_토큰이_두_줄에_있어도_한_번만():
    devices = [_device("same", 10), _device("same", 20, "android")]

    assert tokens_of(target_devices(devices, now=NOW)) == ["same"]


# =============================================================================
# 알림에 실을 글
#
# 알림은 두 줄이 전부다. 그 두 줄이 기계어면 무슨 일로 온 것인지 알 수 없다.
# =============================================================================


def test_되묻는_JSON은_물음만_보여_준다():
    """
    실제로 잠금 화면에 이렇게 떴다:
      Process GPT Agent 우리 회사 조직도 {"user_r...
    """
    raw = (
        '{"user_request_type": "ask_user", '
        '"question": "어떤 목록을 보여드릴까요? 대상이 필요합니다.", '
        '"waiting_for_user_input": true, '
        '"context": "\'list\'만으로는 범위를 알 수 없어요.", '
        '"suggestions": ["프로세스 정의 목록"]}'
    )

    head, _ = notification_text(raw, "대화방")

    assert head == "어떤 목록을 보여드릴까요? 대상이 필요합니다."


def test_보통_글은_그대로_둔다():
    head, body = notification_text("출장 계획 등록", "출장계획 9/7")

    assert head == "출장 계획 등록"
    assert body == "출장계획 9/7"


def test_중괄호로_시작하지만_JSON_이_아니면_원문():
    head, _ = notification_text("{이건 그냥 글입니다}", "")

    assert head == "{이건 그냥 글입니다}"


def test_모르는_모양의_JSON_은_기계어를_보여_주지_않는다():
    """부르는 쪽이 '새 알림' 같은 기본 문구로 채운다."""
    head, _ = notification_text('{"foo": "bar"}', "")

    assert head == ""


def test_본문이_제목과_같으면_두_번_쓰지_않는다():
    head, body = notification_text("휴가 신청", "휴가 신청")

    assert head == "휴가 신청"
    assert body == ""
