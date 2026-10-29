from datetime import datetime

import pytest

from semantic_naming import fallback_semantic_name, generate_semantic_name, source_language
from task_deadline import ensure_minimum_task_due_date


class FakeResponse:
    content = '{"name":"구매 승인 · 노트북 10대 요청"}'


class FakeModel:
    async def ainvoke(self, _prompt):
        return FakeResponse()


class FailingModel:
    async def ainvoke(self, _prompt):
        raise RuntimeError("model unavailable")


class FakeChatResponse:
    content = '{"name":"이면도로 불법주차 민원 처리"}'


class FakeChatModel:
    async def ainvoke(self, _prompt):
        return FakeChatResponse()


@pytest.mark.asyncio
async def test_instance_name_uses_ai_rule():
    name = await generate_semantic_name(
        FakeModel(),
        kind="instance",
        source={"item": "노트북", "quantity": 10},
        process_name="구매 승인",
    )
    assert name == "구매 승인 · 노트북 10대 요청"


@pytest.mark.asyncio
async def test_naming_failure_has_deterministic_fallback():
    name = await generate_semantic_name(
        FailingModel(),
        kind="instance",
        source={"item": "노트북"},
        process_name="구매 승인",
    )
    assert name == fallback_semantic_name("instance", {"item": "노트북"}, "구매 승인")


@pytest.mark.asyncio
async def test_chat_name_summarizes_first_message():
    source = "이면도로 불법주차 문제를 해결하기 위한 민원 처리 프로세스를 만들어줘"
    name = await generate_semantic_name(FakeChatModel(), kind="chat", source=source)
    assert name == "이면도로 불법주차 민원 처리"
    assert name != source


@pytest.mark.asyncio
async def test_chat_fallback_removes_request_ending():
    source = "휴가 신청 프로세스를 만들어줘"
    name = await generate_semantic_name(FailingModel(), kind="chat", source=source)
    assert name == "휴가 신청 프로세스"
    assert name != source


def test_due_date_is_at_least_end_of_creation_day():
    task = {
        "start_date": "2026-07-14T09:30:00+09:00",
        "due_date": "2026-07-14T09:30:00+09:00",
    }
    ensure_minimum_task_due_date(task)
    due = datetime.fromisoformat(task["due_date"])
    assert due.date().isoformat() == "2026-07-14"
    assert due.hour == 23 and due.minute == 59


def test_later_due_date_is_preserved():
    task = {
        "start_date": "2026-07-14T09:30:00+09:00",
        "due_date": "2026-07-16T09:30:00+09:00",
    }
    ensure_minimum_task_due_date(task)
    assert task["due_date"] == "2026-07-16T09:30:00+09:00"


def test_instance_creation_day_wins_over_invalid_earlier_task_date():
    task = {
        "start_date": "2026-07-13T09:30:00+09:00",
        "due_date": "2026-07-13T09:30:00+09:00",
    }
    ensure_minimum_task_due_date(task, "2026-07-14T08:00:00+09:00")
    due = datetime.fromisoformat(task["due_date"])
    assert due.date().isoformat() == "2026-07-14"


class CapturingModel:
    def __init__(self):
        self.prompt = ""

    async def ainvoke(self, prompt):
        self.prompt = prompt
        return FakeChatResponse()


@pytest.mark.asyncio
async def test_chat_name_follows_message_language():
    assert source_language("Create a sales proposal process.\n[ExecutionPolicy]\n- 한국어 지시") == "en"
    assert source_language("영업 제안서 프로세스를 만들어 줘") == "ko"
    en, ko = CapturingModel(), CapturingModel()
    await generate_semantic_name(en, kind="chat", source="Create a sales proposal process for our customers")
    await generate_semantic_name(ko, kind="chat", source="영업 제안서 프로세스를 만들어 줘")
    assert "제목도 영어로" in en.prompt and "한국어 기준 30자" not in en.prompt
    assert "한국어 기준 30자" in ko.prompt and "제목도 영어로" not in ko.prompt
