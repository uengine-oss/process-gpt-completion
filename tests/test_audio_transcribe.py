"""
음성 받아쓰기 경로 검증.

화면은 예전부터 `/completion/upload` 로 녹음을 보냈는데 서버에는 그 경로가
없었다. 그래서 마이크는 눌려도 받아쓴 글이 늘 비어 있었다.
여기서 지키려는 것은 (1) 이름 없는 녹음도 형식을 알아보게 하는 것,
(2) 실패를 조용히 빈 글로 만들지 않는 것이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_transcribe import DEFAULT_STT_MODEL, audio_filename, stt_model, transcript_of  # noqa: E402


def test_name_with_extension_is_kept():
    assert audio_filename("memo.m4a", "audio/mp4") == "memo.m4a"


def test_blank_name_falls_back_to_content_type():
    """브라우저가 만든 녹음(Blob)은 이름 없이 온다. 확장자가 없으면 거절당한다."""
    assert audio_filename("", "audio/webm") == "audio.webm"
    assert audio_filename(None, "audio/mpeg") == "audio.mp3"
    assert audio_filename("blob", "audio/wav") == "audio.wav"


def test_unknown_type_defaults_to_wav():
    """알 수 없으면 가장 흔한 형식으로 시도한다. 아무것도 안 보내는 것보다 낫다."""
    assert audio_filename(None, None) == "audio.wav"
    assert audio_filename("recording", "application/octet-stream") == "audio.wav"


def test_transcript_from_object_and_dict():
    """SDK 버전에 따라 객체이기도 dict 이기도 하다. 한쪽만 보면 조용히 빈 글이 된다."""

    class R:
        text = "  안녕하세요  "

    assert transcript_of(R()) == "안녕하세요"
    assert transcript_of({"text": "반갑습니다"}) == "반갑습니다"
    assert transcript_of(None) == ""
    assert transcript_of({}) == ""


def test_model_is_configurable(monkeypatch):
    assert stt_model() == DEFAULT_STT_MODEL
    monkeypatch.setenv("STT_MODEL", "my-whisper")
    assert stt_model() == "my-whisper"
