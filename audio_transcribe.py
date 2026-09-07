"""
음성을 글로 옮긴다. (`POST /upload`)

왜 이 파일이 새로 필요한가
    화면(포털·앱)은 예전부터 `/completion/upload` 로 녹음을 보내고 `transcript`
    를 받도록 되어 있었다. 그런데 이 서비스에는 그 경로가 **없었다.**
    그래서 마이크를 눌러 녹음까지 되고도 받아쓴 글이 늘 비어 있었고,
    404 가 조용히 삼켜져 "받아쓰지 못했습니다" 로만 보였다.

무엇을 쓰는가
    이미 쓰고 있는 LLM 프록시(LiteLLM)의 OpenAI 호환 `audio/transcriptions` 를
    그대로 부른다. 새 자격 증명을 만들지 않기 위해서다.

주의
    OpenRouter 는 음성 받아쓰기를 제공하지 않는다. 그 경우 502 와 함께 이유를
    돌려준다 — 조용히 빈 글을 주면 화면은 "말이 인식되지 않았다" 로 오해한다.
"""

from __future__ import annotations

import io
import os
from typing import Optional

from fastapi import File, HTTPException, UploadFile

from llm_factory import openai_compatible_client_config

# 받아쓰기 모델. 프록시에 등록된 이름으로 바꿀 수 있게 환경변수로 뺀다.
DEFAULT_STT_MODEL = "whisper-1"

# 너무 큰 녹음은 프록시가 어차피 거절한다. 먼저 걸러 이유를 분명히 말한다.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def stt_model() -> str:
    return (os.getenv("STT_MODEL") or "").strip() or DEFAULT_STT_MODEL


def audio_filename(upload_name: Optional[str], content_type: Optional[str]) -> str:
    """
    프록시에 넘길 파일 이름.

    확장자가 없으면 어떤 형식인지 알지 못해 거절당한다. 화면이 보내는 녹음은
    이름 없이 오는 경우가 많아(브라우저 Blob), content-type 에서 유추한다.
    """
    name = (upload_name or "").strip()
    if "." in name:
        return name

    subtype = (content_type or "").split("/")[-1].split(";")[0].strip().lower()
    known = {
        "mpeg": "mp3",
        "mp3": "mp3",
        "mp4": "mp4",
        "m4a": "m4a",
        "wav": "wav",
        "x-wav": "wav",
        "webm": "webm",
        "ogg": "ogg",
        "flac": "flac",
    }
    return f"audio.{known.get(subtype, 'wav')}"


def transcript_of(result) -> str:
    """
    응답에서 글만 꺼낸다.

    SDK 버전에 따라 객체이기도 하고 dict 이기도 하다. 한쪽만 보면 어떤 환경에서
    조용히 빈 글이 된다.
    """
    if result is None:
        return ""
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    return (text or "").strip()


async def transcribe_audio(audio: UploadFile = File(...)) -> dict:
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="녹음된 소리가 없습니다.")
    if len(raw) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="녹음이 너무 깁니다.")

    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover - 의존성 문제
        raise HTTPException(status_code=500, detail=f"음성 인식 준비 실패: {e}")

    config = openai_compatible_client_config()
    client = OpenAI(api_key=config["api_key"], base_url=config["openai_base_url"])

    payload = io.BytesIO(raw)
    payload.name = audio_filename(audio.filename, audio.content_type)

    try:
        result = client.audio.transcriptions.create(model=stt_model(), file=payload)
    except Exception as e:
        # 조용히 빈 글을 주지 않는다. 화면이 "말이 인식되지 않았다" 로 오해한다.
        raise HTTPException(status_code=502, detail=f"음성 인식에 실패했습니다: {e}")

    return {"transcript": transcript_of(result)}


def add_routes_to_app(app):
    app.add_api_route("/upload", transcribe_audio, methods=["POST"])
