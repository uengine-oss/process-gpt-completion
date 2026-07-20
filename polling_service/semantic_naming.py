import json
import re
from typing import Any


def _compact_text(value: Any, limit: int = 5000) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value or "")
    return re.sub(r"\s+", " ", text).strip()[:limit]


def sanitize_generated_name(value: Any, *, max_length: int = 50) -> str:
    text = _compact_text(value, max_length * 3)
    text = re.sub(r"^[\s\"'`#*-]+|[\s\"'`.,;:!?]+$", "", text)
    return text[:max_length].strip()


def fallback_semantic_name(kind: str, source: Any, process_name: str = "") -> str:
    summary = _compact_text(source, 32)
    if not summary:
        summary = "새 요청" if kind == "chat" else "실행"
    if kind == "instance":
        base = sanitize_generated_name(process_name, max_length=24) or "프로세스"
        return sanitize_generated_name(f"{base} · {summary}", max_length=50)
    summary = re.sub(
        r"\s*(?:만들어|생성해|작성해|분석해|정리해|알려|도와)?\s*(?:줘|주세요|주십시오)[.!?]*$",
        "",
        summary,
    ).strip()
    summary = re.sub(r"(?:을|를|은|는)$", "", summary).strip()
    return sanitize_generated_name(summary, max_length=50) or "새 대화"


def _response_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


async def generate_semantic_name(model: Any, *, kind: str, source: Any, process_name: str = "") -> str:
    """Generate a short name, with a deterministic fallback when AI is unavailable."""
    fallback = fallback_semantic_name(kind, source, process_name)
    source_text = _compact_text(source)
    if not source_text:
        return fallback

    if kind == "instance":
        rule = (
            f'프로세스명 "{sanitize_generated_name(process_name, max_length=80)}"과 최초 폼 입력의 핵심 대상을 조합해 '
            '"프로세스명 · 입력 요약" 형식으로 작성하세요.'
        )
    else:
        rule = "사용자의 첫 요청에서 업무 목적과 핵심 대상을 직접 추출해 제목으로 요약하세요."

    prompt = f"""당신은 업무 목록의 짧은 제목을 만드는 도우미입니다.
{rule}
- 입력 문장은 아래 '첫 메시지'에 있으며, 반드시 그 내용을 직접 읽고 분석합니다.
- 한국어 기준 30자 이내, 2~6개 단어의 구체적인 명사구로 작성합니다.
- 인사말, 설명, 따옴표, 번호, 마침표를 넣지 않습니다.
- 입력에 없는 사실을 만들지 않습니다.
- '목적 미상', '대상 미상', '새 요청', '사용자 요청' 같은 일반 문구는 금지합니다.
- 요청형 어미(해줘, 만들어줘, 알려줘)는 제목에서 제거하거나 업무 명사로 바꿉니다.
- 예: '휴가 신청 프로세스를 만들어줘' → {{"name":"휴가 신청 프로세스 설계"}}
- 예: '이면도로 불법주차 문제를 해결하기 위한 민원 처리 프로세스를 만들어줘' → {{"name":"이면도로 불법주차 민원 처리"}}
- 반드시 {{"name":"제목"}} JSON 하나만 반환합니다.

첫 메시지:
{source_text}
"""
    try:
        response = await model.ainvoke(prompt)
        raw = _response_content(response).strip()
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        match = re.search(r"\{[\s\S]*\}", fenced)
        payload = json.loads(match.group(0) if match else fenced)
        generated = sanitize_generated_name(payload.get("name"), max_length=50)
        return generated or fallback
    except Exception:
        return fallback
