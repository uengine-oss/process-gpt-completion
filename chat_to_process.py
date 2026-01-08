from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from dotenv import load_dotenv
import os
import json
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from llm_factory import create_llm
from database import (
    fetch_chat_history,
    upsert_process_definition,
    subdomain_var,
    supabase_client_var
)
from process_definition import ProcessDefinition
from datetime import datetime
import pytz

if os.getenv("ENV") != "production":
    load_dotenv(override=True)

# LLM 객체 생성
llm = create_llm(model="gpt-4.1", streaming=False)

# 1단계: 채팅 이력에서 프로세스 생성 가능 여부 검사 프롬프트
process_detection_prompt = PromptTemplate.from_template(
    """당신은 채팅방의 대화 이력을 분석하여 프로세스로 만들만한 비즈니스 워크플로우가 있는지 판단하는 AI입니다.

**중요: 이 작업은 제공된 채팅 이력만을 분석해야 합니다. 기존 메모리나 캐시된 정보를 사용하지 말고, 오직 아래 제공된 채팅 이력만을 기반으로 판단하세요.**

## 채팅 이력:
{chat_history}

## 판단 기준:
1. **순차적인 작업 흐름이 있는가?**
   - 여러 단계로 이루어진 작업이 논의되었는가?
   - 각 단계가 명확하게 구분되는가?
   - 단계 간 의존성이나 순서가 있는가?

2. **역할(담당자)이 구분되는가?**
   - 서로 다른 사람이나 역할이 각 단계를 담당하는가?
   - 역할 간 협업이나 승인 프로세스가 있는가?

3. **반복 가능한 워크플로우인가?**
   - 일회성이 아닌 반복적으로 수행되는 업무인가?
   - 표준화할 수 있는 프로세스인가?

4. **프로세스로 만들 가치가 있는가?**
   - 단순한 대화가 아닌 실제 업무 프로세스인가?
   - 프로세스화하면 효율성이 향상되는가?

## 프로세스로 만들만한 내용이 있는 경우:
- 여러 단계의 작업 흐름
- 역할 분담이 명확함
- 승인, 검토, 확인 등의 단계
- 반복 가능한 업무 프로세스

## 프로세스로 만들지 말아야 할 경우:
- 단순한 질문과 답변
- 일회성 대화
- 프로세스가 아닌 일반적인 정보 교환
- 이미 완료된 일회성 작업

## 응답 형식 (JSON만 반환):
{{
    "can_create_process": true 또는 false,
    "confidence": 0.0-1.0 사이의 값,
    "reason": "판단 이유를 간단히 설명",
    "suggested_process_name": "제안하는 프로세스 이름 (can_create_process가 true인 경우만)",
    "suggested_process_description": "제안하는 프로세스 설명 (can_create_process가 true인 경우만)"
}}

**중요: 확실하지 않으면 can_create_process를 false로 설정하세요.**

JSON 형식으로만 응답하세요."""
)

detection_chain = (
    RunnablePassthrough() |
    process_detection_prompt |
    llm |
    StrOutputParser()
)


# BPMN 생성 로직은 프론트엔드의 BPMNXmlGenerator.vue에서 처리
# generate_bpmn_from_process_definition 함수는 더 이상 사용되지 않으므로 삭제됨


def format_chat_history_for_analysis(chat_history: List) -> str:
    """채팅 이력을 분석용 텍스트로 포맷팅"""
    if not chat_history:
        return "채팅 이력이 없습니다."
    
    formatted = []
    for item in chat_history:
        if not item.messages:
            continue
        msg = item.messages
        name = getattr(msg, 'name', 'Unknown')
        content = getattr(msg, 'content', '')
        role = getattr(msg, 'role', 'user')
        
        formatted.append(f"[{role}] {name}: {content}")
    
    return "\n".join(formatted)


async def analyze_chat_for_process(chat_room_id: str) -> Dict[str, Any]:
    """채팅 이력을 분석하여 프로세스 생성 가능 여부 검사"""
    try:
        chat_history = fetch_chat_history(chat_room_id)
        if not chat_history or len(chat_history) < 3:
            return {
                "can_create_process": False,
                "reason": "채팅 이력이 충분하지 않습니다. (최소 3개 이상의 메시지 필요)"
            }
        
        formatted_history = format_chat_history_for_analysis(chat_history)
        
        # 메모리 캐시를 우회하기 위해 프롬프트에 타임스탬프 추가
        timestamp_context = f"\n\n[분석 시점: {datetime.now(pytz.UTC).isoformat()}]\n이 분석은 위의 채팅 이력만을 기반으로 수행되어야 합니다."
        
        result = await detection_chain.ainvoke({
            "chat_history": formatted_history + timestamp_context
        })
        
        # JSON 파싱 시도
        try:
            json_str = result.strip()
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "").replace("```", "").strip()
            elif json_str.startswith("```"):
                json_str = json_str.replace("```", "").strip()
            
            decision = json.loads(json_str)
            return decision
        except Exception as e:
            print(f"JSON 파싱 실패: {str(e)}")
            print(f"원본 응답: {result}")
            return {
                "can_create_process": False,
                "reason": f"분석 결과 파싱 실패: {str(e)}"
            }
    except Exception as e:
        print(f"프로세스 분석 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "can_create_process": False,
            "reason": f"분석 중 오류 발생: {str(e)}"
        }


class GetChatHistoryRequest(BaseModel):
    chat_room_id: str


async def get_chat_history_endpoint(request: GetChatHistoryRequest):
    """채팅 이력 반환 엔드포인트"""
    try:
        chat_history = fetch_chat_history(request.chat_room_id)
        formatted_history = format_chat_history_for_analysis(chat_history)
        
        return {
            "chat_history": formatted_history,
            "raw_history": [
                {
                    "name": getattr(item.messages, 'name', 'Unknown'),
                    "content": getattr(item.messages, 'content', ''),
                    "role": getattr(item.messages, 'role', 'user')
                }
                for item in chat_history if item.messages
            ]
        }
    except Exception as e:
        print(f"Error in get_chat_history_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class SaveProcessRequest(BaseModel):
    process_definition: Dict[str, Any]
    bpmn_xml: str


async def save_process_endpoint(request: SaveProcessRequest):
    """프로세스 저장 엔드포인트"""
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured")
        
        subdomain = subdomain_var.get()
        
        # ProcessDefinition 객체로 변환하여 검증
        process_def = ProcessDefinition(**request.process_definition)
        
        # 데이터베이스에 저장할 형식으로 변환
        definition_data = {
            "id": process_def.processDefinitionId,
            "definition": request.process_definition,
            "bpmn": request.bpmn_xml,
            "tenant_id": subdomain
        }
        
        # upsert_process_definition 사용
        upsert_process_definition(definition_data, subdomain)
        
        return {
            "success": True,
            "process_definition": request.process_definition,
            "message": "프로세스가 성공적으로 저장되었습니다."
        }
    except Exception as e:
        print(f"프로세스 저장 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"프로세스 저장 실패: {str(e)}"
        )


class AnalyzeChatRequest(BaseModel):
    chat_room_id: str


async def analyze_chat_endpoint(request: AnalyzeChatRequest):
    """채팅 이력 분석 엔드포인트"""
    try:
        result = await analyze_chat_for_process(request.chat_room_id)
        return result
    except Exception as e:
        print(f"Error in analyze_chat_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def add_routes_to_app(app: FastAPI):
    """라우트 추가"""
    app.add_api_route(
        "/langchain-chat/process/analyze",
        analyze_chat_endpoint,
        methods=["POST"]
    )
    app.add_api_route(
        "/langchain-chat/process/chat-history",
        get_chat_history_endpoint,
        methods=["POST"]
    )
    app.add_api_route(
        "/langchain-chat/process/save",
        save_process_endpoint,
        methods=["POST"]
    )

