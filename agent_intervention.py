from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import os
import json
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from llm_factory import create_llm
from database import (
    fetch_chat_history, 
    upsert_chat_message, 
    fetch_user_info,
    fetch_assignee_info,
    subdomain_var,
    supabase_client_var
)
import uuid
from datetime import datetime
import pytz
# agent_chat은 직접 호출하지 않고 mem0_agent_client를 사용

if os.getenv("ENV") != "production":
    load_dotenv(override=True)

# LLM 객체 생성
llm = create_llm(model="gpt-4.1", streaming=False)

# 1단계: 개입 여부 판단 프롬프트
intervention_decision_prompt = PromptTemplate.from_template(
    """당신은 채팅방에서 사용자 메시지를 분석하고, 에이전트의 개입이 필요한지 판단하는 AI입니다.
**중요: 확실하지 않으면 개입하지 마세요. 보수적으로 판단하세요.**

## 사용자 메시지:
{user_message}

## 최근 대화 히스토리 (최근 5개):
{recent_history}

## 참여 중인 에이전트 목록:
{agents_info}

## 판단 기준:
1. **사용자가 도움을 요청하는 의도가 있는가?**
   - 질문 형태의 요청 (어떻게, 무엇, 왜 등으로 시작하는 질문)
   - 설명/해석 요청 (이해하지 못하는 내용에 대한 질문)
   - 작업 수행 요청 (작성, 번역, 분석 등)
   - 단순 정보 전달이나 확인은 제외

2. **에이전트의 전문 분야와 사용자 요청이 관련이 있는가?**
   - 사용자 메시지의 의도와 에이전트의 기능 설명을 비교하여 관련성 판단
   - 에이전트 목록에서 제공된 기능 설명을 참고하여 매칭
   - 관련성이 명확하지 않으면 개입하지 않음

3. **에이전트 개입이 실제로 도움이 되는가?**
   - 사용자가 해결하기 어려운 전문적인 도움이 필요한가?
   - 에이전트가 제공할 수 있는 전문 지식이나 기능이 필요한가?
   - 일반적인 대화나 인사는 제외

4. **최근에 이미 같은 에이전트가 개입했는가?**
   - 최근 3개 메시지 내에 같은 에이전트가 개입했다면 중복 방지를 위해 개입하지 않음
   - 단, 다른 에이전트가 개입했어도 현재 요청과 관련된 다른 에이전트는 개입 가능

## 개입하지 말아야 할 경우:
- 단순 정보 전달 (사실 나열, 상태 보고 등)
- 일반적인 대화나 인사
- 에이전트의 전문 분야와 전혀 관련 없는 내용
- 사용자가 이미 해결한 내용에 대한 단순 확인

## 개입해야 할 경우:
- 사용자가 명시적으로 도움을 요청하고, 에이전트의 기능과 관련이 있는 경우
- 사용자가 이해하지 못하는 내용이 있고, 에이전트가 설명할 수 있는 경우
- 사용자가 특정 작업을 요청하고, 에이전트가 그 작업을 수행할 수 있는 경우

## 응답 형식 (JSON만 반환):
{{
    "should_intervene": true 또는 false,
    "reason": "개입 여부 판단 이유를 간단히 설명. 개입하는 경우 어떤 에이전트가 왜 필요한지 명확히 설명"
}}

**중요: 확실하지 않으면 should_intervene을 false로 설정하세요.**

JSON 형식으로만 응답하세요."""
)

# 2단계: 에이전트 선택 프롬프트
agent_selection_prompt = PromptTemplate.from_template(
    """당신은 채팅방에서 사용자 메시지를 분석하고, 가장 적절한 에이전트를 선택하는 AI입니다.

## 사용자 메시지:
{user_message}

## 최근 대화 히스토리:
{recent_history}

## 참여 중인 에이전트 목록:
{agents_info}

## 선택 기준:
1. 사용자 메시지의 내용과 가장 관련이 높은 에이전트를 선택
2. 에이전트의 기능과 역할을 고려
3. 대화 맥락을 고려하여 가장 도움이 되는 에이전트 선택
4. 명확하게 선택할 수 없으면 "default" 선택

## 응답 형식 (JSON만 반환):
{{
    "selected_agent_id": "에이전트 ID 또는 'default'",
    "confidence": 0.0-1.0 사이의 값,
    "reason": "에이전트 선택 이유를 간단히 설명"
}}

JSON 형식으로만 응답하세요."""
)

# 기본 LLM 응답 프롬프트 (에이전트를 선택할 수 없을 때)
default_llm_prompt = PromptTemplate.from_template(
    """당신은 채팅방에서 사용자들의 대화를 돕는 AI 어시스턴트입니다.

## 최근 대화 히스토리:
{recent_history}

## 사용자 메시지:
{user_message}

사용자의 메시지에 대해 도움이 되는 응답을 제공해주세요. 자연스럽고 친절하게 답변하세요."""
)

intervention_chain = (
    RunnablePassthrough() |
    intervention_decision_prompt |
    llm |
    StrOutputParser()
)

agent_selection_chain = (
    RunnablePassthrough() |
    agent_selection_prompt |
    llm |
    StrOutputParser()
)

default_llm_chain = (
    RunnablePassthrough() |
    default_llm_prompt |
    llm |
    StrOutputParser()
)


def get_chat_room_participants(chat_room_id: str) -> Dict[str, List[Dict]]:
    """
    채팅방 참여자 목록을 조회하고 사용자와 에이전트를 분리
    chat_rooms 테이블의 participants 필드에서 직접 조회
    
    Returns:
        {
            "users": [{"id": "...", "email": "...", "username": "..."}, ...],
            "agents": [{"id": "...", "email": "...", "username": "...", "agent_type": "..."}, ...]
        }
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            raise Exception("Supabase client is not configured")
        
        subdomain = subdomain_var.get()
        
        # chat_rooms 테이블에서 participants 조회
        response = supabase.table("chat_rooms").select("participants").eq('id', chat_room_id).eq('tenant_id', subdomain).execute()
        
        users = []
        agents = []
        seen_ids = set()  # 중복 제거용
        
        if response.data and len(response.data) > 0:
            participants_list = response.data[0].get('participants', [])
            
            for participant in participants_list:
                participant_id = participant.get('id')
                if not participant_id or participant_id in seen_ids:
                    continue
                seen_ids.add(participant_id)
                
                email = participant.get('email')
                username = participant.get('username', '')
                
                # email이 null이거나 없으면 에이전트로 판단
                if not email:
                    # 에이전트 정보 조회 시도
                    try:
                        # id로 사용자 정보 조회
                        user_info = fetch_user_info(participant_id)
                        if user_info.get("is_agent") == True:
                            agents.append({
                                "id": participant_id,
                                "email": email,
                                "username": username or user_info.get("username", "에이전트"),
                                "agent_type": user_info.get("agent_type", "agent")
                            })
                        else:
                            users.append({
                                "id": participant_id,
                                "email": email or participant_id,
                                "username": username or user_info.get("username", "사용자")
                            })
                    except:
                        # 사용자 정보를 찾을 수 없으면 username으로 판단
                        # "도우미", "에이전트" 등의 키워드가 있으면 에이전트로 판단
                        if any(keyword in username for keyword in ["도우미", "에이전트", "Agent", "Assistant"]):
                            agents.append({
                                "id": participant_id,
                                "email": email,
                                "username": username,
                                "agent_type": "agent"
                            })
                        else:
                            users.append({
                                "id": participant_id,
                                "email": email or participant_id,
                                "username": username or "사용자"
                            })
                else:
                    # email이 있으면 사용자 정보 조회
                    try:
                        user_info = fetch_user_info(email)
                        if user_info.get("is_agent") == True:
                            agents.append({
                                "id": participant_id,
                                "email": email,
                                "username": username or user_info.get("username", "에이전트"),
                                "agent_type": user_info.get("agent_type", "agent")
                            })
                        else:
                            users.append({
                                "id": participant_id,
                                "email": email,
                                "username": username or user_info.get("username", "사용자")
                            })
                    except:
                        # 사용자 정보를 찾을 수 없으면 일반 사용자로 처리
                        users.append({
                            "id": participant_id,
                            "email": email,
                            "username": username or "사용자"
                        })
        
        return {
            "users": users,
            "agents": agents
        }
    except Exception as e:
        print(f"Error getting chat room participants: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"users": [], "agents": []}


def should_activate_intervention(users: List[Dict], agents: List[Dict]) -> bool:
    """
    에이전트 개입 활성화 조건 확인
    - 사람 1명 + 에이전트 2명 이상
    - 또는 사람 2명 이상 + 에이전트 1명 이상
    """
    user_count = len(users)
    agent_count = len(agents)
    
    condition1 = user_count >= 1 and agent_count >= 2
    condition2 = user_count >= 2 and agent_count >= 1
    
    return condition1 or condition2


def format_recent_history(chat_history: List, limit: int = 5) -> str:
    """최근 대화 히스토리를 포맷팅"""
    if not chat_history:
        return "대화 히스토리가 없습니다."
    
    recent = chat_history[-limit:] if len(chat_history) > limit else chat_history
    formatted = []
    
    for item in recent:
        if not item.messages:
            continue
        msg = item.messages
        name = getattr(msg, 'name', 'Unknown')
        content = getattr(msg, 'content', '')
        role = getattr(msg, 'role', 'user')
        
        formatted.append(f"[{role}] {name}: {content}")
    
    return "\n".join(formatted)


def format_agents_info(agents: List[Dict]) -> str:
    """에이전트 정보를 프롬프트용으로 포맷팅"""
    if not agents:
        return "참여 중인 에이전트가 없습니다."
    
    formatted = []
    for agent in agents:
        agent_id = agent.get("id", "")
        username = agent.get("username", "")
        agent_type = agent.get("agent_type", "agent")
        
        # 에이전트 타입에 따른 기능 설명
        # username에 키워드가 있으면 타입을 보완
        if "이메일" in username or "메일" in username or "email" in username.lower():
            if agent_type == "agent" or not agent_type:
                agent_type = "email"
        elif "번역" in username or "translation" in username.lower():
            if agent_type == "agent" or not agent_type:
                agent_type = "translation"
        elif "비즈니스" in username or "business" in username.lower():
            if agent_type == "agent" or not agent_type:
                agent_type = "business_term"
        
        capabilities = get_agent_capabilities(agent_type, username)
        
        formatted.append(f"- ID: {agent_id}")
        formatted.append(f"  이름: {username}")
        formatted.append(f"  타입: {agent_type}")
        formatted.append(f"  기능: {capabilities}")
        formatted.append("")
    
    return "\n".join(formatted)


def save_intervention_log(
    message_uuid: str,
    chat_room_id: str,
    user_id: str,
    user_message: str,
    context_info: Dict[str, Any],
    should_intervene: bool = None,
    intervention_reason: str = None,
    decision_confidence: float = None,
    selected_agent_id: str = None,
    selected_agent_name: str = None,
    agent_selection_reason: str = None,
    agent_selection_confidence: float = None,
    agent_response_content: str = None,
    agent_response_type: str = None,
    status: str = "checking"
) -> None:
    """
    개입 이력을 agent_intervention_logs 테이블에 저장/업데이트
    ML 학습 데이터 수집을 위한 함수
    """
    try:
        supabase = supabase_client_var.get()
        if supabase is None:
            return
        
        subdomain = subdomain_var.get()
        
        # 기존 로그 확인
        existing_log = supabase.table("agent_intervention_logs").select("*").eq('message_uuid', message_uuid).eq('tenant_id', subdomain).execute()
        
        current_time = datetime.now(pytz.timezone('Asia/Seoul')).isoformat()
        
        log_data = {
            "tenant_id": subdomain,
            "chat_room_id": chat_room_id,
            "message_uuid": message_uuid,
            "user_id": user_id,
            "user_message": user_message,
            "user_message_length": len(user_message) if user_message else 0,
            "context_info": context_info,
            "status": status,
            "updated_at": current_time
        }
        
        # 개입 결정 정보 (should_intervene이 None이면 저장하지 않음)
        if should_intervene is not None:
            log_data["should_intervene"] = should_intervene
        if intervention_reason:
            log_data["intervention_reason"] = intervention_reason
        if decision_confidence is not None:
            log_data["decision_confidence"] = decision_confidence
        
        # 에이전트 선택 정보
        if selected_agent_id:
            log_data["selected_agent_id"] = selected_agent_id
        if selected_agent_name:
            log_data["selected_agent_name"] = selected_agent_name
        if agent_selection_reason:
            log_data["agent_selection_reason"] = agent_selection_reason
        if agent_selection_confidence is not None:
            log_data["agent_selection_confidence"] = agent_selection_confidence
        
        # 에이전트 응답 정보
        if agent_response_content:
            log_data["agent_response_content"] = agent_response_content
            log_data["agent_response_length"] = len(agent_response_content)
        if agent_response_type:
            log_data["agent_response_type"] = agent_response_type
        
        # 상태가 completed면 completed_at 업데이트
        if status == "completed":
            log_data["completed_at"] = current_time
        
        if existing_log.data and len(existing_log.data) > 0:
            # 기존 로그 업데이트
            supabase.table("agent_intervention_logs").update(log_data).eq('id', existing_log.data[0]['id']).execute()
        else:
            # 새 로그 생성
            log_data["id"] = str(uuid.uuid4())
            log_data["created_at"] = current_time
            supabase.table("agent_intervention_logs").insert(log_data).execute()
    except Exception as e:
        print(f" 개입 로그 저장 실패 (무시): {str(e)}")


def get_agent_capabilities(agent_type: str, username: str = "") -> str:
    """에이전트 타입과 이름에 따른 기능 설명"""
    # 기본 타입별 기능 설명
    capabilities_map = {
        "translation": "다국어 번역 및 언어 관련 도움",
        "business_term": "비즈니스 용어 설명 및 정의",
        "email": "이메일/메일 작성 및 답장 작성 도움",
        "email_writing": "이메일/메일 작성 및 답장 작성 도움",
        "default": "일반적인 질문 답변 및 대화 지원"
    }
    
    # agent_type이 매핑되어 있으면 사용
    if agent_type in capabilities_map:
        return capabilities_map[agent_type]
    
    # agent_type이 없거나 매핑되지 않은 경우, username에서 추론
    if username:
        username_lower = username.lower()
        if any(keyword in username_lower for keyword in ["번역", "translation", "translate"]):
            return "다국어 번역 및 언어 관련 도움"
        elif any(keyword in username_lower for keyword in ["비즈니스", "business", "용어", "term"]):
            return "비즈니스 용어 설명 및 정의"
        elif any(keyword in username_lower for keyword in ["이메일", "메일", "email", "mail"]):
            return "이메일/메일 작성 및 답장 작성 도움"
    
    # 기본값: 에이전트의 이름과 타입을 기반으로 일반적인 도움 제공
    return f"에이전트의 전문 분야에 따른 도움 제공 (타입: {agent_type})"


async def check_intervention_needed(
    user_message: str,
    chat_room_id: str,
    agents: List[Dict] = None
) -> Dict[str, Any]:
    """1단계: 개입 여부 판단"""
    try:
        chat_history = fetch_chat_history(chat_room_id)
        recent_history = format_recent_history(chat_history, limit=5)
        
        # 에이전트 정보 포맷팅
        agents_info = ""
        if agents and len(agents) > 0:
            agents_info = format_agents_info(agents)
        else:
            agents_info = "참여 중인 에이전트가 없습니다."
        
        result = await intervention_chain.ainvoke({
            "user_message": user_message,
            "recent_history": recent_history,
            "agents_info": agents_info
        })
        
        # JSON 파싱 시도
        try:
            # JSON 부분만 추출
            json_str = result.strip()
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "").replace("```", "").strip()
            elif json_str.startswith("```"):
                json_str = json_str.replace("```", "").strip()
            
            decision = json.loads(json_str)
            return decision
        except:
            # JSON 파싱 실패 시 기본값
            return {
                "should_intervene": False,
                "reason": "JSON 파싱 실패"
            }
    except Exception as e:
        print(f"Error checking intervention: {str(e)}")
        return {
            "should_intervene": False,
            "reason": f"에러 발생: {str(e)}"
        }


async def select_agent(
    user_message: str,
    chat_room_id: str,
    agents: List[Dict]
) -> Dict[str, Any]:
    """2단계: 적절한 에이전트 선택"""
    try:
        chat_history = fetch_chat_history(chat_room_id)
        recent_history = format_recent_history(chat_history, limit=10)
        agents_info = format_agents_info(agents)
        
        result = await agent_selection_chain.ainvoke({
            "user_message": user_message,
            "recent_history": recent_history,
            "agents_info": agents_info
        })
        
        # JSON 파싱 시도
        try:
            json_str = result.strip()
            if json_str.startswith("```json"):
                json_str = json_str.replace("```json", "").replace("```", "").strip()
            elif json_str.startswith("```"):
                json_str = json_str.replace("```", "").strip()
            
            selection = json.loads(json_str)
            return selection
        except:
            return {
                "selected_agent_id": "default",
                "confidence": 0.0,
                "reason": "JSON 파싱 실패"
            }
    except Exception as e:
        print(f"Error selecting agent: {str(e)}")
        return {
            "selected_agent_id": "default",
            "confidence": 0.0,
            "reason": f"에러 발생: {str(e)}"
        }


async def get_default_llm_response(
    user_message: str,
    chat_room_id: str
) -> str:
    """기본 LLM 응답 생성 (에이전트를 선택할 수 없을 때)"""
    try:
        chat_history = fetch_chat_history(chat_room_id)
        recent_history = format_recent_history(chat_history, limit=10)
        
        response = await default_llm_chain.ainvoke({
            "user_message": user_message,
            "recent_history": recent_history
        })
        
        return response.strip()
    except Exception as e:
        print(f"Error getting default LLM response: {str(e)}")
        return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."


async def process_user_message_with_intervention(
    text: str,
    chat_room_id: str,
    user_id: str
) -> Dict[str, Any]:
    """
    사용자 메시지 처리 및 에이전트 개입 로직
    
    Returns:
        {
            "message_saved": bool,
            "intervention": {
                "should_intervene": bool,
                "selected_agent_id": str | None,
                "agent_response": Dict | None,
                "reason": str,
                "debug_info": Dict (개발 모드에서만)
            }
        }
    """
    try:
        # 디버그 정보 수집
        debug_info = {
            "user_message": text,
            "chat_room_id": chat_room_id,
            "user_id": user_id
        }
        
        # 1. 메시지 저장 (개입 정보는 나중에 업데이트)
        # 개입 상태를 "checking"으로 설정하여 웹단에서 로딩 상태를 표시할 수 있도록 함
        message_data = {
            "email": user_id,
            "command": text,
            "jsonData": {
                "intervention": {
                    "status": "checking",  # checking -> intervening/not_intervening -> completed
                    "should_intervene": None
                }
            }
        }
        upsert_chat_message(chat_room_id, message_data, is_system=False, is_agent=False)
        
        # 저장된 메시지의 UUID 조회 (로그 저장을 위해)
        message_uuid = None
        try:
            from database import supabase_client_var, subdomain_var
            supabase = supabase_client_var.get()
            subdomain = subdomain_var.get()
            response = supabase.table("chats").select("*").eq('id', chat_room_id).eq('tenant_id', subdomain).execute()
            if response.data:
                sorted_messages = sorted(
                    response.data,
                    key=lambda x: x.get('messages', {}).get('timeStamp', 0) if isinstance(x.get('messages'), dict) else 0,
                    reverse=True
                )
                if sorted_messages:
                    latest = sorted_messages[0]
                    if latest.get('messages') and latest['messages'].get('email') == user_id:
                        message_uuid = latest.get('uuid')
        except Exception as e:
            print(f" 메시지 UUID 조회 실패 (무시): {str(e)}")
        
        # 2. 채팅방 참여자 조회
        participants = get_chat_room_participants(chat_room_id)
        users = participants["users"]
        agents = participants["agents"]
        
        debug_info["participants"] = {
            "user_count": len(users),
            "agent_count": len(agents),
            "users": [{"id": u.get("id"), "email": u.get("email")} for u in users],
            "agents": [{"id": a.get("id"), "email": a.get("email"), "type": a.get("agent_type")} for a in agents]
        }
        
        # 컨텍스트 정보 준비 (ML 학습용)
        chat_history = fetch_chat_history(chat_room_id)
        recent_history = format_recent_history(chat_history, limit=5)
        context_info = {
            "user_count": len(users),
            "agent_count": len(agents),
            "available_agents": [{"id": a.get("id"), "name": a.get("username"), "type": a.get("agent_type")} for a in agents],
            "recent_history_length": len(recent_history)
        }
        
        # 초기 로그 생성 (checking 상태)
        if message_uuid:
            save_intervention_log(
                message_uuid=message_uuid,
                chat_room_id=chat_room_id,
                user_id=user_id,
                user_message=text,
                context_info=context_info,
                status="checking"
            )
        
        # 디버그 로그 출력
        print(f"\n{'='*60}")
        print(f" 에이전트 개입 디버그 정보")
        print(f"{'='*60}")
        print(f" 사용자 메시지: {text}")
        print(f" 채팅방 ID: {chat_room_id}")
        print(f" 사용자 ID: {user_id}")
        print(f" 참여자 - 사용자: {len(users)}명, 에이전트: {len(agents)}명")
        if agents:
            print(f" 참여 에이전트:")
            for agent in agents:
                print(f"   - {agent.get('username', 'Unknown')} ({agent.get('id')}) - 타입: {agent.get('agent_type', 'unknown')}")
        
        # 3. 개입 활성화 조건 확인
        condition_met = should_activate_intervention(users, agents)
        debug_info["condition_met"] = condition_met
        print(f" 개입 활성화 조건: {'충족' if condition_met else '미충족'}")
        
        if not condition_met:
            return {
                "message_saved": True,
                "intervention": {
                    "should_intervene": False,
                    "reason": f"개입 활성화 조건 미충족 (사용자: {len(users)}명, 에이전트: {len(agents)}명)",
                    "debug_info": debug_info
                }
            }
        
        # 4. 개입 여부 판단 (에이전트 정보 포함)
        print(f" 1단계: 개입 여부 판단 중...")
        if agents:
            print(f"   참여 에이전트 정보를 포함하여 판단합니다.")
        intervention_decision = await check_intervention_needed(text, chat_room_id, agents)
        debug_info["intervention_decision"] = intervention_decision
        print(f"   결과: {'개입 필요' if intervention_decision.get('should_intervene') else '개입 불필요'}")
        print(f"   이유: {intervention_decision.get('reason', '없음')}")
        
        if not intervention_decision.get("should_intervene", False):
            # 개입하지 않은 경우에도 정보 저장
            try:
                from database import supabase_client_var, subdomain_var
                supabase = supabase_client_var.get()
                subdomain = subdomain_var.get()
                
                # 정렬 없이 모든 메시지를 가져온 후 Python에서 정렬
                response = supabase.table("chats").select("*").eq('id', chat_room_id).eq('tenant_id', subdomain).execute()
                
                if response.data and len(response.data) > 0:
                    # messages.timeStamp를 기준으로 정렬 (timeStamp가 있는 경우)
                    sorted_messages = sorted(
                        response.data,
                        key=lambda x: x.get('messages', {}).get('timeStamp', 0) if isinstance(x.get('messages'), dict) else 0,
                        reverse=True
                    )
                    latest_message = sorted_messages[0] if sorted_messages else response.data[0]
                    if latest_message.get('messages') and latest_message['messages'].get('email') == user_id:
                        intervention_info = {
                            "status": "not_intervening",
                            "should_intervene": False,
                            "reason": intervention_decision.get("reason", "개입 불필요")
                        }
                        
                        existing_json = latest_message.get('messages', {}).get('jsonContent')
                        if existing_json and isinstance(existing_json, dict):
                            existing_json['intervention'] = intervention_info
                        else:
                            existing_json = {"intervention": intervention_info}
                        
                        supabase.table("chats").update({
                            "messages": {
                                **latest_message['messages'],
                                "jsonContent": existing_json
                            }
                        }).eq('uuid', latest_message['uuid']).execute()
                        
                        # 개입 로그 업데이트 (개입하지 않음)
                        if message_uuid:
                            save_intervention_log(
                                message_uuid=message_uuid,
                                chat_room_id=chat_room_id,
                                user_id=user_id,
                                user_message=text,
                                context_info=context_info,
                                should_intervene=False,
                                intervention_reason=intervention_decision.get("reason", "개입 불필요"),
                                status="not_intervening"
                            )
            except Exception as e:
                print(f" 개입 정보 업데이트 실패 (무시): {str(e)}")
            
            print(f" 개입하지 않음")
            print(f"{'='*60}\n")
            return {
                "message_saved": True,
                "intervention": {
                    "should_intervene": False,
                    "reason": intervention_decision.get("reason", "개입 불필요"),
                    "debug_info": debug_info
                }
            }
        
        # 5. 에이전트 선택
        print(f" 2단계: 에이전트 선택 중...")
        agent_selection = await select_agent(text, chat_room_id, agents)
        selected_agent_id = agent_selection.get("selected_agent_id", "default")
        debug_info["agent_selection"] = agent_selection
        print(f"   선택된 에이전트: {selected_agent_id}")
        print(f"   신뢰도: {agent_selection.get('confidence', 0.0):.2f}")
        print(f"   이유: {agent_selection.get('reason', '없음')}")
        
        # 개입이 결정되었으므로 사용자 메시지의 jsonContent 업데이트 (should_intervene: true)
        try:
            from database import supabase_client_var, subdomain_var
            supabase = supabase_client_var.get()
            subdomain = subdomain_var.get()
            
            # 정렬 없이 모든 메시지를 가져온 후 Python에서 정렬
            response = supabase.table("chats").select("*").eq('id', chat_room_id).eq('tenant_id', subdomain).execute()
            
            if response.data and len(response.data) > 0:
                # messages.timeStamp를 기준으로 정렬
                sorted_messages = sorted(
                    response.data,
                    key=lambda x: x.get('messages', {}).get('timeStamp', 0) if isinstance(x.get('messages'), dict) else 0,
                    reverse=True
                )
                latest_message = sorted_messages[0] if sorted_messages else response.data[0]
                if latest_message.get('messages') and latest_message['messages'].get('email') == user_id:
                    intervention_info = {
                        "status": "intervening",  # 에이전트 응답 대기 중
                        "should_intervene": True,  # 개입 결정됨
                        "reason": intervention_decision.get("reason", "개입 필요"),
                        "selected_agent_id": selected_agent_id if selected_agent_id != "default" else None
                    }
                    
                    existing_json = latest_message.get('messages', {}).get('jsonContent')
                    if existing_json and isinstance(existing_json, dict):
                        existing_json['intervention'] = intervention_info
                    else:
                        existing_json = {"intervention": intervention_info}
                    
                    update_result = supabase.table("chats").update({
                        "messages": {
                            **latest_message['messages'],
                            "jsonContent": existing_json
                        }
                    }).eq('uuid', latest_message['uuid']).execute()
                    
                    print(f"✅ 사용자 메시지 업데이트 완료:")
                    print(f"   - UUID: {latest_message['uuid']}")
                    print(f"   - should_intervene: True")
                    print(f"   - status: intervening")
                    print(f"   - selected_agent_id: {selected_agent_id}")
                    print(f"   - 업데이트된 jsonContent: {existing_json}")
        except Exception as e:
            print(f"⚠️ 사용자 메시지 업데이트 실패 (무시): {str(e)}")
        
        # 개입 로그 업데이트 (개입 결정 및 에이전트 선택 정보)
        if message_uuid:
            selected_agent = next((a for a in agents if a.get("id") == selected_agent_id), None) if selected_agent_id != "default" else None
            agent_name = selected_agent.get("username", "에이전트") if selected_agent else None
            
            save_intervention_log(
                message_uuid=message_uuid,
                chat_room_id=chat_room_id,
                user_id=user_id,
                user_message=text,
                context_info=context_info,
                should_intervene=True,
                intervention_reason=intervention_decision.get("reason", ""),
                selected_agent_id=selected_agent_id if selected_agent_id != "default" else None,
                selected_agent_name=agent_name,
                agent_selection_reason=agent_selection.get("reason", ""),
                agent_selection_confidence=agent_selection.get("confidence"),
                status="intervening"
            )
        
        agent_response = None
        
        if selected_agent_id and selected_agent_id != "default":
            # 특정 에이전트 호출
            print(f"🤖 에이전트 호출 중: {selected_agent_id}")
            try:
                # mem0_agent_client의 process_mem0_message_with_history 직접 호출 (히스토리 포함 버전)
                from mem0_agent_client import process_mem0_message_with_history
                
                response_data = await process_mem0_message_with_history(
                    text=text,
                    agent_id=selected_agent_id,
                    chat_room_id=chat_room_id,
                    is_learning_mode=False
                )
                print(f"✅ 에이전트 응답 수신 완료")
                
                agent_response_data = response_data.get("response", {})
                
                # 에이전트 응답 저장
                selected_agent = next((a for a in agents if a.get("id") == selected_agent_id), None)
                agent_name = selected_agent.get("username", "에이전트") if selected_agent else "에이전트"
                
                agent_message_data = {
                    "name": agent_name,
                    "content": agent_response_data.get("content", ""),
                    "html": agent_response_data.get("html_content"),
                    "jsonData": agent_response_data
                }
                upsert_chat_message(chat_room_id, agent_message_data, is_system=False, is_agent=True)
                
                # 사용자 메시지의 개입 상태를 "completed"로 업데이트
                try:
                    from database import supabase_client_var, subdomain_var
                    supabase = supabase_client_var.get()
                    subdomain = subdomain_var.get()
                    
                    # 정렬 없이 모든 메시지를 가져온 후 Python에서 정렬
                    response = supabase.table("chats").select("*").eq('id', chat_room_id).eq('tenant_id', subdomain).execute()
                    if response.data and len(response.data) >= 2:
                        # messages.timeStamp를 기준으로 정렬
                        sorted_messages = sorted(
                            response.data,
                            key=lambda x: x.get('messages', {}).get('timeStamp', 0) if isinstance(x.get('messages'), dict) else 0,
                            reverse=True
                        )
                        # 두 번째 메시지가 사용자 메시지 (첫 번째는 에이전트 응답)
                        user_message = sorted_messages[1] if len(sorted_messages) >= 2 else None
                        if not user_message:
                            return
                        if user_message.get('messages') and user_message['messages'].get('email') == user_id:
                            existing_json = user_message.get('messages', {}).get('jsonContent')
                            if existing_json and isinstance(existing_json, dict) and existing_json.get('intervention'):
                                existing_json['intervention']['status'] = 'completed'
                                supabase.table("chats").update({
                                    "messages": {
                                        **user_message['messages'],
                                        "jsonContent": existing_json
                                    }
                                }).eq('uuid', user_message['uuid']).execute()
                                
                                # 개입 로그 업데이트 (에이전트 응답 완료)
                                if message_uuid:
                                    save_intervention_log(
                                        message_uuid=message_uuid,
                                        chat_room_id=chat_room_id,
                                        user_id=user_id,
                                        user_message=text,
                                        context_info=context_info,
                                        should_intervene=True,
                                        intervention_reason=intervention_decision.get("reason", ""),
                                        selected_agent_id=selected_agent_id,
                                        selected_agent_name=agent_name,
                                        agent_selection_reason=agent_selection.get("reason", ""),
                                        agent_selection_confidence=agent_selection.get("confidence"),
                                        agent_response_content=agent_response_data.get("content", ""),
                                        agent_response_type=agent_response_data.get("type", "response"),
                                        status="completed"
                                    )
                except Exception as e:
                    print(f" 개입 상태 업데이트 실패 (무시): {str(e)}")
                
                agent_response = {
                    "agent_id": selected_agent_id,
                    "agent_name": agent_name,
                    "content": agent_response_data.get("content", ""),
                    "html_content": agent_response_data.get("html_content"),
                    "type": agent_response_data.get("type", "response")
                }
            except Exception as e:
                print(f" 에이전트 호출 실패: {str(e)}")
                print(f" 기본 LLM으로 폴백")
                # 에이전트 호출 실패 시 기본 LLM 사용
                default_response = await get_default_llm_response(text, chat_room_id)
                agent_message_data = {
                    "name": "AI 어시스턴트",
                    "content": default_response
                }
                upsert_chat_message(chat_room_id, agent_message_data, is_system=False, is_agent=True)
                
                # 사용자 메시지의 개입 상태를 "completed"로 업데이트
                try:
                    from database import supabase_client_var, subdomain_var
                    supabase = supabase_client_var.get()
                    subdomain = subdomain_var.get()
                    
                    # 정렬 없이 모든 메시지를 가져온 후 Python에서 정렬
                    response = supabase.table("chats").select("*").eq('id', chat_room_id).eq('tenant_id', subdomain).execute()
                    if response.data and len(response.data) >= 2:
                        # messages.timeStamp를 기준으로 정렬
                        sorted_messages = sorted(
                            response.data,
                            key=lambda x: x.get('messages', {}).get('timeStamp', 0) if isinstance(x.get('messages'), dict) else 0,
                            reverse=True
                        )
                        # 두 번째 메시지가 사용자 메시지 (첫 번째는 에이전트 응답)
                        user_message = sorted_messages[1] if len(sorted_messages) >= 2 else None
                        if not user_message:
                            return
                        if user_message.get('messages') and user_message['messages'].get('email') == user_id:
                            existing_json = user_message.get('messages', {}).get('jsonContent')
                            if existing_json and isinstance(existing_json, dict) and existing_json.get('intervention'):
                                existing_json['intervention']['status'] = 'completed'
                                supabase.table("chats").update({
                                    "messages": {
                                        **user_message['messages'],
                                        "jsonContent": existing_json
                                    }
                                }).eq('uuid', user_message['uuid']).execute()
                                
                                # 개입 로그 업데이트 (에이전트 호출 실패, 기본 LLM 사용)
                                if message_uuid:
                                    save_intervention_log(
                                        message_uuid=message_uuid,
                                        chat_room_id=chat_room_id,
                                        user_id=user_id,
                                        user_message=text,
                                        context_info=context_info,
                                        should_intervene=True,
                                        intervention_reason=intervention_decision.get("reason", ""),
                                        selected_agent_id=selected_agent_id,
                                        selected_agent_name="AI 어시스턴트",
                                        agent_selection_reason=agent_selection.get("reason", ""),
                                        agent_selection_confidence=agent_selection.get("confidence"),
                                        agent_response_content=default_response,
                                        agent_response_type="response",
                                        status="completed"
                                    )
                except Exception as e:
                    print(f" 개입 상태 업데이트 실패 (무시): {str(e)}")
                
                agent_response = {
                    "agent_id": "default",
                    "agent_name": "AI 어시스턴트",
                    "content": default_response
                }
        else:
            # 기본 LLM 사용
            print(f" 기본 LLM 사용 (에이전트 선택 불가)")
            default_response = await get_default_llm_response(text, chat_room_id)
            agent_message_data = {
                "name": "AI 어시스턴트",
                "content": default_response
            }
            upsert_chat_message(chat_room_id, agent_message_data, is_system=False, is_agent=True)
            
            # 사용자 메시지의 개입 상태를 "completed"로 업데이트
            try:
                from database import supabase_client_var, subdomain_var
                supabase = supabase_client_var.get()
                subdomain = subdomain_var.get()
                
                # 정렬 없이 모든 메시지를 가져온 후 Python에서 정렬
                response = supabase.table("chats").select("*").eq('id', chat_room_id).eq('tenant_id', subdomain).execute()
                if response.data and len(response.data) >= 2:
                    # messages.timeStamp를 기준으로 정렬
                    sorted_messages = sorted(
                        response.data,
                        key=lambda x: x.get('messages', {}).get('timeStamp', 0) if isinstance(x.get('messages'), dict) else 0,
                        reverse=True
                    )
                    # 두 번째 메시지가 사용자 메시지 (첫 번째는 에이전트 응답)
                    user_message = sorted_messages[1] if len(sorted_messages) >= 2 else None
                    if not user_message:
                        return
                    if user_message.get('messages') and user_message['messages'].get('email') == user_id:
                        existing_json = user_message.get('messages', {}).get('jsonContent')
                        if existing_json and isinstance(existing_json, dict) and existing_json.get('intervention'):
                            existing_json['intervention']['status'] = 'completed'
                            supabase.table("chats").update({
                                "messages": {
                                    **user_message['messages'],
                                    "jsonContent": existing_json
                                }
                            }).eq('uuid', user_message['uuid']).execute()
                            
                            # 개입 로그 업데이트 (기본 LLM 사용)
                            if message_uuid:
                                save_intervention_log(
                                    message_uuid=message_uuid,
                                    chat_room_id=chat_room_id,
                                    user_id=user_id,
                                    user_message=text,
                                    context_info=context_info,
                                    should_intervene=True,
                                    intervention_reason=intervention_decision.get("reason", ""),
                                    selected_agent_id="default",
                                    selected_agent_name="AI 어시스턴트",
                                    agent_selection_reason=agent_selection.get("reason", ""),
                                    agent_selection_confidence=agent_selection.get("confidence"),
                                    agent_response_content=default_response,
                                    agent_response_type="response",
                                    status="completed"
                                )
            except Exception as e:
                print(f" 개입 상태 업데이트 실패 (무시): {str(e)}")
            
            agent_response = {
                "agent_id": "default",
                "agent_name": "AI 어시스턴트",
                "content": default_response
            }
        
        # 사용자 메시지에 개입 정보 업데이트 (이미 로그는 저장되었으므로 여기서는 chats 테이블만 업데이트)
        try:
            from database import supabase_client_var, subdomain_var
            supabase = supabase_client_var.get()
            subdomain = subdomain_var.get()
            
            # 최근 메시지 찾기 (정렬 없이 가져온 후 Python에서 정렬)
            response = supabase.table("chats").select("*").eq('id', chat_room_id).eq('tenant_id', subdomain).execute()
            
            if response.data and len(response.data) > 0:
                # messages.timeStamp를 기준으로 정렬
                sorted_messages = sorted(
                    response.data,
                    key=lambda x: x.get('messages', {}).get('timeStamp', 0) if isinstance(x.get('messages'), dict) else 0,
                    reverse=True
                )
                latest_message = sorted_messages[0] if sorted_messages else response.data[0]
                if latest_message.get('messages') and latest_message['messages'].get('email') == user_id:
                    # 개입 정보를 jsonContent에 추가
                    intervention_info = {
                        "status": "intervening",  # 에이전트 응답 대기 중
                        "should_intervene": True,
                        "selected_agent_id": selected_agent_id,
                        "reason": agent_selection.get("reason", ""),
                        "agent_name": agent_response.get("agent_name") if agent_response else None
                    }
                    
                    # 기존 jsonContent가 있으면 병합, 없으면 새로 생성
                    existing_json = latest_message.get('messages', {}).get('jsonContent')
                    if existing_json and isinstance(existing_json, dict):
                        existing_json['intervention'] = intervention_info
                    else:
                        existing_json = {"intervention": intervention_info}
                    
                    # 메시지 업데이트
                    supabase.table("chats").update({
                        "messages": {
                            **latest_message['messages'],
                            "jsonContent": existing_json
                        }
                    }).eq('uuid', latest_message['uuid']).execute()
        except Exception as e:
            print(f" 개입 정보 업데이트 실패 (무시): {str(e)}")
        
        print(f" 처리 완료")
        print(f"{'='*60}\n")
        
        return {
            "message_saved": True,
            "intervention": {
                "should_intervene": True,
                "selected_agent_id": selected_agent_id,
                "agent_response": agent_response,
                "reason": agent_selection.get("reason", ""),
                "debug_info": debug_info
            }
        }
        
    except Exception as e:
        print(f"\n 에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=str(e))


class UserMessageRequest(BaseModel):
    text: str
    chat_room_id: str
    user_id: str


async def handle_user_message(message: UserMessageRequest):
    """사용자 메시지 처리 엔드포인트"""
    try:
        result = await process_user_message_with_intervention(
            text=message.text,
            chat_room_id=message.chat_room_id,
            user_id=message.user_id
        )
        return result
    except Exception as e:
        print(f"Error in handle_user_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def add_routes_to_app(app: FastAPI):
    """라우트 추가"""
    app.add_api_route(
        "/completion/chat/message",
        handle_user_message,
        methods=["POST"]
    )

