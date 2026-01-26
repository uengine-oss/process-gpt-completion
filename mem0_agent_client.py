from mem0 import Memory
from dotenv import load_dotenv
import os
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from fastapi import HTTPException
from database import fetch_chat_history
from llm_factory import create_llm

if os.getenv("ENV") != "production":
    load_dotenv(override=True)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# LLM 객체 생성 (공통 팩토리 사용)
llm = create_llm(model="gpt-4o", streaming=True)

intent_analysis_prompt = PromptTemplate.from_template(
    """이전 대화 내역과 사용자의 메시지를 바탕으로 다음 사용자의 의도를 분석해주세요.

응답 형식:
{{
    "intent": "query" 또는 "information". 의도를 알 수 없는 경우 기본적으로 "information",
    "content": "질문의 경우 검색할 내용을 제공, 정보의 경우 저장할 내용을 제공."
}}

예시:
- 입력: "지방에 있는 매출에 설립일 1년 이내 여자 대표이사가 운영하는 회사이다. 이 회사의 법인세 감면율은?"
{{
    "intent": "query",
    "content": "지방에 있는 매출에 설립일 1년 이내 여자 대표이사가 운영하는 회사이다. 이 회사의 법인세 감면율은?"
}}

- 입력: "기본 법인세율: 20%"
{{
    "intent": "information",
    "content": "기본 법인세율은 20% 입니다."
}}

사용자 입력: {message}

이전 대화 내역:
{chat_history}"""
)

response_generation_prompt = PromptTemplate.from_template(
    """당신은 검색 결과를 바탕으로 사용자의 질문에 답변해야 합니다.
다음 형식으로 답변해주세요:
{{
    "content": "답변 내용. 예시: 지방 소재의 IT 기업에 대한 법인세 감면에 대하여 안내드리겠습니다. 기본적으로 최대 감면율은 20%이며, 중복 적용 가능하나 상한선이 존재합니다. 특히 지방에 소재한 기업은 지방세 3% 추가 감면을 받을 수 있습니다. 또한, IT 서비스업인 경우 여성 대표가 있을 때 및 지방 감면이 동시에 적용 가능합니다. 그러나 업종에 따라 제한이 있을 수 있으며, 예를 들어 제조업은 지방세 감면이 제외됩니다.",
    "html_content": "답변 내용을 HTML 태그로 표기, 내용 중 검색 결과를 포함하고 출처(인덱스)를 표기하는데 출처 내용은 span 태그에 'search-result' 라는 class 를 표기하고 인덱스 값은 span 태그에 'search-result-index' 라는 class 를 사용하여 표기. search_results 의 index 값과 동일하게 작성할 것. 답변 예시: <div>지방 소재의 IT 기업에 대한 법인세 감면에 대하여 안내드리겠습니다. 기본적으로 <span class='search-result'>최대 감면율은 20%이며, 중복 적용 가능하나 상한선이 존재<span class="search-result-index">3</span></span>합니다. 특히 지방에 소재한 기업은 <span class='search-result'>지방세 3% 추가 감면<span class="search-result-index">0</span></span>을 받을 수 있습니다. 또한, <span class='search-result'>IT 서비스업인 경우 여성 대표가 있을 때 및 지방 감면이 동시에 적용 가능<span class="search-result-index">2</span></span>합니다. 그러나 업종에 따라 제한이 있을 수 있으며, 예를 들어 <span class='search-result'>제조업은 지방세 감면이 제외<span class="search-result-index">1</span></span>됩니다.</div>",
    "search_results": [
        {{
            "index": 0,
            "score": 0.51,
            "memory": "비수도권 소재 기업은 지방세 3% 추가 감면"
        }},
        {{
            "index": 1,
            "score": 0.61,
            "memory": "업종별 제한: 제조업은 지방세 감면 제외"
        }},
        {{
            "index": 2,
            "score": 0.64,
            "memory": "IT 서비스업은 여성 대표 + 지방 감면 동시 적용 가능"
        }},
        {{
            "index": 3,
            "score": 0.64,
            "memory": "최대 감면율은 20%이며, 중복 적용 가능하나 상한선이 존재함"
        }}
    ]
}}

1. 검색 결과 요약
   - 가장 관련성 높은 정보 2-3개를 간단히 요약

2. 상세 설명
   - 검색 결과를 바탕으로 질문에 대한 상세한 설명
   - 필요한 경우 예시나 추가 설명 포함

3. 추가 정보
   - 관련된 추가 정보나 주의사항이 있다면 언급
   - 더 자세한 정보가 필요한 경우 안내

검색 결과에 없는 내용은 추측하지 마세요.

질문: {message}

검색 결과:
{search_context}"""
)

config = {
    "vector_store": {
        "provider": "supabase",
        "config": {
            "connection_string": connection_string,
            "collection_name": "memories",
            "index_method": "hnsw",
            "index_measure": "cosine_distance"
        }
    }
}

memory = Memory.from_config(config_dict=config)

intent_chain = (
    RunnablePassthrough() |
    intent_analysis_prompt |
    llm |
    StrOutputParser()
)

response_chain = (
    RunnablePassthrough() |
    response_generation_prompt |
    llm |
    StrOutputParser()
)

async def analyze_intent(message: str, chat_room_id: str = None) -> Tuple[str, Optional[Dict]]:
    """OpenAI를 사용하여 메시지의 의도를 분석합니다."""
    try:
        chat_history = fetch_chat_history(chat_room_id)
        if chat_history:
            chat_history = "\n".join([f"{item.messages.content}" for item in chat_history if item.messages])
        else:
            chat_history = ""
        result = await intent_chain.ainvoke({"message": message, "chat_history": chat_history})
        parsed_result = json.loads(result)
        
        intent = parsed_result["intent"]
        info = {
            "content": parsed_result["content"],
            "category": intent,
            "confidence": 0.9,
            "timestamp": datetime.now().isoformat()
        }
        return intent, info
        
    except Exception as e:
        print(f"OpenAI 분석 중 오류 발생: {str(e)}")
        return "other", {"content": "죄송합니다. 이해하지 못했습니다. 다시 요청 해주세요."}

async def generate_response(message: str, search_results: List[Dict]) -> str:
    """검색 결과를 활용하여 응답을 생성합니다."""
    try:
        search_context = "\n".join([f"- {r['memory']} (신뢰도: {r['score']:.2f})" for r in search_results])
        response = await response_chain.ainvoke({
            "message": message,
            "search_context": search_context
        })
        return response
                
    except Exception as e:
        print(f"응답 생성 중 오류 발생: {str(e)}")
        raise

def search_memories(agent_id: str, query: str) -> List[Dict]:
    """mem0에서 관련 정보를 검색합니다."""
    results = memory.search(query, agent_id=agent_id)
    return results["results"][:5]

def store_in_memory(agent_id: str, content: str):
    """유의미한 정보를 mem0에 저장합니다."""
    memory.add(
        content,
        agent_id=agent_id,
        metadata={
            "type": "information",
            "timestamp": datetime.now().isoformat()
        },
        infer=False
    )

async def process_mem0_message(text: str, agent_id: str, chat_room_id: str = None, is_learning_mode: bool = False):
    """Mem0 에이전트를 통해 메시지를 처리합니다."""
    try:
        if is_learning_mode:
            intent = "information"
            store_in_memory(agent_id, text)
            return {
                "task_id": str(datetime.now().timestamp()),
                "response": {
                    "type": intent,
                    "content": text
                }
            }
        else:
            intent = "query"
            search_term = text
            search_results = search_memories(agent_id, search_term)
            
            response = await generate_response(text, search_results)
            try:
                response = json.loads(response)
            except:
                response = {"content": response}
            response["type"] = intent
            
            return {
                "task_id": str(datetime.now().timestamp()),
                "response": response
            }

    except Exception as e:
        print(f"메시지 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))