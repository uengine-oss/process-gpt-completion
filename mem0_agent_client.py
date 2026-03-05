from mem0 import Memory
from dotenv import load_dotenv
import os
from typing import Dict, List, Any
import json
from datetime import datetime
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from fastapi import HTTPException
from llm_factory import create_llm

# 학습 모드 유사도 임계값 (고정 응답은 사용하지 않고 LLM이 학습 유무를 포함해 자연스럽게 답변)
LEARNING_DUPLICATE_THRESHOLD = 0.92

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

learning_response_prompt = PromptTemplate.from_template(
    """당신은 사용자가 입력한 지침이나 기억을 받아들이는 에이전트입니다.
사용자 메시지에 대해 자연스럽게 한두 문장으로 답하되, 반드시 아래 학습 결과를 포함해주세요.

- 저장한 경우: 이 내용을 잘 반영했음/기억했음/학습했음을 자연스럽게 표현해주세요.
- 저장하지 않은 경우: 비슷한 내용이 이미 있어 새로 저장하지 않았음을 자연스럽게 표현해주세요.

답변은 JSON이 아니라 평문 한두 문장으로만 작성해주세요.

학습 결과: {learning_result}
사용자 메시지: {user_message}"""
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

learning_response_chain = (
    RunnablePassthrough() |
    learning_response_prompt |
    llm |
    StrOutputParser()
)

response_chain = (
    RunnablePassthrough() |
    response_generation_prompt |
    llm |
    StrOutputParser()
)

async def generate_learning_response(user_message: str, was_stored: bool) -> str:
    """학습 유무를 포함한 자연스러운 답변을 생성합니다."""
    learning_result = "저장함" if was_stored else "비슷한 내용이 있어 저장하지 않음"
    return await learning_response_chain.ainvoke({
        "user_message": user_message,
        "learning_result": learning_result
    })

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

def _is_duplicate_memory(search_results: List[Dict], threshold: float = LEARNING_DUPLICATE_THRESHOLD) -> bool:
    """검색 결과 상위 1건의 유사도가 임계값 이상이면 유사 메모리 존재로 간주합니다."""
    if not search_results:
        return False
    first = search_results[0]
    score = first.get("score")
    if score is None:
        return False
    return float(score) >= threshold

async def process_mem0_message(text: str, agent_id: str, chat_room_id: str = None, is_learning_mode: bool = False):
    """Mem0 에이전트를 통해 메시지를 처리합니다."""
    try:
        if is_learning_mode:
            search_results = search_memories(agent_id, text)
            is_duplicate = _is_duplicate_memory(search_results)
            if not is_duplicate:
                store_in_memory(agent_id, text)
            content = await generate_learning_response(text, was_stored=not is_duplicate)
            return {
                "task_id": str(datetime.now().timestamp()),
                "response": {
                    "type": "information",
                    "content": content
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