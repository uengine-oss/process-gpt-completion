from fastapi import HTTPException, Request
from typing import List, Optional
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from llm_factory import create_openai_llm
from langchain_core.runnables import RunnableLambda
from langserve import add_routes
from pydantic import BaseModel

from database import fetch_all_process_definitions
from process_var_sql_gen import get_process_definitions


# LLM 지연 초기화 — 키 없는 환경에서도 임포트(=라우트 등록)는 성공해야 한다 (process_engine과 동일 패턴).
_vision_model = None

def get_vision_model():
    global _vision_model
    if _vision_model is None:
        _vision_model = create_openai_llm(max_tokens=4096)
    return _vision_model

parser = SimpleJsonOutputParser()



prompt = PromptTemplate.from_template(
    """
    Now I'm going to create an interactive system that tells you the most similar process to run when you enter an image or message.

    - Process Definition List: {processDefinitionList}
    
    - Entered message: {message}
    
    - Entered image: {image}

    Based on the entered message or image information, return the most similar process definition.
    
    result should be in this JSON format:
    {{
        "processDefinitionList": [{{
            "id": "process definition id",
            "name": "process definition name",
            "description": "process definition description"
        }}]
    }}
    """
    )

import base64
from langchain_core.messages import HumanMessage, AIMessage

def vision_model_chain(input):
    formatted_prompt = prompt.format(**input)
    
    msg = get_vision_model().invoke(
        [   AIMessage(
                content=formatted_prompt
            ),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                           "url": input['image'],
                            'detail': 'high'
                        },
                    },
                ]
            )
        ]
    )
    return msg

class ProcessDefinition(BaseModel):
    id: str
    name: str
    description: Optional[str] = None

class ProcessResult(BaseModel):
    processDefinitionList: Optional[List[ProcessDefinition]] = None

def process_search(process_result_json: dict) -> str:
    try:
        process_result = ProcessResult(**process_result_json)
        # formatted_prompt = prompt.format(query=query, process_definitions=process_definitions)
        # response = model.invoke(formatted_prompt)
        # parsed_response = parser.parse(response)
        # return parsed_response
        
        return json.dumps(process_result_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# (제거) 모듈 레벨 chain/vision_chain — 주석 처리된 langserve add_routes 전용 잔재로,
# 실제 라우트(combine_input_with_process_definition)는 LLM을 호출하지 않는다.

async def combine_input_with_process_definition(request: Request):
    try:
        input = await request.json()
        process_definitions = get_process_definitions(input)

        return process_definitions

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def add_routes_to_app(app) :
    app.add_api_route("/process-search", combine_input_with_process_definition, methods=["POST"])
    app.add_api_route("/vision-process-search", combine_input_with_process_definition, methods=["POST"])
    
    # add_routes(
    #     app,
    #     combine_input_with_process_definition_lambda | prompt | model | parser | process_search,
    #     path="/process-search",
    # )
    
    # add_routes(
    #     app,
    #     combine_input_with_process_definition_lambda | vision_model_chain | parser | process_search,
    #     path="/vision-process-search",
    # )



"""
http :8000/process-search/invoke input[answer]="휴가 신청하고 싶어."
"""
