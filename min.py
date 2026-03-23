from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from llm_factory import create_llm
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langserve import add_routes
from pydantic import BaseModel, Field
from typing import List  # Import List from typing module


prompt2 = ChatPromptTemplate.from_template(
    """
    python 문법을 검증해줘: {pythonCode}

    
    """)

# Prompt Template 생성
prompt_template = ChatPromptTemplate.from_template(
    """
    다음주제에 대한 프로세스 정의를 생성해줘: {topic}
    다음 json format 으로 리턴해:

{{
    "activities": [
        {{
            "id": "액티비티 id",
            "name": "액티비티 이름",
            "description": "액티비티 설명",
            "type": "ScriptActivity" | "UserActivity",
            "pythonCode": "python code to execute the script activity"
        }}
    ]
}}
    """
    )

# LLM 인스턴스 생성 (공통 팩토리 사용)
model = create_llm()

# 출력 파서 생성
output_parser = SimpleJsonOutputParser()
strOutputParser = StrOutputParser()

class Activity(BaseModel):
    id: str = Field(..., alias="id")
    name: str = Field(..., alias="name")
    description: str = Field(..., alias="description")
    type: str = Field(..., alias="type")
    pythonCode: str = Field(None, alias="pythonCode")  # pythonCode를 옵셔널 필드로 변경

class ProcessDefinition(BaseModel):   # class ProcessDefinition extens BaseModel
    activities: List[Activity]  # Define activities as a list of Activity instances

def extract_script_activities(process_definition: dict) -> dict:
    process_definition_obj = ProcessDefinition(**process_definition)  # new ProcessDefinition(...)
    script_activities = [
        activity.pythonCode for activity in process_definition_obj.activities if activity.type == "ScriptActivity"
    ]
    # 결과를 dict 형태로 변경, key는 'pythonCode'로 설정
    return {"pythonCode": "\n - ".join(script_activities)}

# LCEL 체인에 함수 적용
chain = prompt_template | model | output_parser | extract_script_activities | prompt2 | model | strOutputParser

def add_routes_to_app(app) :
    add_routes(app, chain, path="/generate_joke")