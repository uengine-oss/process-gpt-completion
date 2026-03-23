from fastapi import HTTPException, Request
from typing import List, Optional
import json
from langchain.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from llm_factory import create_llm, create_openai_llm
from langchain_core.runnables import RunnableLambda
from langserve import add_routes
from pydantic import BaseModel

from database import fetch_all_process_definitions
from process_var_sql_gen import get_process_definitions


model = create_llm()
vision_model = create_openai_llm(max_tokens=4096)

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
from langchain.schema.messages import HumanMessage, AIMessage

def vision_model_chain(input):
    formatted_prompt = prompt.format(**input)
    
    msg = vision_model.invoke(
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

chain = (
    prompt | model | parser | process_search
)

vision_chain = (
    vision_model_chain | parser | process_search
)

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
