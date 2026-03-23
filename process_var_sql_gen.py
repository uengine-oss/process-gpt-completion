from fastapi import HTTPException, Request
from langchain.prompts import PromptTemplate
from langserve import add_routes
from llm_factory import create_llm
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from database import fetch_all_process_definition_ids, execute_sql, generate_create_statement_for_table, fetch_all_process_definitions, upsert_chat_message, fetch_todolist_by_user_id, fetch_process_instance_list, subdomain_var, fetch_ui_definition, get_vector_store, fetch_all_ui_definition, fetch_organization_chart
from process_engine import submit_workitem
import re
import json
from decimal import Decimal
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser  # JsonOutputParser 임포트
import requests

from datetime import date

# vector
# from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.schema import Document
# embedding_function = OpenAIEmbeddings(model="<embedding_model_alias>")
# persist_directory = "db/speech_embedding_db"

parser = SimpleJsonOutputParser()

# 1. LLM 생성 (공통 팩토리 사용)
model = create_llm(streaming=True)

prompt = PromptTemplate.from_template(
    """
    아래의 데이터베이스 스키마를 참고하여  "{var_name}" 의 값을 얻어올 수 있도록 하는 SQL을 생성해줘:

    - Existing Table Schemas:
    {process_table_schema}
    - 규칙은 이러함:
    {resolution_rule}

    The result should be created in SQL within the following markdown:
    ```
      ...SQL..
    ```
                                      
    """)

form_definition_prompt = PromptTemplate.from_template(
    """
    아래의 사용자 질의와 프로세스 정의 목록을 기반으로 폼 아이디 값을 추출해줘:

    사용자 질의: {query}
    프로세스 정의 목록: {proc_def_list}
    
    결과는 폼 정의 아이디 값만 출력해줘. (ex: leave_request_and_approval_process_submit_leave_request_form)
    """
)

def get_process_definitions(input):
    try:
        query = input.get("query")
        tenant_id = subdomain_var.get()
        
        try:
            # similarity search
            vector_store = get_vector_store()
            proc_def_list = vector_store.similarity_search(
                query, 
                k=3, 
                filter={"tenant_id": tenant_id, "type": "process_definition"}
            )
            return proc_def_list
        except Exception as vector_error:
            print(f"Vector search failed in get_process_definitions: {vector_error}")
            # 벡터 검색에 실패한 경우 빈 리스트 반환
            return []
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
process_definition_prompt = PromptTemplate.from_template(
    """
    다음 질의를 기반으로 프로세스 정의 아이디를 추출해줘:

    사용자 질의: {query}
    프로세스 정의 목록: {proc_def_list}

    결과는 프로세스 정의 아이디만 출력해줘. (ex: vacation_request_process)
    """
)

get_process_definition_id_chain = (
    RunnablePassthrough() | 
    process_definition_prompt | 
    model | 
    StrOutputParser()
)

def get_process_instances(input):
    try:
        query = input.get("query")
        # email = input.get("email")
        # proc_def_list = input.get("proc_def_list")
        
        # # 프롬프트 템플릿을 사용하여 입력 생성
        # process_definition_id = get_process_definition_id_chain.invoke({"query": query, "proc_def_list": proc_def_list})
                
        # # 추출된 프로세스 정의 아이디를 사용하여 인스턴스 조회
        # proc_inst_list = fetch_process_instance_list(email, process_definition_id)
        tenant_id = subdomain_var.get()
        
        try:
            # similarity search
            vector_store = get_vector_store()
            proc_inst_list = vector_store.similarity_search(
                query, 
                k=3, 
                filter={"tenant_id": tenant_id, "type": "process_instance"}
            )
            return proc_inst_list
        except Exception as vector_error:
            print(f"Vector search failed in get_process_instances: {vector_error}")
            # 벡터 검색에 실패한 경우 빈 리스트 반환
            return []
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_chat_history(input):
    try:
        query = input.get("query")
        chat_room_id = input.get("chat_room_id")
        
        # chat_room_id가 없는 경우 빈 리스트 반환
        if not chat_room_id:
            return []
            
        tenant_id = subdomain_var.get()
        
        try:
            # similarity search
            vector_store = get_vector_store()
            chat_history = vector_store.similarity_search(
                query, 
                k=3, 
                filter={"tenant_id": tenant_id, "chat_room_id": chat_room_id}
            )
            return chat_history
        except Exception as vector_error:
            print(f"Vector search failed: {vector_error}")
            # 벡터 검색에 실패한 경우 빈 리스트 반환
            return []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def combine_input_with_process_table_schema(input, path):
    if path == "/process-var-sql":
        var_name = input.get('var_name')
        resolution_rule = input.get('resolution_rule')
        
        process_table_schemas = []
        for process_definition_id in fetch_all_process_definition_ids():
            process_table_schema = generate_create_statement_for_table(process_definition_id)
            process_table_schemas.append(process_table_schema)
        
        process_table_schema = "\n".join(process_table_schemas)
            
        var_sql_input = {
            "var_name": var_name,
            "resolution_rule": resolution_rule,
            "process_table_schema": process_table_schema
        }
        
        return process_var_sql_chain.invoke(var_sql_input)

    elif path == "/process-data-query":
        query = input.get("query")
        email = input.get("user_id")
        proc_def_list = get_process_definitions(input)
        
        # 프로세스 정의 목록을 문자열로 변환
        proc_def_str = json.dumps([doc.page_content for doc in proc_def_list], ensure_ascii=False)
        
        # 프로세스 인스턴스 목록 가져오기
        proc_inst_list = []
        if email:
            instances = fetch_process_instance_list(email)
            if instances:
                proc_inst_list = [inst.model_dump(exclude={'process_definition'}) for inst in instances]
        proc_inst_str = json.dumps(proc_inst_list, ensure_ascii=False, default=default)
        
        # 할일 목록 가져오기
        todo_list = []
        if email:
            todos = fetch_todolist_by_user_id(email)
            if todos:
                todo_list = [todo.model_dump() for todo in todos]
        todo_list_str = json.dumps(todo_list, ensure_ascii=False, default=default)
        
        # 폼 정의 목록 가져오기
        form_def_list = fetch_all_ui_definition()
        form_def_str = json.dumps(form_def_list, ensure_ascii=False, default=default)

        organization_chart = fetch_organization_chart()
        organization_chart_str = json.dumps(organization_chart, ensure_ascii=False, default=default)
        
        # 테이블 생성 체인 설정
        table_chain = (
            draw_table_prompt | 
            model | 
            StrOutputParser() | 
            extract_html_table | 
            clean_html_string
        )
        
        # 테이블 생성 요청
        input_data = {
            "query": query,
            "proc_def_list": proc_def_str,
            "proc_inst_list": proc_inst_str,
            "todo_list": todo_list_str,
            "form_def_list": form_def_str,
            "organization_chart": organization_chart_str
        }
        
        return table_chain.invoke(input_data)


# 프로세스 인스턴스 데이터 조회
def combine_input_with_instance_data_query(input):
    try:
        instance_list = get_process_instances(input)
        query = input["query"]

        return {
            "query": query,
            "instance_list": instance_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 프로세스 인스턴스 시작
def combine_input_with_instance_start(input):
    try:
        query = input["query"]
        chat_room_id = input["chat_room_id"]
        processDefinitionList = input["proc_def_list"]
        if chat_room_id:
            chat_history = get_chat_history(input)

        return {
            "processDefinitionList": processDefinitionList,
            "command": query,
            "chat_history": chat_history,
            "email": input["email"],
            "chat_room_id": chat_room_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 프로세스 정의 조회
def combine_input_with_process_definitions(input):
    try:
        query = input["query"]
        processDefinitionList = input["proc_def_list"]
        
        return {
            "processDefinitionList": processDefinitionList,
            "query": query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 프로세스 인스턴스 할일 목록 조회
def combine_input_with_todolist(input): 
    try:
        query = input["query"]
        user_id = input["email"]
        todolist = fetch_todolist_by_user_id(user_id)
        instance_list = get_process_instances(input)
        
        return {
            "query": query,
            "todolist": todolist,
            "instance_list": instance_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 프로세스 인스턴스 워크아이템
def combine_input_with_workitem_complete(input):
    try:
        query = input["query"]
        email = input["email"]
        chat_room_id = input["chat_room_id"]
        todolist = fetch_todolist_by_user_id(email)
        
        instance_list = get_process_instances(input)
        
        return {
            "command": query,
            "email": email,
            "todolist": todolist,
            "chat_room_id": chat_room_id,
            "instance_list": instance_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 프로세스 인스턴스 입력 데이터
def combine_input_with_process_input_data(input):
    try:
        query = input["query"]
        processDefinitionList = input["proc_def_list"]
        chat_history = get_chat_history(input)
        
        today = str(date.today())
        
        return {
            "command": query,
            "processDefinitionList": processDefinitionList,
            "chat_history": chat_history,
            "today": today
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

combine_input_with_process_table_schema_lambda = RunnableLambda(combine_input_with_process_table_schema)

combine_input_with_instance_data_query_lambda = RunnableLambda(combine_input_with_instance_data_query)
combine_input_with_instance_start_lambda = RunnableLambda(combine_input_with_instance_start)
combine_input_with_process_definitions_lambda = RunnableLambda(combine_input_with_process_definitions)
combine_input_with_todolist_lambda = RunnableLambda(combine_input_with_todolist)
combine_input_with_workitem_complete_lambda = RunnableLambda(combine_input_with_workitem_complete)
combine_input_with_process_input_data_lambda = RunnableLambda(combine_input_with_process_input_data)

def extract_markdown_code_blocks(markdown_text):
    # Extract code blocks from markdown text and concatenate them into a single string
    code_blocks = re.findall(r"```(?:sql)?\n?(.*?)\n?```", markdown_text.content, re.DOTALL)
    single_string_result = "\n".join(code_blocks)
    return single_string_result

def default(obj):
    if isinstance(obj, Decimal):
        return str(obj)  # 또는 float(obj)로 변환
    if isinstance(obj, date):
        return obj.isoformat()  # date 객체를 ISO 포맷 문자열로 변환
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def runsql(sql):
    result = execute_sql(sql)
    return {"result": json.dumps(result, default=default)}

def extract_html_table(markdown_text):
    try:
        if markdown_text is None:
            return None
        # Extract HTML table code block from markdown text
        start = markdown_text.find("```html")
        end = markdown_text.find("```", start + 1)
        if start != -1 and end != -1:
            return markdown_text[start + 7:end].strip()
        return markdown_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def clean_html_string(html_string):
    try:
        if html_string is None:
            return None
        # \n 제거
        cleaned_string = html_string.replace("\n", "")
        # \"를 "로 변환
        cleaned_string = cleaned_string.replace('\\"', '"')
        return cleaned_string
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_form_definition(form_definition_id):
    try:
        form_definition = fetch_ui_definition(form_definition_id)
        if form_definition:
            return form_definition.html
        else:
            return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def execute_process(process_json):
    try:
        process_definition_id = process_json["processDefinitionId"]
        process_instance_id = process_json["processInstanceId"]
        email = process_json["email"]
        answer = process_json["answer"]
        activity_id = process_json["activity_id"]
        chat_room_id = process_json["chatRoomId"]
        
        input = {
            "answer": answer,
            "process_instance_id": process_instance_id,
            "process_definition_id": process_definition_id,
            "userInfo": {
                "email": email
            },
            "activity_id": activity_id,
            "chat_room_id": chat_room_id
        }
        
        response = submit_workitem(input)        
        if response:
            json_data = json.loads(response)
            if json_data:
                return json_data["description"]
            
        else:
            raise HTTPException(status_code=response.status_code, detail=str(response.text))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

draw_table_prompt = PromptTemplate.from_template(
    """
        Please create an HTML table with this data (<table> element only. DO NOT use escape characters like '\"' or '\n'):
        
        User query: {query}
        
        Process definitions:
        {proc_def_list}
        
        Process instances:
        {proc_inst_list}
        
        Todo list:
        {todo_list}
        
        Form definitions:
        {form_def_list}

        Organization chart:
        {organization_chart}
        
        Guidelines:
        1. Analyze the user query carefully and create a table that matches exactly what the user is asking for.

        - If requesting process definition information (process definition, process list):
        Create a table with ID, Name, and Description columns using the process definition data.

        - If requesting process instance information (process instance, ongoing process):
        Create a table with ID, Name, and Status columns using the process instance data.

        - If requesting to-do list information (to-do list, to-dos):
        Create a table with Task, Process, and Status columns using the to-do list data.

        - If requesting form definition information (form definition, form):
        Create a table with Form ID, Name, and Description columns using the form definition data.

        - If requesting organization chart information (organization chart, organization structure, employee information):
        Using the org chart data, create a table with columns for org chart information, such as the team (department) that the team belongs to, who is on that team, and the names of the team members.Using the org chart data, create a table with columns for org chart information, such as the team (department) that the team belongs to, who is on that team, and the names of the team members. You should create the table that best suits your user's question.

        - For other queries, understand the user’s intent and use the most relevant data source.

        2. IMPORTANT: Use only data relevant to the user's specific query. Do not use the process definition as a default.
        3. Make sure the table is responsive and formatted correctly.
        4. Include all relevant data from the data source.
        5. Return only the HTML table code and do not provide a description.
        6. The table must have appropriate Korean column headers that match the data being displayed.

        All of the above guidelines must be followed.
    """
)

describe_result_prompt = PromptTemplate.from_template(
    """
        Based on the following query, please describe the process instance data:
        
        here is user query:
        {query}

        here is the user's instance list:
        {instance_list}

        example: 현재의 프로세스는 영업활동프로세스이며, 진행상태는 영업 제안서 작성단계에서 정체가 발생하고 있으며 담당자는 장진영입니다. 영업 담당자는 강서구입니다. 
        (현재 진행단계 설명, 진행상태 설명, 각 담당자 등 프로세스 인스턴스 테이블에서 얻어진 다양한 정보를 바탕으로 설명)

        * If the data is erroneous, just respond that the information is not available.
        * If the query is about a specific instance, find and describe only the relevant instance.
        * There is no need to explain the instance ID.
        * The result should be described in a narrative form, not in a list.

        * please describe in Korean Language
    """
    )

process_instance_start_prompt = PromptTemplate.from_template(
    """
        다음 질의를 기반으로 어떤 프로세스를 시작해야 할지를 알려줘:

        here is user command:
        {command}

        here is chat history so far:
        {chat_history}

        here is process definitions in our company:
        {processDefinitionList}

        result should be in this JSON format:
        {{
            "processDefinitionId": "the process definition id",
            "processInstanceId": "new",
            "email": "{email}",
            "chatRoomId": "{chat_room_id}",
            "answer": "{command}",
            "activity_id": "The first activity id of the process to start"
        }}
    """
    )

process_input_data_prompt = PromptTemplate.from_template(
    """
        Please describe the input data of the process you want to start like this:

        example: 
        - user: 오늘 프로세스 실행 에러가 발생했어. 해당 내용으로 장애 내역 접수할게.
        - system: 장애 관리 프로세스에 의하면, 장애 내역 접수 활동은 장애 접수폼의 장애 제목, 장애 발생 일자, 장애 유형, 보고자, 장애 설명 작성이 필요합니다. 요청하신 내용을 바탕으로 장애 접수폼을 작성하면 장애 제목은 프로세스 실행 에러, 장애 발생 일자는 오늘 날짜인 2024년 7월 25일, 장애 유형은 소프트웨어, 보고자는 홍길동, 장애 설명은 프로세스 실행 에러 발생 입니다. 수정할 내용이 없으시다면 장애 관리 프로세스를 시작하겠습니다.
        
        here is user command:
        {command}
        
        here is chat history so far:
        {chat_history}

        here is process definitions in our company:
        {processDefinitionList}
        
        today:
        {today}
        
        * 결과는 개조식이 아닌 서술식으로 설명해
        * 만약 사용자의 명령에서 해당 프로세스 정의 활동의 입력 데이터와 일치하는 값이 없다면, 그냥 해당 정보가 없다고 답해
        * 사용자가 수정 요청을 하면 기존 대화 내역을 바탕으로 수정하여 답해
        * 마지막에 수정할 내용이 있는지 확인 받고 없다면 이대로 프로세스를 시작할 건지 확인 받아야 해
        * please describe in Korean Language
    """
    )

process_definition_prompt = PromptTemplate.from_template(
    """
        Please describe the process, activity, checkpoints, transition conditions like this:
    
        example: 
        - user: 영업활동사항을 등록했는데, 다음단계로 뭘 해야 하지?
        - system: 영업활동프로세스에 의하면, 해당 건이 100억 이상인 경우는 팀장승인을 받아야 하고, 제안서를 작성해야 하는 영업기회라면 제안서 작성을 위한 회의를 개최하여야 합니다.

        here is user query:
        {query}

        here is process definitions in our company:
        {processDefinitionList}

        * 결과는 개조식이 아닌 서술식으로 설명해

        * please describe in Korean Language                                    
    """
    )

todolist_prompt = PromptTemplate.from_template(
    """
        아래 정보들과 질의를 기반으로 유저가 요청한 해야할 일에 대한 정보를 간략하게 요약해서 알려줘:
        
        example: 
        - user: 내가 휴가를 언제 신청했더라?
        - system: 휴가 신청 프로세스 인스턴스와 해당 인스턴스의 휴가 신청서 제출 워크아이템에 의하면, 2024년 6월 28일 휴가 신청하였고 사유는 개인 사유, 복귀일은 2024년 7월 1일 입니다.
    
        here is user query:
        {query}
        
        here is the user's instance list:
        {instance_list}

        here is the user's todolist:
        {todolist}

        * 결과는 개조식이 아닌 서술식으로 설명해
        * 인스턴스의 아이디 값은 설명 필요없으니 생략하고 인스턴스 이름으로 설명해
        * please describe in Korean Language
    """
    )

workitem_complete_prompt = PromptTemplate.from_template(
    """
        다음 질의를 기반으로 어떤 할일의 워크아이템을 완료할지를 알려줘:

        here is user command:
        {command}
        
        here is the user's instance list:
        {instance_list}
        
        here is the user's todolist:
        {todolist}

        result should be in this JSON format:
        {{
            "answer": "{command}",
            "processDefinitionId": "the process definition id of the todolist work item",
            "processInstanceId": "the process instance id of the todolist work item",
            "email": "{email}",
            "activity_id": "the activity id of the todolist work item",
            "chatRoomId": "{chat_room_id}"
        }}
    """
    )

process_var_sql_chain = (
    prompt | 
    model | 
    extract_markdown_code_blocks
)

process_data_query_chain = (
    form_definition_prompt | 
    draw_table_prompt |
    model | 
    StrOutputParser() | 
    get_form_definition | 
    extract_html_table | 
    clean_html_string
)


# QUERY_PROCESS_INSTANCE
process_instance_data_query_chain = (
    combine_input_with_instance_data_query_lambda | 
    describe_result_prompt | 
    model | 
    StrOutputParser() 
)

process_instance_start_chain = (
    combine_input_with_instance_start_lambda | 
    process_instance_start_prompt | 
    model | 
    parser | 
    execute_process 
)

process_input_data_chain = (
    combine_input_with_process_input_data_lambda | 
    process_input_data_prompt | 
    model | 
    StrOutputParser()
)

process_definition_query_chain = (
    combine_input_with_process_definitions_lambda | 
    process_definition_prompt | 
    model | 
    StrOutputParser() 
)

todolist_query_chain = (
    combine_input_with_todolist_lambda | 
    todolist_prompt | 
    model | 
    StrOutputParser() 
)

workitem_complete_chain = (
    combine_input_with_workitem_complete_lambda | 
    workitem_complete_prompt | 
    model | 
    parser | 
    execute_process 
)


async def combine_input(request: Request):
    json_data = await request.json()
    input = json_data.get('input')
    return combine_input_with_process_table_schema(input, request.url.path)


def add_routes_to_app(app) :
    # add_routes(
    #     app,
    #     combine_input_with_process_table_schema_lambda | prompt | model | extract_markdown_code_blocks,
    #     path="/process-var-sql",
    # )

    # add_routes(
    #     app,
    #     combine_input_with_process_table_schema_lambda | prompt | model | extract_markdown_code_blocks | runsql | draw_table_prompt | model | StrOutputParser() | extract_html_table | clean_html_string,
    #     path="/process-data-query",
    # )

    app.add_api_route("/process-var-sql", combine_input, methods=["POST"])
    app.add_api_route("/process-data-query", combine_input, methods=["POST"])



 
"""
http :8000/process-data-query/invoke input[var_name]="모든 입사 지원자를 출력해줘"
http :8000/process-data-query/invoke input[var_name]="sw분야 지원한 입사지원자 목록" 
http :8000/process-var-sql/invoke input[var_name]="total_vacation_days_remains" input[resolution_rule]="vacation_addition 테이블의 전체 휴가일수에서 vacation_request 의 사용일수를 제외함. 그리고 10일은 기본적으로 추가"
"""