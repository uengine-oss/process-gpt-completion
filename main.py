import os

os.environ["PYTHONIOENCODING"] = "utf-8"

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from process_db_manager import add_routes_to_app as add_db_manager_routes_to_app
from process_engine import add_routes_to_app as add_process_routes_to_app
from process_image import add_routes_to_app as add_image_routes_to_app
from process_var_sql_gen import add_routes_to_app as add_var_sql_gen_routes_to_app
from audio_input import add_routes_to_app as add_audio_input_routes_to_app
from min import add_routes_to_app as add_min_routes_to_app
from process_def_search import add_routes_to_app as add_process_def_search_routes_to_app
from process_chat import add_routes_to_app as add_process_chat_routes_to_app
from database import update_tenant_id
# notification_polling_task는 FCM 서비스로 분리됨
from mcp_config_api import add_routes_to_app as add_mcp_routes_to_app
from agent_chat import add_routes_to_app as add_agent_chat_routes_to_app
from callbot_api import add_routes_to_app as add_callbot_routes_to_app

from dotenv import load_dotenv

if os.getenv("ENV") != "production":
    load_dotenv(override=True)
    # 캐시 적용
    from langchain.cache import SQLiteCache
    from langchain.globals import set_llm_cache
    set_llm_cache(SQLiteCache(database_path=".langchain.db"))

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

from starlette.middleware.base import BaseHTTPMiddleware
from database import update_tenant_id

class DBConfigMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        host_name = request.headers.get('X-Forwarded-Host')
        if host_name is None or any(substring in host_name for substring in ['localhost']):
            subdomain = 'localhost'
        else:
            subdomain = host_name.split('.')[0]
            
        await update_tenant_id(subdomain)
        # 요청을 다음 미들웨어 또는 엔드포인트로 전달
        response = await call_next(request)
        return response


# app.post("/update_db")(update_db)
    
# 미들웨어 추가
app.add_middleware(DBConfigMiddleware)

app.mount("/static", StaticFiles(directory="static"), name="static")

add_process_routes_to_app(app)
add_db_manager_routes_to_app(app)
add_image_routes_to_app(app)
add_var_sql_gen_routes_to_app(app)
add_audio_input_routes_to_app(app)
add_min_routes_to_app(app)
add_process_def_search_routes_to_app(app)
add_process_chat_routes_to_app(app)
add_mcp_routes_to_app(app)
add_agent_chat_routes_to_app(app)
add_callbot_routes_to_app(app)

import asyncio

@app.on_event("startup")
async def start_background_tasks():
    # 알림 실시간 구독 태스크 시작
    # asyncio.create_task(notification_polling_task())
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)