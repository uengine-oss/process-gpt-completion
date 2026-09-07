import os
import importlib

os.environ["PYTHONIOENCODING"] = "utf-8"

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from process_db_manager import add_routes_to_app as add_db_manager_routes_to_app
from database import update_tenant_id
# notification_polling_task는 FCM 서비스로 분리됨
from mcp_config_api import add_routes_to_app as add_mcp_routes_to_app
from sql_import_api import add_routes_to_app as add_sql_import_routes_to_app
from db_backup_api import add_routes_to_app as add_db_backup_routes_to_app
from bpmn_git_export import add_routes_to_app as add_bpmn_git_export_routes_to_app
from agent_chat import add_routes_to_app as add_agent_chat_routes_to_app
from callbot_api import add_routes_to_app as add_callbot_routes_to_app
from test_mode import add_routes_to_app as add_test_mode_routes_to_app
from process_start_api import add_routes_to_app as add_process_start_routes_to_app
from audio_transcribe import add_routes_to_app as add_audio_routes_to_app
from validate_improve import add_routes_to_app as add_validate_improve_routes_to_app

from dotenv import load_dotenv

if os.getenv("ENV") != "production":
    load_dotenv(override=True)

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
else:
    os.environ["LANGSMITH_TRACING"] = "false"


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


API_PATH_PREFIX = (os.getenv("API_PATH_PREFIX") or "/pi-system-backend").rstrip("/")
DEFAULT_TENANT_ID = (os.getenv("DEFAULT_TENANT_ID") or "").strip()
ENABLE_LANGCHAIN_ROUTES = _env_flag("ENABLE_LANGCHAIN_ROUTES", default=True)


app = FastAPI(
    title="PI System Backend",
    version="1.0",
    description="Backend API server for the PI system",
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

class StripPathPrefixMiddleware:
    def __init__(self, app: FastAPI, prefix: str = ""):
        self.app = app
        self.prefix = prefix.rstrip("/")

    async def __call__(self, scope, receive, send):
        if scope["type"] in {"http", "websocket"} and self.prefix:
            path = scope.get("path", "")
            if path == self.prefix or path.startswith(f"{self.prefix}/"):
                new_scope = dict(scope)
                existing_root_path = scope.get("root_path", "")
                new_scope["root_path"] = f"{existing_root_path}{self.prefix}"
                stripped_path = path[len(self.prefix):] or "/"
                new_scope["path"] = stripped_path

                raw_path = scope.get("raw_path")
                prefix_bytes = self.prefix.encode()
                if raw_path and (raw_path == prefix_bytes or raw_path.startswith(prefix_bytes + b"/")):
                    new_scope["raw_path"] = raw_path[len(prefix_bytes):] or b"/"

                scope = new_scope

        await self.app(scope, receive, send)


class DBConfigMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        host_name = request.headers.get("X-Forwarded-Host")
        if DEFAULT_TENANT_ID:
            tenant_id = DEFAULT_TENANT_ID
        elif host_name is None or "localhost" in host_name:
            tenant_id = "localhost"
        else:
            tenant_id = host_name.split(".")[0]
        await update_tenant_id(tenant_id)
        # 요청을 다음 미들웨어 또는 엔드포인트로 전달
        response = await call_next(request)
        return response


# app.post("/update_db")(update_db)
    
# 미들웨어 추가
app.add_middleware(StripPathPrefixMiddleware, prefix=API_PATH_PREFIX)
app.add_middleware(DBConfigMiddleware)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

app.mount("/static", StaticFiles(directory="static"), name="static")


def _register_optional_route_module(app: FastAPI, module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
        add_routes = getattr(module, "add_routes_to_app", None)
        if callable(add_routes):
            add_routes(app)
    except Exception as exc:
        print(f"[startup] Skipping optional routes from {module_name}: {exc}")

add_db_manager_routes_to_app(app)
add_mcp_routes_to_app(app)
add_sql_import_routes_to_app(app)
add_db_backup_routes_to_app(app)
add_bpmn_git_export_routes_to_app(app)
_register_optional_route_module(app, "process_chat")

if ENABLE_LANGCHAIN_ROUTES:
    for langchain_module in ("process_engine", "process_def_search"):
        _register_optional_route_module(app, langchain_module)


OPTIONAL_ROUTE_MODULES = (
    "process_var_sql_gen",
    "min",
    "callbot_api",
    "test_mode",
    "process_start_api",
    "validate_improve",
    "bpmn_validation_exceptions",
    "process_image",
    "agent_chat",
    "form_designer_chat",
    "tmf_kb_api",
    "organization_api",
    "supplier_api",
    "system_api",
    "project_mappings_api",
    "processes_api",
    "governance_api",
    "operational_board_api",
    "dashboard_api",
)

for optional_module in OPTIONAL_ROUTE_MODULES:
    _register_optional_route_module(app, optional_module)
add_agent_chat_routes_to_app(app)
add_callbot_routes_to_app(app)
add_test_mode_routes_to_app(app)
add_validate_improve_routes_to_app(app)
add_process_start_routes_to_app(app)
# 음성 입력(/completion/upload). 화면은 예전부터 이 경로를 불렀는데 서버에 없었다.
add_audio_routes_to_app(app)

import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from pytz import timezone as pytz_timezone

from db_backup_api import run_scheduled_db_backup
from bpmn_git_export import is_bpmn_git_export_enabled, run_scheduled_bpmn_git_export
from governance_notification_service import run_outbox_processor
KST = pytz_timezone("Asia/Seoul")
DB_BACKUP_CRON_HOUR = int(os.getenv("DB_BACKUP_CRON_HOUR", "4"))
DB_BACKUP_CRON_MINUTE = int(os.getenv("DB_BACKUP_CRON_MINUTE", "0"))
BPMN_GIT_EXPORT_CRON_HOUR = int(os.getenv("BPMN_GIT_EXPORT_CRON_HOUR", "5"))
BPMN_GIT_EXPORT_CRON_MINUTE = int(os.getenv("BPMN_GIT_EXPORT_CRON_MINUTE", "0"))
BPMN_GIT_EXPORT_ENABLED = is_bpmn_git_export_enabled()
# 거버넌스 알림: 변경 발생 시 outbox(트리거 적재) 행을 즉시 발송하는 백그라운드 프로세서 주기(초)
GOVERNANCE_NOTIFY_INTERVAL_SECONDS = int(os.getenv("GOVERNANCE_NOTIFY_INTERVAL_SECONDS", "60"))
GOVERNANCE_NOTIFY_ENABLED = _env_flag("GOVERNANCE_NOTIFY_ENABLED", default=True)
GOVERNANCE_NOTIFY_MAX_ITEMS = int(os.getenv("GOVERNANCE_NOTIFY_MAX_ITEMS", "100"))

scheduler = AsyncIOScheduler(timezone=KST)


def _background_tenant_id() -> str:
    return DEFAULT_TENANT_ID or "localhost"


async def _scheduled_db_backup_job():
    try:
        await update_tenant_id(_background_tenant_id())
        await run_scheduled_db_backup()
    except Exception as exc:
        print(f"[scheduled-db-backup] failed: {exc}")


scheduler.add_job(
    _scheduled_db_backup_job,
    trigger=CronTrigger(hour=DB_BACKUP_CRON_HOUR, minute=DB_BACKUP_CRON_MINUTE),
    id="db-backup",
    replace_existing=True,
)


async def _scheduled_bpmn_git_export_job():
    try:
        await update_tenant_id(_background_tenant_id())
        summary = await run_scheduled_bpmn_git_export()
        print(
            "[scheduled-bpmn-git-export] completed: "
            f"tenant={summary['tenant_id']}, status={summary['status']}, "
            f"exported={summary['exported_files']}, modules={summary['module_files']}, "
            f"missing_bpmn={summary['missing_bpmn']}"
        )
    except Exception as exc:
        print(f"[scheduled-bpmn-git-export] failed: {exc}")


if BPMN_GIT_EXPORT_ENABLED:
    scheduler.add_job(
        _scheduled_bpmn_git_export_job,
        trigger=CronTrigger(hour=BPMN_GIT_EXPORT_CRON_HOUR, minute=BPMN_GIT_EXPORT_CRON_MINUTE),
        id="bpmn-git-export",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )


async def _governance_notification_job():
    try:
        tenant_id = _background_tenant_id()
        await update_tenant_id(tenant_id)
        await run_outbox_processor(tenant_id, max_items=GOVERNANCE_NOTIFY_MAX_ITEMS)
    except Exception as exc:
        print(f"[governance-notify] failed: {exc}")


if GOVERNANCE_NOTIFY_ENABLED:
    scheduler.add_job(
        _governance_notification_job,
        trigger=IntervalTrigger(seconds=GOVERNANCE_NOTIFY_INTERVAL_SECONDS),
        id="governance-notify",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )


@app.on_event("startup")
async def start_background_tasks():
    scheduler.start()
    governance_state = (
        f"every {GOVERNANCE_NOTIFY_INTERVAL_SECONDS}s"
        if GOVERNANCE_NOTIFY_ENABLED
        else "DISABLED"
    )
    bpmn_git_export_state = (
        f"at {BPMN_GIT_EXPORT_CRON_HOUR:02d}:{BPMN_GIT_EXPORT_CRON_MINUTE:02d} KST"
        if BPMN_GIT_EXPORT_ENABLED
        else "DISABLED"
    )
    print(
        f"[scheduler] started – "
        f"db-backup at {DB_BACKUP_CRON_HOUR:02d}:{DB_BACKUP_CRON_MINUTE:02d} KST, "
        f"bpmn-git-export {bpmn_git_export_state}, "
        f"governance-notify {governance_state}"
    )


@app.on_event("shutdown")
async def stop_background_tasks():
    scheduler.shutdown(wait=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
