#!/usr/bin/env python3
"""
Polling Service Main Entry Point

This service handles polling for workitems and processing them.
"""
import logging
import os
from dotenv import load_dotenv
from polling_service import run_polling_service

if os.getenv("ENV") != "production":
    load_dotenv(override=True)

# 이 서비스는 로그를 print 로 남기므로 `logging` 은 설정되어 있지 않다. 그런데 고착화
# (결정론적 코드 생성)는 `logging` 으로만 판정을 남긴다 — 왜 굳혔는지, 왜 보류했는지,
# 어떤 표본을 실패로 제외했는지가 전부 여기 있다. 설정하지 않으면 그 판정이 통째로
# 사라져, 조용히 아무것도 고착화되지 않아도 알아챌 방법이 없다.
# 서비스 전역 로깅을 켜지는 않고(기존 print 출력이 묻힌다) 해당 모듈만 올린다.
_freeze_log = logging.StreamHandler()
_freeze_log.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
for _name in ("deterministic_generator", "mcp_tool_index"):
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.INFO)
    _logger.addHandler(_freeze_log)
    _logger.propagate = False

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

if __name__ == "__main__":
    print("[INFO] Starting Process GPT Polling Service...")
    run_polling_service() 