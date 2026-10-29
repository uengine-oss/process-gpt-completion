#!/usr/bin/env python3
"""
Polling Service Main Entry Point

This service handles polling for workitems and processing them.
"""
import os
from dotenv import load_dotenv
from polling_service import run_polling_service

if os.getenv("ENV") != "production":
    load_dotenv(override=True)

# trace to LangSmith only when a key is set (otherwise every run logs a 401)
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

if __name__ == "__main__":
    print("[INFO] Starting Process GPT Polling Service...")
    run_polling_service() 