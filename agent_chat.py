from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
from uuid import uuid4
from dotenv import load_dotenv
from mem0_agent_client import process_mem0_message
from fastapi.responses import StreamingResponse, JSONResponse

import requests
import os

if os.getenv("ENV") != "production":
    load_dotenv(override=True)

def add_routes_to_app(app):
    app.add_api_route("/multi-agent/chat", chat_message, methods=["POST"])
    app.add_api_route("/multi-agent/health-check", health_check, methods=["GET"])
    app.add_api_route("/multi-agent/fetch-data", fetch_data, methods=["GET"])

class ChatMessage(BaseModel):
    text: str
    chat_room_id: str
    options: Optional[Dict[Any, Any]] = None

class ChatResponse(BaseModel):
    response: Dict[str, Any]

async def chat_message(message: ChatMessage):
    try:
        chat_room_id = message.chat_room_id
        agent_id = message.options.get("agent_id") if message.options else None
        is_learning_mode = message.options.get("is_learning_mode") if message.options else False
        
        if not agent_id: 
            raise HTTPException(status_code=400, detail="agent_id is required for Mem0 agent")
        
        response = await process_mem0_message(
            text=message.text,
            agent_id=agent_id,
            chat_room_id=chat_room_id,
            is_learning_mode=is_learning_mode
        )
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

def fetch_data(agent_url: str = Query(..., description="Agent URL to fetch data from")):
    """Fetch agent data endpoint."""
    try:
        if not agent_url.startswith(('http://', 'https://')):
            agent_url = 'http://' + agent_url
        agent_data = requests.get(
            f'{agent_url}/.well-known/agent.json'
        )
        return agent_data.json()
    except Exception as e:
        print(f"Error in fetch_data endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

