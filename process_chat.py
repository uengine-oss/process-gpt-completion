from features.process_chat import (
    BASE_URL, 
    ChatRequest, TokenCountRequest, EmbeddingRequest, 
    ChatInterface
)
from fastapi import HTTPException
import os


def _resolve_model(model: str | None) -> str:
    resolved_model = model or os.getenv("LLM_MODEL")

    if not resolved_model:
        raise HTTPException(
            status_code=400,
            detail="`model` is required. Provide it in request or set `LLM_MODEL` env.",
        )
    return resolved_model


def _resolve_embedding_model(model: str | None) -> str:
    resolved_model = (
        model
        or os.getenv("LLM_EMBEDDING_MODEL")
        or os.getenv("OPENAI_EMBEDDING_MODEL")
    )
    if not resolved_model:
        raise HTTPException(
            status_code=400,
            detail="`embedding model` is required. Provide it in request or set `LLM_EMBEDDING_MODEL`.",
        )
    return resolved_model

def add_routes_to_app(app):
    app.add_api_route(f"{BASE_URL}/sanity-check", sanity_check, methods=["GET"])
    app.add_api_route(f"{BASE_URL}/messages", process_chat_messages, methods=["POST"])
    app.add_api_route(f"{BASE_URL}/count-tokens", count_tokens, methods=["POST"])
    app.add_api_route(f"{BASE_URL}/embeddings", get_embedding_vector, methods=["POST"])

def sanity_check():
    return {"is_sanity_check": True}

async def process_chat_messages(chat_request: ChatRequest):
    try:
        model = _resolve_model(chat_request.model)

        response = await ChatInterface.messages(
            model=model,
            messages=chat_request.messages,
            stream=chat_request.stream,
            modelConfig=chat_request.modelConfig
        )
        return response

    except ValueError as ve:
        raise HTTPException(status_code=501, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

async def count_tokens(count_request: TokenCountRequest):
    try:
        model = _resolve_model(count_request.model)

        token_count = await ChatInterface.count_tokens(
            model=model,
            messages=count_request.messages
        )
        return {"input_tokens": token_count}

    except ValueError as ve:
        raise HTTPException(status_code=501, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

async def get_embedding_vector(embedding_request: EmbeddingRequest):
    try:
        model = _resolve_embedding_model(embedding_request.model)

        embedding_vector = await ChatInterface.embeddings(
            model=model,
            text=embedding_request.text
        )
        return {"embedding": embedding_vector}

    except ValueError as ve:
        raise HTTPException(status_code=501, detail=str(ve))
    except NotImplementedError as nie:
        raise HTTPException(status_code=501, detail="Embedding not implemented for current model/provider setup")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")
