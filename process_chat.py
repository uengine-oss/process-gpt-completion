import json
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from features.process_chat import (
    BASE_URL,
    ChatRequest,
    TokenCountRequest,
    EmbeddingRequest,
    ChatInterface,
)
from features.process_chat.interfaces.chat_interface.clients import ClientFactory
from features.process_chat.interfaces.chat_interface.factories import LangchainMessageFactory
from process_definition import load_process_definition


DEFAULT_MODEL_CONFIG = {
    "temperature": 0.2,
    "top_p": 0.9,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


class ProcessCopilotRequest(BaseModel):
    question: str
    process_definition: Any
    process_definition_id: Optional[str] = None
    process_name: Optional[str] = None
    activity_id: Optional[str] = None
    activity_name: Optional[str] = None
    selected_element: Optional[Dict[str, Any]] = None
    locale: str = "ko"
    vendor: str = "openai"
    model: str = "gpt-4.1-2025-04-14"
    modelConfig: Dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_MODEL_CONFIG))


def add_routes_to_app(app):
    app.add_api_route(f"{BASE_URL}/sanity-check", sanity_check, methods=["GET"])
    app.add_api_route(f"{BASE_URL}/messages", process_chat_messages, methods=["POST"])
    app.add_api_route(f"{BASE_URL}/count-tokens", count_tokens, methods=["POST"])
    app.add_api_route(f"{BASE_URL}/embeddings", get_embedding_vector, methods=["POST"])
    app.add_api_route(f"{BASE_URL}/process-copilot", process_copilot, methods=["POST"])


def sanity_check():
    return {"is_sanity_check": True}


def _truncate(value: Any, limit: int = 280) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _safe_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _serialize_role(role: Any) -> Dict[str, Any]:
    return {
        "name": getattr(role, "name", "") or "",
        "resolutionRule": getattr(role, "resolutionRule", None),
        "endpoint": getattr(role, "endpoint", None),
    }


def _serialize_data_item(data_item: Any) -> Dict[str, Any]:
    data_source = getattr(data_item, "dataSource", None)
    return {
        "name": getattr(data_item, "name", "") or "",
        "type": getattr(data_item, "type", "") or "",
        "description": getattr(data_item, "description", None),
        "table": getattr(data_item, "table", None),
        "data_source_type": getattr(data_source, "type", None) if data_source else None,
    }


def _describe_data_refs(process_def: Any, refs: Optional[List[str]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    ref_names = refs or []
    data_map = {
        getattr(item, "name", ""): _serialize_data_item(item)
        for item in (getattr(process_def, "data", None) or [])
        if getattr(item, "name", None)
    }

    for ref_name in ref_names:
        if ref_name in data_map:
            result.append(data_map[ref_name])
        else:
            result.append({"name": ref_name})

    return result


def _serialize_activity(process_def: Any, activity: Any) -> Dict[str, Any]:
    properties = _safe_json(getattr(activity, "properties", None))
    return {
        "id": getattr(activity, "id", "") or "",
        "name": getattr(activity, "name", "") or "",
        "type": getattr(activity, "type", "") or "",
        "role": getattr(activity, "role", "") or "",
        "description": _truncate(getattr(activity, "description", "")),
        "instruction": _truncate(getattr(activity, "instruction", "")),
        "tool": getattr(activity, "tool", None),
        "agent": getattr(activity, "agent", None),
        "agent_mode": getattr(activity, "agentMode", None),
        "orchestration": getattr(activity, "orchestration", None),
        "checkpoints": getattr(activity, "checkpoints", None) or [],
        "input_data": _describe_data_refs(process_def, getattr(activity, "inputData", None) or []),
        "output_data": _describe_data_refs(process_def, getattr(activity, "outputData", None) or []),
        "properties": {
            "manualLink": properties.get("manualLink"),
            "systems": properties.get("systems") if isinstance(properties.get("systems"), list) else [],
            "checkpoints": properties.get("checkpoints") if isinstance(properties.get("checkpoints"), list) else [],
        },
    }


def _build_process_context(request: ProcessCopilotRequest) -> Dict[str, Any]:
    definition_json = request.process_definition
    if isinstance(definition_json, str):
        definition_json = json.loads(definition_json)
    if not isinstance(definition_json, dict):
        raise ValueError("process_definition must be a JSON object.")

    process_def = load_process_definition(definition_json)
    current_activity = process_def.find_activity_by_id(request.activity_id) if request.activity_id else None
    previous_activities = process_def.find_immediate_prev_activities(request.activity_id) if request.activity_id else []
    next_activities = process_def.find_next_activities(request.activity_id) if request.activity_id else []
    roles = [_serialize_role(role) for role in (process_def.roles or [])]
    data_items = [_serialize_data_item(item) for item in (process_def.data or [])]
    activity_catalog = [
        {
            "id": getattr(activity, "id", "") or "",
            "name": getattr(activity, "name", "") or "",
            "type": getattr(activity, "type", "") or "",
            "role": getattr(activity, "role", "") or "",
        }
        for activity in (process_def.activities or [])
    ]

    selected_element = request.selected_element or {}
    process_name = (
        request.process_name
        or definition_json.get("processDefinitionName")
        or getattr(process_def, "processDefinitionName", "")
        or "Unknown Process"
    )
    process_definition_id = (
        request.process_definition_id
        or definition_json.get("processDefinitionId")
        or getattr(process_def, "processDefinitionId", "")
        or ""
    )

    return {
        "process": {
            "id": process_definition_id,
            "name": process_name,
            "description": definition_json.get("description"),
            "activity_count": len(process_def.activities or []),
            "sequence_count": len(process_def.sequences or []),
            "roles": roles,
            "data_items": data_items,
            "activity_catalog": activity_catalog,
        },
        "selected_element": selected_element,
        "current_activity": _serialize_activity(process_def, current_activity) if current_activity else None,
        "previous_activities": [_serialize_activity(process_def, activity) for activity in previous_activities],
        "next_activities": [_serialize_activity(process_def, activity) for activity in next_activities],
    }


def _build_process_copilot_messages(context: Dict[str, Any], question: str, locale: str) -> List[Dict[str, str]]:
    response_language = "Korean" if str(locale or "ko").lower().startswith("ko") else "English"
    system_prompt = (
        "You are ProcessGPT BPMN Copilot.\n"
        "Answer only from the BPMN context provided.\n"
        "If information is missing in the context, say so clearly.\n"
        "Explain the current activity, previous/next activities, roles, data, and checkpoints when relevant.\n"
        "Do not invent manuals, systems, approvals, or owners that are not present in the context.\n"
        f"Respond in {response_language}."
    )
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"BPMN context JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def _invoke_llm(vendor: str, model: str, messages: List[Dict[str, Any]], model_config: Dict[str, Any]) -> str:
    client = ClientFactory.get_client(vendor)
    lc_messages = LangchainMessageFactory.create_messages(messages)
    response = await client.invoke(messages=lc_messages, model=model, modelConfig=model_config)
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return message.get("content", "") or ""


async def process_chat_messages(chat_request: ChatRequest):
    try:
        response = await ChatInterface.messages(
            vendor=chat_request.vendor,
            model=chat_request.model,
            messages=chat_request.messages,
            stream=chat_request.stream,
            modelConfig=chat_request.modelConfig,
        )
        return response
    except ValueError as ve:
        raise HTTPException(status_code=501, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


async def count_tokens(count_request: TokenCountRequest):
    try:
        token_count = await ChatInterface.count_tokens(
            vendor=count_request.vendor,
            model=count_request.model,
            messages=count_request.messages,
        )
        return {"input_tokens": token_count}
    except ValueError as ve:
        raise HTTPException(status_code=501, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


async def get_embedding_vector(embedding_request: EmbeddingRequest):
    try:
        embedding_vector = await ChatInterface.embeddings(
            vendor=embedding_request.vendor,
            model=embedding_request.model,
            text=embedding_request.text,
        )
        return {"embedding": embedding_vector}
    except ValueError as ve:
        raise HTTPException(status_code=501, detail=str(ve))
    except NotImplementedError:
        raise HTTPException(status_code=501, detail=f"Embedding not implemented for vendor: {embedding_request.vendor}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")


async def process_copilot(request: ProcessCopilotRequest):
    try:
        context = _build_process_context(request)
        messages = _build_process_copilot_messages(context, request.question, request.locale)
        model_config = dict(DEFAULT_MODEL_CONFIG)
        model_config.update(request.modelConfig or {})
        answer = await _invoke_llm(
            vendor=request.vendor,
            model=request.model,
            messages=messages,
            model_config=model_config,
        )
        return {
            "answer": answer,
            "context": context,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating BPMN copilot answer: {str(e)}")
