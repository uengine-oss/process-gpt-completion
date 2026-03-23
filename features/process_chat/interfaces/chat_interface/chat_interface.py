from typing import List, Dict, Any
from .factories import LangchainMessageFactory
from fastapi.responses import StreamingResponse
from Usage import usage

from langchain.schema import Generation
from langchain.globals import get_llm_cache
from langchain.schema import BaseMessage
from langchain_core.messages import AIMessageChunk
from llm_factory import create_llm, create_embedding

import hashlib, json, asyncio
import os

ENV = os.getenv("ENV")

def build_prompt_for_cache(model: str, messages: list, model_config: dict) -> str:
    return json.dumps({
        "model": model,
        "messages": messages,
        "model_config": model_config
    }, sort_keys=True, ensure_ascii=False)

def build_llm_string(model: str) -> str:
    return model

class ChatInterface:
    @staticmethod
    def _extract_content_from_response(response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, BaseMessage):
            return response.content
        if hasattr(response, "content"):
            return getattr(response, "content") or ""
        return str(response)

    @staticmethod
    def _extract_content_from_chunk(chunk: Any) -> str:
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, AIMessageChunk):
            return chunk.content or ""
        if hasattr(chunk, "content"):
            return getattr(chunk, "content") or ""
        return ""

    @staticmethod
    def _format_non_stream_response(content: str) -> Dict[str, Any]:
        return {
            "id": f"chatcmpl-{os.urandom(8).hex()}",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }

    @staticmethod
    def _format_stream_chunk(response_id: str, chunk_content: str) -> str:
        payload = {
            "id": response_id,
            "choices": [
                {
                    "delta": {"content": chunk_content},
                    "index": 0,
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    async def messages(model: str, messages: List[Dict[str, Any]], stream: bool, modelConfig: Dict[str, Any]):
        lc_messages = LangchainMessageFactory.create_messages(messages)
        # 요청 프롬프트 토큰 계산
        request_tokens = ChatInterface.count_tokens(model, messages)
        print(f"[DEBUG] Request tokens: {request_tokens}")
        
        def record_usage(total_tokens: int, response_text: str = ""):
            """토큰 사용량을 기록하는 헬퍼 함수"""
            raw_data = {
                "serviceId":       "chat_llm", 
                "tenantId":        "localhost", 
                "userId":          "gpt@gpt.org",
                "startAt":         "2025-08-06T09:00:00+09:00",
                "usage": {
                    model: { "request": request_tokens, "response": total_tokens - request_tokens }
                },
                "process_def_id":  None,
                "process_inst_id": None,
                "agent_id":        None
            }
            try:
                # usage(raw_data)
                print(f"[DEBUG] Usage recorded - Total tokens: {total_tokens} (Request: {request_tokens}, Response: {total_tokens - request_tokens})")
            except Exception as e:
                print(f"[ERROR] Failed to record usage: {e}")
        

        if ENV != "production":
            prompt = build_prompt_for_cache(model, messages, modelConfig)
            llm_string = build_llm_string(model)
            
            cache = get_llm_cache()
            cached_generations = cache.lookup(prompt, llm_string) if cache else None
            
            if cached_generations:
                cached_text = cached_generations[0].text

                async def stream_cached_response(text: str):
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(stream_cached_response(cached_text), media_type="text/event-stream")

        if stream:
            llm = create_llm(model=model, streaming=True, **modelConfig)
            result_text = ""
            response_id = f"chatcmpl-{os.urandom(8).hex()}"
            
            async def streaming_response():
                nonlocal result_text
                async for chunk in llm.astream(lc_messages):
                    content = ChatInterface._extract_content_from_chunk(chunk)
                    if not content:
                        continue
                    result_text += content
                    yield ChatInterface._format_stream_chunk(response_id, content)

                # 스트리밍 완료 후 응답 토큰 계산 및 사용량 기록
                if result_text:
                    response_tokens = ChatInterface.count_tokens(model, [{"role": "assistant", "content": result_text}])
                    total_tokens = request_tokens + response_tokens
                    print(f"[DEBUG] Response tokens: {response_tokens}, Total tokens: {total_tokens}")
                    record_usage(total_tokens, result_text)
                else:
                    print(f"[WARNING] No response text in streaming, recording request tokens only")
                    record_usage(request_tokens, "")

                if ENV != "production":
                    if cache:
                        try:
                            cache.update(prompt, llm_string, [Generation(text=result_text)])
                        except Exception as e:
                            print(f"[cache error] {e}")

                yield "data: [DONE]\n\n"

            return StreamingResponse(streaming_response(), media_type="text/event-stream")

        else:
            llm = create_llm(model=model, streaming=False, **modelConfig)
            raw_response = await llm.ainvoke(lc_messages)
            response = ChatInterface._format_non_stream_response(
                ChatInterface._extract_content_from_response(raw_response)
            )
            
            # 비스트리밍 응답에서 텍스트 추출 및 토큰 계산
            try:
                response_text = ""
                if "choices" in response and len(response["choices"]) > 0:
                    if "message" in response["choices"][0]:
                        response_text = response["choices"][0]["message"].get("content", "")
                    elif "text" in response["choices"][0]:
                        response_text = response["choices"][0]["text"]
                
                if response_text:
                    response_tokens = ChatInterface.count_tokens(model, [{"role": "assistant", "content": response_text}])
                    total_tokens = request_tokens + response_tokens
                    print(f"[DEBUG] Response tokens: {response_tokens}, Total tokens: {total_tokens}")
                    record_usage(total_tokens, response_text)
                else:
                    print(f"[WARNING] No response text found, recording request tokens only")
                    record_usage(request_tokens, "")
            except Exception as e:
                print(f"[ERROR] Failed to calculate response tokens: {e}")
                record_usage(request_tokens, "")

            if ENV != "production":
                if cache:
                    try:
                        cache.update(prompt, llm_string, [Generation(text=response_text)])
                    except Exception as e:
                        print(f"[cache error] {e}")

            return response

    @staticmethod
    def count_tokens(model: str, messages: List[Dict[str, Any]]):
        try:
            lc_messages = LangchainMessageFactory.create_messages(messages)
            llm = create_llm(model=model)
            token_count = llm.get_num_tokens_from_messages(messages=lc_messages)
            print(f"[DEBUG] Token count for {model}: {token_count}")
            return token_count
        except Exception as e:
            print(f"[ERROR] Failed to count tokens for {model}: {str(e)}")
            # 토큰 계산 실패 시 대략적인 추정값 반환
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            estimated_tokens = total_chars // 4  # 대략적인 추정 (4글자 ≈ 1토큰)
            print(f"[DEBUG] Using estimated token count: {estimated_tokens}")
            return estimated_tokens
        
    @staticmethod
    async def embeddings(model: str, text: str):
        embedding_client = create_embedding(model=model)
        return await embedding_client.aembed_query(text)
