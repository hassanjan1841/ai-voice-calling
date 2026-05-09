from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from groq_client import groq_client
import json
import time
import uuid
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def _to_langchain(messages: list[dict]) -> list:
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        elif role == "assistant":
            result.append(AIMessage(content=content))
    return result


@router.post("/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    streaming = data.get("stream", False)
    model_name = data.get("model", "llama-3.1-8b-instant")

    logger.info(f"Custom LLM: stream={streaming}, messages={len(messages)}")

    langchain_messages = _to_langchain(messages)
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if streaming:
        async def generate():
            async for chunk in groq_client.llm.astream(langchain_messages):
                if chunk.content:
                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": chunk.content}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            final = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    response = await groq_client.llm.ainvoke(langchain_messages)
    return JSONResponse({
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"content": response.content}, "finish_reason": "stop"}],
    })
