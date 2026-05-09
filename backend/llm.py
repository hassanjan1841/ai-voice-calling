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


def _truncate(text: str, max_len: int = 60) -> str:
    if not text:
        return ""
    first_line = text.split("\n")[0]
    if len(first_line) > max_len:
        return first_line[:max_len] + "..."
    return first_line


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

    turn_id = uuid.uuid4().hex[:8]
    sep = "═" * 55

    # ── Header ──────────────────────────────────────────────
    logger.info("")
    logger.info("╔" + sep + "╗")
    logger.info(f"║  NEW VOICE TURN          turn_{turn_id}  ║")
    logger.info("╚" + sep + "╝")

    # ── Inbound from Vapi ───────────────────────────────────
    mode = "streaming" if streaming else "non-streaming"
    logger.info(f"  ═══════════════════════════════════════════════")
    logger.info(f"  ⬡ VAPI → SERVER   POST /chat/completions ({mode})")
    logger.info(f"  ⬡ VAPI → SERVER   Model: {model_name}")

    last_user_msg = ""
    last_assistant_msg = ""
    history_count = len(messages)
    if messages:
        last = messages[-1]
        if last.get("role") == "user":
            last_user_msg = last.get("content", "")
        for msg in reversed(messages[:-1]):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg.get("content", "")
                break

    if last_user_msg:
        logger.info(f"  🗣 User replied:     {_truncate(last_user_msg)}")
    if last_assistant_msg:
        logger.info(f"  🤖 Vapi just said:   {_truncate(last_assistant_msg)}")
    logger.info(f"  📚 Conversation history: {history_count} messages total")

    # ── Build LangChain messages ─────────────────────────────
    langchain_messages = _to_langchain(messages)
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Outbound to Groq ─────────────────────────────────────
    logger.info(f"  ═══════════════════════════════════════════════")
    logger.info(f"  ⬡ SERVER → GROQ    Sending {history_count} messages to {model_name}...")

    if streaming:
        async def generate():
            chunk_count = 0
            total_chars = 0
            first = True

            async for chunk in groq_client.llm.astream(langchain_messages):
                if chunk.content:
                    chunk_count += 1
                    total_chars += len(chunk.content)

                    if first:
                        logger.info(f"  ⬡ GROQ → SERVER    Response received — streaming...")
                        first = False

                    payload = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": chunk.content}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            # ── Stream complete ──────────────────────────────
            logger.info(f"  ⬡ SERVER → VAPI    Streaming complete: {chunk_count} chunks, {total_chars} chars")
            logger.info("╔" + sep + "╗")
            logger.info(f"║  TURN COMPLETE           turn_{turn_id}            ║")
            logger.info("╚" + sep + "╝")
            logger.info("")

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

    # ── Non-streaming path ────────────────────────────────────
    response = await groq_client.llm.ainvoke(langchain_messages)
    response_text = response.content
    logger.info(f"  ⬡ GROQ → SERVER    Response: {_truncate(response_text)}")
    logger.info(f"  ⬡ SERVER → VAPI    Sent ({len(response_text)} chars)")
    logger.info("╔" + sep + "╗")
    logger.info(f"║  TURN COMPLETE           turn_{turn_id}            ║")
    logger.info("╚" + sep + "╝")
    logger.info("")

    return JSONResponse({
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"content": response_text}, "finish_reason": "stop"}],
    })
