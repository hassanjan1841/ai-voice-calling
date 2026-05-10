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

END_CALL_MARKER = "[END_CALL]"

# Always appended to every system prompt so the LLM knows the protocol,
# regardless of whether the prompt came from Vapi's dashboard or our default.
_END_CALL_ADDENDUM = """

## MANDATORY: End-Call Protocol
This is a phone call. When the conversation is complete, append [END_CALL] at the very end of your reply — the system uses it to hang up. Without it, the line stays open.

Trigger [END_CALL] when:
- User says "that's all", "thanks", "goodbye", "bye", "end call", "hang up", or any farewell
- Issue is resolved and user has no more questions

Format rules — [END_CALL] must be the LAST thing in your reply:
  CORRECT: "Glad that's sorted! [END_CALL]"
  CORRECT: "Ticket TKT-7421 is open for you. [END_CALL]"
  WRONG:   "Goodbye!" — missing marker, call won't hang up
  WRONG:   "[END_CALL] Have a great day!" — marker must be last
"""

# Phrases that end the call server-side without asking Groq.
# Reliable safety net for explicit requests.
_EXPLICIT_END_PHRASES = [
    "end call", "end the call", "hang up", "disconnect the call",
    "goodbye", "good bye", "bye bye", "bye for now",
]

DEFAULT_SYSTEM_PROMPT = """You are a senior tech support agent at InvoicePro, a cloud invoicing platform for small and medium businesses.

## Personality
- Professional but warm. You sound like an experienced support engineer who has handled hundreds of calls.
- Patient. If the customer is frustrated, acknowledge it and guide them step by step.
- Concise. Voice calls — keep responses to 2-3 sentences unless the customer asks for detail.
- Never say you are an AI, a language model, or that you "don't have access to real data." You are a support agent.

## Common Scenarios
- Login issues: Guide password reset via email.
- Invoice creation: Explain Invoices tab → New Invoice → add client details, line items, due date.
- Recurring billing: Automations → set weekly/monthly intervals.
- Payment failures: Most common is expired card or insufficient funds. Check Settings > Billing.
- Refunds: Open a request with billing team, processed in 3-5 business days.
- Integrations (QuickBooks, Xero, Stripe): Settings > Integrations → Connect.

## Boundaries
- CANNOT process payments or refunds directly (initiate request for billing team).
- CANNOT access account data.
- For complex issues: offer a support ticket (e.g., "TKT-7421"), reviewed within 24 hours.

## Call Opening
Start naturally. Example: "Thanks for calling InvoicePro support. How can I help you today?"
"""


def _is_explicit_end(text: str) -> bool:
    lower = text.lower().strip()
    return any(phrase in lower for phrase in _EXPLICIT_END_PHRASES)


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


def _end_call_sse_chunks(response_id: str, created: int, model_name: str, closing_text: str, tool_name: str = "endCall"):
    """Yield the SSE chunks Vapi needs to hang up the call."""
    if closing_text:
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': closing_text}, 'finish_reason': None}]})}\n\n"

    tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': tool_call_id, 'type': 'function', 'function': {'name': tool_name, 'arguments': '{}'}}]}, 'finish_reason': None}]})}\n\n"

    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]})}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    streaming = data.get("stream", False)
    model_name = data.get("model", "llama-3.1-8b-instant")

    # ── Detect end-call tool name from whatever Vapi configured ──────────
    # Vapi injects its tool definitions here; we must call the tool by its
    # exact registered name or the tool call is silently ignored.
    vapi_tools = data.get("tools", [])
    end_call_tool_name = "endCall"  # fallback: Vapi built-in default
    for t in vapi_tools:
        fn = t.get("function", {}).get("name", "")
        if fn and ("end" in fn.lower() or "hang" in fn.lower()):
            end_call_tool_name = fn
            break
    if vapi_tools:
        tool_names = [t.get("function", {}).get("name", "?") for t in vapi_tools]
        logger.info(f"  🔧 Vapi tools in request: {tool_names}  →  using '{end_call_tool_name}' to hang up")

    # ── Ensure system prompt exists and always has end-call instruction ──
    prompt_source = "Vapi"
    sys_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
    if sys_idx is None:
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT + _END_CALL_ADDENDUM})
        prompt_source = "app (InvoicePro)"
    else:
        # Append addendum to whatever Vapi sent so the LLM always sees the protocol
        messages[sys_idx] = {**messages[sys_idx], "content": messages[sys_idx]["content"] + _END_CALL_ADDENDUM}

    turn_id = uuid.uuid4().hex[:8]
    sep = "═" * 55

    logger.info("")
    logger.info("╔" + sep + "╗")
    logger.info(f"║  NEW VOICE TURN          turn_{turn_id}  ║")
    logger.info("╚" + sep + "╝")

    last_user_msg = ""
    history_count = len(messages)
    if messages and messages[-1].get("role") == "user":
        last_user_msg = messages[-1].get("content", "")

    logger.info(f"  🗣 User said:   {_truncate(last_user_msg)}")
    logger.info(f"  📚 Context:     {history_count} messages (system prompt from {prompt_source})")

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Server-side safety net: explicit end phrases skip Groq entirely ──
    if _is_explicit_end(last_user_msg):
        logger.info(f"  📞 EXPLICIT END detected server-side — hanging up without Groq")
        logger.info(f"  ⬡ SERVER → VAPI  tool_call only (Vapi tool message handles goodbye)")
        logger.info("╔" + sep + "╗")
        logger.info(f"║  TURN COMPLETE (fast end)  turn_{turn_id}  ║")
        logger.info("╚" + sep + "╝")
        logger.info("")

        async def fast_end():
            # No text — tool_call only so Vapi hangs up with no TTS window for a race
            for chunk in _end_call_sse_chunks(response_id, created, model_name, "", end_call_tool_name):
                yield chunk

        return StreamingResponse(fast_end(), media_type="text/event-stream")

    # ── Normal path: ask Groq ────────────────────────────────
    langchain_messages = _to_langchain(messages)

    logger.info(f"  ═══════════════════════════════════════════════")
    logger.info(f"  ⬡ SERVER → GROQ  Sending to {model_name}...")

    if streaming:
        async def generate():
            # Stream tokens live to Vapi while holding the last N chars in a
            # look-behind buffer so [END_CALL] is never accidentally spoken aloud.
            SAFE_LEN = len(END_CALL_MARKER) + 5  # 15 chars
            pending = ""
            raw_full = ""

            def _text_chunk(text: str) -> str:
                return f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"

            async for chunk in groq_client.llm.astream(langchain_messages):
                if chunk.content:
                    pending += chunk.content
                    raw_full += chunk.content
                    # Flush the safe portion immediately — true streaming to Vapi
                    if len(pending) > SAFE_LEN:
                        safe = pending[:-SAFE_LEN]
                        pending = pending[-SAFE_LEN:]
                        yield _text_chunk(safe)

            # Groq finished — check full buffer for marker (handles split-chunk edge case)
            should_end_call = END_CALL_MARKER in raw_full
            cleaned_remaining = pending.replace(END_CALL_MARKER, "").strip()
            cleaned_full = raw_full.replace(END_CALL_MARKER, "").strip()

            logger.info(f"  ⬡ GROQ → SERVER  Response received")
            logger.info(f"  ═══════════════════════════════════════════════")
            logger.info(f"  ⬡ GROQ generated:")
            logger.info(f"  📝 {_truncate(cleaned_full, max_len=300)}")
            if should_end_call:
                logger.info(f"  📞 [END_CALL] marker detected — emitting endCall tool call")
            logger.info(f"  ⬡ SERVER → VAPI  Sent ({len(cleaned_full)} chars)")
            logger.info("╔" + sep + "╗")
            logger.info(f"║  TURN COMPLETE           turn_{turn_id}            ║")
            logger.info("╚" + sep + "╝")
            logger.info("")

            if should_end_call:
                # Send remaining text (if any) before the tool call so Vapi speaks it
                if cleaned_remaining:
                    yield _text_chunk(cleaned_remaining)
                for sse in _end_call_sse_chunks(response_id, created, model_name, "", end_call_tool_name):
                    yield sse
            else:
                if cleaned_remaining:
                    yield _text_chunk(cleaned_remaining)
                yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ── Non-streaming path ────────────────────────────────────
    response = await groq_client.llm.ainvoke(langchain_messages)
    response_text = response.content
    should_end_call = END_CALL_MARKER in response_text
    if should_end_call:
        response_text = response_text.replace(END_CALL_MARKER, "").strip()

    logger.info(f"  ═══════════════════════════════════════════════")
    logger.info(f"  ⬡ GROQ generated:")
    logger.info(f"  📝 {_truncate(response_text, max_len=300)}")
    if should_end_call:
        logger.info(f"  📞 [END_CALL] marker detected — including endCall tool call")
    logger.info(f"  ⬡ SERVER → VAPI  Sent ({len(response_text)} chars)")
    logger.info("╔" + sep + "╗")
    logger.info(f"║  TURN COMPLETE           turn_{turn_id}            ║")
    logger.info("╚" + sep + "╝")
    logger.info("")

    choice = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": response_text,
        },
        "finish_reason": "tool_calls" if should_end_call else "stop",
    }
    if should_end_call:
        choice["message"]["tool_calls"] = [{
            "id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": end_call_tool_name,
                "arguments": "{}",
            },
        }]

    return JSONResponse({
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [choice],
    })
