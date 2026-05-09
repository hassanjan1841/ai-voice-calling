from fastapi import APIRouter, HTTPException, Request
import hashlib
import hmac
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def verify_vapi_signature(request: Request, body: bytes) -> bool:
    from config import settings

    if not settings.vapi_verify_key:
        return True

    signature = request.headers.get("x-vapi-signature", "")
    if not signature:
        return False

    computed = hmac.new(
        settings.vapi_verify_key.encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={computed}", signature)


@router.post("/webhook")
async def handle_webhook(request: Request):
    body = await request.body()

    if not verify_vapi_signature(request, body):
        logger.warning("Vapi signature verification failed")

    try:
        data = json.loads(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    msg = data.get("message", {})
    event_type = msg.get("type", "unknown")
    call_id = msg.get("call", {}).get("id", "?") if isinstance(msg.get("call"), dict) else "?"

    icons = {
        "assistant-request": "📞",
        "end-of-call-report": "📋",
        "conversation-update": "💬",
        "speech-update": "🎤",
        "status-update": "🔵",
        "assistant.started": "🤖",
        "user-interrupted": "✋",
        "function-call": "🔧",
    }
    icon = icons.get(event_type, " •")

    logger.info(f"  {icon} VAPI → WEBHOOK  {event_type}  (call: {call_id})")

    if event_type == "assistant-request":
        return {}

    if event_type == "end-of-call-report":
        reason = msg.get("endedReason", "unknown")
        transcript = msg.get("transcript", "")
        logger.info(f"  {icon} VAPI → WEBHOOK  Call ended: {reason} | transcript: {len(transcript)} chars")
        return {}

    if event_type == "conversation-update":
        return {}

    if event_type == "function-call":
        fn_name = msg.get("functionCall", {}).get("name", "?")
        logger.info(f"  {icon} VAPI → WEBHOOK  Function call: {fn_name}")
        return {}

    if event_type in ("status-update", "speech-update", "assistant.started", "user-interrupted"):
        return {}

    logger.info(f"  {icon} VAPI → WEBHOOK  Unhandled: {event_type}")
    return {}
