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

    # Vapi wraps all events in a top-level "message" object with a "type" field
    msg = data.get("message", {})
    event_type = msg.get("type", "unknown")
    logger.info(f"Vapi event: {event_type} | keys: {list(msg.keys())}")

    if event_type == "assistant-request":
        return {}

    if event_type == "end-of-call-report":
        logger.info(f"Call ended: {msg.get('endedReason', 'unknown')} | transcript length: {len(msg.get('transcript', ''))}")
        return {}

    if event_type == "conversation-update":
        return {}

    if event_type == "function-call":
        logger.info(f"Function call: {msg.get('functionCall', {}).get('name')}")
        return {}

    logger.info(f"Unhandled Vapi event: {event_type}")
    return {}
