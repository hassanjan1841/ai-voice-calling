# AGENTS.md

This file provides guidance to AI coding assistants (OpenCode, Cursor, Gemini CLI, Codex, etc.) when working with this repository.

## Commands

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Run dev server (from backend/ — matches Render rootDir)
cd backend && uvicorn main:app --reload

# Health check
curl http://localhost:8000/health

# Test the Custom LLM endpoint directly (what Vapi calls on every voice turn)
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.1-8b-instant", "stream": false, "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"}]}'

# Test Groq client directly
python -c "import asyncio; from groq_client import groq_client; print(asyncio.run(groq_client.generate([], 'Say hello')))"

# Local tunnel for Vapi webhook testing
ngrok http 8000
```

## Architecture

**Call flow (Custom LLM mode — live and tested):**
```
Customer speaks
  → Vapi STT (Deepgram)
  → POST /chat/completions   ← your FastAPI; receives full OpenAI-format messages[]
  → FastAPI calls Groq (llama-3.1-8b-instant) via LangChain astream
  → SSE chunks streamed back to Vapi
  → Vapi TTS (Elliot voice) speaks the response
```

**Vapi dashboard config (Assistant → Model):**
- Provider: Custom LLM
- Custom LLM URL: `https://<ngrok-id>.ngrok-free.app` (Vapi auto-appends `/chat/completions`)
- Model field: `llama-3.1-8b-instant`

**Two separate endpoints on the server:**
- `POST /chat/completions` — the brain; called by Vapi on every user voice turn; receives OpenAI-format `messages[]`; returns streaming SSE (`stream: true`) or JSON; **RAG logic goes here in Phase 3**
- `POST /webhook` — event notifications only; receives Vapi lifecycle events; always returns `{}`; never calls Groq

**Module responsibilities:**
- `backend/config.py` — singleton `Settings` dataclass; reads `GROQ_API_KEY` and `VAPI_VERIFY_KEY` from `.env`
- `backend/groq_client.py` — singleton `groq_client`; wraps `ChatGroq` (langchain-groq); exposes `.llm` for `ainvoke` / `astream`
- `backend/llm.py` — `POST /chat/completions`; converts OpenAI-format messages → LangChain objects via `_to_langchain()`; streams SSE chunks back to Vapi; **Phase 3 RAG insertion point**
- `backend/webhook.py` — `POST /webhook`; dispatches on `message.type`; handles known events, logs unknown ones, always returns `{}`
- `backend/memory.py` — in-memory `dict[conversation_id → list[dict]]`; resets on restart (placeholder for Redis/Supabase in Phase 4)
- `backend/models.py` — Pydantic schemas: `VapiMessage` (role, content) and `VapiWebhookResponse` (response)
- `backend/main.py` — registers both routers; sets `logging.basicConfig(level=logging.INFO)`

**Vapi webhook event types observed on `POST /webhook`:**
- `assistant-request` — call connects
- `end-of-call-report` — call ends; payload has `endedReason`, `transcript`
- `conversation-update` — message history snapshot; has `messagesOpenAIFormatted`
- `speech-update` — user/assistant started or stopped speaking
- `status-update` — call status change
- `assistant.started` — assistant began its speaking turn
- `user-interrupted` — user spoke over the assistant
- `function-call` — tool call (not yet implemented)

**Signature verification:** `x-vapi-signature` is HMAC-SHA256 of raw body using `VAPI_VERIFY_KEY`. Logs a warning but does NOT reject requests when verification fails. Set `VAPI_VERIFY_KEY=` (empty) to silence warnings during local dev.

**Import paths:** All modules use flat imports (`from config import settings`, no `backend.` prefix). Server must be started from `backend/` directory. On Render, `rootDir: backend` handles this automatically.

## Phased Roadmap

| Phase | Status | What it adds |
|---|---|---|
| 1 | **Complete** | Voice skeleton — Vapi Custom LLM → FastAPI → Groq, live and tested |
| 2 | **Next** | Ingestion — PDF/txt → 500-char chunks → FastEmbed (BAAI/bge-small-en-v1.5) → Supabase pgvector |
| 3 | Planned | RAG — embed user query → pgvector similarity search → inject chunks into `/chat/completions` prompt |
| 4 | Planned | Lead capture — entity extraction, SQLAlchemy models (`leads`, `appointments`, `call_logs`), CRUD routes |
| 5 | Planned | Next.js dashboard |

**Phase 3 RAG insertion point:** `backend/llm.py` → `chat_completions()` — before calling `groq_client.llm.astream()`, embed the last user message, query pgvector for top-k chunks, prepend them as a context message.

Full planned file structure and tech stack: `docs/superpowers/specs/2026-05-08-ai-voice-calling-agent-design.md`.

## Environment Variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key (`gsk_...`) |
| `VAPI_VERIFY_KEY` | HMAC key for Vapi webhook signature verification; leave empty during local dev |

## Deployment Target

Render Web Service: `uvicorn main:app --host 0.0.0.0 --port $PORT` (runs from `backend/` rootDir).
