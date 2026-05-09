# Phase 1 — Voice Call Skeleton

## Overview

Phase 1 delivers a working AI voice assistant that answers phone calls. A caller dials a number, speaks, and the AI responds in real time. The entire pipeline runs on your own backend — no vendor lock-in on the LLM, the prompts, or the memory.

**Status:** Complete, live-tested with real calls, deployed to Render.

---

## End-to-End Call Flow

```
Caller dials number
    │
    ▼
Vapi (telephony provider)
    │── Deepgram STT (speech → text)
    │── Sends messages[] to your server
    │
    ▼
POST /chat/completions (your FastAPI server)
    │── Converts OpenAI-format messages → LangChain format
    │── Streams to Groq API (Llama 3.1-8b-instant)
    │── Returns SSE chunks back to Vapi
    │
    ▼
Vapi TTS (Elliot voice, text → speech)
    │
    ▼
Caller hears AI response
```

Round-trip latency is sub-second due to Groq's LPU hardware.

---

## Tech Stack

| Component | Technology | Role |
|---|---|---|
| **Backend** | Python 3.12, FastAPI | HTTP server, API routing |
| **LLM Provider** | Groq (Llama 3.1-8b-instant) | AI response generation |
| **LLM Framework** | LangChain (langchain-groq) | Message formatting, streaming |
| **Voice** | Vapi (Custom LLM mode) | Inbound calls, STT, TTS |
| **STT** | Deepgram (via Vapi) | Speech-to-text |
| **TTS** | ElevenLabs Elliot (via Vapi) | Text-to-speech |
| **Hosting** | Render (Free Tier) | Production deployment |
| **Serialization** | Pydantic v2 | Request/response schemas |
| **Config** | python-dotenv | Environment variable loading |

---

## Module-by-Module Breakdown

### `backend/main.py` — Application Entry Point

The FastAPI app instance. Registers both routers (`webhook` and `llm`), sets up logging, and exposes a `/health` endpoint.

**Responsibilities:**
- Create the `FastAPI` app with title "AI Voice Agent"
- Include `webhook_router` (Vapi events at `/webhook`)
- Include `llm_router` (LLM chat at `/chat/completions`)
- Configure `logging.basicConfig(level=logging.INFO)`

**Endpoints:**
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check; returns `{"status": "ok", "model": "llama-3.1-8b-instant"}` |

---

### `backend/config.py` — Central Configuration

Singleton `Settings` dataclass. Loaded once at import time and shared across all modules via `from config import settings`.

**Fields:**
| Field | Source | Default | Purpose |
|---|---|---|---|
| `groq_api_key` | `GROQ_API_KEY` env var | `""` | Authenticates with Groq API |
| `groq_model` | Hardcoded | `"llama-3.1-8b-instant"` | Model name sent to Groq |
| `vapi_verify_key` | `VAPI_VERIFY_KEY` env var | `""` | HMAC key for webhook verification |
| `host` | Hardcoded | `"0.0.0.0"` | Server bind address |
| `port` | Hardcoded | `8000` | Server port |

Calls `load_dotenv()` at import time to load `.env` file from `backend/` directory.

---

### `backend/groq_client.py` — Groq LLM Wrapper

Wraps LangChain's `ChatGroq` into a singleton client with a business-appropriate system prompt.

**Class `GroqClient`:**
| Method | Input | Output | What It Does |
|---|---|---|---|
| `__init__` | — | — | Creates `ChatGroq` with model, temperature=0.3, max_retries=2 |
| `build_messages` | conversation_history, new_message | LangChain message list | Prepends system prompt + history + new message |
| `generate` | conversation_history, new_message | str (response text) | Calls `llm.ainvoke()` and returns content |

**System prompt:**
```
You are a helpful voice assistant for a business.
You answer questions naturally and conversationally — this is a phone call,
so keep responses concise. If you don't know something, say so.
Do not make up information. Be polite, professional, and friendly.
```

**Singleton:** `groq_client = GroqClient()` is created at module import time.

---

### `backend/llm.py` — Custom LLM API Endpoint

Implements `POST /chat/completions` — the endpoint Vapi calls on every voice turn. It follows the OpenAI chat completions format for compatibility.

**`_to_langchain(messages)` helper:**
Converts OpenAI-format message dicts to LangChain message objects:
- `{role: "system"}` → `SystemMessage`
- `{role: "user"}` → `HumanMessage`
- `{role: "assistant"}` → `AIMessage`

**`POST /chat/completions` — Request format (what Vapi sends):**
```json
{
  "model": "llama-3.1-8b-instant",
  "stream": true,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Tell me about your services"}
  ]
}
```

**Streaming mode (`stream: true`) — Response format (SSE):**
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3.1-8b-instant","choices":[{"index":0,"delta":{"content":"Sure"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3.1-8b-instant","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1234567890,"model":"llama-3.1-8b-instant","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Non-streaming mode (`stream: false`):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama-3.1-8b-instant",
  "choices": [{"index": 0, "delta": {"content": "Sure!"}, "finish_reason": "stop"}]
}
```

**Phase 3 insertion point:** RAG context injection happens here, before `groq_client.llm.astream()` is called.

---

### `backend/webhook.py` — Vapi Event Handler

Processes lifecycle events from Vapi. Vapi sends these as POST requests for call state changes.

**`verify_vapi_signature(request, body)` — Security:**
Computes HMAC-SHA256 of the raw request body using `VAPI_VERIFY_KEY` and compares against the `x-vapi-signature` header. Logs a warning on mismatch but does NOT reject requests (non-blocking in Phase 1).

**`POST /webhook` — Observed event types:**

| Event Type | When It Fires | Payload Highlights |
|---|---|---|
| `assistant-request` | Call connects | Assistant config, model settings |
| `end-of-call-report` | Call ends | `endedReason`, `transcript`, duration |
| `conversation-update` | After each turn | Full message history, `messagesOpenAIFormatted` |
| `speech-update` | User/assistant starts/stops speaking | `status` (started/stopped), `role` |
| `status-update` | Call status changes | `status` value |
| `assistant.started` | Assistant begins its turn | New assistant config |
| `user-interrupted` | User speaks over the assistant | — |
| `function-call` | Tool call triggered | Function name, arguments |

All events return `{}` (200 OK). No event triggers a Groq call — that's what `/chat/completions` is for.

---

### `backend/memory.py` — Conversation Memory (Placeholder)

In-memory dictionary store for conversation history. Keyed by `conversation_id`, each entry is a `list[dict]` of `{role, content}` messages.

**Functions:**
| Function | Purpose |
|---|---|
| `get_history(conversation_id)` | Returns message list for a conversation (creates if absent) |
| `append_message(conversation_id, role, content)` | Appends a message to a conversation |

**Limitations (documented for Phase 4):**
- Data resets on server restart (in-memory, not persistent)
- Not shared across multiple server instances
- Placeholder for Redis / Supabase / PostgreSQL in Phase 4

---

### `backend/models.py` — Pydantic Schemas

Defines data shapes for request/response validation.

```python
class VapiMessage(BaseModel):
    role: str       # "user" | "assistant" | "system"
    content: str    # message text

class VapiWebhookResponse(BaseModel):
    response: str   # webhook acknowledgement
```

Currently lightweight — will grow with entity extraction schemas in Phase 4.

---

### `backend/requirements.txt` — Python Dependencies

```
fastapi>=0.115.0
uvicorn[standard]>=0.30.6
langchain-groq>=1.0
langchain-core>=1.3
pydantic>=2.9.0
python-dotenv>=1.0.1
```

All pins use `>=` (not `==`) to let pip use pre-built wheels and avoid Rust compilation (pydantic-core) during Render builds.

### `backend/runtime.txt` — Python Version Pin

```
python-3.12
```

Pinned for widest wheel compatibility across build environments.

### `backend/__init__.py` — Package Marker

Empty file. Marks `backend/` as a Python package so imports work.

### `.gitignore` — Git Exclusions

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg
.venv/
venv/
env/

# Environment
.env
*.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# ngrok
ngrok.yml
```

---

## API Endpoints Summary

| Method | Path | Called By | Purpose | Response |
|---|---|---|---|---|
| `GET` | `/health` | You (or Render health checks) | Server liveness | `{"status": "ok", "model": "..."}` |
| `POST` | `/chat/completions` | Vapi (every voice turn) | Generate AI response | SSE stream or JSON |
| `POST` | `/webhook` | Vapi (lifecycle events) | Receive call events | `{}` (always) |

---

## Deployment

### Hosting: Render (Free Tier)

- **Service type:** Web Service (non-Blueprint)
- **Root directory:** `backend/`
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Auto-deploy:** On every push to `main` branch
- **URL:** `https://ai-voice-calling-l50m.onrender.com`

### Environment Variables (set in Render dashboard)

| Variable | Value | Purpose |
|---|---|---|
| `GROQ_API_KEY` | `gsk_...` | Authenticate with Groq API |
| `VAPI_VERIFY_KEY` | UUID string | Verify Vapi webhook signatures |

### Git History

```
2e46dd5 Update AGENTS.md: dev command and import convention
b4e8e92 Fix imports: strip backend. prefix for Render rootDir
c8b0198 Remove render.yaml — unused by non-Blueprint Render services
9da0725 Fix render.yaml: paths relative to backend/ root directory
e44da68 Fix Render build: relax version pins, pin Python 3.12
7e4970f Phase 1: voice call skeleton — Vapi Custom LLM -> FastAPI -> Groq, live and tested
```

### Key Deployment Decisions

| Decision | Rationale |
|---|---|
| **Deleted `render.yaml`** | Regular Web Services ignore `render.yaml`. Only Blueprints read it. Dashboard config is source of truth. |
| **Stripped `backend.` import prefix** | Since `rootDir: backend/`, Python runs from inside `backend/`. Flat imports (`from config import settings`) match both local dev and Render. |
| **Relaxed `==` to `>=`** | Pre-built wheels avoid Rust compilation failures on Render's Free tier. |

---

## Vapi Configuration

**Assistant → Model settings:**

| Setting | Value |
|---|---|
| Provider | Custom LLM |
| Custom LLM URL | `https://your-server.com` (Vapi auto-appends `/chat/completions`) |
| Model | `llama-3.1-8b-instant` |
| Stream | Enabled |

**Assistant → Voice settings:**

| Setting | Value |
|---|---|
| Provider | ElevenLabs |
| Voice | Elliot |
| Model | eleven_multilingual_v2 |

**Assistant → Transcriber:**

| Setting | Value |
|---|---|
| Provider | Deepgram |
| Model | nova-2 (general) |

---

## Files in the Repository

```
/
├── .gitignore                  # Git exclusion rules
├── AGENTS.md                   # AI assistant guidance (commands, architecture, roadmap)
│
└── backend/
    ├── __init__.py             # Package marker (empty)
    ├── main.py                 # FastAPI app, /health endpoint, router registration
    ├── config.py               # Settings singleton (env vars → dataclass)
    ├── groq_client.py          # Groq LLM wrapper (ChatGroq + system prompt)
    ├── llm.py                  # POST /chat/completions (streaming SSE + non-streaming JSON)
    ├── webhook.py              # POST /webhook (Vapi lifecycle events)
    ├── memory.py               # In-memory conversation store (placeholder)
    ├── models.py               # Pydantic schemas
    ├── requirements.txt        # Python dependencies (relaxed pins)
    ├── runtime.txt             # Python 3.12 pin
    └── .env                    # Local environment variables (gitignored)
```

---

## What's Next (Roadmap)

| Phase | Status | What It Adds |
|---|---|---|
| **1** | ✅ Complete | Voice skeleton — Vapi → FastAPI → Groq, live and tested |
| **2** | 🔜 Next | Ingestion — PDF/txt → 500-char chunks → FastEmbed → Supabase pgvector |
| **3** | 📋 Planned | RAG — embed user query → pgvector similarity search → inject context into `/chat/completions` |
| **4** | 📋 Planned | Lead capture — entity extraction, SQLAlchemy models, CRUD routes |
| **5** | 📋 Planned | Next.js dashboard — call logs, lead management, analytics |

---

## AGENTS.md Status

**AGENTS.md is updated.** It reflects all Phase 1 changes:

- ✅ Dev command: `cd backend && uvicorn main:app --reload`
- ✅ Flat imports: `from config import settings` (no `backend.` prefix)
- ✅ Render start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- ✅ No `render.yaml` mentioned (removed)
- ✅ Architecture diagram matches live deployment
- ✅ All module descriptions match current code

## Verified Live

Phase 1 was verified end-to-end with a real phone call:

```
GET /health          → 200 OK
POST /webhook        → 200 OK (status-update, conversation-update, speech-update, etc.)
POST /chat/completions → 200 OK (multiple turns, streaming, Groq responses)
```

The server handles voice turns, streaming, and call lifecycle events from Render's production URL.
