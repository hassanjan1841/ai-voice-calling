# AI Voice Calling Agent

> AI-powered voice assistant that answers business calls. Real-time speech-to-text → LLM → text-to-speech pipeline with autonomous call termination. Built with FastAPI, Groq, and Vapi.

![Python](https://img.shields.io/badge/python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green) ![Groq](https://img.shields.io/badge/Groq-Llama%203.1--8b-orange) ![Vapi](https://img.shields.io/badge/Vapi-Custom%20LLM-purple)

---

## Quick Start

```bash
# Clone
git clone https://github.com/hassanjan1841/ai-voice-calling.git
cd ai-voice-calling

# Install
pip install -r backend/requirements.txt

# Set up env
cp .env.example backend/.env
# Edit backend/.env — add GROQ_API_KEY

# Run
uvicorn backend.main:app --reload

# Verify
curl http://localhost:8000/health
# → {"status":"ok","model":"llama-3.1-8b-instant"}

# Test the AI endpoint
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b-instant","stream":false,"messages":[{"role":"user","content":"Hello"}]}'
```

---

## How It Works

```
Caller dials in
    │
    ▼
Vapi answers call
    │── Deepgram STT (speech → text)
    │── POST /chat/completions → your FastAPI server
    │
    ▼
Your Server
    │── Appends end-call instructions to system prompt
    │── Discovers Vapi tool names dynamically
    │── Sends to Groq (Llama 3.1-8b) via LangChain astream
    │── Streams tokens back with look-behind buffer
    │── Detects [END_CALL] marker → converts to tool call
    │
    ▼
Vapi receives SSE stream
    │── ElevenLabs TTS speaks AI response
    │── If tool call: plays farewell & hangs up
    │
    ▼
Caller hears natural conversation
```

---

## Architecture

### File Structure

```
ai-voice-calling/
├── backend/
│   ├── main.py              FastAPI app, /health, router registration
│   ├── config.py            Settings singleton (env vars → dataclass)
│   ├── groq_client.py       Groq LLM wrapper (ChatGroq singleton)
│   ├── llm.py               POST /chat/completions — streaming + non-streaming
│   ├── webhook.py           POST /webhook — Vapi lifecycle events
│   ├── memory.py            In-memory conversation store (placeholder)
│   ├── models.py            Pydantic schemas
│   ├── requirements.txt     Python dependencies (relaxed `>=` pins)
│   └── runtime.txt          Python 3.12 pin
├── docs/
│   ├── architecture-deep-dive.md    Full system architecture (this)
│   ├── end-call-implementation.md   End-call system deep dive
│   ├── invoicepro-system-prompt.md  Live demo system prompt
│   ├── phase-1-complete.md          Phase 1 completion report
│   ├── live-demo-script.md          Phone call demo script
│   └── superpowers/
│       ├── specs/                    Design spec
│       └── plans/                    Implementation plans
├── AGENTS.md                AI assistant guidance (commands, architecture)
├── CLAUDE.md                AI coding agent instructions
├── .env.example             Environment variable template
└── .gitignore
```

### Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Backend** | Python 3.12, FastAPI | Async HTTP server, SSE streaming |
| **LLM** | Groq (Llama 3.1-8b-instant) | Sub-500ms AI inference |
| **Framework** | LangChain (langchain-groq) | Message formatting, streaming abstraction |
| **Voice** | Vapi (Custom LLM mode) | Telephony, STT, TTS, call orchestration |
| **STT** | Deepgram (nova-2, via Vapi) | Speech-to-text |
| **TTS** | ElevenLabs (Elliot, via Vapi) | Text-to-speech |
| **Hosting** | Render (Free Tier) | Production deployment |
| **Config** | python-dotenv | Environment variable loading |

### API Endpoints

| Method | Path | Called By | Purpose | Response |
|---|---|---|---|---|
| `GET` | `/health` | Render / you | Server liveness check | `{"status":"ok","model":"..."}` |
| `POST` | `/chat/completions` | Vapi (every voice turn) | Generate AI response | SSE stream or JSON |
| `POST` | `/webhook` | Vapi (call events) | Receive lifecycle events | `{}` (always) |

---

## The End-Call System (3 Layers of Defense)

The most engineered part of this project. AI voice assistants can't hang up calls themselves — the server must tell Vapi to end it. This system ensures calls never stay open.

### Layer 1 — Server-Side Fast Path
Detects explicit end phrases ("goodbye", "end call", "hang up") in the user's message **before** calling Groq. Emits end-call tool call immediately. Zero LLM latency, zero cost.

```python
_EXPLICIT_END_PHRASES = ["end call", "goodbye", "hang up", "bye bye", ...]
if any(phrase in user_text for phrase in _EXPLICIT_END_PHRASES):
    # Skip Groq entirely. Emit tool call. ~5ms.
```

### Layer 2 — LLM `[END_CALL]` Marker
The system prompt instructs the LLM to append `[END_CALL]` at the end of its response when the conversation is complete. The server strips the marker silently and converts it to a proper tool call.

```
LLM output: "Glad that's sorted! [END_CALL]"
Server:    strips marker → streams "Glad that's sorted!"
           → emits endCall tool call → Vapi hangs up
```

### Layer 3 — Vapi Dashboard (Backup)
Vapi's built-in `endCallPhrases` configuration detects farewell phrases in the AI's speech and hangs up automatically. No code involved — pure configuration safety net.

---

## The Look-Behind Buffer

The key streaming challenge: Groq sends tokens one at a time. If we forward every token to Vapi as it arrives, Vapi's TTS will speak `[END_CALL]` out loud. But if we buffer the entire response, the caller hears 2-3 seconds of silence.

**Solution:** A 15-character sliding window buffer.

```python
SAFE_LEN = 15  # len("[END_CALL]") + 5
pending = ""
raw_full = ""

async for chunk in groq_client.llm.astream(langchain_messages):
    if chunk.content:
        pending += chunk.content
        raw_full += chunk.content                     # accumulate everything
        if len(pending) > SAFE_LEN:
            safe = pending[:-SAFE_LEN]                 # flush safe portion
            pending = pending[-SAFE_LEN:]              # hold last 15 chars
            yield _text_chunk(safe)                    # Vapi speaks this NOW
```

Tokens stream live to Vapi in real time. Only the last 15 characters are held back — enough to detect `[END_CALL]` (10 chars) without ever speaking it.

After Groq finishes, `raw_full` (the complete accumulated text) is checked for the marker. This handles the edge case where the marker splits across chunk boundaries — e.g., `[END_C` in chunk N, `ALL]` in chunk N+1.

---

## End-Call Tool Call Format (Raw SSE)

When `[END_CALL]` is detected, the server emits three SSE chunks that Vapi interprets as a hang-up command:

```
data: {"id":"chatcmpl-abc","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"endCall","arguments":"{}"}}]}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
```

The `finish_reason: "tool_calls"` signal tells Vapi to execute the registered tool and disconnect. The function name is discovered dynamically from Vapi's request payload — so renaming the tool in the dashboard doesn't break anything.

---

## Vapi Dashboard Configuration

| Setting | Value |
|---|---|
| **Provider** | Custom LLM |
| **Custom LLM URL** | `https://your-server.com` (Vapi appends `/chat/completions`) |
| **Model** | `llama-3.1-8b-instant` |
| **Stream** | Enabled |
| **STT Provider** | Deepgram (nova-2) |
| **TTS Provider** | ElevenLabs (Elliot, eleven_multilingual_v2) |

For live demo: the InvoicePro system prompt is in `docs/invoicepro-system-prompt.md`.

---

## Roadmap

| Phase | Status | What It Adds |
|---|---|---|
| **1** | ✅ **Complete** | Voice skeleton — Vapi → FastAPI → Groq, live and tested |
| **2** | 🔜 Next | Document ingestion — PDFs → chunks → embeddings → pgvector |
| **3** | 📋 Planned | RAG — embed query → similarity search → context injection in `/chat/completions` |
| **4** | 📋 Planned | Lead capture — entity extraction, SQLAlchemy models, CRUD |
| **5** | 📋 Planned | Next.js dashboard — call logs, leads, analytics |

---

## Deployment

Render Web Service:

| Setting | Value |
|---|---|
| **Root Directory** | `backend/` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Auto-deploy** | On push to `main` branch |

Environment variables to set in Render dashboard:

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq API key (`gsk_...`) |
| `VAPI_VERIFY_KEY` | HMAC key for Vapi webhook verification |

### Local Tunneling

```bash
ngrok http 8000
# → https://xxxx.ngrok-free.app
# Use this URL in Vapi dashboard as Custom LLM URL
```

---

## Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **Text marker over tool mode** | `[END_CALL]` is simpler and more reliable than LLM-native tool calling. Server converts marker → proper OpenAI tool call format. |
| **No `.bind_tools()`** | Raw SSE construction gives full control over chunk ordering. LangChain's abstraction would add complexity for ~30 lines of JSON. |
| **No database in Phase 1** | In-memory dict for conversation history. Databases are infrastructure — premature setup adds complexity. Placeholder for Phase 4. |
| **Relaxed dependency pins** | `>=` not `==` lets pip use pre-built wheels. Cuts Render build time from minutes to seconds (no Rust compilation). |
| **Flat imports** | `from config import settings` works both locally (`uvicorn backend.main:app`) and on Render (`uvicorn main:app` from `backend/`). |
| **Tool name discovery** | Reads Vapi's `tools[]` array on every request. Rename the tool in dashboard — no redeploy needed. |

---

## Debugging

### Logs

Every voice turn logs a structured block:

```
╔═══════════════════════════════════════════════╗
║  NEW VOICE TURN          turn_a1b2c3d4       ║
╚═══════════════════════════════════════════════╝
  🗣 User said:   I can't log in
  📚 Context:     12 messages (system prompt from app)
  ⬡ SERVER → GROQ  Sending to llama-3.1-8b-instant...
  ⬡ GROQ → SERVER  Response received
  📝 Guide them through password reset...
  ⬡ SERVER → VAPI  Sent (142 chars)
╔═══════════════════════════════════════════════╗
║  TURN COMPLETE           turn_a1b2c3d4       ║
╚═══════════════════════════════════════════════╝
```

End-call markers are logged clearly:

```
📞 EXPLICIT END detected server-side — hanging up without Groq
```

```
📞 [END_CALL] marker detected — emitting endCall tool call
```

### Quick Test Commands

```bash
# Test streaming endpoint
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b-instant","stream":true,"messages":[{"role":"user","content":"Hello"}]}'

# Test end-call via explicit phrase (bypasses Groq)
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b-instant","stream":false,"messages":[{"role":"user","content":"goodbye"}]}'

# Test end-call via LLM marker
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b-instant","stream":false,"messages":[{"role":"system","content":"End the call with [END_CALL]"},{"role":"user","content":"thanks"}]}'

# Test Groq directly
python -c "import asyncio; from groq_client import groq_client; print(asyncio.run(groq_client.generate([], 'Say hello')))"
```

---

## Documentation

| Document | What It Covers |
|---|---|
| `docs/architecture-deep-dive.md` | Complete system architecture, all design decisions |
| `docs/end-call-implementation.md` | End-call system deep dive, bugs found and fixed |
| `docs/invoicepro-system-prompt.md` | Live demo system prompt for Vapi support assistant |
| `docs/phase-1-complete.md` | Phase 1 completion report, endpoints, deployment |
| `docs/live-demo-script.md` | Phone call demo script (InvoicePro) |
| `AGENTS.md` | AI assistant guidance, commands, architecture overview |

---

## License

MIT
