# AI Voice Agent — Architecture Deep Dive

> **Author:** Hassan Jan  
> **Stack:** Python 3.12, FastAPI, Groq (Llama 3.1), Vapi, LangChain, Render  
> **Status:** Phase 1 complete, live-tested with real phone calls

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [File Map](#2-file-map)
3. [Call Flow (Frame-by-Frame)](#3-call-flow-frame-by-frame)
4. [Protocol Deep Dive — Vapi Custom LLM](#4-protocol-deep-dive--vapi-custom-llm)
5. [The 3-Layer End-Call Architecture](#5-the-3-layer-end-call-architecture)
6. [The Look-Behind Buffer — Real-Time Streaming Without Marker Leakage](#6-the-look-behind-buffer)
7. [Dynamic Tool Name Discovery — Zero-Config Coupling](#7-dynamic-tool-name-discovery)
8. [Fast Path Optimization — Cost Savings via Server-Side Intercept](#8-fast-path-optimization)
9. [Module Architecture & Design Patterns](#9-module-architecture--design-patterns)
10. [Key Engineering Decisions](#10-key-engineering-decisions)
11. [The Streaming Challenge — Why This Is Hard](#11-the-streaming-challenge)
12. [OpenAI SSE Tool Call Format (The Raw Bytes)](#12-openai-sse-tool-call-format)
13. [What's Next (Phases 2-5)](#13-whats-next)

---

## 1. System Overview

```
                    ┌──────────────────────────────────────────────────┐
                    │               Vapi Cloud                         │
                    │  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
                    │  │Deepgram  │  │ ElevenLabs│  │ Call         │  │
                    │  │ STT      │  │ TTS      │  │ Orchestrator │  │
                    │  │(speech→  │  │(text→    │  │              │  │
                    │  │ text)    │  │ speech)  │  │              │  │
                    │  └────┬─────┘  └────▲──────┘  └───────┬───────┘  │
                    │       │             │                  │          │
                    │       │  POST       │  SSE stream      │ POST     │
                    │       │  /chat/     │  (OpenAI fmt)    │ /webhook │
                    │       │  completions│                  │          │
                    └───────┼─────────────┼──────────────────┼──────────┘
                            │             │                  │
                    ┌───────▼─────────────┼──────────────────▼──────────┐
                    │              FastAPI Server (Render)              │
                    │                                                   │
                    │  ┌──────────┐  ┌──────────┐  ┌───────────────┐   │
                    │  │ llm.py   │  │webhook.py│  │  memory.py    │   │
                    │  │ /chat/   │  │ /webhook │  │ conv history  │   │
                    │  │completions│  │ events   │  │ (in-memory)  │   │
                    │  └────┬─────┘  └──────────┘  └───────────────┘   │
                    │       │                                            │
                    │       │ LangChain astream                          │
                    │  ┌────▼─────┐                                     │
                    │  │groq_     │                                     │
                    │  │client.py │                                     │
                    │  │ChatGroq  │                                     │
                    │  └────┬─────┘                                     │
                    └───────┼───────────────────────────────────────────┘
                            │
                    ┌───────▼──────────┐
                    │  Groq LPU        │
                    │  (llama-3.1-     │
                    │   8b-instant)    │
                    │  < 1s latency    │
                    └──────────────────┘
```

### The Players

| Component | Role | Why It |
|---|---|---|
| **Vapi** | Telephony provider. Answers the call, transcribes speech, synthesizes voice, manages call state | Handles PSTN/VoIP so we don't have to |
| **Deepgram** (via Vapi) | Speech-to-text. Converts caller audio → text tokens | nova-2 model, ~300ms latency |
| **ElevenLabs** (via Vapi) | Text-to-speech. Converts AI text → natural voice | Elliot voice, multilingual v2 |
| **FastAPI** | Our server. Receives transcribed text, sends back AI responses | Python, async-native, SSE support built-in |
| **Groq** | LLM inference. Generates AI responses from our prompts | LPU hardware — sub-500ms inference on 8B model |
| **LangChain** | LLM framework. Message formatting, streaming abstraction | Wraps Groq's API in standard interfaces |
| **Render** | Hosting. Runs the FastAPI server in production | Free tier, auto-deploys from GitHub |

---

## 2. File Map

```
ai-voice-calling/
│
├── AGENTS.md                AI assistant guidance (commands, architecture)
├── CLAUDE.md                Instructions for AI coding agents
├── .env.example             Template for environment variables
├── .gitignore               Standard Python gitignore
│
└── backend/
    ├── __init__.py          Empty package marker
    ├── main.py              FastAPI app creation, router registration, /health
    ├── config.py            Settings dataclass (env vars → Python)
    ├── groq_client.py       Singleton Groq LLM wrapper
    ├── llm.py               POST /chat/completions — THE BRAIN (273 lines)
    ├── webhook.py           POST /webhook — Vapi lifecycle events
    ├── memory.py            In-memory conversation store (placeholder)
    ├── models.py            Pydantic schemas
    ├── requirements.txt     Python dependencies (relaxed pins)
    └── runtime.txt          Python 3.12 pin
```

**Total backend: 7 files, ~400 lines of Python.** No JavaScript, no database, no Docker.

---

## 3. Call Flow (Frame-by-Frame)

### Normal Voice Turn

```
  CALLER                  VAPI                   OUR SERVER                GROQ
    │                       │                        │                      │
    │  speaks               │                        │                      │
    │──────────────────────►│                        │                      │
    │                       │  Deepgram STT          │                      │
    │                       │  (speech → text)       │                      │
    │                       │                        │                      │
    │                       │  POST /chat/completions │                      │
    │                       │  {messages[], stream:true}                    │
    │                       │───────────────────────►│                      │
    │                       │                        │                      │
    │                       │                        │  1. Append end-call  │
    │                       │                        │     addendum to sys  │
    │                       │                        │     prompt           │
    │                       │                        │                      │
    │                       │                        │  2. Detect end-call  │
    │                       │                        │     tool name from   │
    │                       │                        │     Vapi's tools[]   │
    │                       │                        │                      │
    │                       │                        │  3. Check explicit   │
    │                       │                        │     end phrases      │
    │                       │                        │     (bye/hangup?)    │
    │                       │                        │     ── No ────────►  │
    │                       │                        │                      │
    │                       │                        │  LangChain astream   │
    │                       │                        │─────────────────────►│
    │                       │                        │                      │
    │                       │                        │  ◄── tokens stream ──│
    │                       │                        │                      │
    │                       │                        │  4. Look-behind buf  │
    │                       │                        │     stream live,     │
    │                       │                        │     hold last 15 chr │
    │                       │                        │                      │
    │                       │  ◄── SSE text chunks ──│                      │
    │                       │                        │                      │
    │  ElevenLabs TTS       │                        │                      │
    │  ◄── hears voice ─────│                        │                      │
    │                       │                        │                      │
    │                       │                        │  Groq done:          │
    │                       │                        │  check raw_full      │
    │                       │                        │  for [END_CALL]      │
    │                       │                        │                      │
    │                       │  ◄── SSE tool_call ────│  (if marker found)   │
    │                       │       finish_reason:   │                      │
    │                       │       "tool_calls"     │                      │
    │                       │                        │                      │
    │                       │  executes endCall      │                      │
    │                       │  plays endCallMessage  │                      │
    │  hears goodbye        │                        │                      │
    │  ◄────────────────────│                        │                      │
    │                       │                        │                      │
    │                       │  POST /webhook         │                      │
    │                       │  (end-of-call-report)  │                      │
    │                       │───────────────────────►│                      │
```

### Fast Path (User Says "Goodbye")

```
  CALLER                   VAPI                   OUR SERVER
    │                       │                        │
    │  "Goodbye"            │                        │
    │──────────────────────►│                        │
    │                       │  POST /chat/completions │
    │                       │  {messages[], stream:true}                    │
    │                       │───────────────────────►│                        │
    │                       │                        │                      │
    │                       │                        │  Check explicit      │
    │                       │                        │  end phrases:        │
    │                       │                        │  "goodbye" ∈         │
    │                       │                        │  _EXPLICIT_END_      │
    │                       │                        │  PHRASES → TRUE      │
    │                       │                        │                      │
    │                       │                        │  ⚡ Groq SKIPPED     │
    │                       │                        │  (saves ~$0.0001     │
    │                       │                        │   and ~1-2s latency) │
    │                       │                        │                      │
    │                       │  ◄── SSE tool_call ────│  immediate           │
    │                       │                        │                      │
    │                       │  executes endCall      │                      │
    │                       │  plays endCallMessage  │                      │
    │  hears "Thank you..." │                        │                      │
    │  ◄────────────────────│                        │                      │
```

---

## 4. Protocol Deep Dive — Vapi Custom LLM

Vapi does **not** use webhooks for the AI conversation loop. It uses an OpenAI-compatible `/chat/completions` endpoint.

### Request (Vapi → Our Server)

```json
{
  "model": "llama-3.1-8b-instant",
  "stream": true,
  "messages": [
    {"role": "system", "content": "You are a support agent..."},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "I can't log in"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "endCall",
        "description": "End the current phone call",
        "parameters": {"type": "object", "properties": {}}
      }
    }
  ]
}
```

The `tools[]` array is critical — Vapi injects its registered tools into every request. This is how we discover the exact tool name dynamically (see Section 7).

### Response — Normal (SSE Stream)

```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Sure"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### Response — End Call (SSE with Tool Call)

```
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Glad that's sorted!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"endCall","arguments":"{}"}}]},"finish_reason":null}]}

data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
```

The `finish_reason: "tool_calls"` is the signal. When Vapi sees this, it:
1. Stops waiting for more text
2. Looks up the function name in its registered tools
3. Executes the built-in `endCall` action
4. Plays the `endCallMessage` audio to the caller
5. Disconnects

---

## 5. The 3-Layer End-Call Architecture

This is the most interesting engineering in the project. Call termination has three independent layers. Each catches what the one above misses.

```
                    ┌─────────────────────────────────────┐
                    │  Layer 1: Server-Side Fast Path     │
                    │  _is_explicit_end(last_user_msg)    │
                    │                                     │
                    │  Catches: "goodbye", "end call",    │
                    │  "hang up", "bye bye"               │
                    │                                     │
                    │  Latency: ~5ms. No LLM call.        │
                    │  Groq never contacted.              │
                    └──────────────┬──────────────────────┘
                                   │ fallthrough
                    ┌──────────────▼──────────────────────┐
                    │  Layer 2: LLM [END_CALL] Marker     │
                    │  LLM outputs [END_CALL] in text     │
                    │                                     │
                    │  Catches: natural conversation end  │
                    │  ("thanks", "that's all", resolved) │
                    │                                     │
                    │  Latency: ~1-2s (Groq inference)    │
                    └──────────────┬──────────────────────┘
                                   │ fallthrough
                    ┌──────────────▼──────────────────────┐
                    │  Layer 3: Vapi Dashboard Config     │
                    │  endCallPhrases (audio match)       │
                    │                                     │
                    │  Catches: anything spoken by AI     │
                    │  that sounds like a farewell        │
                    │                                     │
                    │  Latency: Vapi-side processing      │
                    └─────────────────────────────────────┘
```

### Why 3 Layers?

LLMs are stochastic. Even with perfect prompts, the model might not output `[END_CALL]` every time. The server-side fast path catches explicit user requests instantly. The Vapi dashboard config catches edge cases the code misses. Three layers of defense means the call **never** hangs open.

---

## 6. The Look-Behind Buffer

This solves a subtle streaming problem.

### The Problem

Groq streams tokens one at a time. If we forward every token to Vapi as it arrives, Vapi's TTS will speak `[END_CALL]` out loud to the caller. But if we buffer the entire response before sending, the caller hears 2-3 seconds of silence.

### The Solution

A **sliding window buffer** that holds the last 15 characters. Everything before that flushes to Vapi immediately (true real-time streaming). Only the last 15 chars are held back — enough to detect `[END_CALL]` (10 chars) at the boundary.

```python
SAFE_LEN = len(END_CALL_MARKER) + 5  # = 15 chars
pending = ""
raw_full = ""

async for chunk in groq_client.llm.astream(langchain_messages):
    if chunk.content:
        pending += chunk.content
        raw_full += chunk.content                    # accumulates everything
        if len(pending) > SAFE_LEN:
            safe = pending[:-SAFE_LEN]                # flush safe portion
            pending = pending[-SAFE_LEN:]             # keep last 15 chars
            yield _text_chunk(safe)                   # Vapi speaks this NOW
```

After Groq finishes, we check `raw_full` (not `pending`) for the marker:

```python
should_end_call = END_CALL_MARKER in raw_full        # check full text
cleaned_remaining = pending.replace(END_CALL_MARKER, "").strip()

if should_end_call:
    if cleaned_remaining:
        yield _text_chunk(cleaned_remaining)          # last words "Glad that's sorted!"
    for sse in _end_call_sse_chunks(...):
        yield sse                                     # endCall tool call
```

### Why `raw_full` and not just `pending`?

If `[END_CALL]` splits across a TCP chunk boundary (e.g., `[END_C` arrives in chunk N, `ALL]` arrives in chunk N+1), checking only `pending` (the last 15 chars) could miss it. `raw_full` has everything — guaranteed detection.

### The Tradeoff

This approach means we can only detect `[END_CALL]` at the **end** of the response (after Groq finishes). We can't end a call mid-sentence — the caller always hears the complete sentence before the tool call fires. This is the correct behavior for a phone call.

---

## 7. Dynamic Tool Name Discovery

### The Problem

Vapi requires the tool call function name to **exactly** match the tool registered in the dashboard. If the code hardcodes `"endCall"` but Vapi has `"end_call_tool"`, the tool call is silently ignored — the call never ends.

The tool name is a configuration detail that can change between Vapi assistants. Hardcoding creates a fragile coupling.

### The Solution

Every `/chat/completions` request from Vapi includes a `tools[]` array with the registered tool definitions. We read it on every request and discover the correct name:

```python
vapi_tools = data.get("tools", [])
end_call_tool_name = "endCall"  # fallback: Vapi built-in default

for t in vapi_tools:
    fn = t.get("function", {}).get("name", "")
    if fn and ("end" in fn.lower() or "hang" in fn.lower()):
        end_call_tool_name = fn
        break
```

### Why This Matters

- **Zero config coupling**: Rename the tool in Vapi dashboard → it still works. No redeploy.
- **Works across Vapi assistants**: Different assistants can have different tool names. Same backend handles all of them.
- **Self-documenting**: The server logs which tool it found: `🔧 Vapi tools in request: ['endCall'] → using 'endCall' to hang up`

---

## 8. Fast Path Optimization

### The Insight

Most "end call" scenarios are explicit: the user literally says "goodbye" or "end call." Why waste a Groq API call (cost + latency) just to get the model to output `[END_CALL]`?

### The Implementation

```python
_EXPLICIT_END_PHRASES = [
    "end call", "end the call", "hang up", "disconnect the call",
    "goodbye", "good bye", "bye bye", "bye for now",
]

def _is_explicit_end(text: str) -> bool:
    lower = text.lower().strip()
    return any(phrase in lower for phrase in _EXPLICIT_END_PHRASES)
```

If matched, the server skips Groq entirely and directly emits the end-call SSE:

```python
if _is_explicit_end(last_user_msg):
    # No text — tool_call only, Vapi plays endCallMessage
    for chunk in _end_call_sse_chunks(..., "", end_call_tool_name):
        yield chunk
```

### The Savings

| | Normal Path | Fast Path |
|---|---|---|
| **Groq call** | Yes (~$0.0001) | Skipped |
| **Latency** | ~1-2 seconds | ~5 milliseconds |
| **User hears** | AI farewell + hangup | endCallMessage (configured in Vapi) |

---

## 9. Module Architecture & Design Patterns

### Singleton Pattern

Both `config.py` and `groq_client.py` use the singleton pattern — state is initialized once at module import time and shared across all consumers.

```python
# config.py
settings = Settings()  # ← global singleton

# groq_client.py
groq_client = GroqClient()  # ← global singleton, wraps ChatGroq
```

This avoids passing dependencies through every function. Any module can `from config import settings` and get the same instance.

### Router Pattern (FastAPI)

The app uses FastAPI's `APIRouter` to split endpoints across files:

```python
# main.py — just registration
app.include_router(webhook_router)  # POST /webhook
app.include_router(llm_router)      # POST /chat/completions
```

Each router file is self-contained with its own imports and logger. Clean separation of concerns.

### Flat Import Convention

Since Render's `rootDir` is `backend/`, all imports within the backend are flat:

```python
# Inside backend/llm.py:
from config import settings          # NOT from backend.config import settings
from groq_client import groq_client  # NOT from backend.groq_client import ...
```

This works both locally (run from project root with `uvicorn backend.main:app --reload`) and on Render (run from `backend/` with `uvicorn main:app`).

### _end_call_sse_chunks — Generator Pattern

The tool call emission is extracted into a pure generator function:

```python
def _end_call_sse_chunks(response_id, created, model_name, closing_text, tool_name):
    """Yield the SSE chunks Vapi needs to hang up."""
    if closing_text:
        yield text_chunk(closing_text)
    yield tool_call_delta_chunk(tool_call_id, tool_name)
    yield finish_chunk("tool_calls")
    yield done_chunk()
```

Both the streaming path and the fast path consume the same function. No duplication of the SSE format logic.

---

## 10. Key Engineering Decisions

### Decision 1: No LangChain `.bind_tools()`

LangChain has a `bind_tools()` method that auto-formats tool definitions and parses tool calls from LLM output. We deliberately don't use it.

**Why:** We need full control over the SSE stream format. Vapi requires specific chunk ordering (content delta → tool call delta → finish_reason). LangChain's abstraction would add complexity, not remove it. Raw JSON construction at the SSE level is ~30 lines and completely transparent.

### Decision 2: Text Marker Over Tool Mode

We considered running the LLM in "tool calling mode" where it can natively output tool calls. Instead we chose a simple text marker `[END_CALL]`.

**Why:** Tool calling mode adds non-determinism — the model might or might not decide to call the tool. The text marker is simpler, more reliable, and easier to debug (it's just text). The server handles the conversion to proper tool-call format.

### Decision 3: No Database in Phase 1

Conversation memory is an in-memory dict. It resets on every server restart.

**Why:** Databases are infrastructure. Phase 1's goal is a working voice call. Premature database setup adds deployment complexity and debugging surface. Memory.py is explicitly documented as a placeholder for Redis/Supabase in Phase 4.

### Decision 4: Relaxed Dependency Pins

`requirements.txt` uses `>=` not `==`.

**Why:** Render's free tier has limited build resources. Pinned versions can force Rust compilation (pydantic-core, tokenizers). Relaxed pins let pip use pre-built wheels, cutting build time from minutes to seconds.

### Decision 5: HMAC Verification is Non-Blocking

The webhook verifies Vapi's `x-vapi-signature` header but never rejects requests on mismatch.

**Why:** During development, signatures often fail (proxy issues, local tunnels). Rejecting requests would lose debugging data. Production deployment would set `VAPI_VERIFY_KEY` and optionally harden this.

---

## 11. The Streaming Challenge

### Why Real-Time LLM Streaming Is Hard

The fundamental tension: **Vapi needs tokens NOW to speak, but we don't know if the last token is `[END_CALL]` until Groq finishes.**

| Approach | Problem |
|---|---|
| Stream everything live | Vapi TTS says "[END_CALL]" to caller |
| Buffer everything, then send | 2-3 second silence before caller hears anything |
| Hold last N chars (our approach) | ✅ True real-time streaming, no marker leakage |

### The Chunk Boundary Problem

LLM streaming APIs send tokens in unpredictable chunks. On a slow network, `[END_CALL]` might arrive as:

```
Chunk 1: "Glad that's sorted! [END_C"
Chunk 2: "ALL] Have a great day!"
```

Checking only the last 15 characters of the stream for the marker would miss it (it spans two chunks). That's why we check `raw_full` (everything accumulated) instead of just `pending` (the look-behind buffer).

### The SSE Ordering Constraint

Vapi's tool call format requires a specific order of SSE chunks:

1. **Content delta** (optional) — remaining text after stripping marker
2. **Tool call delta** — function name + arguments
3. **Finish chunk** — `finish_reason: "tool_calls"`
4. **`[DONE]`** — stream termination

Each chunk must be a valid OpenAI SSE fragment. The `finish_reason` must be exactly `"tool_calls"` (not `"stop"`, not `"function_call"`). The function name must match the Vapi tool exactly.

---

## 12. OpenAI SSE Tool Call Format

For reference, here's the exact byte-by-byte format Vapi expects for a tool call:

### Chunk 1 — Tool Call Delta
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1715000000,
  "model": "llama-3.1-8b-instant",
  "choices": [{
    "index": 0,
    "delta": {
      "tool_calls": [{
        "index": 0,
        "id": "call_a1b2c3d4e5f6",
        "type": "function",
        "function": {
          "name": "endCall",
          "arguments": "{}"
        }
      }]
    },
    "finish_reason": null
  }]
}
```

### Chunk 2 — Finish Reason
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1715000000,
  "model": "llama-3.1-8b-instant",
  "choices": [{
    "index": 0,
    "delta": {},
    "finish_reason": "tool_calls"
  }]
}
```

### Chunk 3 — Stream End
```
data: [DONE]
```

Each `data: ` prefix and double `\n\n` delimiter is required. The `id` must be consistent across all chunks in a response. The `index` within `tool_calls[]` must be sequential (we use `index: 0` for a single tool call).

---

## 13. What's Next

| Phase | What It Adds | Engineering Challenge |
|---|---|---|
| **2** | Document ingestion — PDF/txt → 500-char chunks → FastEmbed → Supabase pgvector | Chunking strategy, embedding cold start on Render free tier |
| **3** | RAG — embed user query → similarity search → inject context into /chat/completions | Streaming + retrieval latency, prompt budget management |
| **4** | Lead capture — entity extraction, SQLAlchemy models, CRUD routes | Entity extraction from conversation, schema design |
| **5** | Next.js dashboard — call logs, lead management, analytics | Real-time updates, auth, data visualization |

---

## Appendix: Quick Reference

### All Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API authentication (`gsk_...`) |
| `VAPI_VERIFY_KEY` | No (empty during dev) | HMAC key for webhook signature verification |

### All API Endpoints

| Method | Path | Called By | Purpose |
|---|---|---|---|
| `GET` | `/health` | Render / you | Server liveness |
| `POST` | `/chat/completions` | Vapi (every voice turn) | Generate AI response |
| `POST` | `/webhook` | Vapi (lifecycle events) | Receive call events |

### Key Constants (llm.py)

| Constant | Value | Purpose |
|---|---|---|
| `END_CALL_MARKER` | `"[END_CALL]"` | Text marker the LLM appends to end calls |
| `_EXPLICIT_END_PHRASES` | `["end call", "goodbye", ...]` | Triggers server-side fast hangup |
| `SAFE_LEN` | `15` | Look-behind buffer size (len(marker) + 5) |
| `temperature` | `0.3` | LLM temperature — low for consistent tool use |
| `max_retries` | `2` | Groq API retry count |

---

*Document generated 2026-05-10. For implementation details, see `docs/end-call-implementation.md` and `backend/llm.py`.*
