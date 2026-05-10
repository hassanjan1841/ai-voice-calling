# End-Call Implementation

**Date completed:** 2026-05-10  
**File:** `backend/llm.py`  
**Status:** Working

---

## Overview

When using Vapi with a Custom LLM, the AI model itself cannot hang up a call. The call stays open indefinitely unless the server explicitly tells Vapi to end it. This document explains the complete end-call system built into this project — how it detects when a call should end, how it signals Vapi, and how the streaming works without speaking the internal marker out loud.

---

## How Vapi Ends a Call (the Core Mechanism)

Vapi's Custom LLM integration follows the OpenAI chat completions format. To hang up, the server must return a **tool call** response — specifically a function call for the end-call tool configured on the Vapi assistant.

When Vapi receives a response with `finish_reason: "tool_calls"` and a function named `end_call_tool` (or whatever tool is registered), it:

1. Stops waiting for more text
2. Executes the built-in end-call action
3. Plays `endCallMessage` (configured in the Vapi dashboard) to the caller
4. Disconnects the call

This is different from a normal response (`finish_reason: "stop"`), which just plays the text and waits for the user to speak again.

---

## Three-Layer End-Call Architecture

The system uses three independent layers. Each is a fallback for the one before it.

```
Layer 1 — Explicit phrase detection (server-side, no Groq)
    ↓ if user said "goodbye", "hang up", etc.
    → immediately emit endCall tool call, skip Groq entirely

Layer 2 — LLM-driven via [END_CALL] marker (normal conversation path)
    ↓ Groq generates a response ending with [END_CALL]
    → server strips marker, emits text + endCall tool call

Layer 3 — Vapi dashboard endCallPhrases (backup, no code involved)
    ↓ assistant speaks a phrase matching endCallPhrases config
    → Vapi detects it and hangs up automatically
```

---

## Layer 1: Explicit Phrase Detection (Fast Path)

**Where:** `chat_completions()` in `backend/llm.py`, before calling Groq  
**Purpose:** Handle clear "hang up now" requests instantly, without wasting a Groq call

The server checks the last user message against a hardcoded list:

```python
_EXPLICIT_END_PHRASES = [
    "end call", "end the call", "hang up", "disconnect the call",
    "goodbye", "good bye", "bye bye", "bye for now",
]

def _is_explicit_end(text: str) -> bool:
    lower = text.lower().strip()
    return any(phrase in lower for phrase in _EXPLICIT_END_PHRASES)
```

If the user says any of these phrases, the server skips Groq entirely and immediately returns the end-call SSE chunks. No text is sent — Vapi uses its `endCallMessage` setting to say the farewell audio.

**Log output:**
```
📞 EXPLICIT END detected server-side — hanging up without Groq
```

---

## Layer 2: LLM-Driven End Call via `[END_CALL]` Marker

This is the main path for normal conversation endings (issue resolved, user satisfied, natural farewell).

### Step A — System Prompt Addendum

Every request to `/chat/completions` gets the following block appended to the system prompt, regardless of whether the prompt came from the Vapi dashboard or the app default:

```python
_END_CALL_ADDENDUM = """

## MANDATORY: End-Call Protocol
This is a phone call. When the conversation is complete, append [END_CALL] at the very end
of your reply — the system uses it to hang up. Without it, the line stays open.

Trigger [END_CALL] when:
- User says "that's all", "thanks", "goodbye", "bye", "end call", "hang up", or any farewell
- Issue is resolved and user has no more questions

Format rules — [END_CALL] must be the LAST thing in your reply:
  CORRECT: "Glad that's sorted! [END_CALL]"
  CORRECT: "Ticket TKT-7421 is open for you. [END_CALL]"
  WRONG:   "Goodbye!" — missing marker, call won't hang up
  WRONG:   "[END_CALL] Have a great day!" — marker must be last
"""
```

Groq (llama-3.1-8b-instant) follows this instruction reliably. A typical end-of-call response looks like:

```
"Happy to help. Have a great day! [END_CALL]"
```

### Step B — Streaming with Look-Behind Buffer

**The problem:** If we stream Groq tokens directly to Vapi as they arrive, Vapi's TTS will speak `[END_CALL]` out loud. We need to strip it silently.

**The wrong approach (previous code):** Buffer the *entire* Groq response before sending anything. This caused Vapi to sit in silence for 2–3 seconds before the caller heard a single word.

**The solution — look-behind buffer:**

The streaming path holds the last 15 characters (just longer than `[END_CALL]` = 10 chars) in a pending buffer. Everything before that is flushed to Vapi immediately, giving true real-time streaming:

```python
SAFE_LEN = len(END_CALL_MARKER) + 5  # 15 chars

async for chunk in groq_client.llm.astream(langchain_messages):
    if chunk.content:
        pending += chunk.content
        raw_full += chunk.content
        # Flush safe portion immediately — true streaming to Vapi
        if len(pending) > SAFE_LEN:
            safe = pending[:-SAFE_LEN]
            pending = pending[-SAFE_LEN:]
            yield _text_chunk(safe)
```

After Groq finishes, the server checks the full accumulated buffer (`raw_full`) for the marker:

```python
should_end_call = END_CALL_MARKER in raw_full
cleaned_remaining = pending.replace(END_CALL_MARKER, "").strip()
```

**Why `raw_full` and not just `pending`?**  
In rare cases the marker could span a chunk boundary (e.g., `[END_C` in one chunk, `ALL]` in the next). Checking `raw_full` handles this edge case — the marker is always detected even if it was partially flushed.

### Step C — Emitting the End-Call SSE

Once `[END_CALL]` is detected, the server:

1. Sends any remaining cleaned text (so the caller hears the full farewell sentence)
2. Emits three SSE chunks that together form a valid OpenAI tool call response

```python
if should_end_call:
    if cleaned_remaining:
        yield _text_chunk(cleaned_remaining)   # e.g. "Have a great day!"
    for sse in _end_call_sse_chunks(response_id, created, model_name, "", end_call_tool_name):
        yield sse
```

The three SSE chunks from `_end_call_sse_chunks()`:

**Chunk 1 — tool call delta:**
```json
{
  "choices": [{
    "delta": {
      "tool_calls": [{
        "index": 0,
        "id": "call_abc123",
        "type": "function",
        "function": { "name": "end_call_tool", "arguments": "{}" }
      }]
    },
    "finish_reason": null
  }]
}
```

**Chunk 2 — finish:**
```json
{
  "choices": [{ "delta": {}, "finish_reason": "tool_calls" }]
}
```

**Chunk 3:**
```
data: [DONE]
```

Vapi sees `finish_reason: "tool_calls"` + function name `end_call_tool` → recognizes the registered tool → hangs up.

---

## Dynamic Tool Name Detection

**The problem:** The tool name must exactly match what is configured in the Vapi assistant's Tools tab. If the code hardcodes `endCall` but Vapi has `end_call_tool`, the tool call is silently ignored and the call never ends.

**The solution:** On every request, the server reads the `tools` array that Vapi injects into the `/chat/completions` payload and extracts the correct function name:

```python
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
```

This means the code works regardless of what the tool is named in the Vapi dashboard — `endCall`, `end_call_tool`, `hangUp`, or anything else. The fallback `"endCall"` covers the case where Vapi doesn't send a `tools` array (some configurations).

---

## Vapi Dashboard Configuration

For the end-call tool call to actually hang up the call, the Vapi assistant must have the end-call tool registered. In this project the tool is configured under:

**Vapi Dashboard → Assistants → Alex → Tools → `end_call_tool`**

The assistant JSON includes:
```json
"model": {
  "toolIds": ["a65e1a9a-e812-4c60-b532-1d51f22dd600"],
  ...
}
```

And the farewell message is:
```json
"endCallMessage": "Thank you for choosing TechSolutions. I'm glad I could help you today. Have a great day!"
```

Vapi speaks this message after receiving the tool call and before disconnecting.

---

## Complete End-of-Call Flow (Normal Path)

```
User: "Thanks, that's all I needed."
  │
  ▼
Vapi STT (Deepgram) transcribes speech
  │
  ▼
POST /chat/completions
  ├─ reads tools array → end_call_tool_name = "end_call_tool"
  ├─ appends _END_CALL_ADDENDUM to system prompt
  └─ _is_explicit_end("thanks, that's all i needed") → False
  │
  ▼
Groq (llama-3.1-8b-instant) generates:
  "You're very welcome! I'm glad we got that sorted out.
   Don't hesitate to call if anything else comes up. [END_CALL]"
  │
  ▼
Streaming look-behind buffer:
  ├─ "You're very welcome! I'm glad we got that sorted..." → flushed live to Vapi ✓
  └─ " comes up. [END_CALL]" → held in pending buffer
  │
  ▼
Groq stream ends:
  ├─ should_end_call = True  ([END_CALL] found in raw_full)
  ├─ cleaned_remaining = "comes up."  (marker stripped)
  └─ yields: text chunk "comes up." → then endCall SSE chunks
  │
  ▼
Vapi receives:
  ├─ text chunks → TTS speaks full response in real-time
  └─ finish_reason: "tool_calls" + function: "end_call_tool"
  │
  ▼
Vapi executes end_call_tool:
  ├─ plays endCallMessage audio
  └─ disconnects call
  │
  ▼
POST /webhook receives end-of-call-report event
  └─ logs: endedReason, transcript length
```

---

## Complete End-of-Call Flow (Fast Path)

```
User: "Goodbye"
  │
  ▼
POST /chat/completions
  └─ _is_explicit_end("goodbye") → True
  │
  ▼
Groq is skipped entirely
  │
  ▼
_end_call_sse_chunks() emitted immediately (no text, tool_call only)
  │
  ▼
Vapi receives finish_reason: "tool_calls"
  ├─ plays endCallMessage
  └─ disconnects call
```

---

## Non-Streaming Path (Fallback)

When Vapi sends `stream: false`, the same marker detection runs but on the complete `ainvoke()` response. The tool call is returned in a standard JSON response body instead of SSE:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Happy to help! Have a great day!",
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": { "name": "end_call_tool", "arguments": "{}" }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

---

## What Was Broken and How It Was Fixed

### Bug 1 — Wrong server running (root cause of all 404s)

A stale `main.py` at the project root (`/home/hassan-jan/ai-voice-calling/main.py`) was a 14-line test file with no routes. Running `uvicorn main:app` from the project root picked up this file instead of `backend/main.py`. Every Vapi request returned 404 — webhook events were lost and chat completions never reached Groq.

**Fix:** Always run from the project root with `uvicorn backend.main:app --reload`. The stale `main.py` was deleted.

### Bug 2 — Tool name mismatch (silent failure)

The code hardcoded `"endCall"` as the function name in every tool call SSE chunk. The Vapi assistant had the tool registered as `end_call_tool`. Vapi received the tool call, didn't recognize the function name, and silently ignored it — the call never ended.

**Fix:** Added dynamic tool name detection. The server now reads the `tools` array Vapi injects into every `/chat/completions` request and picks the correct name automatically.

### Bug 3 — Full response buffering before streaming

The streaming path accumulated the entire Groq response into `raw_full` before yielding a single SSE chunk to Vapi. Groq could take 1–3 seconds to finish. During that time, the caller heard silence — no voice at all until the whole response was ready.

**Fix:** Look-behind buffer of 15 characters. Tokens flow live to Vapi as they arrive from Groq. Only the last 15 chars are held back to safely detect `[END_CALL]` at the response boundary without ever speaking it aloud.

---

## Key Constants and Where to Find Them

| Constant / Function | Location | Purpose |
|---|---|---|
| `END_CALL_MARKER = "[END_CALL]"` | `llm.py:13` | Internal text marker the LLM appends |
| `_END_CALL_ADDENDUM` | `llm.py:17–31` | Injected into every system prompt |
| `_EXPLICIT_END_PHRASES` | `llm.py:35–38` | Triggers fast-path hang-up |
| `_is_explicit_end()` | `llm.py:66–68` | Checks user message against phrases |
| `_end_call_sse_chunks()` | `llm.py:94–104` | Generates the 3 SSE chunks Vapi needs |
| `end_call_tool_name` detection | `llm.py:117–126` | Reads tool name dynamically from Vapi |
| `SAFE_LEN = 15` | `llm.py:183` | Look-behind buffer size for streaming |

---

## Running the Server

Always run from the **project root**, not from inside `backend/`:

```bash
# Correct
uvicorn backend.main:app --reload

# Wrong — picks up stale root-level main.py or fails with import errors
uvicorn main:app --reload
```
