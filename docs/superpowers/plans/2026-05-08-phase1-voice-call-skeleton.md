# Phase 1: Voice Call Skeleton — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire up a working voice call: customer calls → Vapi transcribes → FastAPI receives → Groq responds → Vapi speaks.

**Architecture:** Single FastAPI server with one `/webhook` endpoint. Vapi sends conversation events as POST requests. FastAPI maintains conversation history and calls Groq via `langchain-groq`. No database or RAG yet.

**Tech Stack:** Python, FastAPI, langchain-groq, Groq API, Vapi AI, Uvicorn

---

### Task 1: Project Scaffold + Config

**Files:**
- Create: `backend/__init__.py` (empty)
- Create: `backend/config.py`
- Create: `backend/requirements.txt`
- Modify: `backend/main.py`

- [ ] **Step 1: Create `backend/__init__.py`**

```bash
touch backend/__init__.py
```

- [ ] **Step 2: Write `backend/config.py`**

```python
import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    groq_model: str = "llama-3.1-8b-instant"
    vapi_verify_key: str = field(default_factory=lambda: os.environ.get("VAPI_VERIFY_KEY", ""))
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
```

- [ ] **Step 3: Write `backend/requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
langchain-groq==0.2.0
langchain-core==0.3.0
pydantic==2.9.0
python-dotenv==1.0.1
```

- [ ] **Step 4: Update `backend/main.py`**

```python
from fastapi import FastAPI
from backend.config import settings
from backend.webhook import router as webhook_router

app = FastAPI(title="AI Voice Agent")

app.include_router(webhook_router)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.groq_model}
```

- [ ] **Step 5: Create `.env.example` at project root**

```
GROQ_API_KEY=gsk_your_key_here
VAPI_VERIFY_KEY=your_vapi_key_here
```

- [ ] **Step 6: Install dependencies and verify server starts**

```bash
cd /home/hassan-jan/ai-voice-calling
pip install -r backend/requirements.txt
```

Run: `uvicorn backend.main:app --reload`
Expected: Server starts on http://0.0.0.0:8000, `GET /health` returns `{"status": "ok"}`

---

### Task 2: Groq Client

**Files:**
- Create: `backend/groq_client.py`

- [ ] **Step 1: Write the Groq client**

```python
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from backend.config import settings

_system_prompt = """You are a helpful voice assistant for a business.
You answer questions naturally and conversationally — this is a phone call, so keep responses concise.
If you don't know something, say so. Do not make up information.
Be polite, professional, and friendly."""


class GroqClient:
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.groq_model,
            temperature=0.3,
            max_retries=2,
        )

    def build_messages(self, conversation_history: list[dict], new_message: str) -> list:
        """Convert conversation history to LangChain message format."""
        messages = [SystemMessage(content=_system_prompt)]
        for msg in conversation_history:
            if msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            else:
                messages.append(HumanMessage(content=msg["content"]))
        messages.append(HumanMessage(content=new_message))
        return messages

    async def generate(self, conversation_history: list[dict], new_message: str) -> str:
        messages = self.build_messages(conversation_history, new_message)
        response = await self.llm.ainvoke(messages)
        return response.content


groq_client = GroqClient()
```

- [ ] **Step 2: Test the client**

```bash
cd /home/hassan-jan/ai-voice-calling && python -c "
import asyncio
from backend.groq_client import groq_client
result = asyncio.run(groq_client.generate([], 'Say hello in one sentence'))
print(result)
"
```

Expected: Prints a greeting response from Groq. If it fails, check `GROQ_API_KEY` is set.

---

### Task 3: Vapi Webhook Handler

**Files:**
- Create: `backend/webhook.py`
- Create: `backend/models.py`

- [ ] **Step 1: Write `backend/models.py` — Pydantic schemas for Vapi**

```python
from pydantic import BaseModel
from typing import Optional


class VapiMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class VapiWebhookRequest(BaseModel):
    """Vapi sends this when a customer speaks."""
    conversation_id: Optional[str] = None
    message: VapiMessage
    messages: list[VapiMessage] = []


class VapiWebhookResponse(BaseModel):
    """The response Vapi expects."""
    response: str
```

- [ ] **Step 2: Write `backend/webhook.py`**

```python
from fastapi import APIRouter, HTTPException
from backend.models import VapiWebhookRequest, VapiWebhookResponse
from backend.groq_client import groq_client
from backend.config import settings

router = APIRouter()


@router.post("/webhook", response_model=VapiWebhookResponse)
async def handle_webhook(request: VapiWebhookRequest):
    try:
        # Build conversation history from Vapi's messages array
        history = []
        for msg in request.messages:
            history.append({"role": msg.role, "content": msg.content})

        # Generate response from Groq
        answer = await groq_client.generate(
            conversation_history=history,
            new_message=request.message.content,
        )

        return VapiWebhookResponse(response=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 3: Test with curl**

Start the server first:
```bash
cd /home/hassan-jan/ai-voice-calling && uvicorn backend.main:app --reload
```

In another terminal:
```bash
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "message": {"role": "user", "content": "What services do you offer?"},
    "messages": []
  }'
```

Expected: Returns JSON with a `response` field containing Groq's answer.

---

### Task 4: Conversation Memory (In-Memory)

**Files:**
- Create: `backend/memory.py`
- Modify: `backend/webhook.py`

- [ ] **Step 1: Write `backend/memory.py`**

```python
"""Simple in-memory conversation store. One day replace with Redis/Supabase."""
from typing import Optional

_store: dict[str, list[dict]] = {}


def get_history(conversation_id: str) -> list[dict]:
    if conversation_id not in _store:
        _store[conversation_id] = []
    return _store[conversation_id]


def append_message(conversation_id: str, role: str, content: str):
    if conversation_id not in _store:
        _store[conversation_id] = []
    _store[conversation_id].append({"role": role, "content": content})
```

- [ ] **Step 2: Update `backend/webhook.py` to use memory**

Replace the existing content with:

```python
from fastapi import APIRouter, HTTPException
from backend.models import VapiWebhookRequest, VapiWebhookResponse
from backend.groq_client import groq_client
from backend.memory import get_history, append_message

router = APIRouter()


@router.post("/webhook", response_model=VapiWebhookResponse)
async def handle_webhook(request: VapiWebhookRequest):
    try:
        conv_id = request.conversation_id or "default"

        # Store incoming message
        append_message(conv_id, request.message.role, request.message.content)

        # Get conversation history
        history = get_history(conv_id)

        # Generate response
        answer = await groq_client.generate(
            conversation_history=history[:-1],  # exclude latest (already in history)
            new_message=request.message.content,
        )

        # Store assistant response
        append_message(conv_id, "assistant", answer)

        return VapiWebhookResponse(response=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 3: Test conversation continuity**

```bash
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "test-1",
    "message": {"role": "user", "content": "My name is Hassan"},
    "messages": []
  }'
```

Then:
```bash
curl -X POST http://localhost:8000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "test-1",
    "message": {"role": "user", "content": "What is my name?"},
    "messages": []
  }'
```

Expected: Second request should respond "Hassan" — proving conversation memory works.

---

### Task 5: Vapi Integration Setup

**No code changes** — this is configuration.

- [ ] **Step 1: Create a Vapi AI account** at https://vapi.ai if you don't have one

- [ ] **Step 2: Create an assistant in Vapi dashboard**
    - Set the model provider to "Custom" or point at your FastAPI webhook URL
    - Set webhook URL to: `https://your-render-app.onrender.com/webhook`
    - Configure voice settings (choose a voice you like)
    - Get your Vapi API key

- [ ] **Step 3: Set up ngrok for local testing** (so Vapi can reach your local server)

```bash
# Install ngrok
pip install pyngrok
# OR download from ngrok.com

# In one terminal — start your server
cd /home/hassan-jan/ai-voice-calling && uvicorn backend.main:app --reload

# In another terminal — expose it
ngrok http 8000
```

Copy the ngrok HTTPS URL (e.g., `https://abc123.ngrok.io`) → paste as your Vapi assistant's webhook URL.

- [ ] **Step 4: Test the full loop**
    - Call your Vapi phone number
    - Say "Hello"
    - Hear the response
    - Say "What services do you offer?"
    - Hear the response

---

### Task 6: Render Deployment Prep

**Files:**
- Create: `render.yaml`
- Create: `Dockerfile` (optional — Render can auto-detect)

- [ ] **Step 1: Create `backend/render.yaml`**

```yaml
services:
  - type: web
    name: ai-voice-agent
    runtime: python
    buildCommand: pip install -r backend/requirements.txt
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: VAPI_VERIFY_KEY
        sync: false
```

- [ ] **Step 2: Deploy to Render**
    - Push code to GitHub
    - In Render dashboard → New Web Service → connect repo
    - Set build command: `pip install -r backend/requirements.txt`
    - Set start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
    - Add `GROQ_API_KEY` as environment variable
    - Deploy

- [ ] **Step 3: Update Vapi assistant with Render URL**
    - After deploy completes, copy your Render URL (e.g., `https://ai-voice-agent.onrender.com`)
    - Set Vapi webhook to: `https://ai-voice-agent.onrender.com/webhook`
    - Test with a real phone call

---

### Verification Checklist

- [ ] `GET /health` returns `200 OK`
- [ ] `POST /webhook` returns Groq response
- [ ] Conversation memory persists across messages in same conversation
- [ ] Server starts with `uvicorn backend.main:app --reload` without errors
- [ ] ngrok exposes local server to internet
- [ ] Vapi webhook connects and responds to calls
- [ ] Render deployment builds and runs successfully
