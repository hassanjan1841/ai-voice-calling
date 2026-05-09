# AI Voice Calling Agent — Design Spec

> **Last updated:** 2026-05-08

## Overview

AI-powered voice calling agent that answers business calls, answers questions from company knowledge base, captures leads, and schedules appointments. Built incrementally for learning.

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Voice | Vapi AI | Receive calls, STT (speech→text), TTS (text→speech) |
| LLM | Groq (llama-3.1-70b/8b) | Generate conversational answers |
| Embeddings | FastEmbed (BAAI/bge-small-en-v1.5) | Convert text chunks to vectors (384-dim) |
| RAG Framework | LangChain | Orchestrate retrieval → augmentation → generation |
| Vector DB | Supabase pgvector | Store document embeddings, similarity search |
| Relational DB | Supabase (PostgreSQL) | Leads, appointments, call logs tables |
| Backend | FastAPI (Python) | Webhook handler, ingestion API, REST endpoints |
| Frontend (later) | Next.js | Dashboard to manage KB, view calls, CRUD leads |
| Deployment | Render (backend) + Vercel (frontend) | Hosting |

## Architecture Flow

```
Customer call → Vapi (STT) → POST /webhook → FastAPI
    → LangChain:
        1. Embed question → search pgvector
        2. Retrieve 3-5 relevant chunks
        3. Build prompt: system + context + history
        4. Call Groq for answer
    → Return answer → Vapi (TTS) → Customer hears
```

## Phased Implementation

### Phase 1: Voice Call Skeleton (Vapi → FastAPI → Groq)

Wire up a working voice call end-to-end. No database, no RAG yet.

**Files:**
```
backend/
├── main.py              # FastAPI app, CORS, lifespan
├── webhook.py           # POST /webhook — Vapi event handler
├── groq_client.py       # Groq LLM wrapper via langchain-groq
├── config.py            # Env vars: GROQ_API_KEY, etc.
└── requirements.txt
```

**Flow:**
1. Customer calls → Vapi transcribes → POST /webhook with `{message, conversation_id}`
2. FastAPI validates request → builds message history → calls Groq
3. Groq returns answer → FastAPI responds → Vapi speaks it

**Learning outcomes:** Webhooks, FastAPI basics, Groq integration, Vapi setup.

### Phase 2: Knowledge Ingestion (Docs → Chunks → Vectors → pgvector)

Build the knowledge base ingestion pipeline.

**Files:**
```
backend/
├── ingestion/
│   ├── loader.py        # Read PDFs (PyMuPDF), txt, markdown
│   ├── chunker.py       # Split text: RecursiveCharacterTextSplitter, 500 chars
│   ├── embedder.py      # FastEmbed: text → 384-dim vector
│   └── store.py         # Store chunks + vectors in Supabase pgvector
├── database/
│   ├── schema.py        # SQL table definitions for pgvector
│   └── vector_store.py  # LangChain SupabaseVectorStore wrapper
├── routes/
│   └── ingest.py        # POST /api/ingest — upload docs
```

**Chunking strategy:** 500-character chunks with 50-char overlap using `RecursiveCharacterTextSplitter`. Balances context preservation with search precision.

**Embedding:** FastEmbed `BAAI/bge-small-en-v1.5` (384-dim, ONNX runtime, ~3-5s cold start on Render).

**Storage:** LangChain's `SupabaseVectorStore` manages the pgvector table automatically.

**Learning outcomes:** Document parsing, chunking strategies, embeddings, pgvector, LangChain abstractions.

### Phase 3: RAG Middleware (Retrieve → Augment → Generate)

Connect the voice call to the knowledge base.

**Files:**
```
backend/
├── rag/
│   ├── retriever.py     # Embed query → similarity search → top-k chunks
│   ├── prompt.py        # RAG prompt template
│   └── chain.py         # LangChain LCEL chain (retriever | prompt | llm)
```

**Prompt design:**
```
You are a helpful business assistant. Answer using ONLY the context below.
If the context doesn't contain the answer, say "I don't know" — don't make up info.
Keep answers concise and natural for a phone conversation.

Context:
{retrieved_chunks}

Conversation history:
{history}

Customer: {query}
```

**RAG flow:**
1. Customer asks a question
2. LangChain embeds the question (via FastEmbed)
3. Supabase pgvector retrieves top-3 most similar chunks
4. LangChain builds the augmented prompt
5. Groq generates the answer grounded in the retrieved context

**Learning outcomes:** RAG architecture, retrieval quality tuning, prompt engineering.

### Phase 4: Structured Data + Lead Capture

Track calls, capture leads, schedule appointments.

**Files:**
```
backend/
├── models/
│   ├── lead.py          # Lead SQLAlchemy model
│   ├── appointment.py   # Appointment model
│   └── call_log.py      # Call log model
├── routes/
│   ├── leads.py         # GET/POST /api/leads
│   ├── appointments.py  # GET/POST /api/appointments
│   └── calls.py         # GET /api/calls
└── webhook.py           # Enhanced — lead detection + capture
```

**Database tables (Supabase):**
```sql
leads (id UUID, name, phone, email, source, notes, created_at);
appointments (id UUID, lead_id FK, scheduled_date, status, notes);
call_logs (id UUID, conversation_id, transcript, summary, lead_id FK, duration, created_at);
```

**Lead capture loop in webhook:**
- After Groq generates a response, check if customer provided name/phone
- If yes → extract via simple regex or LLM extraction → create lead in Supabase
- Return lead ID in webhook response

**Learning outcomes:** SQLAlchemy, relational data modeling, CRUD APIs, entity extraction.

### Phase 5 (Future): Frontend Dashboard

Next.js dashboard consuming the API.

**Planned routes:** `/dashboard` (call logs), `/leads` (manage), `/knowledge` (upload documents), `/settings`.

**API contract already defined by backend routes above.**

## Project Structure (Full)

```
ai-voice-calling/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── groq_client.py
│   ├── webhook.py
│   ├── models/
│   │   ├── lead.py
│   │   ├── appointment.py
│   │   └── call_log.py
│   ├── routes/
│   │   ├── ingest.py
│   │   ├── leads.py
│   │   ├── appointments.py
│   │   └── calls.py
│   ├── rag/
│   │   ├── retriever.py
│   │   ├── prompt.py
│   │   └── chain.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── store.py
│   └── database/
│       ├── schema.py
│       └── vector_store.py
├── frontend/            # Later — Next.js
├── tests/
│   ├── test_webhook.py
│   ├── test_rag.py
│   └── test_ingestion.py
├── .env.example
├── requirements.txt
├── render.yaml
└── docs/
    └── superpowers/
        ├── specs/
        └── plans/
```

## Deployment

- **Backend:** Render Web Service — single `uvicorn main:app` process
- **Database:** Supabase — pgvector enabled, connection via DATABASE_URL
- **Environment:** All config via env vars loaded in `config.py`

## Learning Milestones

| Phase | Milestone | What You Learn |
|---|---|---|
| 1 | Call a number, AI responds | Vapi webhooks, FastAPI, Groq |
| 2 | Upload PDF, query chunks | Document parsing, embeddings, pgvector |
| 3 | AI answers from your data | RAG pipeline, retrieval quality |
| 4 | AI captures leads | Entity extraction, CRUD APIs |
| 5 | Dashboard to manage everything | Next.js, API consumption |
