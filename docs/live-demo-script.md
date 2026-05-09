# Live Demo Script — InvoicePro AI Voice Assistant

## Setup Checklist (Before Demo)

- [ ] Vapi dashboard: System Prompt = InvoicePro prompt (from `invoicepro-system-prompt.md`)
- [ ] Vapi dashboard: Custom LLM URL = `https://ai-voice-calling-l50m.onrender.com`
- [ ] Vapi dashboard: First message = `"Thanks for calling InvoicePro support. This is Alex. How can I help you today?"`
- [ ] Render dashboard logs tab open (shows live logs)
- [ ] Vapi dashboard open on Workflow tab (to show architecture)
- [ ] Phone ready to call the Vapi number
- [ ] Close other browser tabs — only show what matters

---

## Demo Flow (5-7 min)

### Step 1 — Architecture Overview (1 min, no call yet)

**Show Vapi dashboard → Workflow tab → Model section. Point to:**

> "Here's the architecture. Vapi handles the phone infrastructure — they manage the SIP trunking, the phone number, the voice pipeline. Our server runs on Render and connects to Groq's LLM. The flow is: caller speaks → Vapi STT → our server → Groq generates → streamed back → Vapi TTS speaks."

**Show the key config fields:**
- **Custom LLM URL** → `https://ai-voice-calling-l50m.onrender.com`
- **System Prompt** → the InvoicePro prompt
- **Model** → `llama-3.1-8b-instant` (Groq)

> "What makes this powerful: the entire conversation logic is in our custom server. Vapi is just the voice pipeline. We control everything the AI says."

---

### Step 2 — The Code (30 sec)

**Quick switch to the codebase (or just show the file tree):**

> "The code is minimal:
> - `llm.py` — receives Vapi's request, sends to Groq, streams response back
> - `groq_client.py` — single LLM wrapper
> - `webhook.py` — handles call lifecycle events
> - `main.py` — FastAPI entry point
>
> Four files. That's it."

---

### Step 3 — Live Call (3 min)

**Dial the Vapi number. Put on speaker so audience hears.**

| What you say | Expected response | Why it matters |
|---|---|---|
| *(Call connects)* | "Thanks for calling InvoicePro support. This is Alex. How can I help you today?" | Professional opening — sounds like real business |
| **"I can't log into my account. I think I forgot my password."** | "I can help with that. Please go to the InvoicePro login page and click 'Forgot Password.' I'll send a reset link to your registered email." | Realistic SaaS support flow |
| **"I need to create an invoice for a client."** | "Go to the Invoices tab and click 'New Invoice.' Add your client details, line items, and set the due date. It takes about two minutes." | Shows product knowledge |
| **"One of my clients said their payment didn't go through."** | "The most common cause is an expired card or insufficient funds. Ask your client to check their payment method under Settings > Billing, and try again." | Practical troubleshooting |
| **"This is getting complicated. Can someone from your team help me?"** | "Absolutely. I'll create a support ticket for you — your ticket ID is TKT-7421. Our team will review it within 24 hours and follow up." | Graceful escalation |
| **"Thanks, that's all I needed."** | "Happy to help! Is there anything else I can assist with? If not, have a great day." | Natural ending |

---

### Step 4 — Show The Logs (1 min)

**After the call, switch to Render logs tab. Point to the narrative:**

```
╔═══════════════════════════════════════════════════════╗
║  NEW VOICE TURN          turn_a1379c82               ║
╚═══════════════════════════════════════════════════════╝
  🗣 User said:   I can't log into my account...
  📚 Context:     3 messages (incl. system prompt)
  ═══════════════════════════════════════════════
  ⬡ SERVER → GROQ  Sending to llama-3.1-8b-instant...
  ⬡ GROQ → SERVER  Response received — streaming...
  ═══════════════════════════════════════════════
  ⬡ GROQ generated:
  📝 I can help with that. Please go to the login page...
  ⬡ SERVER → VAPI  Sent (22 chunks, 180 chars)
╔═══════════════════════════════════════════════════════╗
║  TURN COMPLETE           turn_a1379c82               ║
╚═══════════════════════════════════════════════════════╝
```

> "Every turn shows: what the user said → what we sent to Groq → what Groq generated → what we sent back to Vapi. You see the AI's text before Vapi speaks it — this proves the system is working end-to-end."

---

### Step 5 — The Tech Stack (30 sec)

**Summarize on one slide:**

| Layer | Technology |
|---|---|
| Phone | Vapi (SIP, STT, TTS) |
| Server | FastAPI on Render |
| LLM | Groq (llama-3.1-8b-instant) |
| Cost | ~$0.002 per call |
| Latency | ~600ms response time |

> "Called a phone number. AI answered professionally. Handled password reset, invoice creation, payment troubleshooting, and ticket escalation. All with ~600ms latency. This is production-ready."

---

### Anticipated Questions

**Q: What about RAG / knowledge base?**
> "Phase 2 adds PDF ingestion and vector search. The AI will answer from company documents — not just the system prompt."

**Q: What about memory across calls?**
> "Phase 4 adds persistent storage. Currently memory resets per call. We'll add Redis or Supabase to remember returning customers."

**Q: What LLMs can it use?**
> "We chose Groq for low latency. Can swap to any OpenAI-compatible provider by changing one config line."

**Q: How much does this cost to run?**
> "Render free tier. Groq has a free tier. Vapi is ~$0.05/minute. A demo call costs pennies."
