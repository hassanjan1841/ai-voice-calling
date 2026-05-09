from fastapi import FastAPI
from backend.config import settings
from backend.webhook import router as webhook_router
from backend.llm import router as llm_router
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Voice Agent")

app.include_router(webhook_router)
app.include_router(llm_router)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.groq_model}
