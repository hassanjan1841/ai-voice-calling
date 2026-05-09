from fastapi import FastAPI
from config import settings
from webhook import router as webhook_router
from llm import router as llm_router
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Voice Agent")

app.include_router(webhook_router)
app.include_router(llm_router)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.groq_model}
