import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    groq_model: str = "llama-3.1-8b-instant"
    vapi_verify_key: str = field(default_factory=lambda: os.environ.get("VAPI_VERIFY_KEY", ""))
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
