from pydantic import BaseModel
from typing import Optional


class VapiMessage(BaseModel):
    role: str
    content: str


class VapiWebhookResponse(BaseModel):
    response: str
