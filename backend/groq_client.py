import logging
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import settings

logger = logging.getLogger(__name__)

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
