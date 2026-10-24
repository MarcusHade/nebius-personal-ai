"""Thin wrapper over Nebius Token Factory (OpenAI-compatible API)."""
from openai import AsyncOpenAI

from .config import Settings


class LLM:
    def __init__(self, settings: Settings):
        self.model = settings.nebius_model
        self.client = AsyncOpenAI(
            api_key=settings.nebius_api_key or "missing-key",
            base_url=settings.nebius_base_url,
        )

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto"):
        """Returns the assistant message object (.content, .tool_calls)."""
        kwargs = {"model": self.model, "messages": messages, "temperature": 0.2}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message
