import json
from typing import Any

from niquests import AsyncSession

from llm_bot.config import Config


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or Config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or Config.LLM_API_KEY
        self.model = model or Config.LLM_MODEL
        self.timeout = timeout or Config.LLM_TIMEOUT

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def create_response(self, input_text: str, instructions: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": input_text,
            "instructions": instructions,
        }

        async with AsyncSession(base_url=self.base_url, headers=self._headers()) as session:
            response = await session.post("/responses", json=payload, timeout=self.timeout)
            response.raise_for_status()
            return json.loads(response.text)
