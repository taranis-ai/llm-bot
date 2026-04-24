import json

from niquests import AsyncSession

from llm_bot.config import Config
from llm_bot.schemas import LookupResponse


class LookupClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or Config.LOOKUP_BASE_URL).rstrip("/")
        self.api_key = api_key or Config.LOOKUP_API_KEY
        self.timeout = timeout or Config.LLM_TIMEOUT

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def lookup(self, query: str, language: str, limit: int) -> LookupResponse:
        params = {
            "q": query,
            "lang": language,
            "limit": limit,
        }

        async with AsyncSession(base_url=self.base_url, headers=self._headers()) as session:
            response = await session.get("/lookup", params=params, timeout=self.timeout)
            response.raise_for_status()
            return LookupResponse.model_validate(json.loads(response.text))
