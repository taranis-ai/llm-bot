import json
from typing import Any

from niquests import AsyncSession
from niquests.exceptions import HTTPError
from pydantic import BaseModel, ConfigDict, Field

from llm_bot.config import Config


class UpstreamEmbeddingError(RuntimeError):
    pass


class _EmbeddingData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    embedding: list[float] = Field(min_length=1)


class _EmbeddingServiceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[_EmbeddingData] = Field(min_length=1)


class EmbeddingClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self.base_url = (base_url or Config.EMBEDDING_BASE_URL).rstrip("/")
        self.api_key = api_key or Config.EMBEDDING_API_KEY
        self.model = model or Config.EMBEDDING_MODEL
        self.timeout = timeout or Config.EMBEDDING_TIMEOUT

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_error_message(response_text: str) -> str:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            return response_text

        if isinstance(payload, dict):
            error_value = payload.get("error")
            if isinstance(error_value, dict):
                for key in ("message", "detail", "error"):
                    if value := error_value.get(key):
                        return str(value)
            if isinstance(error_value, str) and error_value:
                return error_value
            for key in ("message", "detail"):
                if value := payload.get(key):
                    return str(value)

        return response_text

    def _payload(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"input": text}
        if self.model:
            payload["model"] = self.model
        return payload

    async def create_embedding(self, text: str) -> list[float]:
        async with AsyncSession(base_url=self.base_url, headers=self._headers()) as session:
            response = await session.post(
                "/embeddings",
                json=self._payload(text),
                timeout=self.timeout,
            )
            try:
                response.raise_for_status()
            except HTTPError as exc:
                error_message = self._extract_error_message(response.text)
                raise UpstreamEmbeddingError(error_message) from exc

            response_data = _EmbeddingServiceResponse.model_validate_json(response.text)
            return response_data.data[0].embedding
