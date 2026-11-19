from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_bot.config import Config
from llm_bot.schemas import EmbeddingVector, EmbedRequest


class UpstreamEmbeddingError(RuntimeError):
    pass


class _EmbeddingData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int = Field(default=0, ge=0, strict=True)
    embedding: EmbeddingVector


class _EmbeddingServiceResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[_EmbeddingData] = Field(min_length=1)


class EmbeddingClient:
    """OpenAI-compatible embeddings with transport-independent request/response handling."""

    path = "/embeddings"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (Config.EMBEDDING_BASE_URL if base_url is None else base_url).rstrip("/")
        self.api_key = Config.EMBEDDING_API_KEY if api_key is None else api_key
        self.model = Config.EMBEDDING_MODEL if model is None else model
        self.timeout = Config.EMBEDDING_TIMEOUT if timeout is None else timeout

    def build_request(self, text: str | list[str], *, dimensions: int | None = None) -> dict[str, Any]:
        request = EmbedRequest(text=text, dimensions=dimensions)
        payload: dict[str, Any] = {"input": request.text, "encoding_format": "float"}
        if self.model:
            payload["model"] = self.model
        if request.dimensions is not None:
            payload["dimensions"] = request.dimensions
        return payload

    @staticmethod
    def parse_response(body: dict[str, Any], *, count: int = 1, dimensions: int | None = None) -> list[list[float]]:
        data = _EmbeddingServiceResponse.model_validate(body).data
        if len(data) != count or {item.index for item in data} != set(range(count)):
            raise ValueError("Embedding response does not match input indices")
        ordered = sorted(data, key=lambda item: item.index)
        expected_dimensions = dimensions if dimensions is not None else len(ordered[0].embedding)
        if any(len(item.embedding) != expected_dimensions or not any(item.embedding) for item in ordered):
            raise ValueError("Embedding vectors must have matching dimensions and be nonzero")
        return [item.embedding for item in ordered]

    async def _create(self, text: str | list[str], dimensions: int | None) -> list[list[float]]:
        from niquests import AsyncSession
        from niquests.exceptions import HTTPError

        payload = self.build_request(text, dimensions=dimensions)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with AsyncSession(base_url=self.base_url, headers=headers) as session:
            response = await session.post(self.path, json=payload, timeout=self.timeout, allow_redirects=False)
            try:
                response.raise_for_status()
                if not 200 <= response.status_code < 300:
                    raise UpstreamEmbeddingError("Embedding provider returned an unsuccessful response")
            except HTTPError:
                raise UpstreamEmbeddingError("Embedding provider request failed") from None
            return self.parse_response(response.json(), count=1 if isinstance(text, str) else len(text), dimensions=dimensions)

    async def create_embedding(self, text: str, *, dimensions: int | None = None) -> list[float]:
        return (await self._create(text, dimensions))[0]

    async def create_embeddings(self, texts: list[str], *, dimensions: int | None = None) -> list[list[float]]:
        return await self._create(texts, dimensions)
