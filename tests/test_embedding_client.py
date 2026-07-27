import pytest
from niquests.exceptions import HTTPError

from llm_bot.embedding_client import EmbeddingClient, UpstreamEmbeddingError


class FakeResponse:
    def __init__(
        self,
        text: str = '{"data":[{"embedding":[0.25,-0.5,0.75]}]}',
        error: Exception | None = None,
    ):
        self.text = text
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error


class FakeSession:
    def __init__(self, *, base_url=None, headers=None, response=None):
        self.base_url = base_url
        self.headers = headers
        self.response = response or FakeResponse()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, path, json, timeout):
        self.path = path
        self.json = json
        self.timeout = timeout
        return self.response


@pytest.mark.asyncio
async def test_create_embedding_calls_configured_service(monkeypatch):
    session = FakeSession()

    def fake_async_session(*, base_url=None, headers=None):
        session.base_url = base_url
        session.headers = headers
        return session

    monkeypatch.setattr("llm_bot.embedding_client.AsyncSession", fake_async_session)

    client = EmbeddingClient(
        base_url="https://embeddings.example/v1/",
        api_key="embedding-key",
        model="embedding-model",
        timeout=15,
    )

    embedding = await client.create_embedding("Text to embed")

    assert embedding == [0.25, -0.5, 0.75]
    assert session.base_url == "https://embeddings.example/v1"
    assert session.path == "/embeddings"
    assert session.json == {"input": "Text to embed", "model": "embedding-model"}
    assert session.timeout == 15
    assert session.headers["Authorization"] == "Bearer embedding-key"


@pytest.mark.asyncio
async def test_create_embedding_extracts_upstream_error_message(monkeypatch):
    session = FakeSession(
        response=FakeResponse(
            text='{"error":{"message":"Unknown embedding model"}}',
            error=HTTPError("bad request"),
        )
    )

    def fake_async_session(*, base_url=None, headers=None):
        session.base_url = base_url
        session.headers = headers
        return session

    monkeypatch.setattr("llm_bot.embedding_client.AsyncSession", fake_async_session)

    client = EmbeddingClient(base_url="https://embeddings.example/v1", timeout=15)

    with pytest.raises(UpstreamEmbeddingError, match="Unknown embedding model"):
        await client.create_embedding("Text to embed")
