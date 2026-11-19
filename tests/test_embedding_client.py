from unittest.mock import AsyncMock, MagicMock

import pytest
from niquests.exceptions import HTTPError

from llm_bot.embedding_client import EmbeddingClient, UpstreamEmbeddingError


@pytest.mark.parametrize("api_key", ["embedding-key", ""])
async def test_embedding_transport_single_and_batch(monkeypatch, api_key):
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": [{"embedding": [0.25, -0.5]}]}
    session = AsyncMock()
    session.post.return_value = response
    session.__aenter__.return_value = session
    factory = MagicMock(return_value=session)
    monkeypatch.setattr("niquests.AsyncSession", factory)
    monkeypatch.setattr("llm_bot.embedding_client.Config.EMBEDDING_API_KEY", "configured-key")
    client = EmbeddingClient(base_url="https://embeddings.example/v1/", api_key=api_key, model="embed-model", timeout=15)

    assert await client.create_embedding("First", dimensions=2) == [0.25, -0.5]
    assert factory.call_args.kwargs == {
        "base_url": "https://embeddings.example/v1",
        "headers": {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
    }
    session.post.assert_awaited_once_with(
        "/embeddings",
        json={"input": "First", "model": "embed-model", "dimensions": 2, "encoding_format": "float"},
        timeout=15,
        allow_redirects=False,
    )
    response.json.return_value = {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}
    assert await client.create_embeddings(["First", "Second"], dimensions=2) == [[1, 0], [0, 1]]
    assert session.post.call_args.kwargs["json"]["input"] == ["First", "Second"]
    response.raise_for_status.side_effect = HTTPError("private provider error")
    with pytest.raises(UpstreamEmbeddingError, match="^Embedding provider request failed$"):
        await client.create_embedding("First")
    assert session.__aexit__.await_count == 3


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"index": 0, "embedding": [1, 0]}],
        [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [0, 1]}],
        [{"index": 0, "embedding": [1, 0]}, {"index": 2, "embedding": [0, 1]}],
        [{"index": 0, "embedding": [1]}, {"index": 1, "embedding": [0, 1]}],
        [{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [0, 0]}],
        [{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [float("nan"), 1]}],
    ],
)
def test_embedding_response_rejects_invalid_batch(data):
    with pytest.raises(ValueError):
        EmbeddingClient.parse_response({"data": data}, count=2, dimensions=2)
