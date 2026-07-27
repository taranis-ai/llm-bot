from llm_bot.embedding_client import EmbeddingClient
from llm_bot.schemas import EmbedRequest, EmbedResponse


async def embed_text(
    request: EmbedRequest,
    client: EmbeddingClient | None = None,
) -> EmbedResponse:
    embedding_client = client or EmbeddingClient()
    embedding = await embedding_client.create_embedding(request.text)
    return EmbedResponse(embedding=embedding)
