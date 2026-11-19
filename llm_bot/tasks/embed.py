from llm_bot.embedding_client import EmbeddingClient
from llm_bot.schemas import EmbedBatchResponse, EmbedRequest, EmbedResponse


async def embed_text(request: EmbedRequest, client: EmbeddingClient | None = None) -> EmbedResponse | EmbedBatchResponse:
    embedding_client = client or EmbeddingClient()
    if isinstance(request.text, list):
        embeddings = await embedding_client.create_embeddings(request.text, dimensions=request.dimensions)
        return EmbedBatchResponse(embeddings=embeddings)
    embedding = await embedding_client.create_embedding(request.text, dimensions=request.dimensions)
    return EmbedResponse(embedding=embedding)
