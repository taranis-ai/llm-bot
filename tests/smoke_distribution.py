"""Run against an installed distribution with Python's -I flag, outside the checkout."""

import asyncio
import sys
from importlib.metadata import entry_points, version
from importlib.resources import files
from importlib.util import find_spec
from unittest.mock import AsyncMock

from llm_bot import __version__


async def main() -> None:
    assert __version__ == version("taranis-llm-bot")
    arguments = [argument for argument in sys.argv[1:] if argument != "--server"]
    if arguments:
        assert __version__ == arguments[0], (__version__, arguments[0])

    prompts = files("llm_bot").joinpath("prompts")
    adapters = entry_points(group="taranis.image2text", name="describe")
    assert len(adapters) == 1 and callable(adapters["describe"].load())
    assert files("llm_bot").joinpath("py.typed").is_file()
    for name in (
        "chat",
        "classify",
        "cluster",
        "cybersec_classification",
        "entity_relationship_extraction",
        "graph_query_generation",
        "hrag",
        "image2text",
        "ner",
        "sentiment",
        "summarize",
        "title",
        "translate",
    ):
        assert prompts.joinpath(f"{name}.txt").read_text(encoding="utf-8").strip()

    if find_spec("niquests") is None:
        from llm_bot.embedding_client import EmbeddingClient
        from llm_bot.schemas import EntityRelationshipExtractionRequest, GraphQueryGenerationRequest, HragRequest
        from llm_bot.tasks.entity_relationship_extraction import prepare_entity_relationship_extraction
        from llm_bot.tasks.graph_query_generation import prepare_graph_query_generation
        from llm_bot.tasks.hrag import prepare_hrag

        embeddings = EmbeddingClient(model="embed-model")
        assert embeddings.build_request("Example")["input"] == "Example"
        assert embeddings.parse_response({"data": [{"embedding": [1, 0]}]}, dimensions=2) == [[1, 0]]

        schema = {"entity_types": [{"name": "ORG", "description": "Organization"}], "relation_types": []}
        extraction = prepare_entity_relationship_extraction(
            EntityRelationshipExtractionRequest.model_validate({"text": "Example", "schema": schema})
        )
        assert extraction.parse_response({"output_text": '{"entities":[],"relations":[]}'}).relations == []
        plan = prepare_graph_query_generation(GraphQueryGenerationRequest.model_validate({"question": "Who?", "schema": schema}))
        assert plan.parse_response({"output_text": '{"names":[],"entity_types":[],"relationships":[],"hops":0}'}).hops == 0
        answer = prepare_hrag(HragRequest(question="Who?", passages=[], graph_facts=[]))
        assert answer.parse_response(
            {"output_text": '{"answer":"Insufficient evidence","citations":[],"insufficient_evidence":true}'}
        ).insufficient_evidence
        print("Installed base library: extraction, graph planning and HRAG tasks OK")
        return

    from llm_bot.client import LLMClient
    from llm_bot.decisions_client import DecisionsClient
    from llm_bot.schemas import ClassifyRequest, CybersecClassificationRequest, SummarizeRequest
    from llm_bot.tasks.classify import prepare_classify
    from llm_bot.tasks.cybersec_classification import prepare_cybersec_classification
    from llm_bot.tasks.summarize import summarize

    client = AsyncMock(spec=LLMClient)
    client.create_response.return_value = {"output_text": '{"summary": "A short summary."}'}
    result = await summarize(SummarizeRequest(text="Text to summarize.", language="en"), client=client)
    assert result.model_dump() == {"summary": "A short summary."}
    client.create_response.assert_awaited_once()

    decisions_client = DecisionsClient(api_mode="systemone", model="decision-model")
    task = prepare_cybersec_classification(CybersecClassificationRequest(text="A ransomware campaign."), client=decisions_client)
    assert task.build_request(decisions_client)["state"] == "A ransomware campaign."
    probabilities = {"cybersecurity": 0.9, "non-cybersecurity": 0.1}
    decisions_result = task.parse_result(
        {"answers": {"cybersecurity_relevance": {"type": "choice", "choice": "cybersecurity", "probabilities": probabilities}}},
        decisions_client,
    )
    assert decisions_result.model_dump() == probabilities

    task = prepare_classify(ClassifyRequest(text="A ransomware campaign."))
    assert "security incident" in task.build_request(decisions_client)["questions"]["category"]["criteria"]

    checked = "tasks and prompts"
    if "--server" in sys.argv:
        from llm_bot.app import create_app

        app = create_app()
        app.config["TESTING"] = True
        async with app.test_client() as http:
            response = await http.get("/health")
            assert response.status_code == 200
            assert await response.get_json() == {"status": "ok"}
            response = await http.get("/openapi.yaml")
            assert response.status_code == 200
            assert f"version: {__version__}" in await response.get_data(as_text=True)
        checked += ", health and OpenAPI"
    else:
        assert find_spec("granian") is None
        assert find_spec("quart") is None
        checked += " without server dependencies"
    print(f"Installed taranis-llm-bot {__version__}: {checked} OK")


if __name__ == "__main__":
    asyncio.run(main())
