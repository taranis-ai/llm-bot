"""Run against an installed distribution with Python's -I flag, outside the checkout."""

import asyncio
import sys
from importlib.metadata import version
from importlib.resources import files
from importlib.util import find_spec
from unittest.mock import AsyncMock

from llm_bot import __version__
from llm_bot.client import LLMClient
from llm_bot.schemas import SummarizeRequest
from llm_bot.tasks.summarize import summarize


async def main() -> None:
    assert __version__ == version("taranis-llm-bot")
    arguments = [argument for argument in sys.argv[1:] if argument != "--server"]
    if arguments:
        assert __version__ == arguments[0], (__version__, arguments[0])

    prompts = files("llm_bot").joinpath("prompts")
    assert files("llm_bot").joinpath("py.typed").is_file()
    for name in (
        "chat",
        "cluster",
        "cybersec_classification",
        "entity_relationship_extraction",
        "graph_query_generation",
        "hrag",
        "ner",
        "sentiment",
        "summarize",
        "title",
        "translate",
    ):
        assert prompts.joinpath(f"{name}.txt").read_text(encoding="utf-8").strip()

    client = AsyncMock(spec=LLMClient)
    client.create_response.return_value = {"output_text": '{"summary": "A short summary."}'}
    result = await summarize(SummarizeRequest(text="Text to summarize.", language="en"), client=client)
    assert result.model_dump() == {"summary": "A short summary."}
    client.create_response.assert_awaited_once()

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
