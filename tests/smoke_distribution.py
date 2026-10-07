"""Run against an installed distribution with Python's -I flag, outside the checkout."""

import argparse
import asyncio
from importlib import import_module
from importlib.metadata import version
from importlib.resources import files
from importlib.util import find_spec
from pkgutil import iter_modules
from unittest.mock import AsyncMock

from llm_bot import __version__
from llm_bot.client import LLMClient
from llm_bot.schemas import SummarizeRequest
from llm_bot.tasks.summarize import summarize


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected_version", nargs="?")
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()

    assert __version__ == version("taranis-llm-bot")
    if args.expected_version:
        assert __version__ == args.expected_version, (__version__, args.expected_version)

    if not args.server:
        for name in ("granian", "quart", "hypercorn"):
            assert find_spec(name) is None, f"Library installation unexpectedly includes {name}"
    tasks = import_module("llm_bot.tasks")
    for module in iter_modules(tasks.__path__, prefix="llm_bot.tasks."):
        import_module(module.name)

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

    print(f"Installed taranis-llm-bot {__version__}: library tasks and prompts OK")
    if not args.server:
        return

    from llm_bot.app import create_app

    assert find_spec("granian") is not None
    app = create_app()
    app.config["TESTING"] = True
    async with app.test_client() as http:
        response = await http.get("/health")
        assert response.status_code == 200
        assert await response.get_json() == {"status": "ok"}
        response = await http.get("/openapi.yaml")
        assert response.status_code == 200
        assert f"version: {__version__}" in await response.get_data(as_text=True)
    print(f"Installed taranis-llm-bot {__version__}: server health and OpenAPI OK")


if __name__ == "__main__":
    asyncio.run(main())
