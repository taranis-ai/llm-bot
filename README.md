# llm-bot

Async Python library and LLM-backed bot service.

The current implementation exposes stateless chat and grounded HRAG answering, embeddings, sentiment analysis,
title generation, summary, named entity recognition, entity relationship extraction, graph query generation,
translation, linking, clustering, and cybersecurity classification endpoints backed by OpenAI-compatible APIs.

## Requirements

- `uv`
- Python 3.13

## Python library

Install a published release into a Python 3.13 project with `uv add taranis-llm-bot`.
The PyPI distribution is `taranis-llm-bot`; Python imports use `llm_bot`.
This base install contains the reusable task library without the HTTP server runtime.
Before publishing, install a locally built wheel with
`uv add /absolute/path/to/taranis_llm_bot-VERSION-py3-none-any.whl`.

Set `LLM_BASE_URL`, `LLM_API_KEY`, and optionally `LLM_MODEL` and `LLM_API_MODE`
in the environment or `.env` **before importing** the library. Call the async
task functions directly; no bot HTTP server is needed:

```python
import asyncio

from llm_bot.schemas import SummarizeRequest
from llm_bot.tasks.summarize import summarize


async def main():
    result = await summarize(SummarizeRequest(text="Text to summarize", language="en", max_words=80))
    print(result.summary)
    # result.model_dump() returns a dictionary.


asyncio.run(main())
```

In an existing async application, use `await summarize(...)` directly. Other
tasks follow the same pattern: request models in `llm_bot.schemas`, async functions
in `llm_bot.tasks` (for example, `ner.extract_entities` and `translate.translate_text`).
An optional `client=LLMClient(...)` argument overrides the LLM connection for a call;
import it from `llm_bot.client`. Inputs and outputs are Pydantic models, and errors
propagate to the caller. Linking also needs the `LOOKUP_*` configuration; embeddings
use `EMBEDDING_*`.

For the HTTP service, install the `server` extra with
`uv add "taranis-llm-bot[server]"`. Then import `create_app` from `llm_bot.app`
and expose `app = create_app()` in your ASGI entry point.

## Setup

```bash
uv sync --locked --extra dev --extra server
./scripts/check.sh
cp .env.example .env
```

Configure the following values in `.env`:

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL` (optional if your OpenAI-compatible backend provides a default model)
- `LLM_API_MODE`: `responses` (default) or `chat_completions`

See [Configuration](docs/configuration.md) for optional settings.

## Run

```bash
uv sync --locked --extra server
uv run granian --interface asgi app:app --port 5500
```

## API

See the [API reference](docs/api.md) for endpoint examples. Interactive Swagger
docs are available at `GET /docs`; the OpenAPI document is at `GET /openapi.yaml`.

## Guides

- [Batch processing](docs/batch-processing.md): process multiple story texts in one provider batch job.
- [Configuration](docs/configuration.md): LLM, reasoning, embedding, and lookup settings.
- [Build and publish](docs/publishing.md): build distributions and publish releases.

## Development checks

```bash
uv sync --locked --extra dev --extra server
uv run pre-commit install
./scripts/check.sh
```

[The script](scripts/check.sh) installs development dependencies, checks lint and
formatting, and runs the full test suite. The installed pre-commit hook runs the
same Ruff lint and format checks before each commit; run
`uv run pre-commit run --all-files` to check them manually.
