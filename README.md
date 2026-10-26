# llm-bot

Async Python library and LLM-backed bot service.

The current implementation exposes stateless chat and grounded HRAG answering, embeddings, sentiment analysis,
title generation, summary, named entity recognition, entity relationship extraction, graph query generation,
translation, linking, clustering, and classification endpoints backed by OpenAI-compatible APIs and a decisions API.

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
Generative tasks accept an optional `client=LLMClient(...)` to override the connection;
import it from `llm_bot.client`. Inputs and outputs are Pydantic models, and errors
propagate to the caller. Linking also needs the `LOOKUP_*` configuration; embeddings
use `EMBEDDING_*`. Cybersecurity classification uses the independent `DECISIONS_*`
connection when `DECISIONS_URL` is set; you can inject a `DecisionsClient` from
`llm_bot.decisions_client` or an `LLMClient` to explicitly select either transport.
Fine-grained classification uses `ClassifyRequest` and `llm_bot.tasks.classify.classify_text`,
always with a `DecisionsClient` and the `DECISIONS_*` connection. Sentiment analysis
also requires this connection and accepts an optional `DecisionsClient` override.

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

To route `/cybersec-classification` to a decision model, also set `DECISIONS_URL`
to the full endpoint URL, `DECISIONS_MODEL`, and `DECISIONS_API_KEY` if required.
`DECISIONS_API_MODE=systemone` (default) supports OpenRouter's
`https://openrouter.ai/api/alpha/decisions` and llama.cpp/vLLM `/v1/systemone`;
use `openai` for `/v1/decisions`. The public classification response stays unchanged.

`POST /classify` requires the same decisions configuration. Send `{"text":"Article text"}`
to receive probabilities for disinformation, security incident, vulnerability,
threat intelligence, security policy, security news, general news, and other.
These describe the main topic; disinformation is a topic label, not a truth verdict.
See the [classification API](docs/api.md#post-classify) for category definitions.
The [live evaluation](docs/classify-evaluation.md) records 24 examples and how to rerun them.

`POST /sentiment` also requires decisions configuration. Its response keeps the
`sentiment.label` and `sentiment.score` fields, with the score taken from the winning
label's probability. Set `include_emotions: true` for compatible Plutchik emotions,
assessed independently in the same decisions request. Existing sentiment reasoning
fields are accepted but ignored. There is no generative fallback.

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
formatting (including Python examples in Markdown), and runs the full test suite.
The installed pre-commit hook, pull request CI, and release validation all run
this same script. Run `uv run pre-commit run --all-files` to check manually.
