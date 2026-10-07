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
The default installation includes only library dependencies. Granian, Quart,
and their server dependencies are installed only with the `server` extra:
`uv add 'taranis-llm-bot[server]'`.
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

To embed the HTTP service, install the `server` extra, import `create_app` from `llm_bot.app`, and expose
`app = create_app()` in your ASGI entry point.

## Build and upload

From a clean `X.Y.Z` release tag and an empty `dist/` directory:

```bash
./scripts/check.sh
uv build --no-sources
uvx twine check --strict dist/*
uv publish dist/*
```

Authenticate with `UV_PUBLISH_TOKEN`; use `--publish-url <upload-endpoint>` for
another registry. The [release workflow](.github/workflows/release.yml) releases
images and the PyPI package on tag pushes; see [PyPI setup](docs/agents/development-workflow.md#packaging-and-release).

## Setup

```bash
./scripts/check.sh
cp .env.example .env
```

Configure the following values in `.env`:

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL` (optional if your OpenAI-compatible backend provides a default model)
- `LLM_API_MODE`: `responses` (default) or `chat_completions`

Optional:

- `API_KEY`: protects incoming requests to `/chat`, `/hrag`, `/embed`, `/sentiment`, `/title`, `/translate`, `/summarize`, `/ner`, `/ner-link`, `/link`, `/cluster`, `/entity-relation-extraction`, and `/graph-query-generation`
- `LLM_TIMEOUT`
- `LLM_REASONING_PROFILE`: use `none`, `ministral`, or `gemma`
- `LLM_STRIP_REASONING_OUTPUT`: strip `[THINK]...[/THINK]` blocks before parsing model output
- `LLM_PARSE_REASONING_AS_OUTPUT`: use structured reasoning text as fallback output when a provider emits no final message
- `gemma` reasoning is enabled by prefixing the system prompt with `<|think|>` and the service strips Gemma thought-channel output before parsing when output stripping is enabled
- `EMBEDDING_BASE_URL`: base URL for the OpenAI-compatible embedding service; required by `/embed`
- `EMBEDDING_API_KEY`
- `EMBEDDING_MODEL`
- `EMBEDDING_TIMEOUT`
- `LOOKUP_BASE_URL`
- `LOOKUP_API_KEY`
- `LOOKUP_DEFAULT_LANGUAGE`
- `LOOKUP_CANDIDATE_LIMIT`
- `NER_LINKING_ENABLED`
- `NER_LINKING_MODE`: use `deterministic` or `llm`
- `SUMMARY_MAX_INPUT_CHARS`

## Run

```bash
uv run --extra server granian --interface asgi app:app --port 5500
```

## API

Canonical paths are documented below. The service also accepts the same
routes with a trailing slash.

Interactive Swagger docs are available at `GET /docs`.
The raw OpenAPI 3.1 document is available at `GET /openapi.yaml`.

Upstream LLM transport:

- `LLM_API_MODE=responses` sends requests to `/responses`
- `LLM_API_MODE=chat_completions` sends requests to `/chat/completions`
- structured outputs are requested via `text.format` in `responses` mode and `response_format` in `chat_completions` mode
- LLM-backed request payloads may include an optional `reasoning_effort` field. The service forwards it upstream as `reasoning.effort` in `responses` mode and `reasoning_effort` in `chat_completions` mode.
- LLM-backed request payloads may include an optional `thinking_budget_tokens` field, which the service forwards upstream unchanged as a provider-specific extension. This is intended for servers such as `llama.cpp`; other OpenAI-compatible servers may reject it.

When using the Python clients directly, omitting `api_key` (or passing `None`)
uses the configured key. Passing `api_key=""` explicitly disables the authorization
header for that client.

### `POST /chat`

Generates a general chat response. The optional `messages` array contains prior
conversation turns for this request only; the service does not persist conversation
state. Prior messages may use the `user` and `assistant` roles.

Request body:

```json
{
  "message": "What should I do next?",
  "messages": [
    {
      "role": "user",
      "content": "Help me plan a release."
    },
    {
      "role": "assistant",
      "content": "Start by running the validation suite."
    }
  ]
}
```

Response body:

```json
{
  "answer": "Review the validation results, then tag the release.",
  "model": "configured-model"
}
```

`model` is `null` when `LLM_MODEL` is unset and the upstream provider selects
its own default.

### `POST /hrag`

Answers a question using only evidence supplied in the request. The endpoint does
not retrieve documents, query a graph, create embeddings, or persist data. IDs
must be unique across `passages` and `graph_facts`; every item also requires a
caller-supplied source reference.

Request body:

```json
{
  "question": "Who operates the service?",
  "passages": [
    {
      "id": "passage-1",
      "source": "report.pdf#page=2",
      "text": "Example Corp operates the service."
    }
  ],
  "graph_facts": [
    {
      "id": "fact-1",
      "source": "graph://service/42",
      "fact": "Example Corp -[OPERATES]-> Service 42"
    }
  ]
}
```

Response body:

```json
{
  "answer": "Example Corp operates the service.",
  "citations": ["passage-1", "fact-1"],
  "insufficient_evidence": false
}
```

The model is instructed to use no outside knowledge and to report insufficient
evidence explicitly. The service validates that every returned citation is one
of the evidence IDs supplied in the request.

### `POST /embed`

Creates an embedding for one text using the separately configured OpenAI-compatible
embedding service. The service sends the text to its `/embeddings` path and returns
the first embedding vector.

Request body:

```json
{
  "text": "Text to embed"
}
```

Response body:

```json
{
  "embedding": [0.012, -0.034, 0.056]
}
```

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `POST /sentiment`

Sentiment analysis endpoint.

Request body:

```json
{
  "text": "The launch was a success.",
  "include_emotions": true,
  "thinking_budget_tokens": 256
}
```

Response body without emotions:

```json
{
  "sentiment": {
    "label": "positive",
    "score": 0.88
  }
}
```

Response body with emotions:

```json
{
  "sentiment": {
    "label": "negative",
    "score": 0.91,
    "emotions": ["anger", "fear"]
  }
}
```

When `include_emotions` is `false` or omitted, the response must not contain an
`emotions` field.
When it is `true`, `emotions` must be an array; an empty array is valid, `null` is not.

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `POST /cybersec-classification`

Request body:

```json
{
  "text": "The newest development in malware automation is concerning.",
  "reasoning_effort": "high",
  "thinking_budget_tokens": 256
}
```

Response body:

```json
{
  "cybersecurity": 0.9999,
  "non-cybersecurity": 0.0001
}
```

This endpoint is LLM-backed and supports the same optional `reasoning_effort` and
`thinking_budget_tokens` fields as the other LLM routes.

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `POST /title`

Request body:

```json
{
  "text": "Text to title",
  "language": "de",
  "max_chars": 100
}
```

Response body:

```json
{
  "title": "Concise story title"
}
```

`language` is optional. When provided, the title is generated in that language. When omitted and
`news_items` are used, the service uses the majority `news_items[*].language` value when available.
Otherwise it falls back to the input text language. The model is instructed to keep the title
within `max_chars` characters. If omitted, `max_chars` defaults to `100`. The service does not
truncate longer model outputs.

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `POST /translate`

Request body:

```json
{
  "text": "Guten Morgen",
  "target_language": "en",
  "source_language": "de"
}
```

Response body:

```json
{
  "translation": "Good morning"
}
```

`source_language` is optional. When omitted, the model is instructed to detect the source language from the input. `target_language` is required.

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `POST /summarize`

Request body:

```json
{
  "news_items": [
    {
      "title": "Story title",
      "content": "Text to summarize",
      "language": "en"
    }
  ],
  "language": "de",
  "max_words": 80
}
```

Response body:

```json
{
  "summary": "Short summary"
}
```

`language` is optional. When provided, the summary is generated in that language. When omitted and
`news_items` are used, the service uses the majority `news_items[*].language` value when available.
Otherwise it falls back to the input text language.

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `POST /cluster`

Each story requires its original `id` and a name-keyed `tags` dictionary (which may
be empty). `summary` is optional and may be `null` or empty. Clustering uses only
summaries and tags; extra fields such as `news_items` and `title` are ignored.
`CLUSTER_MAX_CONTENT_CHARS_PER_STORY` limits each summary sent to the model
(default: 800 characters). Returned clusters contain the original story IDs.

Request body:

```json
{
  "stories": [
    {
      "id": "s1",
      "tags": {
        "APT29": { "tag_type": "APT" }
      },
      "summary": "APT29 targeted Microsoft users in Vienna."
    },
    {
      "id": "s2",
      "tags": {
        "Microsoft": { "tag_type": "Organization" }
      },
      "summary": "Users in Vienna were targeted in an APT29 campaign."
    }
  ]
}
```

Response body:

```json
{
  "cluster_ids": {
    "event_clusters": [["s1", "s2"]]
  },
  "message": "Clustering completed"
}
```

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `POST /ner`

Request body:

```json
{
  "text": "APT29 used Mimikatz and PowerShell to dump credentials.",
  "cybersecurity": true
}
```

Response body:

```json
{
  "APT29": "GROUP",
  "Mimikatz": "TOOL",
  "PowerShell": "PRODUCT"
}
```

This endpoint performs NER only. It does not run entity linking.
If both the initial response and its repair are truncated, the service returns only complete,
schema-valid entity/type pairs from the repaired response and discards its incomplete suffix.

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `POST /entity-relation-extraction`

Extracts schema-constrained entities and directed relationships using only information explicitly
stated in the supplied text. Entity and relation type names are caller-defined. Relation source
and target types must reference declared entity types.

Request body:

```json
{
  "text": "APT28 exploited CVE-2025-1234.",
  "schema": {
    "entity_types": [
      {"name": "ThreatActor", "description": "A named threat actor"},
      {"name": "Vulnerability", "description": "A named vulnerability"}
    ],
    "relation_types": [
      {
        "name": "EXPLOITS",
        "source_types": ["ThreatActor"],
        "target_types": ["Vulnerability"]
      }
    ]
  }
}
```

Response body:

```json
{
  "entities": [
    {
      "id": "e1",
      "type": "ThreatActor",
      "name": "APT28"
    },
    {
      "id": "e2",
      "type": "Vulnerability",
      "name": "CVE-2025-1234"
    }
  ],
  "relations": [
    {
      "type": "EXPLOITS",
      "source_id": "e1",
      "target_id": "e2",
      "confidence": 0.9
    }
  ]
}
```

If no schema-valid explicit extraction exists, both response lists are empty.

### `POST /graph-query-generation`

Generates one AGE-compatible, read-only Cypher query from a natural-language question and a
caller-supplied graph schema. The service returns parameters separately, validates the query
against the allowed labels, relationship directions, and queryable properties, and requires a
bounded `LIMIT`. It does not execute Cypher, connect to PostgreSQL, or persist anything.

`graph_name` is part of the caller contract but is never embedded in the generated Cypher. The
caller must bind that fixed name separately when it executes the returned query.

Request body:

```json
{
  "question": "Which organization employs Alice?",
  "graph_name": "knowledge_graph",
  "schema": {
    "node_labels": [
      {
        "label": "Person",
        "properties": [
          {"name": "name", "type": "string"}
        ]
      },
      {
        "label": "Organization",
        "properties": [
          {"name": "name", "type": "string"}
        ]
      }
    ],
    "relationship_types": [
      {
        "type": "WORKS_AT",
        "source_labels": ["Person"],
        "target_labels": ["Organization"],
        "properties": []
      }
    ],
    "default_limit": 25,
    "maximum_limit": 100
  }
}
```

Response body:

```json
{
  "cypher": "MATCH (p:Person)-[:WORKS_AT]->(o:Organization) WHERE p.name = $person_name RETURN o.name AS result LIMIT 25",
  "parameters": {
    "person_name": "Alice"
  },
  "explanation": "Returns organizations that employ the named person."
}
```

Only standard unquoted identifiers are accepted in the supplied graph schema. Generated values
must use named `$parameter` placeholders. Mutations, procedures, administration, external data
loading, dynamic schema access, comments, multiple statements, and unbounded results are rejected.
Every relationship must specify an allowed type, and `RETURN` must contain one
expression aliased as `result` (which may be a map or a function call).
Invalid model output receives the service's standard single repair attempt.

### `POST /ner-link`

Request body:

```json
{
  "text": "Apple announced new Mac hardware during its developer event in Cupertino.",
  "language": "en",
  "linking_mode": "llm",
  "cybersecurity": false
}
```

Response body:

```json
{
  "entities": [
    {
      "mention": "Apple",
      "type": "ORG",
      "wikidata_qid": "Q312",
      "wikidata_label": "Apple Inc.",
      "wikidata_description": "American technology company",
      "matched_alias": "Apple",
      "match_type": "alias",
      "score": 0.98,
      "candidate_count": 5
    }
  ]
}
```

This endpoint performs NER first and then links the extracted entities.
When NER finds no entities, it returns `{"entities": []}` without making lookup
or disambiguation requests.

Deterministic example:

```json
{
  "text": "Apple released a new device.",
  "language": "en",
  "linking_mode": "deterministic"
}
```

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `POST /link`

Request body:

```json
{
  "text": "Apple announced new Mac hardware during its developer event in Cupertino.",
  "language": "en",
  "linking_mode": "llm",
  "entities": [
    { "mention": "Apple", "type": "ORG" },
    { "mention": "Cupertino", "type": "GPE" },
    { "mention": "Mac", "type": "PRODUCT" }
  ]
}
```

Response body:

```json
{
  "entities": [
    {
      "mention": "Apple",
      "type": "ORG",
      "wikidata_qid": "Q312",
      "wikidata_label": "Apple Inc.",
      "wikidata_description": "American technology company",
      "matched_alias": "Apple",
      "match_type": "alias",
      "score": 0.98,
      "candidate_count": 5
    }
  ]
}
```

This endpoint performs linking only. It does not run NER first.

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `GET /health`

Returns:

```json
{"status": "ok"}
```

### `GET /info`

Returns discoverable service information and current non-secret feature
configuration, including:

- supported reasoning profiles
- supported linking modes
- canonical endpoint paths
- active non-secret config such as the current reasoning profile and whether
  lookup/linking is configured

## Development checks

```bash
uv sync --locked --extra server --extra dev
uv run pre-commit install
./scripts/check.sh
```

[The script](scripts/check.sh) installs development dependencies, checks lint and
formatting, and runs the full test suite. The installed pre-commit hook runs the
same Ruff lint and format checks before each commit; run
`uv run pre-commit run --all-files` to check them manually.
