# llm-bot

LLM-backed bot service.

The current implementation exposes sentiment analysis, title generation, summary, named entity
recognition, entity relationship extraction, translation, linking, clustering, and cybersecurity classification endpoints backed
by an OpenAI-compatible Responses API or Chat Completions API.

## Requirements

- `uv`
- Python 3.13

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
cp .env.example .env
```

Configure the following values in `.env`:

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL` (optional if your OpenAI-compatible backend provides a default model)
- `LLM_API_MODE`: `responses` (default) or `chat_completions`

Optional:

- `API_KEY`: protects incoming requests to `/sentiment`, `/title`, `/translate`, `/summarize`, `/ner`, `/ner-link`, `/link`, `/cluster`, and `/entity-relationship-extraction`
- `LLM_TIMEOUT`
- `LLM_REASONING_PROFILE`: use `none`, `ministral`, or `gemma`
- `LLM_STRIP_REASONING_OUTPUT`: strip `[THINK]...[/THINK]` blocks before parsing model output
- `LLM_PARSE_REASONING_AS_OUTPUT`: use structured reasoning text as fallback output when a provider emits no final message
- `gemma` reasoning is enabled by prefixing the system prompt with `<|think|>` and the service strips Gemma thought-channel output before parsing when output stripping is enabled
- `LOOKUP_BASE_URL`
- `LOOKUP_API_KEY`
- `LOOKUP_DEFAULT_LANGUAGE`
- `LOOKUP_CANDIDATE_LIMIT`
- `NER_LINKING_ENABLED`
- `NER_LINKING_MODE`: use `deterministic` or `llm`
- `SUMMARY_MAX_INPUT_CHARS`

## Run

```bash
uv run granian --interface asgi app:app --port 5500
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

Request body:

```json
{
  "stories": [
    {
      "id": "s1",
      "tags": {
        "APT29": { "tag_type": "APT" }
      },
      "news_items": [
        {
          "title": "APT29 targets Microsoft users",
          "content": "APT29 targeted Microsoft users in Vienna.",
          "language": "en"
        }
      ]
    },
    {
      "id": "s2",
      "tags": {
        "Microsoft": { "tag_type": "Organization" }
      },
      "news_items": [
        {
          "title": "Microsoft users targeted in Vienna",
          "content": "Users in Vienna were targeted in an APT29 campaign.",
          "language": "en"
        }
      ]
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

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

### `POST /entity-relationship-extraction`

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

## Tests

```bash
uv run --extra dev pytest tests
```
