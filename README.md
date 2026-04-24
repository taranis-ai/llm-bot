# llm-bot

LLM-backed bot service.

The current implementation exposes summary, named entity recognition, and
clustering endpoints backed by an OpenAI-compatible Responses API.

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

Optional:

- `API_KEY`: protects incoming requests to `/summarize`, `/ner`, and `/cluster`
- `LLM_TIMEOUT`
- `LLM_REASONING_PROFILE`: use `none` or `ministral`
- `LLM_REASONING_EFFORT`: optionally send an explicit reasoning effort such as `low`, `medium`, or `high` in the upstream `/responses` payload
- `LLM_STRIP_REASONING_OUTPUT`: strip `[THINK]...[/THINK]` blocks before parsing model output
- `LLM_PARSE_REASONING_AS_OUTPUT`: use structured reasoning text as fallback output when a provider emits no final message
- `SUMMARY_MAX_INPUT_CHARS`
- `SUMMARY_ROUTE_PATH`
- `NER_ROUTE_PATH`

## Run

```bash
uv run granian --interface asgi app:app --port 5500
```

## API

Canonical paths are documented below. The service also accepts the same
routes with a trailing slash.

### `POST /summarize`

Request body:

```json
{
  "text": "Text to summarize",
  "max_words": 80
}
```

Response body:

```json
{
  "summary": "Short summary"
}
```

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
  "APT29": "Group",
  "Mimikatz": "Tool",
  "PowerShell": "CLICommand/CodeSnippet"
}
```

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

Returns non-secret runtime configuration fields such as configured base URL,
model, timeout, and route paths.

## Tests

```bash
uv run --extra dev pytest tests
```
