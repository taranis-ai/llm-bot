# llm-bot

LLM-backed bot service.

The current implementation exposes summary and named entity recognition
endpoints backed by an OpenAI-compatible Responses API.

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
- `LLM_MODEL`

Optional:

- `API_KEY`: protects incoming requests to the configured summary and NER routes
- `LLM_TIMEOUT`
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
