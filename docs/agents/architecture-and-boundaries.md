# Architecture And Boundaries

## When To Load

Read this before changing application structure, routes, schemas, prompts, upstream transports, entity linking, configuration, or deployment behavior.

## Repository Map

- `app.py`: minimal ASGI entry point; exposes the application returned by the factory.
- `llm_bot/app.py`: Quart application factory, logging setup, and blueprint registration.
- `llm_bot/routes.py`: HTTP boundary, optional bearer authentication, common validation/error mapping, health/docs/info endpoints, and task dispatch.
- `llm_bot/schemas.py`: Pydantic request, response, lookup, and internal cluster models. This is the runtime source of truth for payload validation.
- `llm_bot/client.py`: asynchronous OpenAI-compatible transport for Responses and Chat Completions APIs.
- `llm_bot/lookup_client.py`: asynchronous client for the external entity-candidate lookup service.
- `llm_bot/reasoning.py`: provider-specific reasoning prompt and output normalization.
- `llm_bot/tasks/`: task orchestration, prompt construction, structured-output definitions, parsing, validation, and post-processing.
- `llm_bot/prompts/`: task-specific system prompts loaded at request time.
- `llm_bot/config.py`: environment-backed application settings.
- `openapi3_1.yml`: published OpenAPI 3.1 contract served at `/openapi.yaml`.
- `tests/`: route, client, schema, prompt-building, parsing, retry, and task orchestration coverage.
- `Containerfile` and `.github/workflows/`: image construction, test, build, and tagged-release automation.

## Request Data Flow

```text
HTTP request
  -> Quart route and optional API-key check
  -> Pydantic request model
  -> task builds system/user input and JSON Schema
  -> LLMClient sends Responses or Chat Completions request
  -> common output/reasoning normalization
  -> task parser and Pydantic response validation
  -> serialized JSON response
```

`llm_bot.routes._handle_model_request()` owns the common HTTP behavior: request validation failures and supported client errors are `400`, upstream LLM failures are `502` with their message, and unexpected processing failures are logged and returned as a generic `502`. Keep task code independent of Quart request/response objects.

The optional incoming `API_KEY` protects only the LLM-backed POST routes. `/health`, `/info`, `/docs`, and `/openapi.yaml` stay public. The upstream `LLM_API_KEY` and lookup API key are separate credentials.

## Upstream LLM Boundary

`LLMClient` accepts provider-neutral system input, user input, and an optional Responses-style JSON Schema description. It is responsible for translating those into:

- `input`, `text.format`, and `reasoning.effort` for Responses mode
- `messages`, `response_format.json_schema`, and `reasoning_effort` for Chat Completions mode

Provider-specific `thinking_budget_tokens` is forwarded unchanged in either mode. Chat Completions responses are normalized to `output_text` before task parsing. Keep transport translation in the client rather than branching on API mode inside individual tasks.

`llm_bot.tasks.llm_utils.create_and_parse_response()` applies the configured reasoning profile, makes the request, and retries exactly once when JSON decoding, Pydantic validation, or an `InvalidLLMOutputError` rejects model output. The retry includes the invalid output and validation error. Do not add independent retry loops to tasks without an explicit contract change.

## Task Boundaries

Each primary task module generally owns:

- loading its prompt
- constructing system and user messages
- defining the requested JSON Schema
- parsing and validating model output
- creating or accepting an injectable client

Shared output extraction, noisy-JSON recovery, reasoning handling, and repair live in `llm_utils.py` and `reasoning.py`. Shared story formatting, language selection, and truncation live in `task_utils.py`. NER cleanup lives in `ner_postprocessing.py`. Reuse these boundaries rather than duplicating their behavior.

Prompts, JSON Schema definitions, and Pydantic response models form one contract. Change and test them together. Never trust structured-output support alone: all model output must still pass local parsing and Pydantic/task-specific validation.

## Entity Linking

NER extraction and Wikidata linking are intentionally separable:

- `/ner` extracts mention-to-type mappings.
- `/link` links caller-provided mentions and types.
- `/ner-link` composes extraction followed by linking.

The lookup service supplies candidates. Deterministic mode selects the first candidate. LLM mode may choose only a QID from the supplied candidate set and falls back to unresolved entities if batch selection fails. Keep lookup concerns in `LookupClient`/`entity_linking.py`, and do not allow model-generated QIDs that were absent from lookup results.

## Configuration And Runtime State

`Config` is a module-level `Settings` instance created during import. Code reads it directly, and tests commonly monkeypatch its attributes. New configuration should have a safe default and should not cause network access or secret validation at import time.

The service is stateless: it has no database or queue. Its external state boundaries are the LLM provider and, for linking, the lookup service.

## Contract Invariants

- Request models forbid unknown fields unless a schema explicitly documents tolerance; cluster story/tag input intentionally permits extra upstream fields.
- Response models generally forbid unknown fields and serialize aliases where required, such as `non-cybersecurity`.
- Trailing slashes are accepted because the Quart URL map disables strict slashes.
- Story IDs are remapped to compact integer IDs for clustering and validated/remapped back before returning them; every input story must appear exactly once.
- Summary input and output are bounded by configuration. Title input is bounded, but overlong model titles are not truncated by the service.
- Reasoning content is diagnostic output and must be stripped before JSON parsing when configured.
