# LLM Task Development

## When To Load

Read this before adding an endpoint or changing a task, prompt, request/response schema, structured-output contract, parsing rule, or reasoning behavior.

## Adding Or Changing A Task

Follow the existing vertical slice:

1. Define strict request and response models in `llm_bot/schemas.py`. Put shared `reasoning_effort` and `thinking_budget_tokens` behavior on `LLMRequest`.
2. Add or update the system prompt in `llm_bot/prompts/`. Keep dynamic request-specific rules in the task's message builder so they can be tested directly.
3. Implement the task in `llm_bot/tasks/` with separate prompt loading, message building, response-format construction, parsing, and async orchestration functions.
4. Use `create_and_parse_response()` so reasoning profiles, output extraction, validation repair, and retry semantics remain consistent.
5. Register the request model and task in `llm_bot/routes.py`. Use `_handle_model_request()` unless the endpoint genuinely needs different HTTP semantics.
6. Update `/info`, `openapi3_1.yml`, `README.md`, and configuration examples as applicable.
7. Add focused task tests plus route and schema coverage.

## Structured Output Rules

- Keep the task's JSON Schema aligned with its Pydantic response model: required fields, aliases, enums, bounds, array uniqueness, and `additionalProperties` must agree.
- Parse through `get_output_text()` and `loads_json_output()` before Pydantic validation. These functions handle both supported upstream response shapes, configured reasoning removal, and a final JSON object embedded in noisy text.
- Raise `InvalidLLMOutputError` for semantic failures that Pydantic cannot express, such as unknown IDs, missing cluster membership, or a choice outside an allowed candidate set. This makes the common one-time repair path apply.
- Do not silently coerce invalid model output unless normalization is part of the documented behavior and covered by tests.
- Preserve the original exception behavior after the repair attempt; the route boundary converts the final failure to the service's standard `502` response.

## Prompt And Reasoning Rules

- Prompt files contain stable task instructions. Append dynamic constraints such as language, label sets, length limits, or feature flags in message builders.
- Both `ministral` and `gemma` profiles are applied centrally. Tasks should not add provider-specific reasoning tokens themselves.
- `get_output_text()` can read normalized `output_text` and Responses API message items. Keep support for both when changing output parsing.
- Model reasoning may be logged at DEBUG level, stripped before parsing, or used as a configured fallback. Do not include reasoning in API responses.

## Client Injection And Tests

Task entry points accept an optional `LLMClient`; linking paths also accept an optional `LookupClient`. Preserve this injection seam so tests remain network-free.

For a task change, cover the applicable layers:

- message construction, including prompt text and dynamic constraints
- response-format/schema behavior when it affects provider payloads
- successful parsing from `output_text` and Responses message output
- malformed JSON or invalid semantic output followed by one repair attempt
- task-specific normalization and invariants
- request schema acceptance/rejection
- route success, `400` validation/client errors, and `502` upstream errors
- both upstream API modes when changing `LLMClient`

Use small fake clients from `tests/test_helpers.py` where suitable. Assert the system input, user input, response format, and number of calls rather than relying on a live model.

## Endpoint-Specific Pitfalls

- Sentiment responses must include `emotions` only when requested, must not contain duplicates, and must respect the sentiment/emotion compatibility rules.
- NER output is restricted to the per-request allowed entity types. Cybersecurity labels are filtered out unless cybersecurity mode is enabled; URL-like `PRODUCT` values are removed during post-processing.
- LLM entity linking must validate every selected mention and QID against lookup candidates. A failed LLM selection returns unresolved entities rather than inventing links.
- Cluster output must include every input story exactly once, use no unknown or duplicate IDs, and provide exactly one reason for every non-singleton cluster. Only the validated cluster IDs and message are public in the final response.
- Summary text is truncated to `SUMMARY_MAX_OUTPUT_CHARS` after parsing. Title generation only instructs the model to honor `max_chars`; it does not truncate its response.
- Explicit output language wins; otherwise title and summary use the majority news-item language, with first-seen order breaking ties, then fall back to the input language.
