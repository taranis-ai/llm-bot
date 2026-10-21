# Configuration

[Back to README](../README.md)

Copy [`.env.example`](../.env.example) to `.env`, then configure the values below.
For Python library use, set configuration before importing task modules.

## LLM connection

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL` (optional if your OpenAI-compatible backend provides a default model)
- `LLM_API_MODE`: `responses` (default) or `chat_completions`

## Decisions connection

`/cybersec-classification` uses an independent decision model when `DECISIONS_URL`
is nonempty. Other tasks continue to use `LLM_*`. With no decisions URL,
classification retains its existing generative LLM behavior. A configured
decisions service failure returns `502`; it never falls back to the generative model.

- `DECISIONS_URL`: **full endpoint URL**, including its path; no path is appended
- `DECISIONS_API_KEY`: separate bearer token; never inherited from `LLM_API_KEY`
- `DECISIONS_MODEL`: decision-capable model ID; optional only when the server supplies a default
- `DECISIONS_API_MODE`: `systemone` (default) or `openai`; invalid values are rejected
- `DECISIONS_TIMEOUT`: request timeout in seconds, default `120`

| Provider | `DECISIONS_URL` example | `DECISIONS_API_MODE` | Model example |
| --- | --- | --- | --- |
| OpenRouter | `https://openrouter.ai/api/alpha/decisions` | `systemone` | `typesafe/jev-1.13` |
| llama.cpp | `http://localhost:8081/v1/systemone` | `systemone` | Loaded decision model ID |
| vLLM System One | `http://localhost:8000/v1/systemone` | `systemone` | Served decision-capable model ID |
| OpenAI-compatible Decisions, including vLLM | `http://localhost:8000/v1/decisions` | `openai` | Served model ID |

The two formats have different request and response shapes. See the
[OpenRouter reference](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request),
[vLLM reference](https://docs.vllm.ai/en/latest/models/structured_decisions_models/),
[llama.cpp guide](https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp),
and [OpenAI reference](https://developers.openai.com/api/reference/resources/decisions/methods/create).
This client currently supports named choice questions with string labels.

Decisions classification asks one choice question with the labels `cybersecurity`
and `non-cybersecurity`, then returns the provider's probabilities without rounding
or renormalizing. Labels must match exactly, probabilities must be finite numbers
in `[0, 1]`, and their sum must be one within `0.00001`. Duplicate JSON keys,
question names, or choice values are rejected. Invalid output and refusals
fail with `502` without a generative repair request. Reasoning profiles,
`reasoning_effort`, and `thinking_budget_tokens` do not apply in decisions mode.

For a manual OpenRouter check, export the connection before starting the service
or importing the library:

```bash
export DECISIONS_URL=https://openrouter.ai/api/alpha/decisions
export DECISIONS_MODEL=typesafe/jev-1.13
export DECISIONS_API_MODE=systemone
export DECISIONS_API_KEY="$OPENROUTER_TOKEN"
```

Then POST `{"text":"A ransomware campaign targeted hospitals."}` to
`/cybersec-classification`. `OPENROUTER_TOKEN` is only a convenient shell source
for the key; the application reads `DECISIONS_API_KEY`. Automated tests mock the
transport and require no token or live provider. `/info` reports the decisions
model, mode, timeout, and whether the URL is configured; it omits the URL and key.

For rollout, set these variables and restart the service, verify `/health`, check
`/info`, and submit a classification request. To roll back routing, clear
`DECISIONS_URL` and restart; the existing `LLM_*` configuration remains available.

## Other optional settings

- `API_KEY`: protects incoming requests to `/chat`, `/hrag`, `/embed`, `/sentiment`, `/cybersec-classification`, `/title`, `/translate`, `/summarize`, `/ner`, `/ner-link`, `/link`, `/cluster`, `/entity-relation-extraction`, and `/graph-query-generation`
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
