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
is nonempty. `/classify` and `/sentiment` always use this connection and require `DECISIONS_URL`;
with no URL both return `502`. Other tasks continue to use `LLM_*`. With no decisions URL,
`/cybersec-classification` retains its existing generative LLM behavior. A configured
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

`/cybersec-classification` asks one choice question with the labels `cybersecurity`
and `non-cybersecurity`. `/classify` asks one `category` question with the eight
labels defined in the [API reference](api.md#post-classify).
Both classification endpoints return the provider's probabilities without rounding
or renormalizing. Labels must match exactly, probabilities must be finite numbers
in `[0, 1]`, and their sum must be one within `0.00001`. Duplicate JSON keys,
question names, or choice values are rejected. Invalid output and refusals
fail with `502` without a generative repair request. Reasoning profiles,
`reasoning_effort`, and `thinking_budget_tokens` do not apply in decisions mode.

`/sentiment` asks one three-label `sentiment` question and returns its winning
label and probability. With `include_emotions: true`, it adds eight independent
yes/no questions to the same request. Emotions need a `yes` probability above
`0.5` and must be compatible with the selected sentiment. The same probability
validation applies to every question; missing emotion answers fail the request.
Sentiment always uses decisions, including when legacy reasoning fields are supplied.

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

For rollout, set these variables, pull the latest published service image, and
restart the service. Verify `/health`, check `/info` for `/classify` and
`decisions_url_configured: true`, then submit a classification request and check
that all eight probabilities are present and sum to one. `/health` alone does
not test provider readiness. Also verify `/sentiment` with and without
`include_emotions: true`: check the label, score, and optional compatible emotions.
Keep the previous image reference for rollback.
To roll back the endpoint, restore that image and restart. Clearing `DECISIONS_URL`
restores generative routing only for `/cybersec-classification`; it disables `/classify`
and `/sentiment`. To restore generative sentiment, roll back to the previous image.

## Other optional settings

- `API_KEY`: protects incoming requests to `/chat`, `/hrag`, `/embed`, `/sentiment`, `/classify`, `/cybersec-classification`, `/title`, `/translate`, `/summarize`, `/ner`, `/ner-link`, `/link`, `/cluster`, `/entity-relation-extraction`, and `/graph-query-generation`
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
