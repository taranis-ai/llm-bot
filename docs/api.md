# API reference

[Back to README](../README.md) · [Configuration](configuration.md)

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

## `POST /chat`

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

## `POST /hrag`

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

`conversation` and `notices` are optional arrays for question context and retrieval limits.
Facts must still come from the current evidence; negation and uncertainty are preserved.

The model is instructed to use no outside knowledge and to report insufficient
evidence explicitly. The service validates that every returned citation is one
of the evidence IDs supplied in the request.

## `POST /embed`

Creates embeddings through the independent [embedding connection](configuration.md#embedding-connection).
The service sends no generative prompt and does not apply reasoning settings.

For one text:

```json
{"text": "Text to embed", "dimensions": 3}
```

```json
{"embedding": [0.25, -0.5, 0.75]}
```

For a batch:

```json
{"text": ["First text", "Second text"], "dimensions": 3}
```

```json
{"embeddings": [[0.25, -0.5, 0.75], [0.5, 0.25, -0.75]]}
```

`dimensions` is optional and requires provider support. Inputs must be nonblank;
batches must be nonempty. Vectors must be finite, nonzero, and have consistent
lengths matching `dimensions` when supplied. Provider indices determine batch
output order; missing, duplicate, or unexpected indices fail validation.
Invalid requests return `400`; provider or output failures return a static `502`.
If `API_KEY` is configured, send `Authorization: Bearer <API_KEY>`.
Chunking, indexing, storage, and similarity search belong to the caller.

Python callers use `EmbeddingClient.create_embedding()` or `create_embeddings()`
with the `inference` extra. With an application-owned HTTP transport, use
`build_request()` and `parse_response()` from the base package instead.

## `POST /sentiment`

Analyzes sentiment using the independent [decisions connection](configuration.md#decisions-connection).
`DECISIONS_URL` is required. Both `systemone` and `openai` modes are supported.
The label is the most probable of `positive`, `negative`, `neutral`, and `mixed`; `score`
is that label's probability, without rounding or renormalizing. Ties prefer
`neutral`, then `positive`, then `negative`, then `mixed`.

Sentiment describes the tone expressed in the text, not whether a reported event
is harmful or beneficial. Factual reporting of a breach, vulnerability, loss, or
recovery does not by itself express negative or positive sentiment or an emotion.
`mixed` means the text clearly expresses both substantial positive and negative
sentiment (for example, "I love the interface, but I hate the support"). `neutral`
means neither tone is clearly expressed; uncertainty alone does not imply `mixed`.

Request body:

```json
{
  "text": "The launch was a success.",
  "include_emotions": true
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
    "emotions": ["fear", "anger"]
  }
}
```

When `include_emotions` is `false` or omitted, the response must not contain an
`emotions` field.
When it is `true`, `emotions` must be an array; an empty array is valid, `null` is not.
Each of the eight Plutchik emotions is assessed with an independent yes/no question
in the same provider request. An emotion is included only when its `yes` probability
exceeds `0.5` and it is compatible with the winning sentiment: joy/trust for positive,
fear/sadness/disgust/anger for negative, and surprise/anticipation for any sentiment.
All eight emotions are compatible with `mixed`, allowing positive and negative
emotions together; each still needs independent evidence and probability above `0.5`.
The array follows Plutchik order: joy, trust, fear, surprise, sadness, disgust, anger,
anticipation, with no duplicates.

The existing `reasoning_effort` and `thinking_budget_tokens` fields are accepted
for compatibility and ignored. Missing configuration, invalid decisions output,
and upstream failures return `502`, without repair or generative fallback.

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

## `POST /classify`

Classifies the main topic using the independent [decisions connection](configuration.md#decisions-connection).
The only request field is `text`, which must contain at least one non-whitespace character:

```json
{"text": "A ransomware attack shut down a hospital."}
```

Example response (all eight keys are always present):

```json
{
  "disinformation": 0.01,
  "security incident": 0.8,
  "vulnerability": 0.04,
  "threat intelligence": 0.05,
  "security policy": 0.02,
  "security news": 0.03,
  "general news": 0.03,
  "other": 0.02
}
```

| Category | Main subject |
| --- | --- |
| `disinformation` | Disinformation, misinformation, propaganda, fact-checking, or influence campaigns. This labels coverage of the topic; it does not judge the submitted text's truthfulness or establish deceptive intent. |
| `security incident` | A concrete cyberattack, breach, compromise, or its impact and response. |
| `vulnerability` | A flaw, CVE, exploitability analysis, patch, or remediation advisory. |
| `threat intelligence` | Threat actors, malware, campaigns, techniques, or indicators, rather than primarily reporting one incident. |
| `security policy` | Cybersecurity laws, regulations, standards, governance, compliance, or government security strategy. |
| `security news` | Other cybersecurity coverage: industry developments, products, research, or defensive practices. |
| `general news` | Other substantive news outside cybersecurity and disinformation coverage. |
| `other` | Non-news material, advertisements, unrelated instructions, or insufficient context. |

Security categories refer to cybersecurity and information security. The model is
instructed to prefer the most specific category fitting the main topic. Scores form
one distribution over alternative main topics, not independent multi-label scores.
Consumers can take the highest score or apply their own confidence threshold.
The scores are provider estimates, not calibrated guarantees.

`DECISIONS_URL` is required. Both configured decisions API modes are supported;
there is no generative fallback or repair. Probabilities are returned unchanged,
must be finite numbers in `[0, 1]`, and must sum to one within `0.00001`.
Invalid requests (including extra fields or reasoning settings) return `400`.
Missing configuration, invalid provider output, or upstream failures return `502`.
If `API_KEY` is configured, send `Authorization: Bearer <API_KEY>`; unauthorized
requests return `401`. `/cybersec-classification` remains available with its existing contract.

## `POST /cybersec-classification`

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

When `DECISIONS_URL` is configured, this endpoint sends a single choice question
to the separate decisions service and returns its class probabilities in the
same response shape. The optional `reasoning_effort` and `thinking_budget_tokens`
fields are accepted for compatibility and ignored in decisions mode. Invalid
decisions output or upstream failures return `502`, without repair or fallback.
With no decisions URL, the endpoint retains the generative LLM behavior and both
reasoning settings apply. See [decisions configuration](configuration.md#decisions-connection).

If `API_KEY` is configured, send it as:

```http
Authorization: Bearer <API_KEY>
```

## `POST /title`

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

## `POST /translate`

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

## `POST /summarize`

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

## `POST /cluster`

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

## `POST /ner`

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

## `POST /entity-relation-extraction`

Extracts schema-constrained entities and directed relationships using only information explicitly
stated in the supplied text. Reasoning defaults to `low`; an explicit request can override it.
This is initial generation only; Decisions verification belongs to the caller.
Entity and relation type names are caller-defined. Relation source
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

## `POST /graph-query-generation`

Returns a structured retrieval plan. It does not return or execute Cypher or SQL.
The caller supplies the same entity/relationship vocabulary used for extraction,
resolves article filters separately, enforces permissions, and compiles the plan into fixed queries.

```json
{
  "question": "Which organization employs Alice?",
  "schema": {
    "entity_types": [
      {"name": "Person", "description": "A person"},
      {"name": "Organization", "description": "An organization"}
    ],
    "relation_types": [
      {"name": "WORKS_AT", "source_types": ["Person"], "target_types": ["Organization"]}
    ]
  },
  "context": {}
}
```

```json
{
  "names": ["alice"],
  "entity_types": ["Person"],
  "relationships": ["WORKS_AT"],
  "hops": 1
}
```

`context` can contain conversation history to interpret the question. The planner
only returns entity names, entity types, relationships, and hop count; article
filters belong to the caller. Names are stripped and case-folded. Empty lists
mean unrestricted graph search, and hops are limited to zero through three.
Unknown types or extra output fields trigger the common repair attempt.

## `POST /ner-link`

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

## `POST /link`

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

## `GET /health`

Returns:

```json
{"status": "ok"}
```

## `GET /info`

Returns discoverable service information and current non-secret feature
configuration, including:

- supported reasoning profiles
- supported linking modes
- canonical endpoint paths
- active non-secret config such as the current reasoning profile and whether
  lookup/linking is configured
- decisions URL configured status, model, payload mode, and timeout (URL and key omitted)


## Prepared HRAG tasks

Use `prepare_entity_relationship_extraction(request)`,
`prepare_graph_query_generation(request)`, and `prepare_hrag(request)` from their
existing task modules. Each returns an `LLMTask` with prompt, schema, parser,
request reasoning settings, and the common one-repair execution path.

Extraction defaults to low reasoning in immediate and batch execution. Library
callers can pass `result_model=...` and `validate=...` to extraction for exact
source quotes and assertion polarity. The caller computes source offsets and
performs any Decisions verification. The HTTP extraction response includes a
confidence field; that score is not Decisions verification.

Install `[inference]` for bundled clients or `[server]` for the HTTP service.
The base package exposes these prepared tasks with an injected transport and
supports Python 3.13 and newer.
