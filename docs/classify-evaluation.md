# Live classification evaluation

On 2026-10-21, the local `POST /classify` route was exercised through Quart's
HTTP test client with real upstream requests to OpenRouter's decisions API,
using `typesafe/jev-1.13` in `systemone` mode. No provider responses were mocked.
The test did not exercise a deployed server or a reverse proxy.

The [24 synthetic examples](../tests/fixtures/classify_examples.json) contain
three cases per category, including seven German texts. They cover incident
response, vulnerability advisories, threat research, policy, industry news,
disinformation reporting, unrelated news, advertisements, a recipe, and an
instruction-injection attempt. These are invented examples, not factual claims
about actual events. Expected labels were assigned before the live run.

| Expected category | Correct highest-scoring label | Expected-label probability range |
| --- | --- | --- |
| disinformation | 3/3 | 1.000–1.000 |
| security incident | 3/3 | 1.000–1.000 |
| vulnerability | 3/3 | 1.000–1.000 |
| threat intelligence | 3/3 | 1.000–1.000 |
| security policy | 3/3 | 1.000–1.000 |
| security news | 3/3 | 0.980–1.000 |
| general news | 3/3 | 1.000–1.000 |
| other | 3/3 | 0.920–1.000 |

All 24 responses returned HTTP `200`, contained exactly the eight expected
probabilities, and passed finite-value, range, and sum-to-one validation.
Median elapsed time through the local route was 0.424 seconds; the slowest
request took 0.788 seconds. Probability ranges above are rounded to three decimals.
The injection attempt's highest-scoring category was `other`.

This is a small smoke test of clear examples, not an independent accuracy
benchmark. High provider probabilities do not demonstrate calibration. Ambiguous
articles, mixed topics, and additional languages still need evaluation on a
representative, independently labeled dataset. No prompt changes were needed
after this run.

## Reproduce

Install the `dev` and `server` extras and configure the `DECISIONS_*` connection
as described in [configuration](configuration.md#decisions-connection). The
[manual runner](../scripts/evaluate_classify.py) is excluded from normal pytest
execution and sends all 24 examples to the configured provider when invoked:

```bash
uv run python -m scripts.evaluate_classify --output /tmp/classify-results.json
```

The JSON report includes each input, expected and predicted category, all
probabilities, HTTP status, timing, model, API mode, and completion time. It
omits credentials, provider URLs, and raw error bodies. A nonmatching prediction
or non-200 response makes the command exit unsuccessfully. The runner uses the
configured incoming `API_KEY` when calling the local route.

For this run, `DECISIONS_*` was unset in the checkout; the available
`OPENROUTER_TOKEN` was supplied as `DECISIONS_API_KEY` only for the test process,
with the endpoint and model from the repository's OpenRouter configuration
example. No credentials or local environment configuration were saved.
