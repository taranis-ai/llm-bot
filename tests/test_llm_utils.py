import json

import pytest

from llm_bot.client import LLMClient
from llm_bot.schemas import ClusterRequest, CybersecClassificationRequest, NerRequest, SentimentRequest, SummarizeRequest, TitleRequest
from llm_bot.tasks.cluster import prepare_cluster
from llm_bot.tasks.cybersec_classification import prepare_cybersec_classification
from llm_bot.tasks.llm_utils import loads_json_output
from llm_bot.tasks.ner import prepare_ner
from llm_bot.tasks.sentiment import prepare_sentiment
from llm_bot.tasks.summarize import prepare_summary
from llm_bot.tasks.title import prepare_title


@pytest.mark.parametrize(
    "payload",
    [
        {"summary": 'The name contains a quote: " and a brace: {.'},
        {"summary": 'A backslash before a quote: \\" and a closing brace: }.'},
        {"summary": "A Windows path: C:\\reports\\"},
        {"nested": {"summary": "Final result"}},
    ],
)
def test_noisy_json_preserves_strings_and_nested_objects(payload):
    output = 'Draft: {"summary":"Earlier draft"}\nFinal answer:\n```json\n' + json.dumps(payload) + "\n```"

    assert loads_json_output(output) == payload


def test_noisy_json_does_not_salvage_nested_object_from_invalid_final_object():
    with pytest.raises(json.JSONDecodeError):
        loads_json_output('Draft: {"summary":"Earlier draft"}\nFinal: {broken: {"summary":"Nested"}}')


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
@pytest.mark.parametrize(
    "reasoning_settings",
    [{}, {"reasoning_effort": "high"}, {"thinking_budget_tokens": 0}, {"reasoning_effort": "high", "thinking_budget_tokens": 512}],
)
@pytest.mark.parametrize(
    ("prepare", "request_type", "input_data"),
    [
        (prepare_cluster, ClusterRequest, {"stories": [{"id": "s1", "summary": "Story", "tags": {}}]}),
        (prepare_cybersec_classification, CybersecClassificationRequest, {"text": "Story"}),
        (prepare_ner, NerRequest, {"text": "Story"}),
        (prepare_sentiment, SentimentRequest, {"text": "Story"}),
        (prepare_summary, SummarizeRequest, {"text": "Story"}),
        (prepare_title, TitleRequest, {"text": "Story"}),
    ],
)
def test_prepared_tasks_reasoning_settings_preserve_shared_client(api_mode, reasoning_settings, prepare, request_type, input_data):
    client = LLMClient(api_mode=api_mode, model="batch-model", reasoning_effort="low", thinking_budget_tokens=256)
    task = prepare(request_type(**input_data, **reasoning_settings))

    body = task.build_request(client)

    effort = reasoning_settings.get("reasoning_effort", "low")
    if api_mode == "responses":
        assert body["reasoning"] == {"effort": effort}
    else:
        assert body["reasoning_effort"] == effort
    assert body["thinking_budget_tokens"] == reasoning_settings.get("thinking_budget_tokens", 256)
    assert body["model"] == "batch-model"
    assert client.reasoning_effort == "low"
    assert client.thinking_budget_tokens == 256
