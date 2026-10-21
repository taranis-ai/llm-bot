import pytest

from llm_bot.schemas import CybersecClassificationRequest, CybersecClassificationResponse
from llm_bot.tasks.cybersec_classification import (
    build_cybersec_classification_messages,
    classify_cybersecurity_text,
    parse_cybersec_classification_response,
    prepare_cybersec_classification,
)
from tests.test_helpers import StubLLMClient


def test_build_cybersec_classification_messages():
    request = CybersecClassificationRequest(text="APT29 used Mimikatz.", reasoning_effort="medium")

    system_message, user_message = build_cybersec_classification_messages(request)

    assert "cybersecurity relevance classification system" in system_message["content"]
    assert user_message["content"] == "APT29 used Mimikatz."


def test_parse_cybersec_classification_response():
    response = parse_cybersec_classification_response({"output_text": '{"cybersecurity":0.97,"non-cybersecurity":0.03}'})

    assert response == CybersecClassificationResponse.model_validate({"cybersecurity": 0.97, "non-cybersecurity": 0.03})


@pytest.mark.asyncio
async def test_classify_cybersecurity_text_calls_client():
    client = StubLLMClient({"output_text": '{"cybersecurity":0.97,"non-cybersecurity":0.03}'})

    response = await classify_cybersecurity_text(
        CybersecClassificationRequest(
            text="APT29 used Mimikatz.",
            reasoning_effort="high",
            thinking_budget_tokens=128,
        ),
        client=client,
    )

    assert response == CybersecClassificationResponse.model_validate({"cybersecurity": 0.97, "non-cybersecurity": 0.03})
    assert client.calls[0]["user_input"] == "APT29 used Mimikatz."
    assert client.calls[0]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_classify_cybersecurity_text_retries_once_on_invalid_output():
    client = StubLLMClient(
        [
            {"output_text": '{"cybersecurity": 2.0, "non-cybersecurity": -1.0}'},
            {"output_text": '{"cybersecurity":0.61,"non-cybersecurity":0.39}'},
        ]
    )

    response = await classify_cybersecurity_text(
        CybersecClassificationRequest(text="The report discusses phishing campaigns."),
        client=client,
    )

    assert response == CybersecClassificationResponse.model_validate({"cybersecurity": 0.61, "non-cybersecurity": 0.39})
    assert len(client.calls) == 2
    assert "Your previous response was invalid." in client.calls[1]["system_input"]


@pytest.mark.parametrize("api_mode", ["systemone", "openai"])
async def test_decisions_classification_and_prepared_request(monkeypatch, api_mode):
    from llm_bot.decisions_client import DecisionsClient

    monkeypatch.setattr("llm_bot.config.Config.DECISIONS_URL", "https://decisions.example/decisions")
    monkeypatch.setattr("llm_bot.config.Config.LLM_REASONING_PROFILE", "gemma")
    client = DecisionsClient(api_mode=api_mode, model="decision-model")
    probabilities = {"cybersecurity": 0.97, "non-cybersecurity": 0.03}
    calls = []

    async def create_decision(state, questions):
        calls.append(client.build_request(state, questions))
        return {"cybersecurity_relevance": probabilities}

    monkeypatch.setattr(client, "create_decision", create_decision)
    request = CybersecClassificationRequest(text="APT29 used Mimikatz.", reasoning_effort="high", thinking_budget_tokens=128)
    task = prepare_cybersec_classification(request)

    result = await classify_cybersecurity_text(request, client=client)

    assert result.model_dump() == probabilities
    assert calls == [task.build_request(client)]
    assert set(calls[0]) == {"model", "questions", "state" if api_mode == "systemone" else "input"}
    assert calls[0].get("state", calls[0].get("input")) == request.text
    if api_mode == "systemone":
        raw_response = {"answers": {"cybersecurity_relevance": {"type": "choice", "choice": "cybersecurity", "probabilities": probabilities}}}
    else:
        raw_response = {
            "answers": [
                {
                    "type": "choice",
                    "name": "cybersecurity_relevance",
                    "choice": "cybersecurity",
                    "probabilities": [{"value": name, "probability": value} for name, value in probabilities.items()],
                }
            ]
        }
    assert task.parse_result(raw_response, client).model_dump() == probabilities


async def test_explicit_llm_client_overrides_configured_decisions(monkeypatch):
    monkeypatch.setattr("llm_bot.config.Config.DECISIONS_URL", "https://decisions.example/decisions")
    client = StubLLMClient({"output_text": '{"cybersecurity":0.97,"non-cybersecurity":0.03}'})

    result = await classify_cybersecurity_text(CybersecClassificationRequest(text="APT29 used Mimikatz."), client=client)

    assert result.cybersecurity == 0.97
    assert len(client.calls) == 1


async def test_unconfigured_decisions_retains_default_llm_route(monkeypatch):
    monkeypatch.setattr("llm_bot.config.Config.DECISIONS_URL", "")
    client = StubLLMClient({"output_text": '{"cybersecurity":0.97,"non-cybersecurity":0.03}'})

    def make_client(**kwargs):
        assert kwargs == {"reasoning_effort": "high", "thinking_budget_tokens": 128}
        return client

    monkeypatch.setattr("llm_bot.tasks.cybersec_classification.LLMClient", make_client)

    result = await classify_cybersecurity_text(
        CybersecClassificationRequest(text="APT29 used Mimikatz.", reasoning_effort="high", thinking_budget_tokens=128)
    )

    assert result.cybersecurity == 0.97
    assert len(client.calls) == 1
