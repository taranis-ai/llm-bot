import json
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest
from niquests.exceptions import HTTPError

from llm_bot.client import UpstreamLLMError
from llm_bot.decisions_client import ChoiceQuestion, DecisionsClient

QUESTIONS = {"topic": ChoiceQuestion(instructions="Choose the topic", criteria={"security": "Cybersecurity", "other": "Other topics"})}
ANSWER = {"type": "choice", "choice": "security", "probabilities": {"security": 0.9, "other": 0.1}, "confidence": 0.8}
OPENAI_ANSWER = {
    "type": "choice",
    "name": "topic",
    "choice": "security",
    "confidence": 0.8,
    "probabilities": [{"value": "security", "probability": 0.9}, {"value": "other", "probability": 0.1}],
}


@pytest.fixture
def transport(monkeypatch):
    response = Mock(text=json.dumps({"answers": {"topic": ANSWER}}))
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.post.return_value = response
    factory = Mock(return_value=session)
    monkeypatch.setattr("llm_bot.decisions_client.AsyncSession", factory)
    return factory, session, response


@pytest.mark.parametrize("api_mode", ["systemone", "openai"])
@pytest.mark.parametrize("api_key", ["decision-key", ""])
async def test_decision_transport_uses_independent_connection(transport, monkeypatch, api_mode, api_key):
    factory, session, response = transport
    monkeypatch.setattr("llm_bot.config.Config.LLM_API_KEY", "unrelated-llm-key")
    monkeypatch.setattr("llm_bot.config.Config.DECISIONS_URL", "https://decisions.example/api/alpha/decisions")
    monkeypatch.setattr("llm_bot.config.Config.DECISIONS_API_KEY", "configured-decision-key")
    monkeypatch.setattr("llm_bot.config.Config.DECISIONS_MODEL", "decision-model")
    monkeypatch.setattr("llm_bot.config.Config.DECISIONS_TIMEOUT", 17)
    if api_mode == "openai":
        response.text = json.dumps({"answers": [OPENAI_ANSWER]})
    client = DecisionsClient(api_mode=api_mode, api_key=api_key)

    result = await client.create_decision("Untrusted article text", QUESTIONS)

    assert result == {"topic": {"security": 0.9, "other": 0.1}}
    assert factory.call_args.kwargs == {
        "headers": {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
    }
    call = session.post.call_args
    assert call.args == ("https://decisions.example/api/alpha/decisions",)
    assert call.kwargs["timeout"] == 17
    if api_mode == "systemone":
        assert call.kwargs["json"] == {
            "model": "decision-model",
            "state": "Untrusted article text",
            "questions": {"topic": {"type": "choice", "instructions": "Choose the topic", "criteria": QUESTIONS["topic"].criteria}},
        }
    else:
        assert call.kwargs["json"] == {
            "model": "decision-model",
            "input": "Untrusted article text",
            "questions": [
                {
                    "type": "choice",
                    "name": "topic",
                    "instructions": "Choose the topic",
                    "choices": [{"value": "security", "description": "Cybersecurity"}, {"value": "other", "description": "Other topics"}],
                }
            ],
        }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"answers": {}},
        {"answers": {"wrong_question": ANSWER}},
        {"answers": {"topic": {**ANSWER, "type": "refusal"}}},
        {"answers": {"topic": {**ANSWER, "choice": "unknown"}}},
        *[
            {"answers": {"topic": {**ANSWER, "probabilities": probabilities}}}
            for probabilities in [
                {"security": 0.9},
                {"security": 0.9, "other": 0.1, "unknown": 0.0},
                {"security": 0.9, "other": 0.9},
                {"security": 1.1, "other": -0.1},
                {"security": float("nan"), "other": 0.1},
                {"security": "0.9", "other": 0.1},
                {"security": True, "other": 0.0},
            ]
        ],
    ],
)
def test_rejects_invalid_systemone_answers(payload):
    with pytest.raises(UpstreamLLMError, match="Invalid decisions response"):
        DecisionsClient(api_mode="systemone").parse_response(payload, QUESTIONS)


@pytest.mark.parametrize("failure", ["duplicate_question", "duplicate_choice", "refusal", "unknown_choice"])
def test_rejects_invalid_openai_answers(failure):
    answer = deepcopy(OPENAI_ANSWER)
    answers = [answer]
    if failure == "duplicate_question":
        answers.append(answer)
    elif failure == "duplicate_choice":
        answer["probabilities"].append(answer["probabilities"][0])
    elif failure == "refusal":
        answer["type"] = "refusal"
    else:
        answer["probabilities"][0]["value"] = "unexpected"
    with pytest.raises(UpstreamLLMError, match="Invalid decisions response"):
        DecisionsClient(api_mode="openai").parse_response({"answers": answers}, QUESTIONS)


@pytest.mark.parametrize("body", ['{"error":{"message":"Unknown decisions model"}}', "not json"])
async def test_decision_transport_failures_do_not_retry(transport, body):
    _, session, response = transport
    response.text = body
    if body.startswith("{"):
        response.raise_for_status.side_effect = HTTPError("bad request")
        message = "Unknown decisions model"
    else:
        message = "Invalid decisions response"
    with pytest.raises(UpstreamLLMError, match=message):
        await DecisionsClient(url="https://decisions.example/decisions").create_decision("Article", QUESTIONS)
    assert session.post.call_count == 1


@pytest.mark.parametrize(
    "body",
    [
        ('{"answers":{"topic":{"type":"choice","choice":"security","probabilities":{"security":0.1,"security":0.9,"other":0.1}}}}'),
        (
            '{"answers":{"topic":{"type":"refusal"},"topic":'
            '{"type":"choice","choice":"security","probabilities":{"security":0.9,"other":0.1}}}}'
        ),
    ],
)
async def test_duplicate_json_keys_are_rejected_before_validation(transport, body):
    _, session, response = transport
    response.text = body

    with pytest.raises(UpstreamLLMError, match="Invalid decisions response"):
        await DecisionsClient(url="https://decisions.example/decisions", api_mode="systemone").create_decision("Article", QUESTIONS)

    assert session.post.call_count == 1


def test_unknown_mode_fails_before_transport(transport):
    factory, _, _ = transport
    with pytest.raises(ValueError, match="DECISIONS_API_MODE"):
        DecisionsClient(api_mode="typo")
    assert not factory.called


async def test_missing_url_fails_before_transport(transport):
    factory, _, _ = transport
    with pytest.raises(ValueError, match="DECISIONS_URL"):
        await DecisionsClient(url="").create_decision("Article", QUESTIONS)
    assert not factory.called


@pytest.mark.parametrize("failure", [None, "upstream", "invalid"])
@pytest.mark.parametrize("endpoint", ["/cybersec-classification", "/classify", "/sentiment"])
async def test_classification_route_uses_decisions_and_maps_errors(app, transport, monkeypatch, failure, endpoint):
    _, session, response = transport
    monkeypatch.setattr("llm_bot.config.Config.DECISIONS_URL", "https://decisions.example/decisions")
    monkeypatch.setattr("llm_bot.config.Config.DECISIONS_MODEL", "decision-model")
    monkeypatch.setattr("llm_bot.config.Config.DECISIONS_API_MODE", "systemone")
    monkeypatch.setattr("llm_bot.config.Config.API_KEY", "incoming-key")
    probabilities = {"cybersecurity": 0.9, "non-cybersecurity": 0.1}
    question_name, choice = "cybersecurity_relevance", "cybersecurity"
    payload = {"text": "APT29 used Mimikatz.", "reasoning_effort": "high", "thinking_budget_tokens": 256}
    error_prefix = "Failed to classify text for cybersecurity relevance"
    if endpoint == "/classify":
        probabilities = {
            "disinformation": 0.01,
            "security incident": 0.8,
            "vulnerability": 0.04,
            "threat intelligence": 0.05,
            "security policy": 0.02,
            "security news": 0.03,
            "general news": 0.03,
            "other": 0.02,
        }
        question_name, choice = "category", "security incident"
        payload = {"text": "A ransomware attack shut down a hospital."}
        error_prefix = "Failed to classify text"
    if endpoint == "/sentiment":
        probabilities = {"positive": 0.02, "negative": 0.93, "neutral": 0.05}
        question_name, choice = "sentiment", "negative"
        payload["include_emotions"] = True
        error_prefix = "Failed to analyze sentiment"
    answers = {question_name: {**ANSWER, "choice": choice, "probabilities": probabilities}}
    if endpoint == "/sentiment":
        for emotion in ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"):
            present = emotion in {"fear", "anger"}
            answers[emotion] = {
                "type": "choice",
                "choice": "yes" if present else "no",
                "probabilities": {"yes": 0.9, "no": 0.1} if present else {"yes": 0.1, "no": 0.9},
            }
    response.text = json.dumps({"answers": answers})
    if failure == "upstream":
        response.text = '{"error":{"message":"Unknown decisions model"}}'
        response.raise_for_status.side_effect = HTTPError("bad request")
    elif failure == "invalid":
        if endpoint == "/sentiment":
            del answers["fear"]
            response.text = json.dumps({"answers": answers})
        else:
            response.text = '{"answers":{}}'

    result = await app.test_client().post(
        endpoint,
        json=payload,
        headers={"Authorization": "Bearer incoming-key"},
    )

    assert result.status_code == (502 if failure else 200)
    body = await result.get_json()
    if failure:
        assert body == {"error": error_prefix + ": " + ("Unknown decisions model" if failure == "upstream" else "Invalid decisions response")}
    elif endpoint == "/sentiment":
        assert body == {"sentiment": {"label": "negative", "score": 0.93, "emotions": ["fear", "anger"]}}
    else:
        assert body == probabilities
    assert session.post.call_count == 1
    assert set(session.post.call_args.kwargs["json"]) == {"model", "state", "questions"}
