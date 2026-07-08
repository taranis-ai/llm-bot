import pytest

from llm_bot.schemas import CybersecClassificationRequest, CybersecClassificationResponse
from llm_bot.tasks.cybersec_classification import (
    build_cybersec_classification_messages,
    classify_cybersecurity_text,
    parse_cybersec_classification_response,
)
from tests.test_helpers import StubLLMClient


def test_build_cybersec_classification_messages():
    request = CybersecClassificationRequest(text="APT29 used Mimikatz.", reasoning_effort="medium")

    system_message, user_message = build_cybersec_classification_messages(request)

    assert "cybersecurity relevance classification system" in system_message["content"]
    assert user_message["content"] == "APT29 used Mimikatz."


def test_parse_cybersec_classification_response():
    response = parse_cybersec_classification_response(
        {"output_text": '{"cybersecurity":0.97,"non-cybersecurity":0.03}'}
    )

    assert response == CybersecClassificationResponse.model_validate(
        {"cybersecurity": 0.97, "non-cybersecurity": 0.03}
    )


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

    assert response == CybersecClassificationResponse.model_validate(
        {"cybersecurity": 0.97, "non-cybersecurity": 0.03}
    )
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

    assert response == CybersecClassificationResponse.model_validate(
        {"cybersecurity": 0.61, "non-cybersecurity": 0.39}
    )
    assert len(client.calls) == 2
    assert "Your previous response was invalid." in client.calls[1]["system_input"]
