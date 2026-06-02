import pytest

from llm_bot.schemas import SentimentRequest, SentimentResponse
from llm_bot.tasks.sentiment import (
    analyze_sentiment,
    build_sentiment_messages,
    get_sentiment_response_format,
    parse_sentiment_response,
)
from tests.test_helpers import StubLLMClient

def test_build_sentiment_messages_without_emotions():
    request = SentimentRequest(text="The report was factual and dry.")

    system_message, user_message = build_sentiment_messages(request)

    assert "Emotion extraction is disabled." in system_message["content"]
    assert "Do not include an emotions field in the response." in system_message["content"]
    assert user_message["content"] == "The report was factual and dry."


def test_build_sentiment_messages_with_emotions():
    request = SentimentRequest(text="Markets rallied after the announcement.", include_emotions=True)

    system_message, user_message = build_sentiment_messages(request)

    assert "Emotion extraction is enabled." in system_message["content"]
    assert "Allowed emotions:" in system_message["content"]
    assert "Return the emotions field as an array, even if it is empty." in system_message["content"]
    assert user_message["content"] == "Markets rallied after the announcement."


def test_get_sentiment_response_format_without_emotions():
    response_format = get_sentiment_response_format(False)
    sentiment_schema = response_format["schema"]["properties"]["sentiment"]

    assert sentiment_schema["required"] == ["label", "score"]
    assert "emotions" not in sentiment_schema["properties"]


def test_get_sentiment_response_format_with_emotions():
    response_format = get_sentiment_response_format(True)
    sentiment_schema = response_format["schema"]["properties"]["sentiment"]

    assert sentiment_schema["required"] == ["label", "score", "emotions"]
    assert sentiment_schema["properties"]["emotions"]["uniqueItems"] is True


def test_parse_sentiment_response_without_emotions():
    response = parse_sentiment_response({"output_text": '{"sentiment":{"label":"neutral","score":0.62}}'})

    assert response == SentimentResponse.model_validate({"sentiment": {"label": "neutral", "score": 0.62}})


def test_parse_sentiment_response_with_emotions():
    response = parse_sentiment_response(
        {"output_text": '{"sentiment":{"label":"negative","score":0.91,"emotions":["anger","fear"]}}'},
        include_emotions=True,
    )

    assert response == SentimentResponse.model_validate(
        {"sentiment": {"label": "negative", "score": 0.91, "emotions": ["anger", "fear"]}}
    )


@pytest.mark.asyncio
async def test_analyze_sentiment_calls_client_without_emotions():
    client = StubLLMClient({"output_text": '{"sentiment":{"label":"positive","score":0.88}}'})

    response = await analyze_sentiment(SentimentRequest(text="The launch was a success."), client=client)

    assert response == SentimentResponse.model_validate({"sentiment": {"label": "positive", "score": 0.88}})
    assert client.calls[0]["user_input"] == "The launch was a success."
    assert client.calls[0]["response_format"]["type"] == "json_schema"
    assert "emotions" not in client.calls[0]["response_format"]["schema"]["properties"]["sentiment"]["properties"]


@pytest.mark.asyncio
async def test_analyze_sentiment_calls_client_with_emotions():
    client = StubLLMClient(
        {"output_text": '{"sentiment":{"label":"negative","score":0.93,"emotions":["fear","anger"]}}'}
    )

    response = await analyze_sentiment(
        SentimentRequest(text="The attack left investors alarmed and furious.", include_emotions=True),
        client=client,
    )

    assert response == SentimentResponse.model_validate(
        {"sentiment": {"label": "negative", "score": 0.93, "emotions": ["fear", "anger"]}}
    )
    assert "emotions" in client.calls[0]["response_format"]["schema"]["properties"]["sentiment"]["properties"]


@pytest.mark.asyncio
async def test_analyze_sentiment_retries_once_on_invalid_output():
    client = StubLLMClient(
        [
            {"output_text": '{"sentiment":{"label":"positive","score":0.72,"emotions":["joy"]}}'},
            {"output_text": '{"sentiment":{"label":"positive","score":0.72}}'},
        ]
    )

    response = await analyze_sentiment(SentimentRequest(text="The results were encouraging."), client=client)

    assert response == SentimentResponse.model_validate({"sentiment": {"label": "positive", "score": 0.72}})
    assert len(client.calls) == 2
    assert "Your previous response was invalid." in client.calls[1]["system_input"]
