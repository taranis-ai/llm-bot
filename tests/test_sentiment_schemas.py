import pytest
from pydantic import ValidationError

from llm_bot.schemas import SentimentResponse


def test_sentiment_response_allows_valid_emotion_sentiment_combination():
    response = SentimentResponse.model_validate(
        {"sentiment": {"label": "negative", "score": 0.87, "emotions": ["fear", "anger"]}}
    )

    assert response.sentiment.label == "negative"
    assert response.sentiment.emotions == ["fear", "anger"]


def test_sentiment_response_rejects_invalid_emotion_sentiment_combination():
    with pytest.raises(ValidationError, match="Emotions not allowed for sentiment neutral: joy"):
        SentimentResponse.model_validate({"sentiment": {"label": "neutral", "score": 0.51, "emotions": ["joy"]}})


def test_sentiment_response_rejects_duplicate_emotions():
    with pytest.raises(ValidationError, match="Emotions must not contain duplicates"):
        SentimentResponse.model_validate(
            {"sentiment": {"label": "negative", "score": 0.83, "emotions": ["fear", "fear"]}}
        )
