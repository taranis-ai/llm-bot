import pytest
from pydantic import ValidationError

from llm_bot.schemas import SummarizeRequest, TitleRequest


def test_summarize_request_accepts_text_input():
    request = SummarizeRequest.model_validate({"text": "Story text"})

    assert request.text == "Story text"
    assert request.news_items is None


def test_summarize_request_accepts_non_empty_news_items():
    request = SummarizeRequest.model_validate(
        {
            "news_items": [
                {"title": "Story title", "content": ""},
                {"title": "", "content": "Story text"},
            ]
        }
    )

    assert len(request.news_items) == 2


def test_summarize_request_rejects_missing_story_input():
    with pytest.raises(
        ValidationError,
        match="Either text or news_items with at least one non-empty item must be provided",
    ):
        SummarizeRequest.model_validate({})


def test_summarize_request_rejects_only_empty_news_items():
    with pytest.raises(
        ValidationError,
        match="Either text or news_items with at least one non-empty item must be provided",
    ):
        SummarizeRequest.model_validate(
            {
                "news_items": [
                    {"title": "", "content": ""},
                    {"title": "", "content": ""},
                ]
            }
        )


def test_title_request_accepts_text_input():
    request = TitleRequest.model_validate({"text": "Story text"})

    assert request.text == "Story text"
    assert request.max_chars == 100


def test_title_request_accepts_non_empty_news_items():
    request = TitleRequest.model_validate(
        {
            "news_items": [
                {"title": "Story title", "content": ""},
                {"title": "", "content": "Story text"},
            ]
        }
    )

    assert len(request.news_items) == 2


def test_title_request_rejects_missing_story_input():
    with pytest.raises(
        ValidationError,
        match="Either text or news_items with at least one non-empty item must be provided",
    ):
        TitleRequest.model_validate({})


def test_title_request_rejects_only_empty_news_items():
    with pytest.raises(
        ValidationError,
        match="Either text or news_items with at least one non-empty item must be provided",
    ):
        TitleRequest.model_validate(
            {
                "news_items": [
                    {"title": "", "content": ""},
                    {"title": "", "content": ""},
                ]
            }
        )
