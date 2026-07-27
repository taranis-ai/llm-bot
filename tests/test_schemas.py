import pytest
from pydantic import ValidationError

from llm_bot.schemas import (
    EntityRelationshipExtractionRequest,
    GraphQueryGenerationRequest,
    SummarizeRequest,
    TitleRequest,
)


def test_summarize_request_accepts_text_input():
    request = SummarizeRequest.model_validate({"text": "Story text"})

    assert request.text == "Story text"
    assert request.news_items is None
    assert request.reasoning_effort is None
    assert request.thinking_budget_tokens is None


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
    assert request.reasoning_effort is None
    assert request.thinking_budget_tokens is None


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


@pytest.mark.parametrize(
    ("schema_update", "error"),
    [
        (
            {
                "entity_types": [
                    {"name": "Thing", "description": "First"},
                    {"name": "Thing", "description": "Duplicate"},
                ]
            },
            "Entity type names must be unique",
        ),
        (
            {
                "relation_types": [
                    {
                        "name": "USES",
                        "source_types": ["Unknown"],
                        "target_types": ["Thing"],
                    }
                ]
            },
            "reference unknown entity types",
        ),
        (
            {
                "relation_types": [
                    {
                        "name": "USES",
                        "source_types": [],
                        "target_types": ["Thing"],
                    }
                ]
            },
            "at least 1 item",
        ),
    ],
)
def test_entity_relationship_request_rejects_invalid_schema(schema_update, error):
    schema = {
        "entity_types": [{"name": "Thing", "description": "A thing"}],
        "relation_types": [],
    }
    schema.update(schema_update)

    with pytest.raises(ValidationError, match=error):
        EntityRelationshipExtractionRequest.model_validate(
            {"text": "Some text", "schema": schema}
        )


def test_graph_query_request_rejects_unknown_relationship_label():
    with pytest.raises(ValidationError, match="reference unknown node labels"):
        GraphQueryGenerationRequest.model_validate(
            {
                "question": "What is connected?",
                "graph_name": "example",
                "schema": {
                    "node_labels": [
                        {
                            "label": "Known",
                            "properties": [{"name": "name", "type": "string"}],
                        }
                    ],
                    "relationship_types": [
                        {
                            "type": "CONNECTS",
                            "source_labels": ["Known"],
                            "target_labels": ["Unknown"],
                        }
                    ],
                    "default_limit": 10,
                    "maximum_limit": 100,
                },
            }
        )


def test_graph_query_request_rejects_default_limit_above_maximum():
    with pytest.raises(ValidationError, match="default limit must not exceed maximum"):
        GraphQueryGenerationRequest.model_validate(
            {
                "question": "List known nodes",
                "graph_name": "example",
                "schema": {
                    "node_labels": [{"label": "Known", "properties": []}],
                    "relationship_types": [],
                    "default_limit": 101,
                    "maximum_limit": 100,
                },
            }
        )
