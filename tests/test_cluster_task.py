import json

import pytest

from llm_bot.schemas import ClusterRequest, ClusterResponse
from llm_bot.tasks.cluster import (
    build_cluster_messages,
    cluster_stories,
    parse_cluster_response,
)
from llm_bot.tasks.llm_utils import InvalidLLMOutputError
from tests.test_helpers import StubLLMClient


def test_build_cluster_messages_concatenates_all_news_items(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.cluster.Config.CLUSTER_MAX_CONTENT_CHARS_PER_STORY", 30)
    request = ClusterRequest.model_validate(
        {
            "stories": [
                {
                    "id": "existing-story",
                    "title": "Canonical cluster title",
                    "tags": {
                        "APT29": {"name": "APT29", "tag_type": "GROUP"},
                        "Microsoft": {"tag_type": "ORG"},
                    },
                    "news_items": [
                        {
                            "title": "Initial report",
                            "content": "First item content about phishing against Microsoft accounts.",
                            "language": "en",
                        },
                        {
                            "title": "Follow-up report",
                            "content": "Second item confirms Vienna scope and mentions Mimikatz usage.",
                            "language": "de",
                        },
                    ],
                }
            ]
        }
    )

    _, user_message = build_cluster_messages(request)
    payload = json.loads(user_message["content"])
    story = payload["stories"][0]

    assert story["id"] == 1
    assert story["title"] == "Canonical cluster title"
    assert story["source_languages"] == ["de", "en"]
    assert story["tags"] == {"APT29": "GROUP", "Microsoft": "ORG"}
    assert story["content"] == (
        "First item content about phis…\n\n"
        "Second item confirms Vienna s…"
    )


def test_parse_cluster_response_from_output_text():
    response = parse_cluster_response(
        {
            "output_text": (
                '{"cluster_ids":{"event_clusters":[[1,2],[3]]},'
                '"cluster_reasons":[{"story_ids":[1,2],"reason":"Same event"}],'
                '"message":"Clustering completed"}'
            )
        },
        expected_story_ids={"s1", "s2", "s3"},
        story_id_map={1: "s1", 2: "s2", 3: "s3"},
    )

    assert response == ClusterResponse.model_validate(
        {
            "cluster_ids": {"event_clusters": [["s1", "s2"], ["s3"]]},
            "message": "Clustering completed",
        }
    )


def test_parse_cluster_response_rejects_missing_story_ids():
    with pytest.raises(InvalidLLMOutputError, match="Missing story IDs"):
        parse_cluster_response(
            {
                "output_text": (
                    '{"cluster_ids":{"event_clusters":[[1,2]]},'
                    '"cluster_reasons":[{"story_ids":[1,2],"reason":"Same event"}],'
                    '"message":"Clustering completed"}'
                )
            },
            expected_story_ids={"s1", "s2", "s3"},
            story_id_map={1: "s1", 2: "s2", 3: "s3"},
        )


def test_parse_cluster_response_rejects_missing_cluster_reason():
    with pytest.raises(InvalidLLMOutputError, match="Missing cluster_reasons entries"):
        parse_cluster_response(
            {
                "output_text": (
                    '{"cluster_ids":{"event_clusters":[[1,2],[3]]},'
                    '"cluster_reasons":[],'
                    '"message":"Clustering completed"}'
                )
            },
            expected_story_ids={"s1", "s2", "s3"},
            story_id_map={1: "s1", 2: "s2", 3: "s3"},
        )


def test_parse_cluster_response_rejects_reason_for_unknown_cluster():
    with pytest.raises(InvalidLLMOutputError, match="do not match any returned non-singleton cluster"):
        parse_cluster_response(
            {
                "output_text": (
                    '{"cluster_ids":{"event_clusters":[[1,2],[3]]},'
                    '"cluster_reasons":[{"story_ids":[2,3],"reason":"Wrong pair"}],'
                    '"message":"Clustering completed"}'
                )
            },
            expected_story_ids={"s1", "s2", "s3"},
            story_id_map={1: "s1", 2: "s2", 3: "s3"},
        )


def test_parse_cluster_response_rejects_unknown_llm_story_id():
    with pytest.raises(InvalidLLMOutputError, match="Unexpected story IDs in cluster output: 4"):
        parse_cluster_response(
            {
                "output_text": (
                    '{"cluster_ids":{"event_clusters":[[1,4],[3]]},'
                    '"cluster_reasons":[{"story_ids":[1,4],"reason":"Same event"}],'
                    '"message":"Clustering completed"}'
                )
            },
            expected_story_ids={"s1", "s2", "s3"},
            story_id_map={1: "s1", 2: "s2", 3: "s3"},
        )


@pytest.mark.asyncio
async def test_cluster_stories_sends_reduced_story_representation():
    client = StubLLMClient(
        {
            "output_text": (
                '{"cluster_ids":{"event_clusters":[[1,2]]},'
                '"cluster_reasons":[{"story_ids":[1,2],"reason":"Same event"}],'
                '"message":"Clustering completed"}'
            )
        }
    )
    request = ClusterRequest.model_validate(
        {
            "stories": [
                {
                    "id": "s1",
                    "tags": {"APT29": {"name": "APT29", "tag_type": "GROUP"}},
                    "news_items": [
                        {
                            "title": "First report",
                            "content": "First report content.",
                            "language": "en",
                        },
                        {
                            "title": "Second report",
                            "content": "Second report content with more evidence.",
                            "language": "en",
                        },
                    ],
                },
                {
                    "id": "s2",
                    "tags": {"Vienna": {"tag_type": "GPE"}},
                    "news_items": [
                        {
                            "title": "Third report",
                            "content": "Third report content.",
                            "language": "en",
                        }
                    ],
                },
            ]
        }
    )

    response = await cluster_stories(request, client=client)

    assert response == ClusterResponse.model_validate(
        {
            "cluster_ids": {"event_clusters": [["s1", "s2"]]},
            "message": "Clustering completed",
        }
    )
    sent_payload = json.loads(client.calls[0]["user_input"])
    assert sent_payload["stories"][0]["id"] == 1
    assert sent_payload["stories"][1]["id"] == 2
    assert sent_payload["stories"][0]["content"] == "First report content.\n\nSecond report content with more evidence."
    assert sent_payload["stories"][1]["tags"] == {"Vienna": "GPE"}
    assert sent_payload["stories"][1]["source_languages"] == ["en"]
