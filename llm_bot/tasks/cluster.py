import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.log import logger
from llm_bot.schemas import (
    ClusterIds,
    ClusterRequest,
    ClusterReason,
    ClusterResponse,
    LLMClusterResponse,
    StoryClusterItem,
    StoryTag,
)
from llm_bot.tasks.llm_utils import (
    InvalidLLMOutputError,
    create_and_parse_response,
    get_output_text,
    loads_json_output,
)
from llm_bot.tasks.task_utils import truncate_text


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "cluster.txt"


def load_cluster_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def serialize_story_tags(tags: dict[str, StoryTag]) -> dict[str, str]:
    serialized_tags = {
        name: tag.tag_type
        for name, tag in sorted(tags.items(), key=lambda item: (item[0], item[1].tag_type))
    }
    return serialized_tags


def build_story_content(story: StoryClusterItem) -> str:
    content_parts = [
        truncate_text(news_item.content, Config.CLUSTER_MAX_CONTENT_CHARS_PER_STORY)
        for news_item in story.news_items
    ]
    return "\n\n".join(content_parts)


def build_compact_story(story: StoryClusterItem) -> dict[str, object]:
    source_languages = sorted({news_item.language for news_item in story.news_items if news_item.language})
    return {
        "id": story.id,
        "title": getattr(story, "title", "") or story.news_items[0].title,
        "tags": serialize_story_tags(story.tags),
        "source_languages": source_languages,
        "content": build_story_content(story),
    }


def build_story_id_map(stories: list[StoryClusterItem]) -> dict[int, str]:
    return {index: story.id for index, story in enumerate(stories, start=1)}


def build_llm_story(story: StoryClusterItem, llm_story_id: int) -> dict[str, object]:
    compact_story = build_compact_story(story)
    compact_story["id"] = llm_story_id
    return compact_story


def build_cluster_messages(request: ClusterRequest) -> list[dict[str, str]]:
    story_id_map = build_story_id_map(request.stories)
    user_payload = {
        "stories": [
            build_llm_story(story, llm_story_id)
            for llm_story_id, story in zip(story_id_map, request.stories, strict=True)
        ]
    }
    return [
        {"role": "system", "content": load_cluster_prompt()},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
    ]


def remap_story_ids(story_ids: list[int], story_id_map: dict[int, str]) -> list[str]:
    remapped_story_ids: list[str] = []
    unknown_story_ids: list[int] = []
    for story_id in story_ids:
        if story_id not in story_id_map:
            unknown_story_ids.append(story_id)
            continue
        remapped_story_ids.append(story_id_map[story_id])

    if unknown_story_ids:
        raise InvalidLLMOutputError(
            f"Unexpected story IDs in cluster output: {', '.join(str(story_id) for story_id in sorted(unknown_story_ids))}"
        )

    return remapped_story_ids


def remap_cluster_response(
    response: LLMClusterResponse,
    *,
    story_id_map: dict[int, str],
) -> tuple[ClusterIds, list[ClusterReason], str]:
    cluster_ids = ClusterIds.model_validate(
        {
            "event_clusters": [
                remap_story_ids(cluster, story_id_map)
                for cluster in response.cluster_ids.event_clusters
            ]
        }
    )
    cluster_reasons = [
        ClusterReason.model_validate(
            {
                "story_ids": remap_story_ids(reason.story_ids, story_id_map),
                "reason": reason.reason,
            }
        )
        for reason in response.cluster_reasons
    ]
    return cluster_ids, cluster_reasons, response.message


def validate_cluster_response(
    cluster_ids: ClusterIds,
    cluster_reasons: list[ClusterReason],
    message: str,
    *,
    expected_story_ids: set[str],
) -> ClusterResponse:
    assigned_story_ids: list[str] = [
        story_id
        for cluster in cluster_ids.event_clusters
        for story_id in cluster
    ]
    assigned_story_id_set = set(assigned_story_ids)

    if len(assigned_story_ids) != len(assigned_story_id_set):
        duplicate_ids = sorted(
            story_id for story_id, count in Counter(assigned_story_ids).items() if count > 1
        )
        raise InvalidLLMOutputError(
            f"Duplicate story IDs in cluster output: {', '.join(duplicate_ids)}"
        )

    missing_story_ids = sorted(expected_story_ids - assigned_story_id_set)
    if missing_story_ids:
        raise InvalidLLMOutputError(
            f"Missing story IDs in cluster output: {', '.join(missing_story_ids)}"
        )

    unexpected_story_ids = sorted(assigned_story_id_set - expected_story_ids)
    if unexpected_story_ids:
        raise InvalidLLMOutputError(
            f"Unexpected story IDs in cluster output: {', '.join(unexpected_story_ids)}"
        )

    non_singleton_clusters = {frozenset(cluster) for cluster in cluster_ids.event_clusters if len(cluster) >= 2}
    reason_clusters = [frozenset(reason.story_ids) for reason in cluster_reasons]

    if len(reason_clusters) != len(set(reason_clusters)):
        raise InvalidLLMOutputError("Duplicate cluster_reasons entries in cluster output")

    unexpected_reason_clusters = sorted(
        sorted(cluster) for cluster in set(reason_clusters) - non_singleton_clusters
    )
    if unexpected_reason_clusters:
        raise InvalidLLMOutputError(
            "cluster_reasons contains entries that do not match any returned non-singleton cluster"
        )

    missing_reason_clusters = sorted(
        sorted(cluster) for cluster in non_singleton_clusters - set(reason_clusters)
    )
    if missing_reason_clusters:
        raise InvalidLLMOutputError(
            "Missing cluster_reasons entries for returned non-singleton clusters"
        )

    return ClusterResponse.model_validate(
        {
            "cluster_ids": cluster_ids.model_dump(),
            "message": message,
        }
    )


def parse_cluster_response(
    response_data: dict[str, Any],
    *,
    expected_story_ids: set[str],
    story_id_map: dict[int, str],
) -> ClusterResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw cluster output: %s", output_text)
    parsed_output = loads_json_output(output_text)
    cluster_ids, cluster_reasons, message = remap_cluster_response(
        LLMClusterResponse.model_validate(parsed_output),
        story_id_map=story_id_map,
    )
    return validate_cluster_response(
        cluster_ids,
        cluster_reasons,
        message,
        expected_story_ids=expected_story_ids,
    )


def get_cluster_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "cluster_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["cluster_ids", "cluster_reasons", "message"],
            "properties": {
                "cluster_ids": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["event_clusters"],
                    "properties": {
                        "event_clusters": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                        }
                    },
                },
                "cluster_reasons": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["story_ids", "reason"],
                        "properties": {
                            "story_ids": {
                                "type": "array",
                                "minItems": 2,
                                "items": {"type": "integer"},
                            },
                            "reason": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                    },
                },
                "message": {
                    "type": "string",
                    "minLength": 1,
                },
            },
        },
    }


async def cluster_stories(request: ClusterRequest, client: LLMClient | None = None) -> ClusterResponse:
    llm_client = client or LLMClient(
        reasoning_effort=request.reasoning_effort,
        thinking_budget_tokens=request.thinking_budget_tokens,
    )
    story_id_map = build_story_id_map(request.stories)
    system_message, user_message = build_cluster_messages(request)
    expected_story_ids = {story.id for story in request.stories}
    return await create_and_parse_response(
        client=llm_client,
        task_name="cluster",
        user_input=user_message["content"],
        system_input=system_message["content"],
        response_format=get_cluster_response_format(),
        parse_response=lambda response_data: parse_cluster_response(
            response_data,
            expected_story_ids=expected_story_ids,
            story_id_map=story_id_map,
        ),
    )
