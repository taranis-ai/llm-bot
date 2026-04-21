import json
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.log import logger
from llm_bot.schemas import ClusterRequest, ClusterResponse
from llm_bot.tasks.llm_utils import create_and_parse_response, get_output_text


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "cluster.txt"


def load_cluster_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_cluster_messages(request: ClusterRequest) -> list[dict[str, str]]:
    compact_stories: list[dict[str, Any]] = []
    for story in request.stories:
        first_news_item = story.news_items[0]
        tags = list(story.tags.keys())[: Config.CLUSTER_MAX_TAGS_PER_STORY]
        summary_text = first_news_item.review or first_news_item.content
        compact_stories.append(
            {
                "id": story.id,
                "tags": tags,
                "title": first_news_item.title,
                "summary_text": _truncate_text(summary_text, Config.CLUSTER_MAX_CONTENT_CHARS_PER_STORY),
                "language": first_news_item.language,
            }
        )

    user_payload = {"stories": compact_stories}
    return [
        {"role": "system", "content": load_cluster_prompt()},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
    ]

def parse_cluster_response(response_data: dict[str, Any]) -> ClusterResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw cluster output: %s", output_text)
    parsed_output = json.loads(output_text)
    return ClusterResponse.model_validate(parsed_output)


def get_cluster_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "cluster_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["cluster_ids", "message"],
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
                                "items": {"type": "string"},
                            },
                        }
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
    llm_client = client or LLMClient()
    system_message, user_message = build_cluster_messages(request)
    return await create_and_parse_response(
        client=llm_client,
        task_name="cluster",
        input_text=user_message["content"],
        instructions=system_message["content"],
        response_format=get_cluster_response_format(),
        parse_response=parse_cluster_response,
    )
