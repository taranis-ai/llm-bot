import json
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.schemas import ClusterRequest, ClusterResponse


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


def _get_output_text(response_data: dict[str, Any]) -> str:
    if output_text := response_data.get("output_text"):
        return str(output_text)

    for item in response_data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    return str(content["text"])

    raise ValueError("Responses API payload did not contain output text")


def parse_cluster_response(response_data: dict[str, Any]) -> ClusterResponse:
    output_text = _get_output_text(response_data)
    parsed_output = json.loads(output_text)
    return ClusterResponse.model_validate(parsed_output)


async def cluster_stories(request: ClusterRequest, client: LLMClient | None = None) -> ClusterResponse:
    llm_client = client or LLMClient()
    system_message, user_message = build_cluster_messages(request)
    response_data = await llm_client.create_response(user_message["content"], system_message["content"])
    return parse_cluster_response(response_data)
