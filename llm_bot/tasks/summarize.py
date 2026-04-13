import json
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.log import logger
from llm_bot.schemas import SummarizeRequest, SummarizeResponse


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "summarize.txt"


def load_summary_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_summary_messages(request: SummarizeRequest) -> list[dict[str, str]]:
    system_prompt = load_summary_prompt()
    if request.max_words is not None:
        system_prompt = f"{system_prompt}\n- The summary must not exceed {request.max_words} words."
    truncated_text = _truncate_text(request.text, Config.SUMMARY_MAX_INPUT_CHARS)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": truncated_text},
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


def parse_summary_response(response_data: dict[str, Any]) -> SummarizeResponse:
    output_text = _get_output_text(response_data)
    logger.debug("Raw summarize output: %s", output_text)
    parsed_output = json.loads(output_text)
    return SummarizeResponse.model_validate(parsed_output)


def get_summary_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "summary_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary"],
            "properties": {
                "summary": {
                    "type": "string",
                    "minLength": 1,
                }
            },
        },
    }


async def summarize(request: SummarizeRequest, client: LLMClient | None = None) -> SummarizeResponse:
    llm_client = client or LLMClient()
    system_message, user_message = build_summary_messages(request)
    response_data = await llm_client.create_response(
        user_message["content"],
        system_message["content"],
        get_summary_response_format(),
    )
    return parse_summary_response(response_data)
