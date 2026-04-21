import json
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.log import logger
from llm_bot.schemas import SummarizeRequest, SummarizeResponse
from llm_bot.tasks.llm_utils import create_and_parse_response, get_output_text


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "summarize.txt"


def load_summary_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_summary_messages(request: SummarizeRequest) -> list[dict[str, str]]:
    system_prompt = load_summary_prompt()
    system_prompt = f"{system_prompt}\n- The summary must not exceed {Config.SUMMARY_MAX_OUTPUT_CHARS} characters."
    if request.max_words is not None:
        system_prompt = f"{system_prompt}\n- The summary must not exceed {request.max_words} words."
    truncated_text = _truncate_text(request.text, Config.SUMMARY_MAX_INPUT_CHARS)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": truncated_text},
    ]

def parse_summary_response(response_data: dict[str, Any]) -> SummarizeResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw summarize output: %s", output_text)
    parsed_output = json.loads(output_text)
    response = SummarizeResponse.model_validate(parsed_output)
    response.summary = _truncate_text(response.summary, Config.SUMMARY_MAX_OUTPUT_CHARS)
    return response


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
    return await create_and_parse_response(
        client=llm_client,
        task_name="summary",
        input_text=user_message["content"],
        instructions=system_message["content"],
        response_format=get_summary_response_format(),
        parse_response=parse_summary_response,
    )
