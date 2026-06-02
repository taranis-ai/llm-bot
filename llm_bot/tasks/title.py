from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.log import logger
from llm_bot.schemas import TitleRequest, TitleResponse
from llm_bot.tasks.llm_utils import create_and_parse_response, get_output_text, loads_json_output
from llm_bot.tasks.task_utils import truncate_text


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "title.txt"


def load_title_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_title_messages(request: TitleRequest) -> list[dict[str, str]]:
    system_prompt = load_title_prompt()
    system_prompt = f"{system_prompt}\n- The title must not exceed {request.max_chars} characters."
    truncated_text = truncate_text(request.text, Config.SUMMARY_MAX_INPUT_CHARS)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": truncated_text},
    ]


def parse_title_response(response_data: dict[str, Any]) -> TitleResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw title output: %s", output_text)
    parsed_output = loads_json_output(output_text)
    return TitleResponse.model_validate(parsed_output)


def get_title_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "title_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["title"],
            "properties": {
                "title": {
                    "type": "string",
                    "minLength": 1,
                }
            },
        },
    }


async def generate_title(request: TitleRequest, client: LLMClient | None = None) -> TitleResponse:
    llm_client = client or LLMClient()
    system_message, user_message = build_title_messages(request)
    return await create_and_parse_response(
        client=llm_client,
        task_name="title",
        user_input=user_message["content"],
        system_input=system_message["content"],
        response_format=get_title_response_format(),
        parse_response=lambda response_data: parse_title_response(response_data),
    )
