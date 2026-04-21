import json
from typing import Any, Callable, TypeVar

from pydantic import ValidationError

from llm_bot.client import LLMClient
from llm_bot.log import logger


T = TypeVar("T")


def get_output_text(response_data: dict[str, Any]) -> str:
    if output_text := response_data.get("output_text"):
        return str(output_text)

    for item in response_data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    return str(content["text"])

    raise ValueError("Responses API payload did not contain output text")


def _build_repair_input(input_text: str, invalid_output_text: str, error: Exception) -> str:
    return json.dumps(
        {
            "original_input_text": input_text,
            "previous_invalid_output": invalid_output_text,
            "validation_error": str(error),
        },
        ensure_ascii=True,
    )


def _build_repair_instructions(instructions: str, error: Exception) -> str:
    return (
        f"{instructions}\n\n"
        "Your previous response was invalid.\n"
        f"Validation error: {error}\n"
        "Return corrected valid JSON only.\n"
        "Do not include explanations, comments, or markdown.\n"
        "The corrected response must match the required schema exactly."
    )


async def create_and_parse_response(
    *,
    client: LLMClient,
    task_name: str,
    input_text: str,
    instructions: str,
    response_format: dict[str, Any] | None,
    parse_response: Callable[[dict[str, Any]], T],
) -> T:
    response_data = await client.create_response(input_text, instructions, response_format)
    try:
        return parse_response(response_data)
    except (json.JSONDecodeError, ValidationError) as error:
        invalid_output_text = get_output_text(response_data)
        logger.warning("Invalid %s output, retrying once: %s", task_name, error)
        repair_response_data = await client.create_response(
            _build_repair_input(input_text, invalid_output_text, error),
            _build_repair_instructions(instructions, error),
            response_format,
        )
        return parse_response(repair_response_data)
