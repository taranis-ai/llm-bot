import json
from typing import Any, Callable, TypeVar

from pydantic import ValidationError

from llm_bot.client import LLMClient
from llm_bot.log import logger
from llm_bot.reasoning import (
    apply_reasoning_profile,
    extract_inline_reasoning,
    extract_structured_reasoning,
    strip_reasoning_output,
)


T = TypeVar("T")


class InvalidLLMOutputError(ValueError):
    pass


class MissingOutputTextError(RuntimeError):
    pass


def extract_last_json_object(text: str) -> str:
    end_index = text.rfind("}")
    if end_index == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    depth = 0
    in_string = False
    escaped = False
    for index in range(end_index, -1, -1):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = in_string
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "}":
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0:
                return text[index : end_index + 1]

    raise json.JSONDecodeError("No balanced JSON object found", text, 0)


def loads_json_output(output_text: str) -> Any:
    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        extracted_json = extract_last_json_object(output_text)
        logger.debug("Extracted JSON object from noisy LLM output: %s", extracted_json)
        return json.loads(extracted_json)


def _log_reasoning_output(response_data: dict[str, Any], output_text: str | None = None) -> None:
    reasoning_text = extract_structured_reasoning(response_data)
    if output_text:
        inline_reasoning_text = extract_inline_reasoning(output_text)
        if inline_reasoning_text:
            reasoning_text = f"{reasoning_text}\n\n{inline_reasoning_text}".strip()
    if reasoning_text:
        logger.debug("LLM reasoning output: %s", reasoning_text)


def get_output_text(response_data: dict[str, Any]) -> str:
    if output_text := response_data.get("output_text"):
        output_text = str(output_text)
        _log_reasoning_output(response_data, output_text)
        return strip_reasoning_output(output_text)

    for item in response_data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    output_text = str(content["text"])
                    _log_reasoning_output(response_data, output_text)
                    return strip_reasoning_output(output_text)

    logger.debug("Responses API payload without output text: %s", json.dumps(response_data, ensure_ascii=True, default=str))
    raise MissingOutputTextError("Responses API payload did not contain output text")


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
    instructions = apply_reasoning_profile(instructions)
    response_data = await client.create_response(input_text, instructions, response_format)
    try:
        return parse_response(response_data)
    except (json.JSONDecodeError, ValidationError, InvalidLLMOutputError) as error:
        invalid_output_text = get_output_text(response_data)
        logger.warning("Invalid %s output, retrying once: %s", task_name, error)
        repair_response_data = await client.create_response(
            _build_repair_input(input_text, invalid_output_text, error),
            _build_repair_instructions(instructions, error),
            response_format,
        )
        return parse_response(repair_response_data)
