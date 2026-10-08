import json
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.log import logger
from llm_bot.reasoning import (
    apply_reasoning_profile,
    extract_inline_reasoning,
    extract_structured_reasoning,
    strip_reasoning_output,
)


class InvalidLLMOutputError(ValueError):
    pass


class MissingOutputTextError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMTask[T]:
    """A prepared task that can run immediately or be submitted to a batch API."""

    task_name: str
    user_input: str
    system_input: str
    response_format: dict[str, Any] | None
    parse_response: Callable[[dict[str, Any]], T]
    recover_response: Callable[[dict[str, Any]], T] | None = None
    reasoning_effort: str | None = None
    thinking_budget_tokens: int | None = None

    async def run(self, client: LLMClient) -> T:
        return await create_and_parse_response(
            client=client,
            task_name=self.task_name,
            user_input=self.user_input,
            system_input=self.system_input,
            response_format=self.response_format,
            parse_response=self.parse_response,
            recover_response=self.recover_response,
        )

    def build_request(self, client: LLMClient) -> dict[str, Any]:
        client = copy(client)
        if self.reasoning_effort is not None:
            client.reasoning_effort = self.reasoning_effort
        if self.thinking_budget_tokens is not None:
            client.thinking_budget_tokens = self.thinking_budget_tokens
        _, body = client._request_target(apply_reasoning_profile(self.system_input), self.user_input, self.response_format)
        return body

    def parse_result(self, response_data: dict[str, Any], client: LLMClient) -> T:
        if client.api_mode == "chat_completions":
            response_data = client._normalize_chat_completions_response(response_data)
        return self.parse_response(response_data)


def _log_response_payload(task_name: str, response_data: dict[str, Any], *, attempt: str = "initial") -> None:
    logger.debug(
        "LLM %s response payload (%s): %s",
        task_name,
        attempt,
        json.dumps(response_data, ensure_ascii=True, default=str),
    )


def extract_last_json_object(text: str) -> str:
    end_index = text.rfind("}")
    if end_index == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    decoder = json.JSONDecoder()
    for start_index, char in enumerate(text[:end_index]):
        if char != "{":
            continue
        try:
            _, parsed_end = decoder.raw_decode(text, start_index)
        except json.JSONDecodeError:
            continue
        # Only accept a decoded object ending at the final closing brace.
        if parsed_end == end_index + 1:
            return text[start_index:parsed_end]

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

    if Config.LLM_PARSE_REASONING_AS_OUTPUT:
        reasoning_text = extract_structured_reasoning(response_data)
        if reasoning_text:
            logger.debug("Using structured reasoning output as fallback output text")
            return strip_reasoning_output(reasoning_text)

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


async def create_and_parse_response[T](
    *,
    client: LLMClient,
    task_name: str,
    user_input: str,
    system_input: str,
    response_format: dict[str, Any] | None,
    parse_response: Callable[[dict[str, Any]], T],
    recover_response: Callable[[dict[str, Any]], T] | None = None,
) -> T:
    system_input = apply_reasoning_profile(system_input)
    response_data = await client.create_response(system_input, user_input, response_format)
    _log_response_payload(task_name, response_data)
    try:
        return parse_response(response_data)
    except (json.JSONDecodeError, ValidationError, InvalidLLMOutputError) as error:
        invalid_output_text = get_output_text(response_data)
        logger.warning("Invalid %s output, retrying once: %s", task_name, error)
        repair_response_data = await client.create_response(
            _build_repair_instructions(system_input, error),
            _build_repair_input(user_input, invalid_output_text, error),
            response_format,
        )
        _log_response_payload(task_name, repair_response_data, attempt="repair")
        try:
            return parse_response(repair_response_data)
        except (json.JSONDecodeError, ValidationError, InvalidLLMOutputError) as repair_error:
            if recover_response is None:
                raise
            logger.warning("Invalid %s repair output, attempting recovery: %s", task_name, repair_error)
            try:
                return recover_response(repair_response_data)
            except (json.JSONDecodeError, ValidationError, InvalidLLMOutputError) as recovery_error:
                logger.warning("%s output recovery failed: %s", task_name, recovery_error)
                raise repair_error from recovery_error
