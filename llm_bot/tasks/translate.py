from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.local_inference import validate_local_text
from llm_bot.log import logger
from llm_bot.schemas import LocalTextRequest, TranslateRequest, TranslateResponse
from llm_bot.tasks.language import detect_language
from llm_bot.tasks.llm_utils import create_and_parse_response, get_output_text, loads_json_output

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "translate.txt"


def load_translate_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_translate_messages(request: TranslateRequest) -> list[dict[str, str]]:
    system_prompt = load_translate_prompt()
    if request.source_language:
        system_prompt = f"{system_prompt}\n- The source language is {request.source_language}.\n- Translate into {request.target_language}.\n"
    else:
        system_prompt = f"{system_prompt}\n- Detect the source language from the input text.\n- Translate into {request.target_language}.\n"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.text},
    ]


def parse_translate_response(response_data: dict[str, Any]) -> TranslateResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw translate output: %s", output_text)
    parsed_output = loads_json_output(output_text)
    return TranslateResponse.model_validate(parsed_output)


def get_translate_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "translate_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["translation"],
            "properties": {
                "translation": {
                    "type": "string",
                    "minLength": 1,
                }
            },
        },
    }


async def translate_text(
    request: TranslateRequest,
    client: LLMClient | None = None,
) -> TranslateResponse:
    if not request.source_language:
        validate_local_text(request.text)
        detected = await detect_language(LocalTextRequest(text=request.text))
        if detected.language != "und":
            request = request.model_copy(update={"source_language": detected.language})
    llm_client = client or LLMClient(
        reasoning_effort=request.reasoning_effort,
        thinking_budget_tokens=request.thinking_budget_tokens,
    )
    system_message, user_message = build_translate_messages(request)
    return await create_and_parse_response(
        client=llm_client,
        task_name="translation",
        user_input=user_message["content"],
        system_input=system_message["content"],
        response_format=get_translate_response_format(),
        parse_response=parse_translate_response,
    )
