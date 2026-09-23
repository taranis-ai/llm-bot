import json
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.log import logger
from llm_bot.schemas import ChatRequest, ChatResponse
from llm_bot.tasks.llm_utils import (
    create_and_parse_response,
    get_output_text,
    loads_json_output,
)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "chat.txt"


def load_chat_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_chat_messages(request: ChatRequest) -> list[dict[str, str]]:
    user_payload = {
        "messages": [message.model_dump() for message in request.messages],
        "message": request.message,
    }
    return [
        {"role": "system", "content": load_chat_prompt()},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
    ]


def get_chat_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "chat_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {
                "answer": {
                    "type": "string",
                    "minLength": 1,
                }
            },
        },
    }


def parse_chat_response(
    response_data: dict[str, Any],
    *,
    model: str | None,
) -> ChatResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw chat output: %s", output_text)
    parsed_output = loads_json_output(output_text)
    if not isinstance(parsed_output, dict):
        return ChatResponse.model_validate(parsed_output)
    return ChatResponse.model_validate({**parsed_output, "model": model})


async def chat(
    request: ChatRequest,
    client: LLMClient | None = None,
) -> ChatResponse:
    llm_client = client or LLMClient(
        reasoning_effort=request.reasoning_effort,
        thinking_budget_tokens=request.thinking_budget_tokens,
    )
    system_message, user_message = build_chat_messages(request)
    model = getattr(llm_client, "model", None) or Config.LLM_MODEL or None
    return await create_and_parse_response(
        client=llm_client,
        task_name="chat",
        user_input=user_message["content"],
        system_input=system_message["content"],
        response_format=get_chat_response_format(),
        parse_response=lambda response_data: parse_chat_response(
            response_data,
            model=model,
        ),
    )
