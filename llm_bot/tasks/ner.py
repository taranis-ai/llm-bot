import json
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.log import logger
from llm_bot.schemas import NerRequest, NerResponse


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "ner.txt"
ALLOWED_ENTITY_TYPES = {
    "Person",
    "Location",
    "Organization",
    "Product",
    "Address",
    "CLICommand/CodeSnippet",
    "Con",
    "Group",
    "Malware",
    "Sector",
    "Tactic",
    "Technique",
    "Tool",
    "Misc",
}


def load_ner_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def resolve_entity_types(request: NerRequest) -> list[str]:
    entity_types = request.entity_types or Config.NER_ENTITY_TYPES
    unknown_entity_types = sorted(set(entity_types) - ALLOWED_ENTITY_TYPES)
    if unknown_entity_types:
        raise ValueError(f"Unsupported entity types requested: {', '.join(unknown_entity_types)}")
    return entity_types


def build_ner_messages(request: NerRequest) -> list[dict[str, str]]:
    system_prompt = load_ner_prompt()
    entity_types = resolve_entity_types(request)
    system_prompt = f"{system_prompt}\nUse only these entity types: {', '.join(entity_types)}."
    if request.cybersecurity:
        system_prompt = (
            f"{system_prompt}\n"
            "Cybersecurity mode is enabled. Extract general named entities and cybersecurity-relevant entities when present."
        )
    else:
        system_prompt = f"{system_prompt}\nCybersecurity mode is disabled."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.text},
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


def parse_ner_response(response_data: dict[str, Any]) -> NerResponse:
    output_text = _get_output_text(response_data)
    logger.debug("Raw NER output: %s", output_text)
    parsed_output = json.loads(output_text)
    return NerResponse.model_validate(parsed_output)


def get_ner_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "ner_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    }


async def extract_entities(request: NerRequest, client: LLMClient | None = None) -> NerResponse:
    llm_client = client or LLMClient()
    system_message, user_message = build_ner_messages(request)
    response_data = await llm_client.create_response(
        user_message["content"],
        system_message["content"],
        get_ner_response_format(),
    )
    return parse_ner_response(response_data)
