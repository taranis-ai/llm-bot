import json
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.log import logger
from llm_bot.schemas import HragRequest, HragResponse
from llm_bot.tasks.llm_utils import (
    InvalidLLMOutputError,
    create_and_parse_response,
    get_output_text,
    loads_json_output,
)


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "hrag.txt"


def load_hrag_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_hrag_messages(request: HragRequest) -> list[dict[str, str]]:
    user_payload = {
        "question": request.question,
        "passages": [passage.model_dump() for passage in request.passages],
        "graph_facts": [fact.model_dump() for fact in request.graph_facts],
    }
    return [
        {"role": "system", "content": load_hrag_prompt()},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
    ]


def get_hrag_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "hrag_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "citations", "insufficient_evidence"],
            "properties": {
                "answer": {
                    "type": "string",
                    "minLength": 1,
                },
                "citations": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "insufficient_evidence": {"type": "boolean"},
            },
        },
    }


def parse_hrag_response(
    response_data: dict[str, Any],
    request: HragRequest,
) -> HragResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw HRAG output: %s", output_text)
    parsed_output = loads_json_output(output_text)
    response = HragResponse.model_validate(parsed_output)

    supplied_ids = {item.id for item in request.passages}
    supplied_ids.update(item.id for item in request.graph_facts)
    unknown_citations = sorted(set(response.citations) - supplied_ids)
    if unknown_citations:
        raise InvalidLLMOutputError("Citations reference unknown evidence IDs: " + ", ".join(unknown_citations))
    if not response.insufficient_evidence and not response.citations:
        raise InvalidLLMOutputError("A sufficient grounded answer must cite at least one supplied evidence ID")
    return response


async def answer_with_hrag(
    request: HragRequest,
    client: LLMClient | None = None,
) -> HragResponse:
    llm_client = client or LLMClient(
        reasoning_effort=request.reasoning_effort,
        thinking_budget_tokens=request.thinking_budget_tokens,
    )
    system_message, user_message = build_hrag_messages(request)
    return await create_and_parse_response(
        client=llm_client,
        task_name="HRAG",
        user_input=user_message["content"],
        system_input=system_message["content"],
        response_format=get_hrag_response_format(),
        parse_response=lambda response_data: parse_hrag_response(
            response_data,
            request,
        ),
    )
