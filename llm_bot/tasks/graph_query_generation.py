from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_bot.schemas import GraphQueryGenerationRequest, GraphQueryGenerationResponse
from llm_bot.tasks.llm_utils import InvalidLLMOutputError, LLMTask, get_output_text, loads_json_output

if TYPE_CHECKING:
    from llm_bot.client import LLMClient

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "graph_query_generation.txt"


def load_graph_query_generation_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_graph_query_generation_messages(request: GraphQueryGenerationRequest) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_graph_query_generation_prompt()},
        {
            "role": "user",
            "content": json.dumps(request.model_dump(exclude={"reasoning_effort", "thinking_budget_tokens"}), ensure_ascii=False),
        },
    ]


def get_graph_query_generation_response_format(request: GraphQueryGenerationRequest) -> dict[str, Any]:
    schema = GraphQueryGenerationResponse.model_json_schema()
    for field, values in (
        ("entity_types", [item.name for item in request.graph_schema.entity_types]),
        ("relationships", [item.name for item in request.graph_schema.relation_types]),
    ):
        if values:
            schema["properties"][field]["items"]["enum"] = values
        else:
            schema["properties"][field]["maxItems"] = 0
    return {"type": "json_schema", "name": "graph_query_generation_response", "strict": False, "schema": schema}


def parse_graph_query_generation_response(body: dict[str, Any], request: GraphQueryGenerationRequest) -> GraphQueryGenerationResponse:
    result = GraphQueryGenerationResponse.model_validate(loads_json_output(get_output_text(body)))
    if set(result.entity_types) - {item.name for item in request.graph_schema.entity_types}:
        raise InvalidLLMOutputError("Plan references an unknown entity type")
    if set(result.relationships) - {item.name for item in request.graph_schema.relation_types}:
        raise InvalidLLMOutputError("Plan references an unknown relationship type")
    return result


def prepare_graph_query_generation(request: GraphQueryGenerationRequest) -> LLMTask[GraphQueryGenerationResponse]:
    system_message, user_message = build_graph_query_generation_messages(request)
    return LLMTask(
        task_name="graph query generation",
        system_input=system_message["content"],
        user_input=user_message["content"],
        response_format=get_graph_query_generation_response_format(request),
        parse_response=lambda body: parse_graph_query_generation_response(body, request),
        reasoning_effort=request.reasoning_effort,
        thinking_budget_tokens=request.thinking_budget_tokens,
    )


async def generate_graph_query(request: GraphQueryGenerationRequest, client: LLMClient | None = None) -> GraphQueryGenerationResponse:
    if client is None:
        from llm_bot.client import LLMClient

        client = LLMClient(reasoning_effort=request.reasoning_effort, thinking_budget_tokens=request.thinking_budget_tokens)
    return await prepare_graph_query_generation(request).run(client)
