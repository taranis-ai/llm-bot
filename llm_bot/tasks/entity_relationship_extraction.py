import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.log import logger
from llm_bot.schemas import (
    EntityRelationshipExtractionRequest,
    EntityRelationshipExtractionResponse,
)
from llm_bot.tasks.llm_utils import (
    InvalidLLMOutputError,
    create_and_parse_response,
    get_output_text,
    loads_json_output,
)


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "entity_relationship_extraction.txt"


def load_entity_relationship_extraction_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_entity_relationship_extraction_messages(
    request: EntityRelationshipExtractionRequest,
) -> list[dict[str, str]]:
    user_payload = {
        "text": request.text,
        "schema": request.schema.model_dump(),
    }
    return [
        {"role": "system", "content": load_entity_relationship_extraction_prompt()},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
    ]


def validate_entity_relationship_extraction(
    response: EntityRelationshipExtractionResponse,
    request: EntityRelationshipExtractionRequest,
) -> EntityRelationshipExtractionResponse:
    allowed_entity_types = {entity_type.name for entity_type in request.schema.entity_types}
    relation_types = {relation_type.name: relation_type for relation_type in request.schema.relation_types}

    entity_ids = [entity.id for entity in response.entities]
    duplicate_ids = sorted(entity_id for entity_id, count in Counter(entity_ids).items() if count > 1)
    if duplicate_ids:
        raise InvalidLLMOutputError(f"Duplicate entity IDs in extraction output: {', '.join(duplicate_ids)}")

    entities_by_id = {entity.id: entity for entity in response.entities}
    unknown_entity_types = sorted({entity.type for entity in response.entities} - allowed_entity_types)
    if unknown_entity_types:
        raise InvalidLLMOutputError(f"Unknown entity types in extraction output: {', '.join(unknown_entity_types)}")

    for relation in response.relations:
        relation_schema = relation_types.get(relation.type)
        if relation_schema is None:
            raise InvalidLLMOutputError(f"Unknown relation type in extraction output: {relation.type}")
        if relation.source_id not in entities_by_id or relation.target_id not in entities_by_id:
            raise InvalidLLMOutputError(f"Relation {relation.type} contains a dangling source or target entity ID")

        source_type = entities_by_id[relation.source_id].type
        target_type = entities_by_id[relation.target_id].type
        if source_type not in relation_schema.source_types or target_type not in relation_schema.target_types:
            raise InvalidLLMOutputError(f"Relation {relation.type} does not allow source type {source_type} and target type {target_type}")
    return response


def parse_entity_relationship_extraction_response(
    response_data: dict[str, Any],
    request: EntityRelationshipExtractionRequest,
) -> EntityRelationshipExtractionResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw entity relationship extraction output: %s", output_text)
    parsed_output = loads_json_output(output_text)
    response = EntityRelationshipExtractionResponse.model_validate(parsed_output)
    return validate_entity_relationship_extraction(response, request)


def get_entity_relationship_extraction_response_format(
    request: EntityRelationshipExtractionRequest,
) -> dict[str, Any]:
    entity_types = [entity_type.name for entity_type in request.schema.entity_types]
    relation_types = [relation_type.name for relation_type in request.schema.relation_types]
    return {
        "type": "json_schema",
        "name": "entity_relationship_extraction_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["entities", "relations"],
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "type", "name"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "type": {"type": "string", "enum": entity_types},
                            "name": {"type": "string", "minLength": 1},
                        },
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "type",
                            "source_id",
                            "target_id",
                            "confidence",
                        ],
                        "properties": {
                            "type": {"type": "string", "enum": relation_types},
                            "source_id": {"type": "string", "minLength": 1},
                            "target_id": {"type": "string", "minLength": 1},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                    },
                },
            },
        },
    }


async def extract_entity_relationships(
    request: EntityRelationshipExtractionRequest,
    client: LLMClient | None = None,
) -> EntityRelationshipExtractionResponse:
    llm_client = client or LLMClient(
        reasoning_effort=request.reasoning_effort,
        thinking_budget_tokens=request.thinking_budget_tokens,
    )
    system_message, user_message = build_entity_relationship_extraction_messages(request)
    return await create_and_parse_response(
        client=llm_client,
        task_name="entity relationship extraction",
        user_input=user_message["content"],
        system_input=system_message["content"],
        response_format=get_entity_relationship_extraction_response_format(request),
        parse_response=lambda response_data: parse_entity_relationship_extraction_response(response_data, request),
    )
