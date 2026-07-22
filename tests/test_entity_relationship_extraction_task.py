import json

import pytest

from llm_bot.schemas import EntityRelationshipExtractionRequest
from llm_bot.tasks.entity_relationship_extraction import (
    build_entity_relationship_extraction_messages,
    extract_entity_relationships,
    parse_entity_relationship_extraction_response,
)
from llm_bot.tasks.llm_utils import InvalidLLMOutputError
from tests.test_helpers import StubLLMClient


TEXT = "APT28 exploited CVE-2025-1234."
REQUEST_PAYLOAD = {
    "text": TEXT,
    "schema": {
        "entity_types": [
            {"name": "ThreatActor", "description": "A named threat actor"},
            {"name": "Vulnerability", "description": "A named vulnerability"},
        ],
        "relation_types": [
            {
                "name": "EXPLOITS",
                "source_types": ["ThreatActor"],
                "target_types": ["Vulnerability"],
            }
        ],
    },
}
VALID_OUTPUT = {
    "entities": [
        {
            "id": "e1",
            "type": "ThreatActor",
            "name": "APT28",
        },
        {
            "id": "e2",
            "type": "Vulnerability",
            "name": "CVE-2025-1234",
        },
    ],
    "relations": [
        {
            "type": "EXPLOITS",
            "source_id": "e1",
            "target_id": "e2",
            "confidence": 0.9,
        }
    ],
}


def make_request() -> EntityRelationshipExtractionRequest:
    return EntityRelationshipExtractionRequest.model_validate(REQUEST_PAYLOAD)


def test_build_messages_include_text_schema_and_explicit_only_rules():
    system_message, user_message = build_entity_relationship_extraction_messages(make_request())
    payload = json.loads(user_message["content"])

    assert payload == REQUEST_PAYLOAD
    assert "Use only information explicitly stated" in system_message["content"]
    assert "Never infer facts" in system_message["content"]
    assert "Return empty entities and relations lists" in system_message["content"]


def test_parse_valid_extraction():
    response = parse_entity_relationship_extraction_response({"output_text": json.dumps(VALID_OUTPUT)}, make_request())

    assert response.model_dump() == VALID_OUTPUT


def test_parse_empty_extraction():
    response = parse_entity_relationship_extraction_response({"output_text": '{"entities":[],"relations":[]}'}, make_request())

    assert response.entities == []
    assert response.relations == []


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda output: output["entities"][0].update(type="Malware"), "Unknown entity types"),
        (lambda output: output["relations"][0].update(type="TARGETS"), "Unknown relation type"),
        (
            lambda output: output["relations"][0].update(source_id="missing"),
            "dangling source or target",
        ),
        (lambda output: output["entities"][1].update(id="e1"), "Duplicate entity IDs"),
        (
            lambda output: output["relations"][0].update(source_id="e2", target_id="e1"),
            "does not allow source type",
        ),
    ],
)
def test_parse_rejects_semantically_invalid_output(mutate, error):
    output = json.loads(json.dumps(VALID_OUTPUT))
    mutate(output)

    with pytest.raises(InvalidLLMOutputError, match=error):
        parse_entity_relationship_extraction_response({"output_text": json.dumps(output)}, make_request())


@pytest.mark.asyncio
async def test_extraction_repairs_malformed_llm_output_once():
    client = StubLLMClient(
        [
            {"output_text": '{"entities":"invalid","relations":[]}'},
            {"output_text": json.dumps(VALID_OUTPUT)},
        ]
    )

    response = await extract_entity_relationships(make_request(), client=client)

    assert response.entities[0].name == "APT28"
    assert len(client.calls) == 2
    response_schema = client.calls[0]["response_format"]["schema"]
    entity_type_schema = response_schema["properties"]["entities"]["items"]["properties"]["type"]
    relation_type_schema = response_schema["properties"]["relations"]["items"]["properties"]["type"]
    assert entity_type_schema["enum"] == ["ThreatActor", "Vulnerability"]
    assert relation_type_schema["enum"] == ["EXPLOITS"]
