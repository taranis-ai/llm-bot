import json

import pytest

from llm_bot.schemas import GraphQueryGenerationRequest
from llm_bot.tasks.graph_query_generation import (
    build_graph_query_generation_messages,
    generate_graph_query,
    parse_graph_query_generation_response,
)
from llm_bot.tasks.llm_utils import InvalidLLMOutputError
from tests.test_helpers import StubLLMClient


REQUEST_PAYLOAD = {
    "question": "Which organization employs Alice?",
    "graph_name": "knowledge_graph",
    "schema": {
        "node_labels": [
            {
                "label": "Person",
                "properties": [
                    {"name": "name", "type": "string"},
                    {"name": "age", "type": "integer"},
                ],
            },
            {
                "label": "Organization",
                "properties": [{"name": "name", "type": "string"}],
            },
        ],
        "relationship_types": [
            {
                "type": "WORKS_AT",
                "source_labels": ["Person"],
                "target_labels": ["Organization"],
                "properties": [],
            }
        ],
        "default_limit": 25,
        "maximum_limit": 100,
    },
}
VALID_OUTPUT = {
    "cypher": ("MATCH (p:Person)-[:WORKS_AT]->(o:Organization) WHERE p.name = $person_name RETURN o.name AS result LIMIT 25"),
    "parameters": {"person_name": "Alice"},
    "explanation": "Returns organizations that employ the named person.",
}


def make_request() -> GraphQueryGenerationRequest:
    return GraphQueryGenerationRequest.model_validate(REQUEST_PAYLOAD)


def test_build_messages_include_question_schema_graph_name_and_safety_rules():
    system_message, user_message = build_graph_query_generation_messages(make_request())
    payload = json.loads(user_message["content"])

    assert payload == REQUEST_PAYLOAD
    assert "Apache AGE" in system_message["content"]
    assert "Never put a graph name in the Cypher" in system_message["content"]
    assert "parameter placeholders" in system_message["content"]
    assert "Do not generate mutations" in system_message["content"]


def test_parse_valid_graph_query():
    response = parse_graph_query_generation_response({"output_text": json.dumps(VALID_OUTPUT)}, make_request())

    assert response.model_dump() == VALID_OUTPUT


def test_parse_accepts_a_map_with_a_key_matching_its_graph_variable():
    output = {
        **VALID_OUTPUT,
        "cypher": "MATCH (p:Person) RETURN {p: p.name} AS result LIMIT 25",
        "parameters": {},
    }

    response = parse_graph_query_generation_response({"output_text": json.dumps(output)}, make_request())

    assert response.model_dump() == output


def test_parse_rejects_a_return_value_without_the_required_result_alias():
    output = {**VALID_OUTPUT, "cypher": "MATCH (p:Person) RETURN p.name AS name LIMIT 25", "parameters": {}}

    with pytest.raises(InvalidLLMOutputError, match="aliased as result"):
        parse_graph_query_generation_response({"output_text": json.dumps(output)}, make_request())


@pytest.mark.parametrize(
    ("cypher", "error"),
    [
        (
            "MATCH (p:Account) RETURN p.name AS name LIMIT 25",
            "unknown node label",
        ),
        (
            "MATCH (p:Person) RETURN p.secret AS secret LIMIT 25",
            "unknown property",
        ),
        (
            "MATCH (p:Person)-[:OWNS]->(o:Organization) RETURN o.name AS name LIMIT 25",
            "unknown relationship type",
        ),
        (
            "MATCH (o:Organization)-[:WORKS_AT]->(p:Person) RETURN o.name AS name LIMIT 25",
            "does not allow source label",
        ),
    ],
)
def test_parse_rejects_schema_violations(cypher, error):
    output = {**VALID_OUTPUT, "cypher": cypher, "parameters": {}}

    with pytest.raises(InvalidLLMOutputError, match=error):
        parse_graph_query_generation_response({"output_text": json.dumps(output)}, make_request())


@pytest.mark.parametrize(
    ("cypher", "error"),
    [
        (
            "MATCH (p:Person) DELETE p RETURN p.name AS name LIMIT 25",
            "prohibited clauses",
        ),
        (
            "CALL db.labels() RETURN $value AS value LIMIT 25",
            "prohibited clauses",
        ),
        (
            "LOAD CSV FROM $url AS row RETURN row LIMIT 25",
            "prohibited clauses",
        ),
        (
            "MATCH (p:Person) RETURN p.name AS name",
            "exactly one integer LIMIT",
        ),
        (
            "MATCH (p:Person) RETURN p.name AS name LIMIT 101",
            "between 1 and 100",
        ),
        (
            "MATCH (p:Person) WHERE p.name = 'Alice' RETURN p.name AS name LIMIT 25",
            "quoted literals",
        ),
        (
            "MATCH (p:Person), (anything) RETURN p.name AS name LIMIT 25",
            "Every node in a MATCH pattern",
        ),
        (
            "MATCH (p:Person) RETURN p LIMIT 25",
            "must not return whole graph variable",
        ),
        (
            "MATCH (p:Person) WHERE p.age = 42 RETURN p.name AS name LIMIT 25",
            "parameters instead of numeric literal",
        ),
        (
            "MATCH (p:Person) RETURN p.name AS name LIMIT 25 RETURN p.name",
            "LIMIT must be the final clause",
        ),
    ],
)
def test_parse_rejects_unsafe_or_unbounded_cypher(cypher, error):
    output = {**VALID_OUTPUT, "cypher": cypher, "parameters": {}}

    with pytest.raises(InvalidLLMOutputError, match=error):
        parse_graph_query_generation_response({"output_text": json.dumps(output)}, make_request())


@pytest.mark.parametrize(
    ("parameters", "error"),
    [
        ({}, "placeholders have no parameter values"),
        (
            {"person_name": "Alice", "unused": "value"},
            "Parameters are not used",
        ),
    ],
)
def test_parse_rejects_missing_or_extra_parameters(parameters, error):
    output = {**VALID_OUTPUT, "parameters": parameters}

    with pytest.raises(InvalidLLMOutputError, match=error):
        parse_graph_query_generation_response({"output_text": json.dumps(output)}, make_request())


@pytest.mark.asyncio
async def test_generation_repairs_malformed_provider_output_once():
    client = StubLLMClient(
        [
            {"output_text": "not JSON"},
            {"output_text": json.dumps(VALID_OUTPUT)},
        ]
    )

    response = await generate_graph_query(make_request(), client=client)

    assert response.model_dump() == VALID_OUTPUT
    assert len(client.calls) == 2
    assert client.calls[0]["response_format"]["name"] == ("graph_query_generation_response")
