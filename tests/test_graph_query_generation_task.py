import json

import pytest

from llm_bot.schemas import GraphQueryGenerationRequest
from llm_bot.tasks.graph_query_generation import generate_graph_query, prepare_graph_query_generation
from tests.test_helpers import StubLLMClient

SCHEMA = {
    "entity_types": [{"name": "Person", "description": "A person"}, {"name": "Organization", "description": "An organization"}],
    "relation_types": [{"name": "WORKS_AT", "source_types": ["Person"], "target_types": ["Organization"]}],
}
REQUEST = {
    "question": "Which organization employs Alice?",
    "schema": SCHEMA,
    "context": {"conversation": []},
}
PLAN = {"names": ["alice"], "entity_types": ["Person"], "relationships": ["WORKS_AT"], "hops": 1}


def test_prepared_graph_query_is_a_plan_with_caller_vocabulary():
    task = prepare_graph_query_generation(GraphQueryGenerationRequest.model_validate(REQUEST))
    assert json.loads(task.user_input) == REQUEST
    assert "Never generate SQL, Cypher" in task.system_input
    assert task.response_format["schema"]["properties"]["relationships"]["items"]["enum"] == ["WORKS_AT"]
    assert task.parse_response({"output_text": json.dumps(PLAN)}).model_dump() == PLAN


@pytest.mark.parametrize(
    "invalid",
    [
        {"cypher": "MATCH (n) RETURN n", "parameters": {}, "explanation": "Query"},
        {**PLAN, "entity_types": ["Unknown"]},
        {**PLAN, "relationships": ["Unknown"]},
        {**PLAN, "hops": 4},
        {**PLAN, "filters": {"read": False}},
    ],
)
async def test_graph_query_repairs_unsupported_or_executable_output(invalid):
    client = StubLLMClient([{"output_text": json.dumps(invalid)}, {"output_text": json.dumps(PLAN)}])
    result = await generate_graph_query(GraphQueryGenerationRequest.model_validate(REQUEST), client=client)
    assert result.model_dump() == PLAN
    assert len(client.calls) == 2
