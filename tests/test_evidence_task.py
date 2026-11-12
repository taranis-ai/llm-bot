import pytest
from pydantic import BaseModel, ConfigDict

from llm_bot.tasks.evidence import prepare_evidence_extraction, prepare_retrieval_plan
from llm_bot.tasks.llm_utils import InvalidLLMOutputError
from tests.test_helpers import StubLLMClient


class Passage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quote: str


@pytest.mark.asyncio
async def test_evidence_task_repairs_unsupported_quote_and_keeps_caller_vocabulary():
    def validate(result):
        if result.quote != "A did not attack B.":
            raise InvalidLLMOutputError("Unsupported passage")
        return result

    task = prepare_evidence_extraction("A did not attack B.", {"entities": ["ORG"]}, Passage, validate=validate)
    client = StubLLMClient([{"output_text": '{"quote":"A attacked B."}'}, {"output_text": '{"quote":"A did not attack B."}'}])
    assert (await task.run(client)).quote == "A did not attack B."
    assert len(client.calls) == 2
    assert '"entities": ["ORG"]' in client.calls[0]["user_input"]
    assert "negation and uncertainty" in client.calls[0]["system_input"]
    plan = prepare_retrieval_plan({"question": "Who attacked B?"}, Passage)
    assert "never SQL or Cypher" in plan.system_input
    assert plan.response_format["schema"] == Passage.model_json_schema()
