from pydantic import BaseModel, ConfigDict

from llm_bot.tasks.evidence import prepare_retrieval_plan


class Passage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quote: str


def test_retrieval_plan_uses_caller_schema():
    plan = prepare_retrieval_plan({"question": "Who attacked B?"}, Passage)
    assert "never SQL or Cypher" in plan.system_input
    assert plan.response_format["schema"] == Passage.model_json_schema()
    assert plan.parse_response({"output_text": '{"quote":"Example"}'}).quote == "Example"
