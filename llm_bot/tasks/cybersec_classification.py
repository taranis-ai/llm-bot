from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.decisions_client import ChoiceProbabilities, ChoiceQuestion, DecisionsClient
from llm_bot.log import logger
from llm_bot.schemas import CybersecClassificationRequest, CybersecClassificationResponse
from llm_bot.tasks.decision_utils import DecisionsTask
from llm_bot.tasks.llm_utils import LLMTask, get_output_text, loads_json_output

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "cybersec_classification.txt"


def load_cybersec_classification_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_cybersec_classification_messages(
    request: CybersecClassificationRequest,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_cybersec_classification_prompt()},
        {"role": "user", "content": request.text},
    ]


def parse_cybersec_classification_response(response_data: dict[str, Any]) -> CybersecClassificationResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw cybersec classification output: %s", output_text)
    parsed_output = loads_json_output(output_text)
    return CybersecClassificationResponse.model_validate(parsed_output)


def get_cybersec_classification_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "cybersec_classification_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["cybersecurity", "non-cybersecurity"],
            "properties": {
                "cybersecurity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "non-cybersecurity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
        },
    }


def parse_cybersec_decision(probabilities: ChoiceProbabilities) -> CybersecClassificationResponse:
    return CybersecClassificationResponse.model_validate(probabilities["cybersecurity_relevance"])


def prepare_cybersec_classification(
    request: CybersecClassificationRequest,
    *,
    client: LLMClient | DecisionsClient | None = None,
) -> LLMTask[CybersecClassificationResponse] | DecisionsTask[CybersecClassificationResponse]:
    if isinstance(client, DecisionsClient) or (client is None and Config.DECISIONS_URL):
        return DecisionsTask(
            state=request.text,
            questions={
                "cybersecurity_relevance": ChoiceQuestion(
                    instructions=(
                        "Classify the main subject of the text. Focus on the main subject, not isolated keywords. "
                        "Mentions of software vendors, products, or infrastructure are not enough by themselves."
                    ),
                    criteria={
                        "cybersecurity": (
                            "The text is substantially about cyber threats, incidents, malware, vulnerabilities, exploits, "
                            "threat actors, attacks, defenses, security tooling, security operations, or cyber policy."
                        ),
                        "non-cybersecurity": "The text is not mainly about cybersecurity, even if it contains incidental technical words.",
                    },
                ),
            },
            parse_response=parse_cybersec_decision,
        )
    system_message, user_message = build_cybersec_classification_messages(request)
    return LLMTask(
        task_name="cybersec classification",
        reasoning_effort=request.reasoning_effort,
        thinking_budget_tokens=request.thinking_budget_tokens,
        user_input=user_message["content"],
        system_input=system_message["content"],
        response_format=get_cybersec_classification_response_format(),
        parse_response=parse_cybersec_classification_response,
    )


async def classify_cybersecurity_text(
    request: CybersecClassificationRequest,
    client: LLMClient | DecisionsClient | None = None,
) -> CybersecClassificationResponse:
    if client is None:
        client = (
            DecisionsClient()
            if Config.DECISIONS_URL
            else LLMClient(reasoning_effort=request.reasoning_effort, thinking_budget_tokens=request.thinking_budget_tokens)
        )
    return await prepare_cybersec_classification(request, client=client).run(client)
