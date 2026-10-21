from pathlib import Path

from llm_bot.decisions_client import ChoiceProbabilities, ChoiceQuestion, DecisionsClient
from llm_bot.schemas import ClassifyRequest, ClassifyResponse
from llm_bot.tasks.decision_utils import DecisionsTask

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "classify.txt"


def parse_classify_response(probabilities: ChoiceProbabilities) -> ClassifyResponse:
    return ClassifyResponse.model_validate(probabilities["category"])


def prepare_classify(request: ClassifyRequest) -> DecisionsTask[ClassifyResponse]:
    return DecisionsTask(
        state=request.text,
        questions={
            "category": ChoiceQuestion(
                instructions=PROMPT_PATH.read_text(encoding="utf-8").strip(),
                criteria={
                    "disinformation": "Coverage of disinformation, misinformation, propaganda, fact-checking, or influence campaigns.",
                    "security incident": "A concrete cyberattack, breach, compromise, or security incident and its impact or response.",
                    "vulnerability": "A security flaw, CVE, exploitability analysis, patch, or remediation advisory as the main subject.",
                    "threat intelligence": (
                        "Analysis of threat actors, malware, attack campaigns, techniques, or indicators of compromise, "
                        "rather than primarily reporting a concrete incident."
                    ),
                    "security policy": "Cybersecurity laws, regulations, standards, governance, compliance, or government security strategy.",
                    "security news": (
                        "Other cybersecurity coverage, including industry developments, security products, research, "
                        "or defensive practices, that does not fit a more specific security category."
                    ),
                    "general news": "Substantive news whose main subject is neither cybersecurity nor disinformation coverage.",
                    "other": "Non-news material, advertisements, unrelated instructions, or text with too little context to classify.",
                },
            ),
        },
        parse_response=parse_classify_response,
    )


async def classify_text(request: ClassifyRequest, client: DecisionsClient | None = None) -> ClassifyResponse:
    if client is None:
        client = DecisionsClient()
    return await prepare_classify(request).run(client)
