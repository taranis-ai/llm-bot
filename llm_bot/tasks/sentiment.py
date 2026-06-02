from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.log import logger
from llm_bot.schemas import (
    PLUTCHIK_8,
    EmotionLabel,
    SentimentLabel,
    SentimentRequest,
    SentimentResponse,
)
from llm_bot.tasks.llm_utils import (
    InvalidLLMOutputError,
    create_and_parse_response,
    get_output_text,
    loads_json_output,
)


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "sentiment.txt"


def load_sentiment_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_sentiment_messages(request: SentimentRequest) -> list[dict[str, str]]:
    system_prompt = load_sentiment_prompt()
    if request.include_emotions:
        emotions = ", ".join(emotion.value for emotion in PLUTCHIK_8)
        system_prompt = (
            f"{system_prompt}\n"
            "Emotion extraction is enabled.\n"
            f"Allowed emotions: {emotions}.\n"
            "- Return the emotions field as an array, even if it is empty.\n"
            "- Only include emotions that are clearly supported by the text.\n"
        )
    else:
        system_prompt = (
            f"{system_prompt}\n"
            "Emotion extraction is disabled.\n"
            "- Do not include an emotions field in the response.\n"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.text},
    ]


def parse_sentiment_response(response_data: dict[str, Any], include_emotions: bool = False) -> SentimentResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw sentiment output: %s", output_text)
    parsed_output = loads_json_output(output_text)
    sentiment = parsed_output.get("sentiment")
    if not isinstance(sentiment, dict):
        raise InvalidLLMOutputError("Response did not contain a sentiment object")

    has_emotions = "emotions" in sentiment
    if include_emotions and not has_emotions:
        raise InvalidLLMOutputError("Emotion extraction was enabled but the response omitted emotions")
    if not include_emotions and has_emotions:
        raise InvalidLLMOutputError("Emotion extraction was disabled but the response included emotions")

    return SentimentResponse.model_validate(parsed_output)


def get_sentiment_response_format(include_emotions: bool) -> dict[str, Any]:
    sentiment_properties: dict[str, Any] = {
        "label": {
            "type": "string",
            "enum": [label.value for label in SentimentLabel],
        },
        "score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    }
    required_fields = ["label", "score"]

    if include_emotions:
        sentiment_properties["emotions"] = {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [emotion.value for emotion in EmotionLabel],
            },
            "uniqueItems": True,
        }
        required_fields.append("emotions")

    return {
        "type": "json_schema",
        "name": "sentiment_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["sentiment"],
            "properties": {
                "sentiment": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": required_fields,
                    "properties": sentiment_properties,
                }
            },
        },
    }


async def analyze_sentiment(
    request: SentimentRequest,
    client: LLMClient | None = None,
) -> SentimentResponse:
    llm_client = client or LLMClient()
    system_message, user_message = build_sentiment_messages(request)
    return await create_and_parse_response(
        client=llm_client,
        task_name="sentiment",
        user_input=user_message["content"],
        system_input=system_message["content"],
        response_format=get_sentiment_response_format(request.include_emotions),
        parse_response=lambda response_data: parse_sentiment_response(response_data, request.include_emotions),
    )
