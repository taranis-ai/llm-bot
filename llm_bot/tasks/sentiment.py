from pathlib import Path

from llm_bot.decisions_client import ChoiceProbabilities, ChoiceQuestion, DecisionsClient
from llm_bot.schemas import (
    ALLOWED_SENTIMENTS_BY_EMOTION,
    EmotionLabel,
    SentimentLabel,
    SentimentRequest,
    SentimentResponse,
    SentimentResult,
)
from llm_bot.tasks.decision_utils import DecisionsTask

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "sentiment.txt"

EMOTION_DESCRIPTIONS = {
    EmotionLabel.JOY: "Happiness, delight, or satisfaction.",
    EmotionLabel.TRUST: "Confidence, acceptance, or a sense of safety and reliability.",
    EmotionLabel.FEAR: "Anxiety, alarm, or apprehension about danger or harm.",
    EmotionLabel.SURPRISE: "Astonishment or a reaction to something unexpected.",
    EmotionLabel.SADNESS: "Sorrow, grief, disappointment, or a sense of loss.",
    EmotionLabel.DISGUST: "Revulsion, aversion, or strong repulsion.",
    EmotionLabel.ANGER: "Rage, frustration, outrage, or hostility.",
    EmotionLabel.ANTICIPATION: "Expectation, interest, or looking forward to a future event.",
}


def parse_sentiment_response(probabilities: ChoiceProbabilities, include_emotions: bool = False) -> SentimentResponse:
    distribution = probabilities["sentiment"]
    # Prefer neutral on a tie, consistently across provider response ordering.
    label = max((SentimentLabel.NEUTRAL, SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE), key=distribution.__getitem__)
    emotions = None
    if include_emotions:
        emotions = [
            emotion
            for emotion in EmotionLabel
            if probabilities[emotion.value]["yes"] > 0.5 and label in ALLOWED_SENTIMENTS_BY_EMOTION[emotion]
        ]
    return SentimentResponse(sentiment=SentimentResult(label=label, score=distribution[label], emotions=emotions))


def prepare_sentiment(request: SentimentRequest) -> DecisionsTask[SentimentResponse]:
    prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
    questions = {
        "sentiment": ChoiceQuestion(
            instructions=prompt,
            criteria={
                "positive": "The overall tone is clearly favorable, approving, optimistic, or satisfied.",
                "negative": "The overall tone is clearly unfavorable, critical, pessimistic, or distressed.",
                "neutral": "The text is factual, mixed, or does not clearly support a positive or negative tone.",
            },
        ),
    }
    if request.include_emotions:
        for emotion, description in EMOTION_DESCRIPTIONS.items():
            questions[emotion.value] = ChoiceQuestion(
                instructions=f"{prompt}\nIs {emotion.value} clearly expressed in the text? {description}",
                criteria={
                    "yes": f"The text clearly expresses {emotion.value}; it is not merely mentioned or weakly implied.",
                    "no": f"The text does not clearly express {emotion.value}, or the evidence is ambiguous.",
                },
            )
    return DecisionsTask(
        state=request.text,
        questions=questions,
        parse_response=lambda probabilities: parse_sentiment_response(probabilities, request.include_emotions),
    )


async def analyze_sentiment(request: SentimentRequest, client: DecisionsClient | None = None) -> SentimentResponse:
    if client is None:
        client = DecisionsClient()
    return await prepare_sentiment(request).run(client)
