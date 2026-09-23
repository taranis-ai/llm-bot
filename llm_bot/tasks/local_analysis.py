"""Local questions and score mapping shared by the existing text-analysis tasks."""

from math import isclose, isfinite

from llm_bot.local_inference import LayaRuntime
from llm_bot.schemas import (
    _ALLOWED_SENTIMENTS_BY_EMOTION,
    PLUTCHIK_8,
    CybersecClassificationRequest,
    CybersecClassificationResponse,
    SentimentRequest,
    SentimentResponse,
)

CYBER_QUESTION = {
    "type": "choice",
    "instructions": "Is this text relevant to cybersecurity, regardless of its primary topic? Cybersecurity regulation is relevant.",
    "criteria": {
        "cybersecurity": "Digital security, vulnerabilities, attacks, breaches, defenses or cybersecurity policy",
        "non-cybersecurity": "No substantive connection to digital security",
    },
}

SENTIMENT_QUESTION = {
    "type": "choice",
    "instructions": "What attitude does the author express? Judge tone, not event severity or urgency. Objective news about harm is neutral.",
    "criteria": {
        "positive": "The author expresses praise, approval, happiness or optimism",
        "negative": "The author expresses criticism, disapproval, anger, fear or sadness",
        "neutral": "Objective factual news, including breaches, losses and attacks; no author opinion or emotion",
    },
}


def choice_probabilities(result: dict, question: str, labels) -> dict[str, float]:
    answer = result["answers"][question]
    scores = answer["probabilities"]
    if set(scores) != set(labels) or any(not isinstance(p, (int, float)) or not isfinite(p) or not 0 <= p <= 1 for p in scores.values()):
        raise ValueError("Invalid Laya probability distribution")
    total = sum(scores.values())
    # Laya rounds each probability to four decimal places. Correct only rounding drift.
    if not isclose(total, 1, abs_tol=len(scores) * 0.00005 + 1e-8):
        raise ValueError("Laya probabilities do not sum to one")
    if answer["choice"] not in scores or scores[answer["choice"]] != max(scores.values()):
        raise ValueError("Laya choice is inconsistent with probabilities")
    return {key: value / total for key, value in scores.items()}


async def local_cybersecurity(request: CybersecClassificationRequest, inference: LayaRuntime) -> CybersecClassificationResponse:
    result = await inference.predict(request.text, {"relevance": CYBER_QUESTION})
    return CybersecClassificationResponse.model_validate(choice_probabilities(result, "relevance", CYBER_QUESTION["criteria"]))


async def local_sentiment(request: SentimentRequest, inference: LayaRuntime) -> SentimentResponse:
    questions = {"sentiment": SENTIMENT_QUESTION}
    if request.include_emotions:
        questions.update(
            {
                emotion.value: {
                    "type": "choice",
                    "instructions": f"Does the author clearly express {emotion.value}? Events alone do not imply an emotion.",
                    "criteria": {"absent": "Not clearly expressed", "present": f"Clearly expressed {emotion.value}"},
                }
                for emotion in PLUTCHIK_8
            }
        )
    result = await inference.predict(request.text, questions)
    scores = choice_probabilities(result, "sentiment", SENTIMENT_QUESTION["criteria"])
    label = result["answers"]["sentiment"]["choice"]
    sentiment = {"label": label, "score": scores[label]}
    if request.include_emotions:
        # Ask independently, then retain only emotions compatible with the selected sentiment.
        compatible = {emotion.value for emotion, labels in _ALLOWED_SENTIMENTS_BY_EMOTION.items() if label in labels}
        emotion_scores = {emotion.value: choice_probabilities(result, emotion.value, ("absent", "present")) for emotion in PLUTCHIK_8}
        sentiment["emotions"] = [emotion for emotion, values in emotion_scores.items() if emotion in compatible and values["present"] > 0.5]
    return SentimentResponse.model_validate({"sentiment": sentiment})
