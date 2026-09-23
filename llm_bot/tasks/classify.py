from llm_bot.local_inference import LayaRuntime, runtime
from llm_bot.schemas import ClassificationResponse, LocalTextRequest
from llm_bot.tasks.language import require_english_text
from llm_bot.tasks.local_analysis import choice_probabilities

TOPICS = {
    "vulnerabilities": "Software flaws, CVEs, disclosures, patches and exploit availability",
    "attacks": "Attack techniques, malware campaigns, threat actors and exploitation activity",
    "incidents": "Concrete breaches, compromises, data leaks or disruptions affecting victims",
    "politics": "Government, elections, geopolitics, legislation and public policy",
    "business": "Companies, markets, acquisitions, finance and commercial developments",
    "other": "Content outside these categories",
}

TOPIC_QUESTION = {
    "type": "choice",
    "instructions": (
        "Choose the article's primary focus. Newly disclosed flaw: vulnerabilities. "
        "Ransomware campaign analysis: attacks. Hospital reporting a ransomware breach: incidents. "
        "Cybersecurity legislation: politics. Security company acquisition: business."
    ),
    "criteria": TOPICS,
}


async def classify_text(request: LocalTextRequest, inference: LayaRuntime | None = None) -> ClassificationResponse:
    await require_english_text(request.text)
    result = await (inference or runtime).predict(request.text, {"topic": TOPIC_QUESTION})
    scores = choice_probabilities(result, "topic", TOPICS)
    return ClassificationResponse(category=result["answers"]["topic"]["choice"], scores=scores)
