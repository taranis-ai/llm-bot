import asyncio
import re
from functools import lru_cache

from llm_bot.local_inference import validate_local_text
from llm_bot.schemas import LanguageResponse, LocalTextRequest


@lru_cache(maxsize=1)
def _detector():
    from lingua import LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().with_minimum_relative_distance(0.2).build()


def identify_language(text: str) -> str:
    """ISO 639-1, or und for short, technical, uncertain or substantially mixed text."""
    validate_local_text(text)
    without_urls = re.sub(r"(?:https?://|www\.)\S+|\S+@\S+", " ", text)
    without_urls = re.sub(r"([。！？、，：「」『』（）؟،])", r"\1 ", without_urls)
    # Drop identifiers, IPs, hashes, dotted hostnames and mixed alphanumeric tokens.
    words = [word for word in without_urls.split() if word.strip(".,;:!?()[]{}\"'“”‘’—-。！？、，：「」『』（）؟،").isalpha()]
    clean = " ".join(words)
    letters = sum(char.isalpha() for char in clean)
    if letters < 20 or letters < sum(not char.isspace() for char in text) * 0.5:
        return "und"
    detector = _detector()
    language = detector.detect_language_of(clean)
    if language is None:
        return "und"
    # Lingua's short mixed-language spans misidentify ordinary shared vocabulary.
    # Detect whole sentences instead; tiny fragments do not establish a language switch.
    sections = [
        (part, detector.detect_language_of(part)) for part in re.split(r"[.!?。！？\n]+", clean) if sum(char.isalpha() for char in part) >= 20
    ]
    total = sum(len(part) for part, detected in sections if detected is not None)
    dominant = sum(len(part) for part, detected in sections if detected == language)
    if not total or dominant / total < 0.8:
        return "und"
    return language.iso_code_639_1.name.lower()


async def detect_language(request: LocalTextRequest) -> LanguageResponse:
    return LanguageResponse(language=await asyncio.to_thread(identify_language, request.text))
