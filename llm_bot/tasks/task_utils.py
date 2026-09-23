from collections import Counter

from langcodes import Language
from langcodes.tag_parser import LanguageTagError

from llm_bot.schemas import StoryInputNewsItem, StoryRequest


def build_story_input_text(request: StoryRequest) -> str:
    if request.news_items:
        formatted_items = []
        for index, news_item in enumerate(request.news_items, start=1):
            formatted_items.append(
                "\n".join(
                    [
                        f"News item {index}",
                        f"Title: {news_item.title or '-'}",
                        "Content:",
                        news_item.content or "-",
                    ]
                )
            )
        return "\n\n".join(formatted_items)
    return request.text or ""


def resolve_majority_language(news_items: list[StoryInputNewsItem] | None) -> str | None:
    if not news_items:
        return None

    counts: Counter[str] = Counter()
    for news_item in news_items:
        if language := (news_item.language or "").strip():
            counts[language] += 1

    if not counts:
        return None

    return counts.most_common(1)[0][0]


def resolve_language_name(language_code: str) -> str:
    normalized_code = language_code.strip()
    try:
        language_name = Language.get(normalized_code).language_name()
    except (LanguageTagError, ValueError):
        return f'language code "{normalized_code}"'

    if not language_name or language_name.startswith("Unknown language"):
        return f'language code "{normalized_code}"'

    return language_name


def build_output_language_instruction(
    request: StoryRequest,
    *,
    output_name: str,
) -> str:
    if request.language:
        return f"- Write the {output_name} in {resolve_language_name(request.language)}."

    majority_language = resolve_majority_language(request.news_items)
    if majority_language:
        return f"- Write the {output_name} in {resolve_language_name(majority_language)}."

    return f"- Write the {output_name} in the same language as the input text."


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
