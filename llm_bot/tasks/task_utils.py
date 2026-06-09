from llm_bot.schemas import SummarizeRequest, TitleRequest

def build_story_input_text(request: SummarizeRequest | TitleRequest) -> str:
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

def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"
