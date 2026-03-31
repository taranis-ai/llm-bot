from pathlib import Path

from llm_bot.schemas import SummarizeRequest


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "summarize.txt"


def load_summary_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_summary_messages(request: SummarizeRequest) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": load_summary_prompt()},
        {"role": "user", "content": request.text},
    ]
