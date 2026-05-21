import re
from typing import Any

from llm_bot.config import Config


MINISTRAL_REASONING_PROMPT = """# HOW YOU SHOULD THINK AND ANSWER
First draft your thinking process (inner monologue) until you arrive at a response.
Format your response using Markdown, and use LaTeX for any mathematical equations.
Write both your thoughts and the response in the same language as the input.
Your thinking process must follow the template below:
[THINK]
Your thoughts or/and draft, like working through an exercise on scratch paper.
Be as casual and as long as you want until you are confident to generate the response to the user.
[/THINK]
Here, provide a self-contained response."""

GEMMA_THINK_TOKEN = "<|think|>"
THINK_BLOCK_PATTERN = re.compile(r"\[THINK\].*?\[/THINK\]", re.DOTALL | re.IGNORECASE)
GEMMA_THOUGHT_BLOCK_PATTERN = re.compile(
    r"<\|channel\>thought\s*\n?.*?<channel\|>",
    re.DOTALL | re.IGNORECASE,
)


def apply_reasoning_profile(instructions: str) -> str:
    if Config.LLM_REASONING_PROFILE == "ministral":
        return (
            f"{MINISTRAL_REASONING_PROMPT}\n\n"
            "After the [/THINK] block, follow the task-specific instructions exactly.\n"
            "For this service, the final response after [/THINK] must be valid JSON only when the task requires JSON.\n\n"
            f"{instructions}"
        )
    if Config.LLM_REASONING_PROFILE == "gemma":
        if instructions.startswith(GEMMA_THINK_TOKEN):
            return instructions
        return f"{GEMMA_THINK_TOKEN}\n{instructions}"
    return instructions


def strip_reasoning_output(output_text: str) -> str:
    if not Config.LLM_STRIP_REASONING_OUTPUT:
        return output_text
    output_text = THINK_BLOCK_PATTERN.sub("", output_text)
    output_text = GEMMA_THOUGHT_BLOCK_PATTERN.sub("", output_text)
    return output_text.strip()


def extract_inline_reasoning(output_text: str) -> str:
    reasoning_blocks = [match.group(0).strip() for match in THINK_BLOCK_PATTERN.finditer(output_text)]
    reasoning_blocks.extend(match.group(0).strip() for match in GEMMA_THOUGHT_BLOCK_PATTERN.finditer(output_text))
    return "\n\n".join(reasoning_blocks)


def extract_structured_reasoning(response_data: dict[str, Any]) -> str:
    reasoning_texts: list[str] = []
    for item in response_data.get("output", []):
        if item.get("type") != "reasoning":
            continue
        for content in item.get("content", []):
            if content.get("text"):
                reasoning_texts.append(str(content["text"]))
    return "\n\n".join(reasoning_texts)
