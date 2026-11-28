from __future__ import annotations

from base64 import b64encode
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_bot.schemas import Image2TextRequest, Image2TextResponse, ImageInput
from llm_bot.tasks.llm_utils import LLMTask, get_output_text, loads_json_output

if TYPE_CHECKING:
    from llm_bot.client import LLMClient

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "image2text.txt"


def parse_image2text_response(body: dict[str, Any]) -> Image2TextResponse:
    return Image2TextResponse.model_validate(loads_json_output(get_output_text(body)))


def prepare_image2text(request: Image2TextRequest) -> LLMTask[Image2TextResponse]:
    encoded = b64encode(request.data).decode("ascii")
    return LLMTask(
        task_name="image2text",
        system_input=PROMPT_PATH.read_text(encoding="utf-8").strip(),
        user_input=ImageInput("Describe this image.", f"data:{request.mime_type};base64,{encoded}"),
        response_format={
            "type": "json_schema",
            "name": "image2text_response",
            "strict": True,
            "schema": Image2TextResponse.model_json_schema(),
        },
        parse_response=parse_image2text_response,
    )


async def describe_image(request: Image2TextRequest, client: LLMClient | None = None) -> Image2TextResponse:
    if client is None:
        from llm_bot.client import LLMClient

        client = LLMClient()
    return await prepare_image2text(request).run(client)
