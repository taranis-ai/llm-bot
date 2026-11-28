import json

import pytest
from pydantic import ValidationError

from llm_bot.client import LLMClient
from llm_bot.schemas import Image2TextRequest, Image2TextResponse
from llm_bot.tasks.image2text import describe_image, prepare_image2text
from tests.test_helpers import PNG_BASE64, PNG_BYTES, StubLLMClient


@pytest.mark.parametrize("mode", ["responses", "chat_completions"])
def test_image2text_provider_payload_and_result(mode):
    client = LLMClient(api_mode=mode, model="vision")
    task = prepare_image2text(Image2TextRequest(data=PNG_BYTES, mime_type="image/png"))
    payload = task.build_request(client)
    image_url = f"data:image/png;base64,{PNG_BASE64}"
    caption = '{"description":"One white pixel."}'
    if mode == "responses":
        messages = payload["input"]
        assert messages[1]["content"] == [
            {"type": "input_text", "text": "Describe this image."},
            {"type": "input_image", "image_url": image_url},
        ]
        result = {"output": [{"type": "message", "content": [{"type": "output_text", "text": caption}]}]}
    else:
        messages = payload["messages"]
        assert messages[1]["content"] == [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        result = {"choices": [{"message": {"content": caption}}]}
    assert "legible" in messages[0]["content"]
    assert payload["model"] == "vision"
    assert task.parse_result(result, client).description == "One white pixel."


@pytest.mark.asyncio
async def test_image2text_repair_resends_original_image():
    client = StubLLMClient([{"output_text": '{"description":" "}'}, {"output_text": '{"description":"  Two boats. "}'}])
    image = Image2TextRequest(data=PNG_BYTES, mime_type="image/png")
    assert (await describe_image(image, client)).description == "Two boats."
    first, repair = [call["user_input"] for call in client.calls]
    assert repair.data_url == first.data_url == f"data:image/png;base64,{PNG_BASE64}"
    assert json.loads(repair.text)["original_input_text"] == first.text
    assert PNG_BASE64 not in repair.text


@pytest.mark.parametrize("description", [" \n", "\x1c", "caption\x00", "x" * 16001])
def test_worker_caption_constraints(description):
    with pytest.raises(ValidationError):
        Image2TextResponse(description=description)


def test_worker_caption_limit_applies_after_trimming():
    assert Image2TextResponse(description=" " + "ü" * 16000 + "\n").description == "ü" * 16000


def test_request_accepts_other_worker_image_formats():
    assert Image2TextRequest(data=b"\xff\xd8\xff\xe0", mime_type="image/jpeg").mime_type == "image/jpeg"
    assert Image2TextRequest(data=b"RIFF\x08\x00\x00\x00WEBPVP8 ", mime_type="image/webp").mime_type == "image/webp"
    with pytest.raises(ValidationError):
        Image2TextRequest(data=b"RIFF\x08\x00\x00\x00WAVE", mime_type="image/webp")
