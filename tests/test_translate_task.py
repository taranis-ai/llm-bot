import pytest

from llm_bot.schemas import TranslateRequest, TranslateResponse
from llm_bot.tasks.translate import (
    build_translate_messages,
    parse_translate_response,
    translate_text,
)
from tests.test_helpers import StubLLMClient


@pytest.mark.parametrize(
    ("translate_request", "expected_source_instruction"),
    [
        (
            TranslateRequest(text="Guten Morgen", source_language="de", target_language="en"),
            "The source language is de.",
        ),
        (
            TranslateRequest(text="Bonjour", target_language="en"),
            "Detect the source language from the input text.",
        ),
    ],
)
def test_build_translate_messages(translate_request, expected_source_instruction):
    system_message, user_message = build_translate_messages(translate_request)

    assert "Translate faithfully into the requested target language." in system_message["content"]
    assert expected_source_instruction in system_message["content"]
    assert "Translate into en." in system_message["content"]
    assert user_message["content"] == translate_request.text


def test_parse_translate_response_from_output_text():
    response = parse_translate_response({"output_text": '{"translation":"Good morning"}'})

    assert response == TranslateResponse(translation="Good morning")


def test_parse_translate_response_from_output_messages():
    response = parse_translate_response(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"translation":"Good morning"}',
                        }
                    ],
                }
            ]
        }
    )

    assert response == TranslateResponse(translation="Good morning")


@pytest.mark.asyncio
async def test_translate_text_calls_client_and_returns_validated_response():
    client = StubLLMClient({"output_text": '{"translation":"Good morning"}'})

    response = await translate_text(
        TranslateRequest(text="Guten Morgen", source_language="de", target_language="en"),
        client=client,
    )

    assert response == TranslateResponse(translation="Good morning")
    assert client.calls[0]["user_input"] == "Guten Morgen"
    assert "The source language is de." in client.calls[0]["system_input"]
    assert "Translate into en." in client.calls[0]["system_input"]
    assert client.calls[0]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_translate_text_applies_reasoning_profile(monkeypatch):
    monkeypatch.setattr("llm_bot.reasoning.Config.LLM_REASONING_PROFILE", "ministral")
    client = StubLLMClient({"output_text": '[THINK]draft[/THINK]{"translation":"Good morning"}'})

    response = await translate_text(TranslateRequest(text="Guten Morgen", target_language="en"), client=client)

    assert response == TranslateResponse(translation="Good morning")
    assert "# HOW YOU SHOULD THINK AND ANSWER" in client.calls[0]["system_input"]
