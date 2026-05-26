import pytest

from llm_bot.schemas import TitleRequest, TitleResponse
from llm_bot.tasks.title import build_title_messages, generate_title, parse_title_response
from tests.test_helpers import StubLLMClient


def test_build_title_messages():
    system_message, user_message = build_title_messages(TitleRequest(text="Story text", max_chars=150))

    assert "same language as the input text" in system_message["content"]
    assert "must not exceed 150 characters" in system_message["content"]
    assert user_message["content"] == "Story text"


def test_build_title_messages_truncates_input_text(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.title.Config.SUMMARY_MAX_INPUT_CHARS", 12)

    _, user_message = build_title_messages(TitleRequest(text="This is a very long story text"))

    assert user_message["content"] == "This is a v…"


@pytest.mark.asyncio
async def test_generate_title_calls_client_and_returns_validated_response():
    client = StubLLMClient({"output_text": '{"title":"Short title"}'})

    response = await generate_title(TitleRequest(text="Story text", max_chars=55), client=client)

    assert response == TitleResponse(title="Short title")
    assert client.calls[0]["input_text"] == "Story text"
    assert "must not exceed 55 characters" in client.calls[0]["instructions"]
    assert client.calls[0]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_generate_title_retries_once_on_invalid_json():
    client = StubLLMClient([
        {"output_text": "Short title"},
        {"output_text": '{"title":"Short title"}'},
    ])

    response = await generate_title(TitleRequest(text="Story text"), client=client)

    assert response == TitleResponse(title="Short title")
    assert len(client.calls) == 2
    assert "Your previous response was invalid." in client.calls[1]["instructions"]
