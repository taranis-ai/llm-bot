import pytest

from llm_bot.schemas import SummarizeRequest, SummarizeResponse
from llm_bot.tasks.summarize import build_summary_messages, parse_summary_response, summarize


class StubLLMClient:
    def __init__(self, response_data):
        self.response_data = response_data
        self.calls = []

    async def create_response(self, input_text: str, instructions: str, response_format=None):
        self.calls.append({"input_text": input_text, "instructions": instructions, "response_format": response_format})
        return self.response_data


def test_build_summary_messages_without_max_words():
    request = SummarizeRequest(text="Example story text")

    system_message, user_message = build_summary_messages(request)

    assert "same language as the input text" in system_message["content"]
    assert "must not exceed 1000 characters" in system_message["content"]
    assert "words" not in system_message["content"]
    assert user_message["content"] == "Example story text"


def test_build_summary_messages_with_max_words():
    request = SummarizeRequest(text="Example story text", max_words=80)

    system_message, user_message = build_summary_messages(request)

    assert "must not exceed 80 words" in system_message["content"]
    assert user_message["content"] == "Example story text"


def test_build_summary_messages_truncates_input_text(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.summarize.Config.SUMMARY_MAX_INPUT_CHARS", 12)
    request = SummarizeRequest(text="This is a very long story text")

    _, user_message = build_summary_messages(request)

    assert user_message["content"] == "This is a v…"


def test_parse_summary_response_from_output_text():
    response = parse_summary_response({"output_text": '{"summary":"Short summary"}'})

    assert response == SummarizeResponse(summary="Short summary")


def test_parse_summary_response_truncates_summary(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.summarize.Config.SUMMARY_MAX_OUTPUT_CHARS", 12)

    response = parse_summary_response({"output_text": '{"summary":"This is a very long summary"}'})

    assert response == SummarizeResponse(summary="This is a v…")


def test_parse_summary_response_from_output_messages():
    response = parse_summary_response(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"summary":"Short summary"}',
                        }
                    ],
                }
            ]
        }
    )

    assert response == SummarizeResponse(summary="Short summary")


@pytest.mark.asyncio
async def test_summarize_calls_client_and_returns_validated_response():
    client = StubLLMClient({"output_text": '{"summary":"Short summary"}'})

    response = await summarize(SummarizeRequest(text="Story text", max_words=40), client=client)

    assert response == SummarizeResponse(summary="Short summary")
    assert client.calls[0]["input_text"] == "Story text"
    assert "must not exceed 40 words" in client.calls[0]["instructions"]
    assert client.calls[0]["response_format"]["type"] == "json_schema"
