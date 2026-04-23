import pytest

from llm_bot.schemas import SummarizeRequest, SummarizeResponse
from llm_bot.reasoning import apply_reasoning_profile, extract_structured_reasoning, strip_reasoning_output
from llm_bot.tasks.llm_utils import get_output_text
from llm_bot.tasks.summarize import build_summary_messages, parse_summary_response, summarize


class StubLLMClient:
    def __init__(self, response_data):
        self.response_data = response_data
        self.calls = []

    async def create_response(self, input_text: str, instructions: str, response_format=None):
        self.calls.append({"input_text": input_text, "instructions": instructions, "response_format": response_format})
        if isinstance(self.response_data, list):
            return self.response_data.pop(0)
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


def test_strip_reasoning_output_removes_think_block():
    output_text = "[THINK]\nI should summarize this.\n[/THINK]\n{\"summary\":\"Short summary\"}"

    assert strip_reasoning_output(output_text) == '{"summary":"Short summary"}'


def test_extract_structured_reasoning_from_responses_output():
    response_data = {
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "I should summarize this."}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"summary":"Short summary"}'}],
            },
        ]
    }

    assert extract_structured_reasoning(response_data) == "I should summarize this."


def test_get_output_text_logs_structured_reasoning(caplog):
    response_data = {
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "I should summarize this."}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"summary":"Short summary"}'}],
            },
        ]
    }

    with caplog.at_level("DEBUG", logger="llm_bot"):
        output_text = get_output_text(response_data)

    assert output_text == '{"summary":"Short summary"}'
    assert "LLM reasoning output: I should summarize this." in caplog.text


def test_get_output_text_logs_inline_reasoning(caplog):
    response_data = {"output_text": '[THINK]I should summarize this.[/THINK]{"summary":"Short summary"}'}

    with caplog.at_level("DEBUG", logger="llm_bot"):
        output_text = get_output_text(response_data)

    assert output_text == '{"summary":"Short summary"}'
    assert "LLM reasoning output: [THINK]I should summarize this.[/THINK]" in caplog.text


def test_apply_reasoning_profile_adds_ministral_instructions(monkeypatch):
    monkeypatch.setattr("llm_bot.reasoning.Config.LLM_REASONING_PROFILE", "ministral")

    instructions = apply_reasoning_profile("Return JSON only.")

    assert "# HOW YOU SHOULD THINK AND ANSWER" in instructions
    assert "[THINK]" in instructions
    assert "Return JSON only." in instructions
    assert "final response after [/THINK] must be valid JSON only" in instructions


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


@pytest.mark.asyncio
async def test_summarize_applies_reasoning_profile(monkeypatch):
    monkeypatch.setattr("llm_bot.reasoning.Config.LLM_REASONING_PROFILE", "ministral")
    client = StubLLMClient({"output_text": '[THINK]draft[/THINK]{"summary":"Short summary"}'})

    response = await summarize(SummarizeRequest(text="Story text"), client=client)

    assert response == SummarizeResponse(summary="Short summary")
    assert "# HOW YOU SHOULD THINK AND ANSWER" in client.calls[0]["instructions"]


@pytest.mark.asyncio
async def test_summarize_retries_once_on_invalid_json():
    client = StubLLMClient(
        [
            {"output_text": "Short summary"},
            {"output_text": '{"summary":"Short summary"}'},
        ]
    )

    response = await summarize(SummarizeRequest(text="Story text"), client=client)

    assert response == SummarizeResponse(summary="Short summary")
    assert len(client.calls) == 2
    assert "Your previous response was invalid." in client.calls[1]["instructions"]


@pytest.mark.asyncio
async def test_summarize_retries_once_on_validation_error():
    client = StubLLMClient(
        [
            {"output_text": '{"summary":""}'},
            {"output_text": '{"summary":"Recovered summary"}'},
        ]
    )

    response = await summarize(SummarizeRequest(text="Story text"), client=client)

    assert response == SummarizeResponse(summary="Recovered summary")
    assert len(client.calls) == 2
