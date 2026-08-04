import json

import pytest

from llm_bot.schemas import ChatRequest
from llm_bot.tasks.chat import build_chat_messages, chat
from tests.test_helpers import StubLLMClient


def make_request() -> ChatRequest:
    return ChatRequest.model_validate(
        {
            "message": "What should I do next?",
            "messages": [
                {"role": "user", "content": "Help me plan a release."},
                {
                    "role": "assistant",
                    "content": "Start by running the validation suite.",
                },
            ],
        }
    )


def test_build_messages_include_prior_conversation_and_latest_message():
    system_message, user_message = build_chat_messages(make_request())

    assert "general-purpose chat assistant" in system_message["content"]
    assert json.loads(user_message["content"]) == {
        "messages": [
            {"role": "user", "content": "Help me plan a release."},
            {
                "role": "assistant",
                "content": "Start by running the validation suite.",
            },
        ],
        "message": "What should I do next?",
    }


@pytest.mark.asyncio
async def test_chat_calls_client_with_structured_output_and_model_metadata(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.chat.Config.LLM_MODEL", "configured-model")
    client = StubLLMClient({"output_text": '{"answer":"Run the checks first."}'})

    response = await chat(make_request(), client=client)

    assert response.model_dump() == {
        "answer": "Run the checks first.",
        "model": "configured-model",
    }
    assert len(client.calls) == 1
    assert client.calls[0]["response_format"]["name"] == "chat_response"
