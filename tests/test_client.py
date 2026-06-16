import pytest
from niquests.exceptions import HTTPError

from llm_bot.client import LLMClient, UpstreamLLMError


class FakeResponse:
    def __init__(self, text='{"ok":true}', status_code=200, error: Exception | None = None):
        self.text = text
        self.status_code = status_code
        self.error = error

    def raise_for_status(self):
        if self.error is not None:
            raise self.error
        return None


class FakeSession:
    def __init__(self, *, base_url=None, headers=None, response=None):
        self.base_url = base_url
        self.headers = headers
        self.response = response or FakeResponse()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, path, json, timeout):
        self.path = path
        self.json = json
        self.timeout = timeout
        return self.response


@pytest.mark.asyncio
async def test_create_response_includes_reasoning_effort(monkeypatch):
    session = FakeSession()

    def fake_async_session(*, base_url=None, headers=None):
        session.base_url = base_url
        session.headers = headers
        return session

    monkeypatch.setattr("llm_bot.client.AsyncSession", fake_async_session)

    client = LLMClient(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
        api_mode="responses",
        timeout=30,
        reasoning_effort="high",
    )

    response = await client.create_response("Return JSON only.", "Story text")

    assert response == {"ok": True}
    assert session.path == "/responses"
    assert session.timeout == 30
    assert session.json["input"] == [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "Story text"},
    ]
    assert session.json["model"] == "test-model"
    assert session.json["reasoning"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_create_response_includes_thinking_budget_tokens(monkeypatch):
    session = FakeSession()

    def fake_async_session(*, base_url=None, headers=None):
        session.base_url = base_url
        session.headers = headers
        return session

    monkeypatch.setattr("llm_bot.client.AsyncSession", fake_async_session)

    client = LLMClient(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
        api_mode="responses",
        timeout=30,
        reasoning_effort="high",
        thinking_budget_tokens=512,
    )

    await client.create_response("Return JSON only.", "Story text")

    assert session.json["reasoning"] == {"effort": "high"}
    assert session.json["thinking_budget_tokens"] == 512


@pytest.mark.asyncio
async def test_create_response_omits_reasoning_effort_when_unset(monkeypatch):
    session = FakeSession()

    def fake_async_session(*, base_url=None, headers=None):
        session.base_url = base_url
        session.headers = headers
        return session

    monkeypatch.setattr("llm_bot.client.AsyncSession", fake_async_session)

    client = LLMClient(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
        api_mode="responses",
        timeout=30,
        reasoning_effort="",
    )

    await client.create_response("Return JSON only.", "Story text")

    assert "reasoning" not in session.json
    assert "thinking_budget_tokens" not in session.json


@pytest.mark.asyncio
async def test_create_response_extracts_nested_upstream_error_message(monkeypatch):
    session = FakeSession(
        response=FakeResponse(
            text='{"error":{"message":"Unsupported parameter: text.format"}}',
            status_code=400,
            error=HTTPError("bad request"),
        )
    )

    def fake_async_session(*, base_url=None, headers=None):
        session.base_url = base_url
        session.headers = headers
        return session

    monkeypatch.setattr("llm_bot.client.AsyncSession", fake_async_session)

    client = LLMClient(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
        api_mode="responses",
        timeout=30,
        reasoning_effort="",
    )

    with pytest.raises(UpstreamLLMError, match="Unsupported parameter: text.format"):
        await client.create_response("Return JSON only.", "Story text")


@pytest.mark.asyncio
async def test_create_response_falls_back_to_raw_upstream_error_text(monkeypatch):
    session = FakeSession(
        response=FakeResponse(
            text="provider exploded",
            status_code=500,
            error=HTTPError("server error"),
        )
    )

    def fake_async_session(*, base_url=None, headers=None):
        session.base_url = base_url
        session.headers = headers
        return session

    monkeypatch.setattr("llm_bot.client.AsyncSession", fake_async_session)

    client = LLMClient(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
        api_mode="responses",
        timeout=30,
        reasoning_effort="",
    )

    with pytest.raises(UpstreamLLMError, match="provider exploded"):
        await client.create_response("Return JSON only.", "Story text")


@pytest.mark.asyncio
async def test_create_response_uses_chat_completions_payload(monkeypatch):
    session = FakeSession(
        response=FakeResponse(
            text='{"choices":[{"message":{"content":"{\\"summary\\":\\"Short summary\\"}"}}]}'
        )
    )

    def fake_async_session(*, base_url=None, headers=None):
        session.base_url = base_url
        session.headers = headers
        return session

    monkeypatch.setattr("llm_bot.client.AsyncSession", fake_async_session)

    client = LLMClient(
        base_url="https://example.invalid/v1",
        api_key="test-key",
        model="test-model",
        api_mode="chat_completions",
        timeout=30,
        reasoning_effort="high",
        thinking_budget_tokens=256,
    )

    response = await client.create_response(
        "Return JSON only.",
        "Story text",
        {
            "type": "json_schema",
            "name": "summary_response",
            "strict": True,
            "schema": {"type": "object"},
        },
    )

    assert response == {"output_text": '{"summary":"Short summary"}'}
    assert session.path == "/chat/completions"
    assert session.json["messages"] == [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": "Story text"},
    ]
    assert session.json["reasoning_effort"] == "high"
    assert "reasoning" not in session.json
    assert session.json["thinking_budget_tokens"] == 256
    assert session.json["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "summary_response",
            "strict": True,
            "schema": {"type": "object"},
        },
    }
