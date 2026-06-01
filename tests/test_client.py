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
        timeout=30,
        reasoning_effort="high",
    )

    response = await client.create_response("Story text", "Return JSON only.")

    assert response == {"ok": True}
    assert session.path == "/responses"
    assert session.timeout == 30
    assert session.json["input"] == "Story text"
    assert session.json["instructions"] == "Return JSON only."
    assert session.json["model"] == "test-model"
    assert session.json["reasoning"] == {"effort": "high"}


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
        timeout=30,
        reasoning_effort="",
    )

    await client.create_response("Story text", "Return JSON only.")

    assert "reasoning" not in session.json


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
        timeout=30,
        reasoning_effort="",
    )

    with pytest.raises(UpstreamLLMError, match="Unsupported parameter: text.format"):
        await client.create_response("Story text", "Return JSON only.")


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
        timeout=30,
        reasoning_effort="",
    )

    with pytest.raises(UpstreamLLMError, match="provider exploded"):
        await client.create_response("Story text", "Return JSON only.")
