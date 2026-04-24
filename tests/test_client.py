import pytest

from llm_bot.client import LLMClient


class FakeResponse:
    text = '{"ok":true}'

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, *, base_url=None, headers=None):
        self.base_url = base_url
        self.headers = headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, path, json, timeout):
        self.path = path
        self.json = json
        self.timeout = timeout
        return FakeResponse()


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
