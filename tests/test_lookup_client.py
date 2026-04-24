import pytest

from llm_bot.lookup_client import LookupClient
from llm_bot.schemas import LookupResponse


class FakeResponse:
    text = (
        '{"query":"Apple","language":"en","limit":5,"candidates":['
        '{"qid":"Q312","label":"Apple Inc.","description":"American technology company",'
        '"matched_alias":"Apple","match_type":"alias","language":"en","score":0.98,'
        '"is_label":true,"type_tags":["organization","company"]}'
        "]}"
    )

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

    async def get(self, path, params, timeout):
        self.path = path
        self.params = params
        self.timeout = timeout
        return FakeResponse()


@pytest.mark.asyncio
async def test_lookup_client_calls_lookup_endpoint(monkeypatch):
    session = FakeSession()

    def fake_async_session(*, base_url=None, headers=None):
        session.base_url = base_url
        session.headers = headers
        return session

    monkeypatch.setattr("llm_bot.lookup_client.AsyncSession", fake_async_session)

    client = LookupClient(
        base_url="https://example.invalid",
        api_key="lookup-key",
        timeout=15,
    )

    response = await client.lookup("Apple", "en", 5)

    assert isinstance(response, LookupResponse)
    assert session.path == "/lookup"
    assert session.params == {"q": "Apple", "lang": "en", "limit": 5}
    assert session.timeout == 15
    assert session.headers["Authorization"] == "Bearer lookup-key"


def test_lookup_response_validates_candidate_shape():
    response = LookupResponse.model_validate(
        {
            "query": "Apple",
            "language": "en",
            "limit": 5,
            "candidates": [
                {
                    "qid": "Q312",
                    "label": "Apple Inc.",
                    "description": "American technology company",
                    "matched_alias": "Apple",
                    "match_type": "alias",
                    "language": "en",
                    "score": 0.98,
                    "is_label": True,
                    "type_tags": ["organization", "company"],
                }
            ],
        }
    )

    assert response.candidates[0].qid == "Q312"
    assert response.candidates[0].type_tags == ["organization", "company"]
