import pytest

from llm_bot.schemas import LinkedNerResponse, LookupResponse, NerLinkRequest
from llm_bot.tasks.ner_link import extract_and_link
from tests.test_helpers import StubLLMClient, StubLookupClient

@pytest.mark.asyncio
async def test_extract_and_link_returns_linked_response_in_deterministic_mode(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.entity_linking.Config.LOOKUP_CANDIDATE_LIMIT", 3)
    client = StubLLMClient({"output_text": '{"Apple":"ORG","GitHub":"PRODUCT"}'})
    lookup_client = StubLookupClient(
        {
            "Apple": LookupResponse.model_validate(
                {
                    "query": "Apple",
                    "language": "en",
                    "limit": 3,
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
            ),
            "GitHub": LookupResponse.model_validate(
                {
                    "query": "GitHub",
                    "language": "en",
                    "limit": 3,
                    "candidates": [],
                }
            ),
        }
    )

    response = await extract_and_link(
        NerLinkRequest(
            text="Apple published on GitHub.",
            linking_mode="deterministic",
            language="en",
        ),
        client=client,
        lookup_client=lookup_client,
    )

    assert isinstance(response, LinkedNerResponse)
    assert response.entities[0].mention == "Apple"
    assert response.entities[0].wikidata_qid == "Q312"
    assert response.entities[1].mention == "GitHub"
    assert response.entities[1].wikidata_qid is None
    assert lookup_client.calls == [
        {"query": "Apple", "language": "en", "limit": 3},
        {"query": "GitHub", "language": "en", "limit": 3},
    ]


@pytest.mark.asyncio
async def test_extract_and_link_returns_linked_response_in_llm_mode(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.entity_linking.Config.LOOKUP_CANDIDATE_LIMIT", 3)
    client = StubLLMClient(
        [
            {"output_text": '{"Apple":"ORG"}'},
            {"output_text": '{"decisions":{"Apple":"Q312"}}'},
        ]
    )
    lookup_client = StubLookupClient(
        {
            "Apple": LookupResponse.model_validate(
                {
                    "query": "Apple",
                    "language": "en",
                    "limit": 3,
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
        }
    )

    response = await extract_and_link(
        NerLinkRequest(
            text="Apple released a new device.",
            linking_mode="llm",
            language="en",
        ),
        client=client,
        lookup_client=lookup_client,
    )

    assert isinstance(response, LinkedNerResponse)
    assert response.entities[0].mention == "Apple"
    assert response.entities[0].wikidata_qid == "Q312"
    assert len(client.calls) == 2
