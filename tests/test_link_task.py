import pytest

from llm_bot.schemas import LinkRequest, LinkedNerResponse, LookupResponse
from llm_bot.tasks.link_task import build_linking_ner_response, link_entities


class StubLLMClient:
    def __init__(self, response_data):
        self.response_data = response_data
        self.calls = []

    async def create_response(self, input_text: str, instructions: str, response_format=None):
        self.calls.append({"input_text": input_text, "instructions": instructions, "response_format": response_format})
        if isinstance(self.response_data, list):
            return self.response_data.pop(0)
        return self.response_data


class StubLookupClient:
    def __init__(self, responses_by_query):
        self.responses_by_query = responses_by_query
        self.calls = []

    async def lookup(self, query: str, language: str, limit: int) -> LookupResponse:
        self.calls.append({"query": query, "language": language, "limit": limit})
        return self.responses_by_query[query]


def test_build_linking_ner_response_converts_entities_to_map():
    request = LinkRequest.model_validate(
        {
            "text": "Apple announced a new device in Cupertino.",
            "entities": [
                {"mention": "Apple", "type": "ORG"},
                {"mention": "Cupertino", "type": "GPE"},
            ],
        }
    )

    response = build_linking_ner_response(request)

    assert response.root == {"Apple": "ORG", "Cupertino": "GPE"}


@pytest.mark.asyncio
async def test_link_entities_returns_linked_response_in_deterministic_mode(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.entity_linking.Config.LOOKUP_CANDIDATE_LIMIT", 3)
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
            "Cupertino": LookupResponse.model_validate(
                {
                    "query": "Cupertino",
                    "language": "en",
                    "limit": 3,
                    "candidates": [
                        {
                            "qid": "Q189471",
                            "label": "Cupertino",
                            "description": "city in California",
                            "matched_alias": "Cupertino",
                            "match_type": "label",
                            "language": "en",
                            "score": 0.97,
                            "is_label": True,
                            "type_tags": ["city"],
                        }
                    ],
                }
            ),
        }
    )

    response = await link_entities(
        LinkRequest.model_validate(
            {
                "text": "Apple announced a new device in Cupertino.",
                "language": "en",
                "linking_mode": "deterministic",
                "entities": [
                    {"mention": "Apple", "type": "ORG"},
                    {"mention": "Cupertino", "type": "GPE"},
                ],
            }
        ),
        lookup_client=lookup_client,
    )

    assert isinstance(response, LinkedNerResponse)
    assert response.entities[0].wikidata_qid == "Q312"
    assert response.entities[1].wikidata_qid == "Q189471"


@pytest.mark.asyncio
async def test_link_entities_returns_linked_response_in_llm_mode(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.entity_linking.Config.LOOKUP_CANDIDATE_LIMIT", 3)
    client = StubLLMClient({"output_text": '{"decisions":{"Apple":"Q312","Cupertino":"Q189471"}}'})
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
            "Cupertino": LookupResponse.model_validate(
                {
                    "query": "Cupertino",
                    "language": "en",
                    "limit": 3,
                    "candidates": [
                        {
                            "qid": "Q189471",
                            "label": "Cupertino",
                            "description": "city in California",
                            "matched_alias": "Cupertino",
                            "match_type": "label",
                            "language": "en",
                            "score": 0.97,
                            "is_label": True,
                            "type_tags": ["city"],
                        }
                    ],
                }
            ),
        }
    )

    response = await link_entities(
        LinkRequest.model_validate(
            {
                "text": "Apple announced a new device in Cupertino.",
                "language": "en",
                "linking_mode": "llm",
                "entities": [
                    {"mention": "Apple", "type": "ORG"},
                    {"mention": "Cupertino", "type": "GPE"},
                ],
            }
        ),
        client=client,
        lookup_client=lookup_client,
    )

    assert isinstance(response, LinkedNerResponse)
    assert response.entities[0].wikidata_qid == "Q312"
    assert response.entities[1].wikidata_qid == "Q189471"
    assert len(client.calls) == 1
