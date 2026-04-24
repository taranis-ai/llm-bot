import pytest

from llm_bot.schemas import LookupResponse, NerRequest, NerResponse
from llm_bot.tasks.linking import (
    build_deterministic_linked_response,
    is_linking_enabled,
    lookup_entity_candidates,
    resolve_linking_mode,
    resolve_lookup_language,
    select_deterministic_candidate,
)


class StubLookupClient:
    def __init__(self):
        self.calls = []

    async def lookup(self, query: str, language: str, limit: int) -> LookupResponse:
        self.calls.append({"query": query, "language": language, "limit": limit})
        return LookupResponse.model_validate(
            {
                "query": query,
                "language": language,
                "limit": limit,
                "candidates": [],
            }
        )


def test_is_linking_enabled_uses_request_override():
    request = NerRequest(text="Apple released a new device.", link_entities=True)

    assert is_linking_enabled(request) is True


def test_is_linking_enabled_falls_back_to_config(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.linking.Config.NER_LINKING_ENABLED", True)
    request = NerRequest(text="Apple released a new device.")

    assert is_linking_enabled(request) is True


def test_resolve_lookup_language_uses_request_language():
    request = NerRequest(text="Apple released a new device.", language="de")

    assert resolve_lookup_language(request) == "de"


def test_resolve_lookup_language_falls_back_to_config(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.linking.Config.LOOKUP_DEFAULT_LANGUAGE", "fr")
    request = NerRequest(text="Apple released a new device.")

    assert resolve_lookup_language(request) == "fr"


def test_resolve_linking_mode_uses_request_override():
    request = NerRequest(text="Apple released a new device.", linking_mode="deterministic")

    assert resolve_linking_mode(request) == "deterministic"


def test_resolve_linking_mode_rejects_unknown_mode():
    request = NerRequest(text="Apple released a new device.", linking_mode="magic")

    with pytest.raises(ValueError, match="Unsupported linking mode requested: magic"):
        resolve_linking_mode(request)


@pytest.mark.asyncio
async def test_lookup_entity_candidates_uses_deduplicated_mentions(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.linking.Config.LOOKUP_CANDIDATE_LIMIT", 3)
    client = StubLookupClient()
    response = NerResponse({"Apple": "ORG", "GitHub": "PRODUCT"})
    request = NerRequest(text="Apple published GitHub content.", language="en")

    lookup_results = await lookup_entity_candidates(response, request, client=client)

    assert set(lookup_results.keys()) == {"Apple", "GitHub"}
    assert client.calls == [
        {"query": "Apple", "language": "en", "limit": 3},
        {"query": "GitHub", "language": "en", "limit": 3},
    ]


def test_select_deterministic_candidate_returns_top_candidate():
    lookup_response = LookupResponse.model_validate(
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
                },
                {
                    "qid": "Q89",
                    "label": "apple",
                    "description": "fruit of the apple tree",
                    "matched_alias": "Apple",
                    "match_type": "alias",
                    "language": "en",
                    "score": 0.42,
                    "is_label": True,
                    "type_tags": ["food"],
                },
            ],
        }
    )

    candidate = select_deterministic_candidate(lookup_response)

    assert candidate is not None
    assert candidate.qid == "Q312"


def test_build_deterministic_linked_response_uses_top_candidate_and_preserves_unresolved_entities():
    ner_response = NerResponse({"Apple": "ORG", "GitHub": "PRODUCT"})
    lookup_results = {
        "Apple": LookupResponse.model_validate(
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
        ),
        "GitHub": LookupResponse.model_validate(
            {
                "query": "GitHub",
                "language": "en",
                "limit": 5,
                "candidates": [],
            }
        ),
    }

    linked_response = build_deterministic_linked_response(ner_response, lookup_results)

    assert linked_response.entities[0].mention == "Apple"
    assert linked_response.entities[0].wikidata_qid == "Q312"
    assert linked_response.entities[0].candidate_count == 1
    assert linked_response.entities[1].mention == "GitHub"
    assert linked_response.entities[1].wikidata_qid is None
    assert linked_response.entities[1].candidate_count == 0
