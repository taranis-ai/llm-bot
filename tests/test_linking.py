import pytest

from llm_bot.schemas import LookupResponse, NerRequest, NerResponse
from llm_bot.tasks.linking import (
    build_llm_linked_response,
    build_deterministic_linked_response,
    is_linking_enabled,
    lookup_entity_candidates,
    parse_linking_decision_map,
    resolve_linking_mode,
    resolve_lookup_language,
    select_llm_candidates,
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


class StubLLMClient:
    def __init__(self, response_data):
        self.response_data = response_data
        self.calls = []

    async def create_response(self, input_text: str, instructions: str, response_format=None):
        self.calls.append({"input_text": input_text, "instructions": instructions, "response_format": response_format})
        if isinstance(self.response_data, list):
            return self.response_data.pop(0)
        return self.response_data


def test_is_linking_enabled_uses_request_override():
    request = NerRequest(text="Apple released a new device.", link_entities=True)

    assert is_linking_enabled(request) is True


def test_is_linking_enabled_falls_back_to_config(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.linking.Config.NER_LINKING_ENABLED", True)
    monkeypatch.setattr("llm_bot.tasks.linking.Config.LOOKUP_BASE_URL", "https://example.invalid")
    request = NerRequest(text="Apple released a new device.")

    assert is_linking_enabled(request) is True


def test_is_linking_disabled_when_lookup_base_url_is_missing(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.linking.Config.LOOKUP_BASE_URL", "")
    request = NerRequest(text="Apple released a new device.", link_entities=True)

    assert is_linking_enabled(request) is False


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


def test_parse_linking_decision_map_rejects_qid_outside_candidate_set():
    with pytest.raises(ValueError, match="Response selected unsupported Wikidata QID for Apple: Q999"):
        parse_linking_decision_map({"output_text": '{"decisions":{"Apple":"Q999"}}'}, {"Apple": {"Q312"}})


@pytest.mark.asyncio
async def test_select_llm_candidates_chooses_valid_candidates():
    client = StubLLMClient({"output_text": '{"decisions":{"Apple":"Q312","GitHub":null}}'})
    response = NerResponse({"Apple": "ORG", "GitHub": "PRODUCT"})
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

    selected = await select_llm_candidates(
        client=client,
        source_text="Apple announced a new device.",
        response=response,
        lookup_results=lookup_results,
    )

    assert selected["Apple"] is not None
    assert selected["Apple"].qid == "Q312"
    assert selected["GitHub"] is None
    assert client.calls[0]["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_build_llm_linked_response_falls_back_to_unresolved_on_invalid_qid():
    client = StubLLMClient({"output_text": '{"decisions":{"Apple":"Q999"}}'})
    ner_response = NerResponse({"Apple": "ORG"})
    request = NerRequest(text="Apple announced a new device.")
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
        )
    }

    linked_response = await build_llm_linked_response(ner_response, request, lookup_results, client=client)

    assert linked_response.entities[0].mention == "Apple"
    assert linked_response.entities[0].wikidata_qid is None
    assert linked_response.entities[0].candidate_count == 1
