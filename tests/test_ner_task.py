import pytest

from llm_bot.schemas import LinkedNerResponse, LookupResponse, NerRequest, NerResponse
from llm_bot.tasks.ner_postprocessing import is_url_like, normalize_entity_name, postprocess_entities
from llm_bot.tasks.ner import (
    build_ner_messages,
    extract_entities,
    parse_ner_response,
    resolve_entity_types,
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


class StubLookupClient:
    def __init__(self, responses_by_query):
        self.responses_by_query = responses_by_query
        self.calls = []

    async def lookup(self, query: str, language: str, limit: int) -> LookupResponse:
        self.calls.append({"query": query, "language": language, "limit": limit})
        return self.responses_by_query[query]


@pytest.fixture(autouse=True)
def use_current_ner_entity_types(monkeypatch):
    monkeypatch.setattr(
        "llm_bot.tasks.ner.Config.NER_ENTITY_TYPES",
        "PER,ORG,GPE,PRODUCT,EVENT,GROUP,MALWARE,TOOL,TACTIC,TECHNIQUE,SECTOR,INDICATOR",
    )


def test_build_ner_messages_without_cybersecurity():
    request = NerRequest(text="Microsoft announced a new Outlook update.")

    system_message, user_message = build_ner_messages(request)

    assert "Cybersecurity mode is disabled." in system_message["content"]
    assert "Allowed labels for this request:" in system_message["content"]
    assert "PER" in system_message["content"]
    assert "ORG" in system_message["content"]
    assert "GPE" in system_message["content"]
    assert "PRODUCT" in system_message["content"]
    assert "EVENT" in system_message["content"]
    assert "GROUP" not in system_message["content"]
    assert "MALWARE" not in system_message["content"]
    assert "Cybersecurity examples:" not in system_message["content"]
    assert "Do not split compounds or larger derived expressions into smaller embedded entities." in system_message["content"]
    assert '"Einsteins" -> {"Einstein": "PER"}' in system_message["content"]
    assert user_message["content"] == "Microsoft announced a new Outlook update."


def test_build_ner_messages_with_cybersecurity():
    request = NerRequest(text="APT29 used Mimikatz.", cybersecurity=True)

    system_message, user_message = build_ner_messages(request)

    assert "Cybersecurity mode is enabled." in system_message["content"]
    assert "Allowed labels for this request:" in system_message["content"]
    assert "GROUP" in system_message["content"]
    assert "MALWARE" in system_message["content"]
    assert "TOOL" in system_message["content"]
    assert "Cybersecurity examples:" in system_message["content"]
    assert user_message["content"] == "APT29 used Mimikatz."


def test_resolve_entity_types_uses_request_override():
    request = NerRequest(text="Joe Biden visited Vienna.", entity_types=["PER", "ORG"])

    resolved = resolve_entity_types(request)

    assert resolved == ["PER", "ORG"]


def test_resolve_entity_types_uses_request_override_with_cybersecurity():
    request = NerRequest(text="APT29 used Mimikatz.", cybersecurity=True, entity_types=["GROUP", "TOOL"])

    resolved = resolve_entity_types(request)

    assert resolved == ["GROUP", "TOOL"]


def test_build_ner_messages_includes_request_entity_types():
    request = NerRequest(text="Joe Biden visited Vienna.", entity_types=["PER", "ORG"])

    system_message, _ = build_ner_messages(request)

    assert "Allowed labels for this request: PER, ORG." in system_message["content"]


def test_build_ner_messages_includes_request_entity_types_with_cybersecurity():
    request = NerRequest(text="APT29 used Mimikatz.", cybersecurity=True, entity_types=["GROUP", "TOOL"])

    system_message, _ = build_ner_messages(request)

    assert "Allowed labels for this request: GROUP, TOOL." in system_message["content"]


def test_resolve_entity_types_rejects_unknown_request_entity_types():
    request = NerRequest(text="APT29 used Mimikatz.", entity_types=["GROUP", "AlienType"])

    with pytest.raises(ValueError, match="Unsupported entity types requested: AlienType"):
        resolve_entity_types(request)


def test_normalize_entity_name_strips_markdown_emphasis():
    assert normalize_entity_name(" **Microsoft** ") == "Microsoft"
    assert normalize_entity_name("_Vienna_") == "Vienna"


def test_is_url_like_detects_urls():
    assert is_url_like("https://example.com/login") is True
    assert is_url_like("www.example.com") is True
    assert is_url_like("Wikipedia") is False


def test_postprocess_entities_normalizes_entity_names():
    processed = postprocess_entities({"**Microsoft**": "ORG", "_Vienna_": "GPE"})

    assert processed == {"Microsoft": "ORG", "Vienna": "GPE"}


def test_postprocess_entities_drops_url_like_products():
    processed = postprocess_entities(
        {
            "https://mail.google.com": "PRODUCT",
            "Wikipedia": "PRODUCT",
            "https://example.org": "ORG",
        }
    )

    assert processed == {"Wikipedia": "PRODUCT", "https://example.org": "ORG"}


def test_parse_ner_response_from_output_text():
    response = parse_ner_response({"output_text": '{"Microsoft":"ORG","Outlook":"PRODUCT"}'}, ["ORG", "PRODUCT"])

    assert response == NerResponse({"Microsoft": "ORG", "Outlook": "PRODUCT"})


def test_parse_ner_response_strips_markdown_emphasis_from_entity_names():
    response = parse_ner_response({"output_text": '{"**Microsoft**":"ORG","_Vienna_":"GPE"}'}, ["ORG", "GPE"])

    assert response == NerResponse({"Microsoft": "ORG", "Vienna": "GPE"})


def test_parse_ner_response_drops_url_like_products():
    response = parse_ner_response(
        {"output_text": '{"https://mail.google.com":"PRODUCT","Wikipedia":"PRODUCT","https://example.org":"ORG"}'},
        ["PRODUCT", "ORG"],
    )

    assert response == NerResponse({"Wikipedia": "PRODUCT", "https://example.org": "ORG"})


def test_parse_ner_response_from_output_messages():
    response = parse_ner_response(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"APT29":"GROUP","Mimikatz":"TOOL"}',
                        }
                    ],
                }
            ]
        },
        ["GROUP", "TOOL"],
    )

    assert response == NerResponse({"APT29": "GROUP", "Mimikatz": "TOOL"})


def test_parse_ner_response_rejects_unsupported_entity_types():
    with pytest.raises(ValueError, match="Response contained unsupported entity types"):
        parse_ner_response({"output_text": '{"Wikipedia":"PRODUCT","Foo":"*Omitted*"}'}, ["PRODUCT"])


def test_parse_ner_response_extracts_final_json_from_noisy_output():
    response = parse_ner_response(
        {
            "output_text": (
                'Thinking: Chrome Dev is a product. Final plan.{"Chrome Dev":"PRODUCT",'
                '"Android":"PRODUCT","Google Play":"ORG"}'
            )
        },
        ["ORG", "PRODUCT"],
    )

    assert response == NerResponse({"Chrome Dev": "PRODUCT", "Android": "PRODUCT", "Google Play": "ORG"})


@pytest.mark.asyncio
async def test_extract_entities_calls_client_and_returns_validated_response():
    client = StubLLMClient({"output_text": '{"Microsoft":"ORG","Outlook":"PRODUCT"}'})

    response = await extract_entities(NerRequest(text="Microsoft announced Outlook."), client=client)

    assert response == NerResponse({"Microsoft": "ORG", "Outlook": "PRODUCT"})
    assert client.calls[0]["input_text"] == "Microsoft announced Outlook."
    assert client.calls[0]["response_format"]["type"] == "json_schema"
    assert set(client.calls[0]["response_format"]["schema"]["additionalProperties"]["enum"]) == {
        "PER",
        "ORG",
        "GPE",
        "PRODUCT",
        "EVENT",
    }


@pytest.mark.asyncio
async def test_extract_entities_retries_once_on_invalid_json():
    client = StubLLMClient(
        [
            {"output_text": "APT29 GROUP"},
            {"output_text": '{"APT29":"GROUP","Mimikatz":"TOOL"}'},
        ]
    )

    response = await extract_entities(NerRequest(text="APT29 used Mimikatz.", cybersecurity=True), client=client)

    assert response == NerResponse({"APT29": "GROUP", "Mimikatz": "TOOL"})
    assert len(client.calls) == 2
    assert "Your previous response was invalid." in client.calls[1]["instructions"]


@pytest.mark.asyncio
async def test_extract_entities_retries_once_on_validation_error():
    client = StubLLMClient(
        [
            {"output_text": '{"Microsoft":["ORG"]}'},
            {"output_text": '{"Microsoft":"ORG"}'},
        ]
    )

    response = await extract_entities(NerRequest(text="Microsoft announced Outlook."), client=client)

    assert response == NerResponse({"Microsoft": "ORG"})
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_extract_entities_retries_once_on_unsupported_entity_type():
    client = StubLLMClient(
        [
            {"output_text": '{"Wikipedia":"PRODUCT","Foo":"*Omitted*"}'},
            {"output_text": '{"Wikipedia":"PRODUCT"}'},
        ]
    )

    response = await extract_entities(NerRequest(text="Wikipedia was mentioned."), client=client)

    assert response == NerResponse({"Wikipedia": "PRODUCT"})
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_extract_entities_returns_linked_response_in_deterministic_mode(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.linking.Config.LOOKUP_CANDIDATE_LIMIT", 3)
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

    response = await extract_entities(
        NerRequest(
            text="Apple published on GitHub.",
            link_entities=True,
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
async def test_extract_entities_returns_linked_response_in_llm_mode(monkeypatch):
    monkeypatch.setattr("llm_bot.tasks.linking.Config.LOOKUP_CANDIDATE_LIMIT", 3)
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

    response = await extract_entities(
        NerRequest(
            text="Apple released a new device.",
            link_entities=True,
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
