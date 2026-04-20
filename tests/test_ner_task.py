import pytest

from llm_bot.schemas import NerRequest, NerResponse
from llm_bot.tasks.ner import (
    build_ner_messages,
    extract_entities,
    normalize_entity_types,
    parse_ner_response,
    resolve_entity_types,
)


class StubLLMClient:
    def __init__(self, response_data):
        self.response_data = response_data
        self.calls = []

    async def create_response(self, input_text: str, instructions: str, response_format=None):
        self.calls.append({"input_text": input_text, "instructions": instructions, "response_format": response_format})
        return self.response_data


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


def test_normalize_entity_types_supports_legacy_labels():
    normalized = normalize_entity_types(["Person", "Organization", "CLICommand/CodeSnippet", "Con", "Tool"])

    assert normalized == ["PER", "ORG", "TOOL", "INDICATOR"]


def test_parse_ner_response_from_output_text():
    response = parse_ner_response({"output_text": '{"Microsoft":"ORG","Outlook":"PRODUCT"}'})

    assert response == NerResponse({"Microsoft": "ORG", "Outlook": "PRODUCT"})


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
        }
    )

    assert response == NerResponse({"APT29": "GROUP", "Mimikatz": "TOOL"})


@pytest.mark.asyncio
async def test_extract_entities_calls_client_and_returns_validated_response():
    client = StubLLMClient({"output_text": '{"Microsoft":"ORG","Outlook":"PRODUCT"}'})

    response = await extract_entities(NerRequest(text="Microsoft announced Outlook."), client=client)

    assert response == NerResponse({"Microsoft": "ORG", "Outlook": "PRODUCT"})
    assert client.calls[0]["input_text"] == "Microsoft announced Outlook."
    assert client.calls[0]["response_format"]["type"] == "json_schema"
