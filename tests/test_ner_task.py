import pytest

from llm_bot.schemas import NerRequest, NerResponse
from llm_bot.tasks.ner import build_ner_messages, extract_entities, parse_ner_response, resolve_entity_types


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

    assert "Use only these entity types:" in system_message["content"]
    assert "Cybersecurity mode is disabled." in system_message["content"]
    assert "CLICommand/CodeSnippet" not in system_message["content"]
    assert "Malware" not in system_message["content"]
    assert user_message["content"] == "Microsoft announced a new Outlook update."


def test_build_ner_messages_with_cybersecurity():
    request = NerRequest(text="APT29 used Mimikatz.", cybersecurity=True)

    system_message, user_message = build_ner_messages(request)

    assert "Cybersecurity mode is enabled." in system_message["content"]
    assert "CLICommand/CodeSnippet" in system_message["content"]
    assert "Malware" in system_message["content"]
    assert user_message["content"] == "APT29 used Mimikatz."


def test_resolve_entity_types_uses_request_override():
    request = NerRequest(text="APT29 used Mimikatz.", entity_types=["Group", "Tool"])

    resolved = resolve_entity_types(request)

    assert resolved == []


def test_resolve_entity_types_uses_request_override_with_cybersecurity():
    request = NerRequest(text="APT29 used Mimikatz.", cybersecurity=True, entity_types=["Group", "Tool"])

    resolved = resolve_entity_types(request)

    assert resolved == ["Group", "Tool"]


def test_build_ner_messages_includes_request_entity_types():
    request = NerRequest(text="APT29 used Mimikatz.", entity_types=["Group", "Tool"])

    system_message, _ = build_ner_messages(request)

    assert "Use only these entity types: ." in system_message["content"]


def test_build_ner_messages_includes_request_entity_types_with_cybersecurity():
    request = NerRequest(text="APT29 used Mimikatz.", cybersecurity=True, entity_types=["Group", "Tool"])

    system_message, _ = build_ner_messages(request)

    assert "Use only these entity types: Group, Tool." in system_message["content"]


def test_resolve_entity_types_rejects_unknown_request_entity_types():
    request = NerRequest(text="APT29 used Mimikatz.", entity_types=["Group", "AlienType"])

    with pytest.raises(ValueError, match="Unsupported entity types requested: AlienType"):
        resolve_entity_types(request)


def test_parse_ner_response_from_output_text():
    response = parse_ner_response({"output_text": '{"Microsoft":"Organization","Outlook":"Product"}'})

    assert response == NerResponse({"Microsoft": "Organization", "Outlook": "Product"})


def test_parse_ner_response_from_output_messages():
    response = parse_ner_response(
        {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"APT29":"Group","Mimikatz":"Tool"}',
                        }
                    ],
                }
            ]
        }
    )

    assert response == NerResponse({"APT29": "Group", "Mimikatz": "Tool"})


@pytest.mark.asyncio
async def test_extract_entities_calls_client_and_returns_validated_response():
    client = StubLLMClient({"output_text": '{"Microsoft":"Organization","Outlook":"Product"}'})

    response = await extract_entities(NerRequest(text="Microsoft announced Outlook."), client=client)

    assert response == NerResponse({"Microsoft": "Organization", "Outlook": "Product"})
    assert client.calls[0]["input_text"] == "Microsoft announced Outlook."
    assert client.calls[0]["response_format"]["type"] == "json_schema"
