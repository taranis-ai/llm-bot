import json
from pathlib import Path
from typing import Any

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.log import logger
from llm_bot.schemas import NerRequest, NerResponse
from llm_bot.tasks.llm_utils import create_and_parse_response, get_output_text


PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "ner.txt"
GENERAL_ENTITY_TYPES = {
    "PER",
    "ORG",
    "GPE",
    "PRODUCT",
    "EVENT",
}
CYBERSECURITY_ENTITY_TYPES = {
    "GROUP",
    "MALWARE",
    "TOOL",
    "TACTIC",
    "TECHNIQUE",
    "SECTOR",
    "INDICATOR",
}
ALLOWED_ENTITY_TYPES = GENERAL_ENTITY_TYPES | CYBERSECURITY_ENTITY_TYPES
LEGACY_ENTITY_TYPE_MAP = {
    "Person": "PER",
    "Organization": "ORG",
    "Location": "GPE",
    "Product": "PRODUCT",
    "Event": "EVENT",
    "Group": "GROUP",
    "Malware": "MALWARE",
    "Tool": "TOOL",
    "Tactic": "TACTIC",
    "Technique": "TECHNIQUE",
    "Sector": "SECTOR",
    "Con": "INDICATOR",
    "CLICommand/CodeSnippet": "TOOL",
}
CYBERSECURITY_EXAMPLES = """

Cybersecurity examples:

In addition to the base labels, you may also use these cybersecurity labels when they are explicitly named in the text:

- GROUP
- MALWARE
- TOOL
- TACTIC
- TECHNIQUE
- SECTOR
- INDICATOR

Use cybersecurity labels only when they are supported by the text and are more specific than a general label.

Cybersecurity label definitions:

GROUP
Use for named threat actors, hacking groups, intrusion sets, or tracked adversary clusters.

Examples:
- APT29
- Lazarus Group
- FIN7
- Sandworm

Do not use for:
- generic references such as "attackers", "hackers", or "researchers"

MALWARE
Use for named malware families, ransomware families, worms, trojans, botnets, implants, or backdoors.

Examples:
- Emotet
- TrickBot
- WannaCry
- PlugX

TOOL
Use for named offensive or defensive tools, frameworks, utilities, or exploit kits.

Examples:
- Mimikatz
- Metasploit
- Cobalt Strike
- Nmap

Do not use for:
- generic words such as "tool", "script", or "malware"

TACTIC
Use for explicitly named high-level adversary goals or tactical categories.

Examples:
- Lateral Movement
- Initial Access
- Persistence

TECHNIQUE
Use for explicitly named attack techniques or tradecraft methods.

Examples:
- Spearphishing Attachment
- Credential Dumping
- DLL Search Order Hijacking

SECTOR
Use for explicitly named industry or public sectors when referred to as targets or domains.

Examples:
- healthcare sector
- energy sector
- financial sector

INDICATOR
Use for explicitly named technical indicators when they appear as concrete indicators in the text.

Includes:
- domain names
- IP addresses
- file hashes
- named email addresses

Examples:
- example-malware.com
- 185.10.10.5
- admin@phish-example.com

Cybersecurity rules:
- General labels are still valid when they are the better fit.
- A named company remains ORG, not SECTOR.
- A named software product may be PRODUCT or TOOL depending on context.
- If an entity could fit both a general and cybersecurity label, choose the more contextually specific label.
- Do not force cybersecurity labels onto ordinary entities.

Input text:
APT29 used Mimikatz against government systems in Vienna.
Expected output:
{"APT29": "GROUP", "Mimikatz": "TOOL", "Vienna": "GPE"}

Input text:
Microsoft warned that Emotet campaigns targeted the healthcare sector through Gmail accounts.
Expected output:
{"Microsoft": "ORG", "Emotet": "MALWARE", "healthcare sector": "SECTOR", "Gmail accounts": "PRODUCT"}
""".strip()


def load_ner_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def normalize_entity_types(entity_types: list[str]) -> list[str]:
    normalized_entity_types: list[str] = []
    for entity_type in entity_types:
        normalized_entity_type = LEGACY_ENTITY_TYPE_MAP.get(entity_type, entity_type)
        if normalized_entity_type not in normalized_entity_types:
            normalized_entity_types.append(normalized_entity_type)
    return normalized_entity_types


def resolve_entity_types(request: NerRequest) -> list[str]:
    entity_types = normalize_entity_types(request.entity_types or Config.ner_entity_types)
    unknown_entity_types = sorted(set(entity_types) - ALLOWED_ENTITY_TYPES)
    if unknown_entity_types:
        raise ValueError(f"Unsupported entity types requested: {', '.join(unknown_entity_types)}")
    if request.cybersecurity:
        return entity_types
    return [entity_type for entity_type in entity_types if entity_type in GENERAL_ENTITY_TYPES]


def build_ner_messages(request: NerRequest) -> list[dict[str, str]]:
    system_prompt = load_ner_prompt()
    entity_types = resolve_entity_types(request)
    if request.cybersecurity:
        system_prompt = (
            f"{system_prompt}\n"
            f"{CYBERSECURITY_EXAMPLES}\n"
            f"Cybersecurity mode is enabled.\n"
            f"Allowed labels for this request: {', '.join(entity_types)}."
        )
    else:
        system_prompt = (
            f"{system_prompt}\n"
            f"Cybersecurity mode is disabled.\n"
            f"Allowed labels for this request: {', '.join(entity_types)}."
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.text},
    ]

def parse_ner_response(response_data: dict[str, Any]) -> NerResponse:
    output_text = get_output_text(response_data)
    logger.debug("Raw NER output: %s", output_text)
    parsed_output = json.loads(output_text)
    return NerResponse.model_validate(parsed_output)


def get_ner_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "ner_response",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    }


async def extract_entities(request: NerRequest, client: LLMClient | None = None) -> NerResponse:
    llm_client = client or LLMClient()
    system_message, user_message = build_ner_messages(request)
    return await create_and_parse_response(
        client=llm_client,
        task_name="NER",
        input_text=user_message["content"],
        instructions=system_message["content"],
        response_format=get_ner_response_format(),
        parse_response=parse_ner_response,
    )
