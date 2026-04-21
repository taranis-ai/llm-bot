import re
from typing import Any


URL_PATTERN = re.compile(
    r"^(?:https?://|www\.)[^\s/$.?#].[^\s]*$",
    re.IGNORECASE,
)


def strip_markdown_emphasis(value: str) -> str:
    normalized_value = value.strip()
    for marker in ("**", "__", "*", "_"):
        if normalized_value.startswith(marker) and normalized_value.endswith(marker):
            normalized_value = normalized_value[len(marker) : -len(marker)].strip()
    return normalized_value


def normalize_entity_name(value: str) -> str:
    return strip_markdown_emphasis(value)


def is_url_like(value: str) -> bool:
    return URL_PATTERN.match(value.strip()) is not None


def postprocess_entities(parsed_output: dict[str, Any]) -> dict[str, Any]:
    processed_entities: dict[str, Any] = {}
    for entity, entity_type in parsed_output.items():
        normalized_entity = normalize_entity_name(entity)
        if entity_type == "PRODUCT" and is_url_like(normalized_entity):
            continue
        processed_entities[normalized_entity] = entity_type
    return processed_entities
