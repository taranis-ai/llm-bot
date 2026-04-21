from typing import Any


def strip_markdown_emphasis(value: str) -> str:
    normalized_value = value.strip()
    for marker in ("**", "__", "*", "_"):
        if normalized_value.startswith(marker) and normalized_value.endswith(marker):
            normalized_value = normalized_value[len(marker) : -len(marker)].strip()
    return normalized_value


def normalize_entity_name(value: str) -> str:
    return strip_markdown_emphasis(value)


def postprocess_entities(parsed_output: dict[str, Any]) -> dict[str, Any]:
    return {normalize_entity_name(entity): entity_type for entity, entity_type in parsed_output.items()}
