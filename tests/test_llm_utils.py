import json

import pytest

from llm_bot.tasks.llm_utils import loads_json_output


@pytest.mark.parametrize(
    "payload",
    [
        {"summary": 'The name contains a quote: " and a brace: {.'},
        {"summary": 'A backslash before a quote: \\" and a closing brace: }.'},
        {"summary": "A Windows path: C:\\reports\\"},
        {"nested": {"summary": "Final result"}},
    ],
)
def test_noisy_json_preserves_strings_and_nested_objects(payload):
    output = 'Draft: {"summary":"Earlier draft"}\nFinal answer:\n```json\n' + json.dumps(payload) + "\n```"

    assert loads_json_output(output) == payload


def test_noisy_json_does_not_salvage_nested_object_from_invalid_final_object():
    with pytest.raises(json.JSONDecodeError):
        loads_json_output('Draft: {"summary":"Earlier draft"}\nFinal: {broken: {"summary":"Nested"}}')
