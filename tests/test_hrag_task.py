import json

import pytest

from llm_bot.schemas import HragRequest
from llm_bot.tasks.hrag import (
    answer_with_hrag,
    build_hrag_messages,
    parse_hrag_response,
)
from llm_bot.tasks.llm_utils import InvalidLLMOutputError
from tests.test_helpers import StubLLMClient


REQUEST_PAYLOAD = {
    "question": "Who operates the service?",
    "passages": [
        {
            "id": "passage-1",
            "source": "report.pdf#page=2",
            "text": "Example Corp operates the service.",
        }
    ],
    "graph_facts": [
        {
            "id": "fact-1",
            "source": "graph://service/42",
            "fact": "Example Corp -[OPERATES]-> Service 42",
        }
    ],
}


def make_request() -> HragRequest:
    return HragRequest.model_validate(REQUEST_PAYLOAD)


def test_build_messages_include_evidence_and_strict_grounding_rules():
    system_message, user_message = build_hrag_messages(make_request())

    assert json.loads(user_message["content"]) == REQUEST_PAYLOAD
    assert "Never use outside knowledge" in system_message["content"]
    assert "Never invent entities, facts, relationships" in system_message["content"]
    assert "Cite only evidence IDs" in system_message["content"]
    assert "insufficient" in system_message["content"]


def test_parse_accepts_citations_from_both_evidence_lists():
    output = {
        "answer": "Example Corp operates the service.",
        "citations": ["passage-1", "fact-1"],
        "insufficient_evidence": False,
    }

    response = parse_hrag_response(
        {"output_text": json.dumps(output)},
        make_request(),
    )

    assert response.model_dump() == output


def test_parse_accepts_insufficient_answer_without_evidence_or_citations():
    request = HragRequest.model_validate(
        {
            "question": "Who operates the service?",
            "passages": [],
            "graph_facts": [],
        }
    )
    output = {
        "answer": "The supplied evidence is insufficient to answer the question.",
        "citations": [],
        "insufficient_evidence": True,
    }

    response = parse_hrag_response(
        {"output_text": json.dumps(output)},
        request,
    )

    assert response.model_dump() == output


def test_parse_rejects_citation_not_supplied_by_caller():
    output = {
        "answer": "Example Corp operates the service.",
        "citations": ["invented-id"],
        "insufficient_evidence": False,
    }

    with pytest.raises(
        InvalidLLMOutputError,
        match="unknown evidence IDs: invented-id",
    ):
        parse_hrag_response({"output_text": json.dumps(output)}, make_request())


def test_parse_rejects_sufficient_answer_without_a_citation():
    output = {
        "answer": "Example Corp operates the service.",
        "citations": [],
        "insufficient_evidence": False,
    }

    with pytest.raises(InvalidLLMOutputError, match="must cite at least one"):
        parse_hrag_response({"output_text": json.dumps(output)}, make_request())


@pytest.mark.asyncio
async def test_hrag_repairs_invented_citation_once():
    client = StubLLMClient(
        [
            {
                "output_text": json.dumps(
                    {
                        "answer": "Example Corp operates the service.",
                        "citations": ["invented-id"],
                        "insufficient_evidence": False,
                    }
                )
            },
            {
                "output_text": json.dumps(
                    {
                        "answer": "Example Corp operates the service.",
                        "citations": ["passage-1"],
                        "insufficient_evidence": False,
                    }
                )
            },
        ]
    )

    response = await answer_with_hrag(make_request(), client=client)

    assert response.citations == ["passage-1"]
    assert len(client.calls) == 2
    assert client.calls[0]["response_format"]["name"] == "hrag_response"
