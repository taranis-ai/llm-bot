import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.lookup_client import LookupClient
from llm_bot.log import logger
from llm_bot.schemas import LinkedEntity, LinkedNerResponse, LookupCandidate, LookupResponse, NerRequest, NerResponse
from llm_bot.tasks.llm_utils import InvalidLLMOutputError, create_and_parse_response, get_output_text, loads_json_output


ALLOWED_LINKING_MODES = {"llm", "deterministic"}


class UnsupportedLinkingModeError(ValueError):
    pass


class LinkingDecisionMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: dict[str, str | None]


def is_linking_enabled(request: NerRequest) -> bool:
    if not Config.LOOKUP_BASE_URL:
        return False
    if not Config.NER_LINKING_ENABLED:
        return False
    if request.link_entities is not None:
        return request.link_entities
    return True


def resolve_lookup_language(request: NerRequest) -> str:
    return request.language or Config.LOOKUP_DEFAULT_LANGUAGE


def resolve_linking_mode(request: NerRequest) -> str:
    linking_mode = request.linking_mode or Config.NER_LINKING_MODE
    if linking_mode not in ALLOWED_LINKING_MODES:
        raise UnsupportedLinkingModeError(
            f"Unsupported linking mode requested: {linking_mode}. "
            f"Allowed linking modes: {', '.join(sorted(ALLOWED_LINKING_MODES))}"
        )
    return linking_mode


async def lookup_entity_candidates(
    response: NerResponse,
    request: NerRequest,
    client: LookupClient | None = None,
) -> dict[str, LookupResponse]:
    lookup_client = client or LookupClient()
    language = resolve_lookup_language(request)
    limit = Config.LOOKUP_CANDIDATE_LIMIT

    lookup_results: dict[str, LookupResponse] = {}
    for mention in response.root:
        if mention in lookup_results:
            continue
        lookup_results[mention] = await lookup_client.lookup(mention, language, limit)
    return lookup_results


def select_deterministic_candidate(lookup_response: LookupResponse) -> LookupCandidate | None:
    if not lookup_response.candidates:
        return None
    return lookup_response.candidates[0]


def build_deterministic_linked_response(
    response: NerResponse,
    lookup_results: dict[str, LookupResponse],
) -> LinkedNerResponse:
    entities: list[LinkedEntity] = []
    for mention, entity_type in response.root.items():
        lookup_response = lookup_results.get(mention)
        candidate = select_deterministic_candidate(lookup_response) if lookup_response else None
        entities.append(
            LinkedEntity(
                mention=mention,
                type=entity_type,
                wikidata_qid=candidate.qid if candidate else None,
                wikidata_label=candidate.label if candidate else None,
                wikidata_description=candidate.description if candidate else None,
                matched_alias=candidate.matched_alias if candidate else None,
                match_type=candidate.match_type if candidate else None,
                score=candidate.score if candidate else None,
                candidate_count=len(lookup_response.candidates) if lookup_response else 0,
            )
        )
    return LinkedNerResponse(entities=entities)


def build_linking_instructions() -> str:
    return (
        "You are an entity linking system.\n\n"
        "Task:\n"
        "Choose the single best Wikidata candidate for each provided entity mention.\n\n"
        "Rules:\n"
        "- You must choose only from the provided candidates.\n"
        "- Use the full source text and the entity type for disambiguation.\n"
        "- If none of the candidates fit confidently for a mention, return null for that mention.\n"
        "- Return valid JSON only.\n\n"
        'Output format:\nReturn exactly one JSON object in the form {"decisions": {"mention": "<candidate qid>|null"}}.'
    )


def build_linking_input(
    source_text: str,
    response: NerResponse,
    lookup_results: dict[str, LookupResponse],
) -> str:
    return json.dumps(
        {
            "source_text": source_text,
            "entities": [
                {
                    "mention": mention,
                    "entity_type": entity_type,
                    "candidates": [candidate.model_dump() for candidate in lookup_results[mention].candidates],
                }
                for mention, entity_type in response.root.items()
                if mention in lookup_results
            ],
        },
        ensure_ascii=True,
    )


def get_linking_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "linking_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decisions": {
                    "type": "object",
                    "additionalProperties": {"type": ["string", "null"]},
                },
            },
            "required": ["decisions"],
        },
    }


def parse_linking_decision_map(
    response_data: dict[str, Any],
    allowed_qids_by_mention: dict[str, set[str]],
) -> LinkingDecisionMap:
    output_text = get_output_text(response_data)
    parsed_output = loads_json_output(output_text)
    decision_map = LinkingDecisionMap.model_validate(parsed_output)
    for mention, qid in decision_map.decisions.items():
        if mention not in allowed_qids_by_mention:
            raise InvalidLLMOutputError(
                f"Response selected unsupported entity mention: {mention}. "
                f"Allowed mentions: {', '.join(sorted(allowed_qids_by_mention))}"
            )
        allowed_qids = allowed_qids_by_mention[mention]
        if qid is not None and qid not in allowed_qids:
            raise InvalidLLMOutputError(
                f"Response selected unsupported Wikidata QID for {mention}: {qid}. "
                f"Allowed QIDs: {', '.join(sorted(allowed_qids))}"
            )
    return decision_map


async def select_llm_candidates(
    *,
    client: LLMClient,
    source_text: str,
    response: NerResponse,
    lookup_results: dict[str, LookupResponse],
) -> dict[str, LookupCandidate | None]:
    allowed_qids_by_mention = {
        mention: {candidate.qid for candidate in lookup_response.candidates}
        for mention, lookup_response in lookup_results.items()
    }
    decision_map = await create_and_parse_response(
        client=client,
        task_name="entity linking",
        input_text=build_linking_input(source_text, response, lookup_results),
        instructions=build_linking_instructions(),
        response_format=get_linking_response_format(),
        parse_response=lambda response_data: parse_linking_decision_map(response_data, allowed_qids_by_mention),
    )
    selected_candidates: dict[str, LookupCandidate | None] = {}
    for mention, lookup_response in lookup_results.items():
        qid = decision_map.decisions.get(mention)
        if qid is None:
            selected_candidates[mention] = None
            continue
        selected_candidates[mention] = next(
            (candidate for candidate in lookup_response.candidates if candidate.qid == qid),
            None,
        )
    return selected_candidates


async def build_llm_linked_response(
    response: NerResponse,
    request: NerRequest,
    lookup_results: dict[str, LookupResponse],
    client: LLMClient,
) -> LinkedNerResponse:
    try:
        selected_candidates = await select_llm_candidates(
            client=client,
            source_text=request.text,
            response=response,
            lookup_results=lookup_results,
        )
    except Exception as exc:
        logger.warning("Entity linking batch failed: %s", exc)
        selected_candidates = {mention: None for mention in response.root}

    entities: list[LinkedEntity] = []
    for mention, entity_type in response.root.items():
        lookup_response = lookup_results.get(mention)
        candidate = selected_candidates.get(mention)
        entities.append(
            LinkedEntity(
                mention=mention,
                type=entity_type,
                wikidata_qid=candidate.qid if candidate else None,
                wikidata_label=candidate.label if candidate else None,
                wikidata_description=candidate.description if candidate else None,
                matched_alias=candidate.matched_alias if candidate else None,
                match_type=candidate.match_type if candidate else None,
                score=candidate.score if candidate else None,
                candidate_count=len(lookup_response.candidates) if lookup_response else 0,
            )
        )
    return LinkedNerResponse(entities=entities)
