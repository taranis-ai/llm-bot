from llm_bot.client import LLMClient
from llm_bot.lookup_client import LookupClient
from llm_bot.schemas import LinkRequest, LinkedNerResponse, NerRequest, NerResponse
from llm_bot.tasks.linking import (
    build_deterministic_linked_response,
    build_llm_linked_response,
    lookup_entity_candidates,
    resolve_linking_mode,
)


def build_linking_ner_response(request: LinkRequest) -> NerResponse:
    return NerResponse({entity.mention: entity.type for entity in request.entities})


async def link_entities(
    request: LinkRequest,
    client: LLMClient | None = None,
    lookup_client: LookupClient | None = None,
) -> LinkedNerResponse:
    llm_client = client or LLMClient()
    linking_mode = resolve_linking_mode(NerRequest(text=request.text, language=request.language, linking_mode=request.linking_mode))
    ner_response = build_linking_ner_response(request)
    ner_request = NerRequest(
        text=request.text,
        language=request.language,
        linking_mode=request.linking_mode,
    )
    lookup_results = await lookup_entity_candidates(ner_response, ner_request, client=lookup_client)
    if linking_mode == "deterministic":
        return build_deterministic_linked_response(ner_response, lookup_results)
    return await build_llm_linked_response(ner_response, ner_request, lookup_results, client=llm_client)
