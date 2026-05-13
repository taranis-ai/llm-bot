from llm_bot.client import LLMClient
from llm_bot.lookup_client import LookupClient
from llm_bot.schemas import LinkRequest, LinkedNerResponse, NerLinkRequest, NerRequest
from llm_bot.tasks.link_task import link_entities
from llm_bot.tasks.ner import extract_entities


async def extract_and_link(
    request: NerLinkRequest,
    client: LLMClient | None = None,
    lookup_client: LookupClient | None = None,
) -> LinkedNerResponse:
    llm_client = client or LLMClient()
    ner_response = await extract_entities(
        NerRequest(
            text=request.text,
            cybersecurity=request.cybersecurity,
            entity_types=request.entity_types,
        ),
        client=llm_client,
    )
    link_request = LinkRequest(
        text=request.text,
        language=request.language,
        linking_mode=request.linking_mode,
        entities=[
            {"mention": mention, "type": entity_type}
            for mention, entity_type in ner_response.root.items()
        ],
    )
    return await link_entities(link_request, client=llm_client, lookup_client=lookup_client)
