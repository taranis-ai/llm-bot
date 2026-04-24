from llm_bot.config import Config
from llm_bot.lookup_client import LookupClient
from llm_bot.schemas import LookupResponse, NerRequest, NerResponse


ALLOWED_LINKING_MODES = {"llm", "deterministic"}


class UnsupportedLinkingModeError(ValueError):
    pass


def is_linking_enabled(request: NerRequest) -> bool:
    if request.link_entities is not None:
        return request.link_entities
    return Config.NER_LINKING_ENABLED


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
