import json
from collections.abc import Awaitable, Callable
from functools import wraps
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from quart import Blueprint, Response, request

from llm_bot import __version__
from llm_bot.client import UpstreamLLMError
from llm_bot.config import Config
from llm_bot.embedding_client import UpstreamEmbeddingError
from llm_bot.log import logger
from llm_bot.schemas import (
    ChatRequest,
    ClusterRequest,
    CybersecClassificationRequest,
    EmbedRequest,
    EntityRelationshipExtractionRequest,
    GraphQueryGenerationRequest,
    HragRequest,
    LinkRequest,
    NerLinkRequest,
    NerRequest,
    SentimentRequest,
    SummarizeRequest,
    TitleRequest,
    TranslateRequest,
)
from llm_bot.tasks.chat import chat
from llm_bot.tasks.cluster import cluster_stories
from llm_bot.tasks.cybersec_classification import classify_cybersecurity_text
from llm_bot.tasks.embed import embed_text
from llm_bot.tasks.entity_linking import UnsupportedLinkingModeError
from llm_bot.tasks.entity_relationship_extraction import extract_entity_relationships
from llm_bot.tasks.graph_query_generation import generate_graph_query
from llm_bot.tasks.hrag import answer_with_hrag
from llm_bot.tasks.link_task import link_entities
from llm_bot.tasks.ner import UnsupportedEntityTypesError, extract_entities
from llm_bot.tasks.ner_link import extract_and_link
from llm_bot.tasks.sentiment import analyze_sentiment
from llm_bot.tasks.summarize import summarize
from llm_bot.tasks.title import generate_title
from llm_bot.tasks.translate import translate_text

OPENAPI_PATH = Path(__file__).resolve().with_name("openapi3_1.yml")

SWAGGER_UI_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>llm-bot API docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
    <style>
      html { box-sizing: border-box; overflow-y: scroll; }
      *, *:before, *:after { box-sizing: inherit; }
      body { margin: 0; background: #faf7f2; }
    </style>
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
      window.ui = SwaggerUIBundle({
        url: "/openapi.yaml",
        dom_id: "#swagger-ui",
      });
    </script>
  </body>
</html>
"""


def _load_openapi_spec_text() -> str:
    spec_text = OPENAPI_PATH.read_text(encoding="utf-8")
    return spec_text.replace("version: 0.0.0", f"version: {__version__}", 1)


def api_key_required(view_func):
    @wraps(view_func)
    async def wrapped(*args, **kwargs):
        if not Config.API_KEY:
            return await view_func(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        expected_header = f"Bearer {Config.API_KEY}"
        if auth_header != expected_header:
            logger.warning("Unauthorized request for %s", request.path)
            return {"error": "Unauthorized"}, 401

        return await view_func(*args, **kwargs)

    return wrapped


async def _handle_model_request[RequestModel: BaseModel](
    *,
    log_prefix: str,
    validation_error_message: str,
    processing_error_message: str,
    request_model_factory: Callable[[object], RequestModel],
    task: Callable[[RequestModel], Awaitable[BaseModel]],
    client_error_exceptions: tuple[type[Exception], ...] = (),
) -> tuple[dict[str, Any], int]:
    try:
        payload = await request.get_json()
        if Config.DEBUG:
            logger.debug("%s payload: %s", log_prefix, json.dumps(payload, ensure_ascii=True))
        request_model = request_model_factory(payload)
    except ValidationError:
        logger.warning(validation_error_message)
        return {"error": validation_error_message}, 400

    try:
        response_model = await task(request_model)
    except client_error_exceptions as exc:
        logger.warning("%s client error: %s", log_prefix, exc)
        return {"error": str(exc)}, 400
    except (UpstreamLLMError, UpstreamEmbeddingError) as exc:
        logger.error("%s upstream error: %s", log_prefix, exc)
        return {"error": f"{processing_error_message}: {exc}"}, 502
    # Keep unexpected task failures out of HTTP responses.
    except Exception:  # noqa: BLE001
        logger.exception(processing_error_message)
        return {"error": processing_error_message}, 502

    logger.info("%s successfully", log_prefix)
    return response_model.model_dump(), 200


def build_info_response() -> dict[str, object]:
    return {
        "package_name": Config.PACKAGE_NAME,
        "package_version": __version__,
        "reasoning_profiles": {
            "none": {"description": "No special reasoning prompt handling"},
            "ministral": {"description": "Adds [THINK]...[/THINK] reasoning instructions"},
            "gemma": {"description": "Prefixes the system prompt with <|think|>"},
        },
        "linking_modes": ["deterministic", "llm"],
        "endpoints": {
            "docs": "/docs",
            "openapi": "/openapi.yaml",
            "sentiment": "/sentiment",
            "cybersec_classification": "/cybersec-classification",
            "summarize": "/summarize",
            "title": "/title",
            "translate": "/translate",
            "ner": "/ner",
            "ner_link": "/ner-link",
            "link": "/link",
            "cluster": "/cluster",
            "entity_relation_extraction": "/entity-relation-extraction",
            "graph_query_generation": "/graph-query-generation",
            "embed": "/embed",
            "chat": "/chat",
            "hrag": "/hrag",
        },
        "current": {
            "llm_base_url": Config.LLM_BASE_URL,
            "llm_model": Config.LLM_MODEL,
            "llm_api_mode": Config.LLM_API_MODE,
            "llm_timeout": Config.LLM_TIMEOUT,
            "llm_reasoning_profile": Config.LLM_REASONING_PROFILE,
            "llm_strip_reasoning_output": Config.LLM_STRIP_REASONING_OUTPUT,
            "llm_parse_reasoning_as_output": Config.LLM_PARSE_REASONING_AS_OUTPUT,
            "embedding_base_url_configured": bool(Config.EMBEDDING_BASE_URL),
            "embedding_model": Config.EMBEDDING_MODEL,
            "embedding_timeout": Config.EMBEDDING_TIMEOUT,
            "lookup_base_url_configured": bool(Config.LOOKUP_BASE_URL),
            "lookup_default_language": Config.LOOKUP_DEFAULT_LANGUAGE,
            "lookup_candidate_limit": Config.LOOKUP_CANDIDATE_LIMIT,
            "ner_linking_enabled": Config.NER_LINKING_ENABLED,
            "ner_linking_mode": Config.NER_LINKING_MODE,
        },
    }


def create_api_blueprint() -> Blueprint:
    api = Blueprint("api", __name__)

    @api.get("/health")
    async def health() -> tuple[dict[str, Any], int]:
        return {"status": "ok"}, 200

    @api.get("/openapi.yaml")
    async def openapi_spec() -> Response:
        return Response(_load_openapi_spec_text(), mimetype="application/yaml")

    @api.get("/docs")
    async def swagger_docs() -> Response:
        return Response(SWAGGER_UI_HTML, mimetype="text/html")

    @api.get("/info")
    async def info() -> tuple[dict[str, object], int]:
        return build_info_response(), 200

    @api.post("/sentiment")
    @api_key_required
    async def sentiment_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Sentiment",
            validation_error_message="Invalid sentiment request payload",
            processing_error_message="Failed to analyze sentiment",
            request_model_factory=SentimentRequest.model_validate,
            task=analyze_sentiment,
        )

    @api.post("/chat")
    @api_key_required
    async def chat_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Chat",
            validation_error_message="Invalid chat request payload",
            processing_error_message="Failed to generate chat response",
            request_model_factory=ChatRequest.model_validate,
            task=chat,
        )

    @api.post("/hrag")
    @api_key_required
    async def hrag_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="HRAG",
            validation_error_message="Invalid HRAG request payload",
            processing_error_message="Failed to generate grounded answer",
            request_model_factory=HragRequest.model_validate,
            task=answer_with_hrag,
        )

    @api.post("/embed")
    @api_key_required
    async def embed_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Embedding",
            validation_error_message="Invalid embedding request payload",
            processing_error_message="Failed to create embedding",
            request_model_factory=EmbedRequest.model_validate,
            task=embed_text,
        )

    @api.post("/cybersec-classification")
    @api_key_required
    async def cybersec_classification_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Cybersec classification",
            validation_error_message="Invalid cybersec classification request payload",
            processing_error_message="Failed to classify text for cybersecurity relevance",
            request_model_factory=CybersecClassificationRequest.model_validate,
            task=classify_cybersecurity_text,
        )

    @api.post("/title")
    @api_key_required
    async def title_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Title",
            validation_error_message="Invalid title request payload",
            processing_error_message="Failed to generate title",
            request_model_factory=TitleRequest.model_validate,
            task=generate_title,
        )

    @api.post("/translate")
    @api_key_required
    async def translate_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Translate",
            validation_error_message="Invalid translate request payload",
            processing_error_message="Failed to translate text",
            request_model_factory=TranslateRequest.model_validate,
            task=translate_text,
        )

    @api.post("/summarize")
    @api_key_required
    async def summarize_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Summarize",
            validation_error_message="Invalid summarize request payload",
            processing_error_message="Failed to generate summary",
            request_model_factory=SummarizeRequest.model_validate,
            task=summarize,
        )

    @api.post("/ner")
    @api_key_required
    async def ner_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="NER",
            validation_error_message="Invalid NER request payload",
            processing_error_message="Failed to extract entities",
            request_model_factory=NerRequest.model_validate,
            task=extract_entities,
            client_error_exceptions=(UnsupportedEntityTypesError,),
        )

    @api.post("/ner-link")
    @api_key_required
    async def ner_link_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="NER link",
            validation_error_message="Invalid NER link request payload",
            processing_error_message="Failed to extract and link entities",
            request_model_factory=NerLinkRequest.model_validate,
            task=extract_and_link,
            client_error_exceptions=(UnsupportedEntityTypesError, UnsupportedLinkingModeError),
        )

    @api.post("/link")
    @api_key_required
    async def link_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Link",
            validation_error_message="Invalid link request payload",
            processing_error_message="Failed to link entities",
            request_model_factory=LinkRequest.model_validate,
            task=link_entities,
            client_error_exceptions=(UnsupportedLinkingModeError,),
        )

    @api.post("/cluster")
    @api_key_required
    async def cluster_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Cluster",
            validation_error_message="Invalid Cluster request payload",
            processing_error_message="Failed to cluster stories",
            request_model_factory=ClusterRequest.model_validate,
            task=cluster_stories,
        )

    @api.post("/entity-relation-extraction")
    @api_key_required
    async def entity_relationship_extraction_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Entity relationship extraction",
            validation_error_message="Invalid entity relationship extraction request payload",
            processing_error_message="Failed to extract entity relationships",
            request_model_factory=EntityRelationshipExtractionRequest.model_validate,
            task=extract_entity_relationships,
        )

    @api.post("/graph-query-generation")
    @api_key_required
    async def graph_query_generation_view() -> tuple[dict[str, Any], int]:
        return await _handle_model_request(
            log_prefix="Graph query generation",
            validation_error_message="Invalid graph query generation request payload",
            processing_error_message="Failed to generate graph query",
            request_model_factory=GraphQueryGenerationRequest.model_validate,
            task=generate_graph_query,
        )

    return api
