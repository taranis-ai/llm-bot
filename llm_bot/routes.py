import json
from functools import wraps
from typing import Awaitable, Callable

from pydantic import ValidationError
from quart import Blueprint, request

from llm_bot.config import Config
from llm_bot.log import logger
from llm_bot.schemas import NerRequest, SummarizeRequest
from llm_bot.tasks.ner import extract_entities
from llm_bot.tasks.summarize import summarize


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


async def _handle_model_request(
    *,
    log_prefix: str,
    validation_error_message: str,
    processing_error_message: str,
    request_model_factory: Callable[[object], object],
    task: Callable[[object], Awaitable[object]],
) -> tuple[dict[str, str], int]:
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
    except Exception:
        logger.exception(processing_error_message)
        return {"error": processing_error_message}, 502

    logger.info("%s successfully", log_prefix)
    return response_model.model_dump(), 200


def create_api_blueprint() -> Blueprint:
    api = Blueprint("api", __name__)

    @api.get("/health")
    async def health() -> tuple[dict[str, str], int]:
        return {"status": "ok"}, 200

    @api.get("/info")
    async def info() -> tuple[dict[str, str | int], int]:
        return {
            "package_name": Config.PACKAGE_NAME,
            "llm_base_url": Config.LLM_BASE_URL,
            "llm_model": Config.LLM_MODEL,
            "llm_timeout": Config.LLM_TIMEOUT,
            "summary_route_path": Config.SUMMARY_ROUTE_PATH,
            "ner_route_path": Config.NER_ROUTE_PATH,
        }, 200

    @api.post(Config.SUMMARY_ROUTE_PATH)
    @api_key_required
    async def summarize_view() -> tuple[dict[str, str], int]:
        return await _handle_model_request(
            log_prefix="Summarize",
            validation_error_message="Invalid summarize request payload",
            processing_error_message="Failed to generate summary",
            request_model_factory=SummarizeRequest.model_validate,
            task=summarize,
        )

    @api.post(Config.NER_ROUTE_PATH)
    @api_key_required
    async def ner_view() -> tuple[dict[str, str], int]:
        return await _handle_model_request(
            log_prefix="NER",
            validation_error_message="Invalid NER request payload",
            processing_error_message="Failed to extract entities",
            request_model_factory=NerRequest.model_validate,
            task=extract_entities,
        )

    return api
