from functools import wraps

from pydantic import ValidationError
from quart import Quart, request

from llm_bot.config import Config
from llm_bot.schemas import SummarizeRequest
from llm_bot.tasks.summarize import summarize


def api_key_required(view_func):
    @wraps(view_func)
    async def wrapped(*args, **kwargs):
        if not Config.API_KEY:
            return await view_func(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        expected_header = f"Bearer {Config.API_KEY}"
        if auth_header != expected_header:
            return {"error": "Unauthorized"}, 401

        return await view_func(*args, **kwargs)

    return wrapped


def create_app() -> Quart:
    app = Quart(__name__)
    app.config.from_mapping(
        DEBUG=Config.DEBUG,
        PACKAGE_NAME=Config.PACKAGE_NAME,
    )

    @app.get("/health")
    async def health() -> tuple[dict[str, str], int]:
        return {"status": "ok"}, 200

    @app.get("/info")
    async def info() -> tuple[dict[str, str | int], int]:
        return {
            "package_name": Config.PACKAGE_NAME,
            "llm_base_url": Config.LLM_BASE_URL,
            "llm_model": Config.LLM_MODEL,
            "llm_timeout": Config.LLM_TIMEOUT,
            "summary_route_path": Config.SUMMARY_ROUTE_PATH,
        }, 200

    @app.post(Config.SUMMARY_ROUTE_PATH)
    @api_key_required
    async def summarize_view() -> tuple[dict[str, str], int]:
        try:
            payload = await request.get_json()
            request_model = SummarizeRequest.model_validate(payload)
        except ValidationError:
            return {"error": "Invalid summarize request payload"}, 400

        try:
            response_model = await summarize(request_model)
        except Exception:
            return {"error": "Failed to generate summary"}, 502

        return response_model.model_dump(), 200

    return app


app = create_app()
