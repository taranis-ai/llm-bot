from quart import Quart, request

from llm_bot.config import Config
from llm_bot.schemas import SummarizeRequest
from llm_bot.tasks.summarize import summarize


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
    async def summarize_view() -> tuple[dict[str, str], int]:
        payload = await request.get_json()
        request_model = SummarizeRequest.model_validate(payload)
        response_model = await summarize(request_model)
        return response_model.model_dump(), 200

    return app


app = create_app()
