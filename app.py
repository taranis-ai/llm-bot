from quart import Quart

from llm_bot.config import Config


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

    return app


app = create_app()
