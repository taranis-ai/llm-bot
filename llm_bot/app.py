from quart import Quart

from llm_bot.config import Config
from llm_bot.local_inference import runtime
from llm_bot.log import configure_logging
from llm_bot.routes import create_api_blueprint


def create_app() -> Quart:
    app = Quart(__name__)
    app.url_map.strict_slashes = False
    app.config.from_mapping(
        DEBUG=Config.DEBUG,
        PACKAGE_NAME=Config.PACKAGE_NAME,
    )
    configure_logging(debug=Config.DEBUG)
    app.register_blueprint(create_api_blueprint())

    @app.before_serving
    async def prepare_local_models():
        if Config.LAYA_PRELOAD:
            await runtime.preload()

    return app
