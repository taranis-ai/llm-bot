from quart import Quart

from llm_bot.config import Config


def create_app() -> Quart:
    app = Quart(__name__)
    app.config.from_mapping(
        DEBUG=Config.DEBUG,
        PACKAGE_NAME=Config.PACKAGE_NAME,
    )
    return app


app = create_app()
