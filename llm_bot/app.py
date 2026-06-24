from quart import Quart
from quart_schema import HttpSecurityScheme, Info, License, QuartSchema, Tag

from llm_bot import __version__
from llm_bot.config import Config
from llm_bot.log import configure_logging
from llm_bot.openapi import LlmBotOpenAPIProvider
from llm_bot.routes import create_api_blueprint


def create_app() -> Quart:
    app = Quart(__name__)
    app.url_map.strict_slashes = False
    app.config.from_mapping(
        DEBUG=Config.DEBUG,
        PACKAGE_NAME=Config.PACKAGE_NAME,
    )
    configure_logging(debug=Config.DEBUG)
    QuartSchema(
        app,
        openapi_path="/openapi.json",
        redoc_ui_path=None,
        scalar_ui_path=None,
        swagger_ui_path="/docs",
        info=Info(
            title="LLM Bot for Taranis AI",
            version=__version__,
            description="Multi-route LLM-backed bot service for Taranis AI.",
            license=License(name="EUPL-1.2", url="https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12"),
        ),
        tags=[
            Tag(name="Sentiment", description="Sentiment analysis operations"),
            Tag(name="Title", description="Title generation operations"),
            Tag(name="Translation", description="Translation operations"),
            Tag(name="Summarization", description="Summarization operations"),
            Tag(name="NER", description="Named entity recognition operations"),
            Tag(name="Linking", description="Entity linking operations"),
            Tag(name="Clustering", description="Story clustering operations"),
            Tag(name="System", description="Health and info endpoints"),
        ],
        security_schemes={
            "bearerAuth": HttpSecurityScheme(scheme="bearer", bearer_format="API key"),
        },
        openapi_provider_class=LlmBotOpenAPIProvider,
    )
    app.register_blueprint(create_api_blueprint())
    return app
