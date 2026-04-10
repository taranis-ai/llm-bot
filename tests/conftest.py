import pytest

from llm_bot.app import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config.update(TESTING=True)
    return app
