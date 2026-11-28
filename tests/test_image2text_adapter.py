from importlib.metadata import entry_points
from unittest.mock import Mock

import pytest
from niquests.exceptions import HTTPError

from llm_bot.config import Config
from tests.test_helpers import PNG_BYTES


def test_worker_discovers_and_calls_adapter(monkeypatch):
    adapters = entry_points(group="taranis.image2text", name="describe")
    assert len(adapters) == 1
    describe = adapters["describe"].load()
    png = PNG_BYTES
    monkeypatch.setattr(Config, "IMAGE2TEXT_URL", "")
    with pytest.raises(ValueError, match="IMAGE2TEXT_URL"):
        describe(png, "image/png")

    monkeypatch.setattr(Config, "IMAGE2TEXT_URL", "https://caption.example/image2text")
    monkeypatch.setattr(Config, "IMAGE2TEXT_API_KEY", "service-key")
    monkeypatch.setattr(Config, "IMAGE2TEXT_TIMEOUT", 42)
    response = Mock(status_code=200, content=b'{"description":" A map. "}')
    post = Mock(return_value=response)
    monkeypatch.setattr("niquests.post", post)
    assert describe(png, "image/png") == "A map."
    post.assert_called_once_with(
        "https://caption.example/image2text",
        data=png,
        headers={"Content-Type": "image/png", "Authorization": "Bearer service-key"},
        timeout=42,
        allow_redirects=False,
    )
    response.content = b'{"description":" "}'
    with pytest.raises(ValueError):
        describe(png, "image/png")
    response.raise_for_status.side_effect = HTTPError("service unavailable")
    with pytest.raises(HTTPError):
        describe(png, "image/png")
