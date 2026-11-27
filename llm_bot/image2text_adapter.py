from llm_bot.config import Config
from llm_bot.schemas import Image2TextRequest, Image2TextResponse


def describe(data: bytes, mime_type: str) -> str:
    """Taranis worker entry point for the llm-bot image caption service."""
    from niquests import post

    if not Config.IMAGE2TEXT_URL:
        raise ValueError("Configure IMAGE2TEXT_URL in the Taranis worker")
    image = Image2TextRequest(data=data, mime_type=mime_type)
    headers = {"Content-Type": image.mime_type.value}
    if Config.IMAGE2TEXT_API_KEY:
        headers["Authorization"] = f"Bearer {Config.IMAGE2TEXT_API_KEY}"
    response = post(Config.IMAGE2TEXT_URL, data=image.data, headers=headers, timeout=Config.IMAGE2TEXT_TIMEOUT, allow_redirects=False)
    response.raise_for_status()
    if response.status_code != 200:
        raise ValueError(f"Unexpected Image2Text service status {response.status_code}")
    return Image2TextResponse.model_validate_json(response.content).description
