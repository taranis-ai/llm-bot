from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DEBUG: bool = False
    API_KEY: str = ""

    PACKAGE_NAME: str = "llm_bot"

    LLM_BASE_URL: str = ""
    LLM_API_KEY: str = ""
    LLM_MODEL: str = ""
    LLM_TIMEOUT: int = 120

    SUMMARY_MAX_INPUT_CHARS: int = 50000
    SUMMARY_ROUTE_PATH: str = "/summarize"


Config = Settings()
