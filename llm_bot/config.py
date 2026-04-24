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
    LLM_REASONING_PROFILE: str = "none"
    LLM_REASONING_EFFORT: str = ""
    LLM_STRIP_REASONING_OUTPUT: bool = True
    LLM_PARSE_REASONING_AS_OUTPUT: bool = False

    SUMMARY_MAX_INPUT_CHARS: int = 50000
    SUMMARY_MAX_OUTPUT_CHARS: int = 1000
    SUMMARY_ROUTE_PATH: str = "/summarize"
    NER_ROUTE_PATH: str = "/ner"
    NER_ENTITY_TYPES: str = (
        "PER,ORG,GPE,PRODUCT,EVENT,GROUP,MALWARE,TOOL,TACTIC,TECHNIQUE,SECTOR,INDICATOR"
    )
    CLUSTER_ROUTE_PATH: str = "/cluster"
    CLUSTER_MAX_CONTENT_CHARS_PER_STORY: int = 800
    CLUSTER_MAX_TAGS_PER_STORY: int = 10

    @property
    def ner_entity_types(self) -> list[str]:
        return [item.strip() for item in self.NER_ENTITY_TYPES.split(",") if item.strip()]


Config = Settings()
