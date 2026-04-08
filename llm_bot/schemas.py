from pydantic import BaseModel, ConfigDict, Field, RootModel


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    max_words: int | None = Field(default=None, ge=1)


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)


class NerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    cybersecurity: bool = False


class NerResponse(RootModel[dict[str, str]]):
    root: dict[str, str]
