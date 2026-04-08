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


class StoryTag(BaseModel):
    model_config = ConfigDict(extra="allow")

    tag_type: str


class StoryNewsItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str
    content: str
    review: str | None = None
    language: str


class StoryClusterItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    tags: dict[str, StoryTag]
    news_items: list[StoryNewsItem] = Field(min_length=1)


class ClusterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stories: list[StoryClusterItem] = Field(min_length=1)


class ClusterIds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_clusters: list[list[str]] = Field(min_length=1)


class ClusterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_ids: ClusterIds
    message: str = Field(min_length=1)
