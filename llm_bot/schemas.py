from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class EmotionLabel(StrEnum):
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


PLUTCHIK_8: tuple[EmotionLabel, ...] = tuple(EmotionLabel)

_ALLOWED_SENTIMENTS_BY_EMOTION: dict[EmotionLabel, set[SentimentLabel]] = {
    EmotionLabel.JOY: {SentimentLabel.POSITIVE},
    EmotionLabel.TRUST: {SentimentLabel.POSITIVE},
    EmotionLabel.FEAR: {SentimentLabel.NEGATIVE},
    EmotionLabel.SURPRISE: {
        SentimentLabel.POSITIVE,
        SentimentLabel.NEUTRAL,
        SentimentLabel.NEGATIVE,
    },
    EmotionLabel.SADNESS: {SentimentLabel.NEGATIVE},
    EmotionLabel.DISGUST: {SentimentLabel.NEGATIVE},
    EmotionLabel.ANGER: {SentimentLabel.NEGATIVE},
    EmotionLabel.ANTICIPATION: {
        SentimentLabel.POSITIVE,
        SentimentLabel.NEUTRAL,
        SentimentLabel.NEGATIVE,
    },
}


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    max_words: int | None = Field(default=None, ge=1)


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)


class SentimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    include_emotions: bool = False


class SentimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: SentimentLabel
    score: float = Field(ge=0, le=1)
    emotions: list[EmotionLabel] | None = None

    @model_validator(mode="after")
    def validate_emotions(self) -> "SentimentResult":
        if self.emotions is None:
            return self

        if len(self.emotions) != len(set(self.emotions)):
            raise ValueError("Emotions must not contain duplicates")

        invalid_emotions = [
            emotion
            for emotion in self.emotions
            if self.label not in _ALLOWED_SENTIMENTS_BY_EMOTION[emotion]
        ]
        if invalid_emotions:
            invalid_names = ", ".join(emotion.value for emotion in invalid_emotions)
            raise ValueError(
                f"Emotions not allowed for sentiment {self.label.value}: {invalid_names}"
            )

        return self


class SentimentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: SentimentResult

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)


class NerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    cybersecurity: bool = False
    entity_types: list[str] | None = None


class NerLinkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    cybersecurity: bool = False
    entity_types: list[str] | None = None
    language: str | None = None
    linking_mode: str | None = None


class NerResponse(RootModel[dict[str, str]]):
    root: dict[str, str]


class LinkedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention: str = Field(min_length=1)
    type: str = Field(min_length=1)
    wikidata_qid: str | None = None
    wikidata_label: str | None = None
    wikidata_description: str | None = None
    matched_alias: str | None = None
    match_type: str | None = None
    score: float | None = None
    candidate_count: int | None = Field(default=None, ge=0)


class LinkedNerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[LinkedEntity]


class LinkRequestEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention: str = Field(min_length=1)
    type: str = Field(min_length=1)


class LinkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    entities: list[LinkRequestEntity] = Field(min_length=1)
    language: str | None = None
    linking_mode: str | None = None


class LookupCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qid: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str | None = None
    matched_alias: str | None = None
    match_type: str | None = None
    language: str = Field(min_length=1)
    score: float
    is_label: bool
    type_tags: list[str]


class LookupResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    language: str = Field(min_length=1)
    limit: int = Field(ge=1, le=100)
    candidates: list[LookupCandidate]


class StoryTag(BaseModel):
    model_config = ConfigDict(extra="allow")

    tag_type: str


class StoryNewsItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str
    content: str
    review: str | None = None
    language: str | None = None


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
