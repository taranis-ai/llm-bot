from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, RootModel, field_validator, model_validator


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
    MIXED = "mixed"


PLUTCHIK_8: tuple[EmotionLabel, ...] = tuple(EmotionLabel)

ALLOWED_SENTIMENTS_BY_EMOTION: dict[EmotionLabel, set[SentimentLabel]] = {
    EmotionLabel.JOY: {SentimentLabel.POSITIVE, SentimentLabel.MIXED},
    EmotionLabel.TRUST: {SentimentLabel.POSITIVE, SentimentLabel.MIXED},
    EmotionLabel.FEAR: {SentimentLabel.NEGATIVE, SentimentLabel.MIXED},
    EmotionLabel.SURPRISE: {
        SentimentLabel.POSITIVE,
        SentimentLabel.NEUTRAL,
        SentimentLabel.NEGATIVE,
        SentimentLabel.MIXED,
    },
    EmotionLabel.SADNESS: {SentimentLabel.NEGATIVE, SentimentLabel.MIXED},
    EmotionLabel.DISGUST: {SentimentLabel.NEGATIVE, SentimentLabel.MIXED},
    EmotionLabel.ANGER: {SentimentLabel.NEGATIVE, SentimentLabel.MIXED},
    EmotionLabel.ANTICIPATION: {
        SentimentLabel.POSITIVE,
        SentimentLabel.NEUTRAL,
        SentimentLabel.NEGATIVE,
        SentimentLabel.MIXED,
    },
}


class StoryInputNewsItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = ""
    content: str = ""
    language: str | None = None


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning_effort: str | None = Field(default=None, min_length=1)
    thinking_budget_tokens: int | None = Field(default=None, ge=0)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(LLMRequest):
    message: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    model: str | None


class HragDocumentPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)


class HragGraphFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    fact: str = Field(min_length=1)


class HragRequest(LLMRequest):
    question: str = Field(min_length=1)
    passages: list[HragDocumentPassage]
    graph_facts: list[HragGraphFact]
    conversation: list[ChatMessage] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> Self:
        evidence_ids = [item.id for item in self.passages]
        evidence_ids.extend(item.id for item in self.graph_facts)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("Evidence IDs must be unique across passages and graph_facts")
        return self


class HragResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    citations: list[str]
    insufficient_evidence: bool

    @field_validator("citations")
    @classmethod
    def validate_unique_citations(cls, citations: list[str]) -> list[str]:
        if len(citations) != len(set(citations)):
            raise ValueError("Citations must not contain duplicates")
        if any(not citation for citation in citations):
            raise ValueError("Citations must not contain empty IDs")
        return citations


EmbeddingVector = Annotated[list[FiniteFloat], Field(min_length=1)]


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | list[str] = Field(min_length=1)
    dimensions: int | None = Field(default=None, gt=0)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str | list[str]) -> str | list[str]:
        texts = [value] if isinstance(value, str) else value
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input must not be blank")
        return value


class EmbedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding: EmbeddingVector


class EmbedBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embeddings: list[EmbeddingVector] = Field(min_length=1)


class StoryRequest(LLMRequest):
    text: str | None = Field(default=None, min_length=1)
    news_items: list[StoryInputNewsItem] | None = None
    language: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_story_input(self) -> Self:
        if self.text or any(item.title or item.content for item in self.news_items or []):
            return self
        raise ValueError("Either text or news_items with at least one non-empty item must be provided")


class SummarizeRequest(StoryRequest):
    max_words: int | None = Field(default=None, ge=1)


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)


class TitleRequest(StoryRequest):
    max_chars: int = Field(default=100, ge=1)


class TitleResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)


class TranslateRequest(LLMRequest):
    text: str = Field(min_length=1)
    target_language: str = Field(min_length=1)
    source_language: str | None = None


class TranslateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    translation: str = Field(min_length=1)


class SentimentRequest(LLMRequest):
    text: str = Field(min_length=1)
    include_emotions: bool = False
    include_mixed: bool = False


class CybersecClassificationRequest(LLMRequest):
    text: str = Field(min_length=1)


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, pattern=r"\S")


class ClassifyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    disinformation: float = Field(ge=0, le=1, strict=True)
    security_incident: float = Field(alias="security incident", ge=0, le=1, strict=True)
    vulnerability: float = Field(ge=0, le=1, strict=True)
    threat_intelligence: float = Field(alias="threat intelligence", ge=0, le=1, strict=True)
    security_policy: float = Field(alias="security policy", ge=0, le=1, strict=True)
    security_news: float = Field(alias="security news", ge=0, le=1, strict=True)
    general_news: float = Field(alias="general news", ge=0, le=1, strict=True)
    other: float = Field(ge=0, le=1, strict=True)


class CybersecClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    cybersecurity: float = Field(ge=0, le=1)
    non_cybersecurity: float = Field(alias="non-cybersecurity", ge=0, le=1)


class SentimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: SentimentLabel
    score: float = Field(ge=0, le=1)
    emotions: list[EmotionLabel] | None = None

    @model_validator(mode="after")
    def validate_emotions(self) -> Self:
        if self.emotions is None:
            return self

        if len(self.emotions) != len(set(self.emotions)):
            raise ValueError("Emotions must not contain duplicates")

        invalid_emotions = [emotion for emotion in self.emotions if self.label not in ALLOWED_SENTIMENTS_BY_EMOTION[emotion]]
        if invalid_emotions:
            invalid_names = ", ".join(emotion.value for emotion in invalid_emotions)
            raise ValueError(f"Emotions not allowed for sentiment {self.label.value}: {invalid_names}")

        return self


class SentimentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: SentimentResult

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(*args, **kwargs)


class NerRequest(LLMRequest):
    text: str = Field(min_length=1)
    cybersecurity: bool = False
    entity_types: list[str] | None = None


class NerLinkRequest(NerRequest):
    language: str | None = None
    linking_mode: str | None = None


class NerResponse(RootModel[dict[str, str]]):
    root: dict[str, str]


class ExtractionEntityType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ExtractionRelationType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    source_types: list[str] = Field(min_length=1)
    target_types: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_type_lists(self) -> Self:
        if len(self.source_types) != len(set(self.source_types)):
            raise ValueError("Relation source_types must not contain duplicates")
        if len(self.target_types) != len(set(self.target_types)):
            raise ValueError("Relation target_types must not contain duplicates")
        if any(not entity_type for entity_type in self.source_types + self.target_types):
            raise ValueError("Relation source_types and target_types must not contain empty names")
        return self


class EntityRelationshipSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_types: list[ExtractionEntityType] = Field(min_length=1)
    relation_types: list[ExtractionRelationType]

    @model_validator(mode="after")
    def validate_schema_references(self) -> Self:
        entity_type_names = [entity_type.name for entity_type in self.entity_types]
        if len(entity_type_names) != len(set(entity_type_names)):
            raise ValueError("Entity type names must be unique")

        relation_type_names = [relation_type.name for relation_type in self.relation_types]
        if len(relation_type_names) != len(set(relation_type_names)):
            raise ValueError("Relation type names must be unique")

        known_entity_types = set(entity_type_names)
        referenced_entity_types = {
            entity_type for relation_type in self.relation_types for entity_type in relation_type.source_types + relation_type.target_types
        }
        unknown_entity_types = sorted(referenced_entity_types - known_entity_types)
        if unknown_entity_types:
            raise ValueError("Relation constraints reference unknown entity types: " + ", ".join(unknown_entity_types))
        return self


class EntityRelationshipExtractionRequest(LLMRequest):
    model_config = ConfigDict(serialize_by_alias=True)

    text: str = Field(min_length=1)
    extraction_schema: EntityRelationshipSchema = Field(alias="schema")
    reasoning_effort: str | None = Field(default="low", min_length=1)


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    name: str = Field(min_length=1)


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1, strict=True)


class EntityRelationshipExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


class GraphQueryGenerationRequest(LLMRequest):
    model_config = ConfigDict(serialize_by_alias=True)

    question: str = Field(min_length=1)
    graph_schema: EntityRelationshipSchema = Field(alias="schema")
    context: dict[str, Any] = Field(default_factory=dict)


class GraphQueryGenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(default_factory=list, max_length=12)
    entity_types: list[str] = Field(default_factory=list, max_length=20)
    relationships: list[str] = Field(default_factory=list, max_length=18)
    hops: int = Field(default=1, ge=0, le=3)

    @field_validator("names")
    @classmethod
    def normalize_names(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 200 for value in values):
            raise ValueError("Invalid entity search name")
        return [value.strip().casefold() for value in values]


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


class LinkRequest(LLMRequest):
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


class StoryClusterItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    tags: dict[str, StoryTag]
    summary: str | None = None


class ClusterRequest(LLMRequest):
    stories: list[StoryClusterItem] = Field(min_length=1)


class ClusterIds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_clusters: list[list[str]] = Field(min_length=1)


class ClusterReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    story_ids: list[str] = Field(min_length=2)
    reason: str = Field(min_length=1)


class LLMClusterIds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_clusters: list[list[int]] = Field(min_length=1)


class LLMClusterReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    story_ids: list[int] = Field(min_length=2)
    reason: str = Field(min_length=1)


class LLMClusterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_ids: LLMClusterIds
    cluster_reasons: list[LLMClusterReason]
    message: str = Field(min_length=1)


class ClusterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_ids: ClusterIds
    message: str = Field(min_length=1)
