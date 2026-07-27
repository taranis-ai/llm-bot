import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator


GRAPH_IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"


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

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> "HragRequest":
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


class EmbedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


class EmbedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding: list[float] = Field(min_length=1)


class SummarizeRequest(LLMRequest):

    text: str | None = Field(default=None, min_length=1)
    news_items: list[StoryInputNewsItem] | None = None
    language: str | None = Field(default=None, min_length=1)
    max_words: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_story_input(self) -> "SummarizeRequest":
        if self.text:
            return self
        if self.news_items:
            if any(item.title or item.content for item in self.news_items):
                return self
        raise ValueError("Either text or news_items with at least one non-empty item must be provided")


class SummarizeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)


class TitleRequest(LLMRequest):

    text: str | None = Field(default=None, min_length=1)
    news_items: list[StoryInputNewsItem] | None = None
    language: str | None = Field(default=None, min_length=1)
    max_chars: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def validate_story_input(self) -> "TitleRequest":
        if self.text:
            return self
        if self.news_items:
            if any(item.title or item.content for item in self.news_items):
                return self
        raise ValueError("Either text or news_items with at least one non-empty item must be provided")


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


class CybersecClassificationRequest(LLMRequest):

    text: str = Field(min_length=1)


class CybersecClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cybersecurity: float = Field(ge=0, le=1)
    non_cybersecurity: float = Field(alias="non-cybersecurity", ge=0, le=1)

    def model_dump(self, *args, **kwargs):
        kwargs.setdefault("by_alias", True)
        return super().model_dump(*args, **kwargs)


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


class NerRequest(LLMRequest):

    text: str = Field(min_length=1)
    cybersecurity: bool = False
    entity_types: list[str] | None = None


class NerLinkRequest(LLMRequest):

    text: str = Field(min_length=1)
    cybersecurity: bool = False
    entity_types: list[str] | None = None
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
    def validate_type_lists(self) -> "ExtractionRelationType":
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
    def validate_schema_references(self) -> "EntityRelationshipSchema":
        entity_type_names = [entity_type.name for entity_type in self.entity_types]
        if len(entity_type_names) != len(set(entity_type_names)):
            raise ValueError("Entity type names must be unique")

        relation_type_names = [relation_type.name for relation_type in self.relation_types]
        if len(relation_type_names) != len(set(relation_type_names)):
            raise ValueError("Relation type names must be unique")

        known_entity_types = set(entity_type_names)
        referenced_entity_types = {
            entity_type
            for relation_type in self.relation_types
            for entity_type in relation_type.source_types + relation_type.target_types
        }
        unknown_entity_types = sorted(referenced_entity_types - known_entity_types)
        if unknown_entity_types:
            raise ValueError(
                "Relation constraints reference unknown entity types: "
                + ", ".join(unknown_entity_types)
            )
        return self


class EntityRelationshipExtractionRequest(LLMRequest):
    text: str = Field(min_length=1)
    schema: EntityRelationshipSchema


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


class GraphQueryableProperty(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=GRAPH_IDENTIFIER_PATTERN)
    type: str | None = Field(default=None, min_length=1)
    description: str | None = Field(default=None, min_length=1)


class GraphNodeLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, pattern=GRAPH_IDENTIFIER_PATTERN)
    properties: list[GraphQueryableProperty]

    @model_validator(mode="after")
    def validate_property_names(self) -> "GraphNodeLabel":
        property_names = [prop.name for prop in self.properties]
        if len(property_names) != len(set(property_names)):
            raise ValueError(f"Property names for node label {self.label} must be unique")
        return self


class GraphRelationshipType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1, pattern=GRAPH_IDENTIFIER_PATTERN)
    source_labels: list[str] = Field(min_length=1)
    target_labels: list[str] = Field(min_length=1)
    properties: list[GraphQueryableProperty] = Field(default_factory=list)

    @field_validator("source_labels", "target_labels")
    @classmethod
    def validate_label_names(cls, labels: list[str]) -> list[str]:
        if len(labels) != len(set(labels)):
            raise ValueError("Relationship source_labels and target_labels must not contain duplicates")
        for label in labels:
            if not label or not re.fullmatch(GRAPH_IDENTIFIER_PATTERN, label):
                raise ValueError("Relationship source_labels and target_labels must contain valid identifiers")
        return labels

    @model_validator(mode="after")
    def validate_property_names(self) -> "GraphRelationshipType":
        property_names = [prop.name for prop in self.properties]
        if len(property_names) != len(set(property_names)):
            raise ValueError(f"Property names for relationship type {self.type} must be unique")
        return self


class GraphQuerySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_labels: list[GraphNodeLabel] = Field(min_length=1)
    relationship_types: list[GraphRelationshipType]
    default_limit: int = Field(ge=1)
    maximum_limit: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_schema_references(self) -> "GraphQuerySchema":
        node_labels = [node.label for node in self.node_labels]
        if len(node_labels) != len(set(node_labels)):
            raise ValueError("Graph node labels must be unique")

        relationship_types = [relationship.type for relationship in self.relationship_types]
        if len(relationship_types) != len(set(relationship_types)):
            raise ValueError("Graph relationship types must be unique")

        known_labels = set(node_labels)
        referenced_labels = {
            label for relationship in self.relationship_types for label in relationship.source_labels + relationship.target_labels
        }
        unknown_labels = sorted(referenced_labels - known_labels)
        if unknown_labels:
            raise ValueError("Relationship constraints reference unknown node labels: " + ", ".join(unknown_labels))
        if self.default_limit > self.maximum_limit:
            raise ValueError("Graph query default limit must not exceed maximum")
        return self


class GraphQueryGenerationRequest(LLMRequest):
    question: str = Field(min_length=1)
    graph_name: str = Field(min_length=1)
    schema: GraphQuerySchema


class GraphQueryGenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cypher: str = Field(min_length=1)
    parameters: dict[str, Any]
    explanation: str = Field(min_length=1, max_length=500)


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
