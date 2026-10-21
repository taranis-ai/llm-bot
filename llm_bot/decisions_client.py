import json
from dataclasses import dataclass
from math import isclose
from typing import Annotated, Any, Literal

from niquests import AsyncSession
from niquests.exceptions import HTTPError
from pydantic import BaseModel, Field

from llm_bot.client import LLMClient, UpstreamLLMError
from llm_bot.config import Config

type Probability = Annotated[float, Field(ge=0, le=1, strict=True)]
type ChoiceProbabilities = dict[str, dict[str, float]]


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: dict[str, str]


class _ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]


class _SystemOneResponse(BaseModel):
    answers: dict[str, _ChoiceAnswer]


class _ChoiceProbability(BaseModel):
    value: str
    probability: Probability


class _OpenAIChoiceAnswer(BaseModel):
    type: Literal["choice"]
    name: str
    choice: str
    probabilities: list[_ChoiceProbability]


class _OpenAIResponse(BaseModel):
    answers: list[_OpenAIChoiceAnswer]


class DecisionsClient:
    """Transport for named choice questions, with independent provider settings."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        api_mode: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.url = Config.DECISIONS_URL if url is None else url
        self.api_key = Config.DECISIONS_API_KEY if api_key is None else api_key
        self.model = Config.DECISIONS_MODEL if model is None else model
        self.api_mode = Config.DECISIONS_API_MODE if api_mode is None else api_mode
        self.timeout = Config.DECISIONS_TIMEOUT if timeout is None else timeout
        if self.api_mode not in {"systemone", "openai"}:
            raise ValueError("DECISIONS_API_MODE must be systemone or openai")

    def build_request(self, state: str, questions: dict[str, ChoiceQuestion]) -> dict[str, Any]:
        payload: dict[str, Any]
        if self.api_mode == "openai":
            payload = {
                "input": state,
                "questions": [
                    {
                        "type": "choice",
                        "name": name,
                        "instructions": question.instructions,
                        "choices": [{"value": value, "description": description} for value, description in question.criteria.items()],
                    }
                    for name, question in questions.items()
                ],
            }
        else:
            payload = {
                "state": state,
                "questions": {
                    name: {"type": "choice", "instructions": question.instructions, "criteria": question.criteria}
                    for name, question in questions.items()
                },
            }
        if self.model:
            payload["model"] = self.model
        return payload

    def parse_response(self, response_data: dict[str, Any], questions: dict[str, ChoiceQuestion]) -> ChoiceProbabilities:
        try:
            if self.api_mode == "openai":
                response = _OpenAIResponse.model_validate(response_data)
                answers = {}
                for answer in response.answers:
                    probabilities = {item.value: item.probability for item in answer.probabilities}
                    if answer.name in answers or len(probabilities) != len(answer.probabilities):
                        raise ValueError("Duplicate question or choice")
                    answers[answer.name] = _ChoiceAnswer(type="choice", choice=answer.choice, probabilities=probabilities)
            else:
                answers = _SystemOneResponse.model_validate(response_data).answers
            if answers.keys() != questions.keys():
                raise ValueError("Unexpected or missing questions")
            for name, answer in answers.items():
                if answer.probabilities.keys() != questions[name].criteria.keys() or answer.choice not in answer.probabilities:
                    raise ValueError("Unexpected or missing choices")
                if not isclose(sum(answer.probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-5):
                    raise ValueError("Choice probabilities must sum to one")
        except ValueError as exc:
            raise UpstreamLLMError("Invalid decisions response") from exc
        return {name: answer.probabilities for name, answer in answers.items()}

    async def create_decision(self, state: str, questions: dict[str, ChoiceQuestion]) -> ChoiceProbabilities:
        if not self.url:
            raise ValueError("DECISIONS_URL is required")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with AsyncSession(headers=headers) as session:
            response = await session.post(self.url, json=self.build_request(state, questions), timeout=self.timeout)
            try:
                response.raise_for_status()
            except HTTPError as exc:
                raise UpstreamLLMError(LLMClient._extract_error_message(response.text)) from exc
            try:
                response_data = json.loads(response.text, object_pairs_hook=_unique_json_object)
            except ValueError as exc:
                raise UpstreamLLMError("Invalid decisions response") from exc
            return self.parse_response(response_data, questions)
