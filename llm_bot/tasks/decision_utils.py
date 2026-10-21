from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from llm_bot.decisions_client import ChoiceProbabilities, ChoiceQuestion, DecisionsClient


@dataclass(frozen=True)
class DecisionsTask[T]:
    """A prepared decisions request; invalid provider output fails without repair."""

    state: str
    questions: dict[str, ChoiceQuestion]
    parse_response: Callable[[ChoiceProbabilities], T]

    async def run(self, client: DecisionsClient) -> T:
        return self.parse_response(await client.create_decision(self.state, self.questions))

    def build_request(self, client: DecisionsClient) -> dict[str, Any]:
        return client.build_request(self.state, self.questions)

    def parse_result(self, response_data: dict[str, Any], client: DecisionsClient) -> T:
        return self.parse_response(client.parse_response(response_data, self.questions))
