import json
from typing import Any

from niquests import AsyncSession
from niquests.exceptions import HTTPError

from llm_bot.config import Config


class UpstreamLLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        reasoning_effort: str | None = None,
    ):
        self.base_url = (base_url or Config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or Config.LLM_API_KEY
        self.model = model or Config.LLM_MODEL
        self.timeout = timeout or Config.LLM_TIMEOUT
        self.reasoning_effort = (
            Config.LLM_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _extract_error_message(self, response_text: str) -> str:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            return response_text

        if isinstance(payload, dict):
            error_value = payload.get("error")
            if isinstance(error_value, dict):
                for key in ("message", "detail", "error"):
                    value = error_value.get(key)
                    if value:
                        return str(value)
            if isinstance(error_value, str) and error_value:
                return error_value
            for key in ("message", "detail", "error"):
                value = payload.get(key)
                if value:
                    return str(value)

        return response_text

    async def create_response(
        self,
        system_input: str,
        user_input: str,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "input": [
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_input},
            ],
        }
        if self.model:
            payload["model"] = self.model
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if response_format is not None:
            payload["text"] = {"format": response_format}

        async with AsyncSession(base_url=self.base_url, headers=self._headers()) as session:
            response = await session.post("/responses", json=payload, timeout=self.timeout)
            try:
                response.raise_for_status()
            except HTTPError as exc:
                error_message = self._extract_error_message(response.text)
                raise UpstreamLLMError(error_message) from exc
            return json.loads(response.text)
