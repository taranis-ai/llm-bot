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
        api_mode: str | None = None,
        timeout: int | None = None,
        reasoning_effort: str | None = None,
    ):
        self.base_url = (base_url or Config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or Config.LLM_API_KEY
        self.model = model or Config.LLM_MODEL
        self.api_mode = api_mode or Config.LLM_API_MODE
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

    def _build_responses_payload(
        self,
        system_input: str,
        user_input: str,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        return payload

    def _build_chat_completions_payload(
        self,
        system_input: str,
        user_input: str,
        response_format: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_input},
            ],
        }
        if self.model:
            payload["model"] = self.model
        if response_format is not None and response_format.get("type") == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format["name"],
                    "strict": response_format.get("strict", False),
                    "schema": response_format["schema"],
                },
            }
        return payload

    def _normalize_chat_completions_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            return response_data
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return {"output_text": content}
        if isinstance(content, list):
            text_parts = [
                str(item.get("text"))
                for item in content
                if isinstance(item, dict) and item.get("text")
            ]
            if text_parts:
                return {"output_text": "".join(text_parts)}
        return response_data

    def _request_target(
        self,
        system_input: str,
        user_input: str,
        response_format: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        if self.api_mode == "chat_completions":
            return "/chat/completions", self._build_chat_completions_payload(
                system_input, user_input, response_format
            )
        return "/responses", self._build_responses_payload(system_input, user_input, response_format)

    async def create_response(
        self,
        system_input: str,
        user_input: str,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path, payload = self._request_target(system_input, user_input, response_format)

        async with AsyncSession(base_url=self.base_url, headers=self._headers()) as session:
            response = await session.post(path, json=payload, timeout=self.timeout)
            try:
                response.raise_for_status()
            except HTTPError as exc:
                error_message = self._extract_error_message(response.text)
                raise UpstreamLLMError(error_message) from exc
            response_data = json.loads(response.text)
            if self.api_mode == "chat_completions":
                return self._normalize_chat_completions_response(response_data)
            return response_data
