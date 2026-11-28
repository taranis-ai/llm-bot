from base64 import b64decode

from llm_bot.schemas import LookupResponse

# One white pixel, encoded as a grayscale PNG.
PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP4DwABAQEAsTj2FAAAAABJRU5ErkJggg=="
PNG_BYTES = b64decode(PNG_BASE64)


class StubLLMClient:
    def __init__(self, response_data):
        self.response_data = response_data
        self.calls = []

    async def create_response(self, system_input: str, user_input: str, response_format=None):
        self.calls.append(
            {
                "system_input": system_input,
                "user_input": user_input,
                "response_format": response_format,
            }
        )
        if isinstance(self.response_data, list):
            return self.response_data.pop(0)
        return self.response_data


class StubLookupClient:
    def __init__(self, responses_by_query=None):
        self.responses_by_query = responses_by_query or {}
        self.calls = []

    async def lookup(self, query: str, language: str, limit: int) -> LookupResponse:
        self.calls.append({"query": query, "language": language, "limit": limit})
        if query in self.responses_by_query:
            return self.responses_by_query[query]
        return LookupResponse.model_validate(
            {
                "query": query,
                "language": language,
                "limit": limit,
                "candidates": [],
            }
        )
