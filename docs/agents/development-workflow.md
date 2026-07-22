# Development Workflow

## When To Load

Read this before editing application code, tests, configuration, packaging, CI, or local-development instructions.

## Environment

- The project targets Python 3.13 and uses `uv` for dependency management. Do not use `pip` or edit `uv.lock` by hand.
- Runtime and development dependencies are declared in `pyproject.toml`; install them with `uv sync --extra dev`.
- Copy `.env.example` to `.env` for local configuration. Never commit secrets or copy values from an existing `.env` into documentation, tests, or logs.
- Settings are loaded by `llm_bot.config.Config` from the process environment and `.env`. When adding a setting, update `Settings`, `.env.example`, and the relevant README/API metadata together.

## Common Commands

Run commands from the repository root:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

For a focused test while iterating, run its file or node directly, for example:

```bash
uv run pytest tests/test_client.py
uv run pytest tests/test_app.py::test_health_endpoint
```

Run the full test suite and Ruff checks before handing off a code change. Run `uv build` when changing packaging, package data, or release inputs.

## Local Startup

Start the development server with:

```bash
uv run granian --interface asgi app:app --port 5500
```

The root `app.py` is the ASGI entry point and delegates construction to `llm_bot.app.create_app()`. Container startup uses `granian app` and the environment defaults in `Containerfile`.

The service needs an OpenAI-compatible backend for LLM routes. `LLM_API_MODE` selects either the Responses API (`/responses`) or Chat Completions (`/chat/completions`). Entity-linking routes additionally need the lookup service configured through `LOOKUP_*` settings.

## Test Conventions

- Tests live in `tests/` and mirror the application concern: route integration in `test_app.py`, transport behavior in client tests, schema validation in schema tests, and task behavior in `test_*_task.py` files.
- Use the Quart application fixture from `tests/conftest.py` for route tests.
- Inject `LLMClient` or `LookupClient` into task functions, or monkeypatch the imported dependency at its use site. Unit tests must not call live LLM or lookup services.
- Keep prompt construction and response parsing testable as pure functions. Cover valid output, invalid output, the one repair attempt, and task-specific invariants when applicable.
- When changing a request or response, test both Pydantic validation and the HTTP status/body exposed by the route.
- Preserve async tests and mark standalone async task/client tests with `pytest.mark.asyncio`; the project config uses pytest's auto asyncio mode.

## API And Documentation Changes

The API contract is represented in several places. When behavior changes, keep these synchronized:

- `llm_bot/schemas.py` for runtime validation and serialization
- `llm_bot/routes.py` for routing, errors, `/info`, and Swagger/OpenAPI serving
- `openapi3_1.yml` for the published contract
- `README.md` for operator-facing examples and configuration
- `.env.example` for new or changed settings
- focused tests under `tests/`

Prompt changes in `llm_bot/prompts/` are behavior changes. Update the corresponding task tests even when no Python signature changes.

## Packaging And Release

- Versioning is tag-driven through `setuptools_scm`; release tags use `X.Y.Z`.
- `llm_bot.__version__` resolves the latest Git tag at runtime and falls back to `0.0.0` outside a Git checkout.
- `Containerfile` creates the runtime image. Ensure every runtime file, especially prompts and `openapi3_1.yml`, is present in both the installed distribution and container path when packaging changes.
- `.github/workflows/test.yml` delegates Python validation to the shared Taranis AI workflow. The build workflow publishes multi-architecture images, and the release workflow retags the existing `latest` image and publishes build artifacts.

## Change Discipline

- Keep changes focused and preserve unrelated work in a dirty worktree.
- Prefer the nearest existing pattern over introducing a new abstraction for one endpoint.
- Do not log API keys or authorization headers. Treat request text and model reasoning as potentially sensitive; DEBUG logging is opt-in for that reason.
- Do not manually edit generated build outputs such as `dist/`, `*.egg-info`, caches, or bytecode.
