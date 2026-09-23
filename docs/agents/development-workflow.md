# Development Workflow

## When To Load

Read this before editing application code, tests, configuration, packaging, CI, or local-development instructions.

## Environment

- The project targets Python 3.13 and uses `uv` for dependency management. Do not use `pip` or edit `uv.lock` by hand.
- Runtime and development dependencies are declared in `pyproject.toml`; `scripts/check.sh` installs them from the lockfile before running checks.
- Copy `.env.example` to `.env` for local configuration. Never commit secrets or copy values from an existing `.env` into documentation, tests, or logs.
- Settings are loaded by `llm_bot.config.Config` from the process environment and `.env`. When adding a setting, update `Settings`, `.env.example`, and the relevant README/API metadata together.

## Common Commands

Run commands from the repository root:

```bash
./scripts/check.sh
uv build
```

`scripts/check.sh` runs dependency sync, Ruff lint/format checks, and the full test
suite. It can also be invoked by absolute path from another directory.

For a focused test while iterating, run its file or node directly, for example:

```bash
uv run pytest tests/test_client.py
uv run pytest tests/test_app.py::test_health_endpoint
```

Run the full test suite and Ruff checks before handing off a code change. Run `uv build` when changing packaging, package data, or release inputs.
Ruff also checks import order (`I`), Python modernization (`UP`), and common bugs (`B`).

## Local Startup

Start the development server with:

```bash
uv run granian --interface asgi app:app --port 5500
```

The root `app.py` is the ASGI entry point and delegates construction to `llm_bot.app.create_app()`. Container startup uses `granian app` and the environment defaults in `Containerfile`.

The service needs an OpenAI-compatible backend for LLM routes. `LLM_API_MODE` selects either the Responses API (`/responses`) or Chat Completions (`/chat/completions`). Entity-linking routes additionally need the lookup service configured through `LOOKUP_*` settings.

Before serving, prepare the pinned local models using
`uv run python -m llm_bot.local_inference --download`. Startup preloads the English Laya
checkpoint offline by default; `/ready` reports its readiness and `/health`
remains liveness. For HTTP development without model tasks, `LAYA_PRELOAD=false`
skips startup loading. See [deployment](../deployment.md) for memory and storage.
Unit tests inject inference and must not download weights. Real model evaluation
is separate: `uv run python -m evaluation.evaluate`; see [evaluation](../../evaluation/README.md).

## Test Conventions

- Tests live in `tests/` and mirror the application concern: route integration in `test_app.py`, transport behavior in client tests, schema validation in schema tests, and task behavior in `test_*_task.py` files.
- Use the Quart application fixture from `tests/conftest.py` for route tests.
- Inject `LLMClient` or `LookupClient` into task functions, or monkeypatch the imported dependency at its use site. Unit tests must not call live LLM or lookup services.
- Keep prompt construction and response parsing testable as pure functions. Cover valid output, invalid output, the one repair attempt, and task-specific invariants when applicable.
- When changing a request or response, test both Pydantic validation and the HTTP status/body exposed by the route.
- Test behavior owned by this project, not behavior already guaranteed by Pydantic, Python, or another dependency. A test that only proves `model_validate()` rejects a primitive with the wrong shape, or that `dict.update()` works, adds no useful coverage.
- Avoid duplicate tests across schema, task, and route layers. Keep more than one only when each protects a distinct boundary, such as a custom model invariant at the schema layer and its documented `400` mapping at the HTTP layer.
- Prefer direct assertions against the expected public data. Do not build the expected result by calling the same validator or helper used by the code under test.
- Shared retry and parsing mechanics need task-level coverage only where the task adds behavior to that path. Do not repeat assertions about generic repair-instruction wording in every task test.
- Preserve async tests and mark standalone async task/client tests with `pytest.mark.asyncio`; the project config uses pytest's auto asyncio mode.

## API And Documentation Changes

The API contract is represented in several places. When behavior changes, keep these synchronized:

- `llm_bot/schemas.py` for runtime validation and serialization
- `llm_bot/routes.py` for routing, errors, `/info`, and Swagger/OpenAPI serving
- `llm_bot/openapi3_1.yml` for the published contract
- `README.md` for operator-facing examples and configuration
- `.env.example` for new or changed settings
- focused tests under `tests/`

Prompt changes in `llm_bot/prompts/` are behavior changes. Update the corresponding task tests even when no Python signature changes.

## Packaging And Release

- Versioning is tag-driven through `setuptools_scm`; release tags use `X.Y.Z`.
- `llm_bot.__version__` reads the installed distribution metadata and falls back to `0.0.0` only when distribution metadata is unavailable. Git is needed for release builds, not at runtime. Release tags support multi-digit `X.Y.Z` components.
- `Containerfile` creates the runtime image. Ensure every runtime file, especially prompts and `llm_bot/openapi3_1.yml`, is present in both the installed distribution and container path when packaging changes.
- The OpenAPI source lives inside `llm_bot` and is included as package data. After packaging changes, smoke-test `/health`, `/openapi.yaml`, and prompt loading from the built wheel outside the checkout. The release workflow uploads the same packaged source as its OpenAPI artifact.
- `.github/workflows/test.yml` delegates Python validation to the shared Taranis AI workflow. The build workflow publishes multi-architecture images on branch pushes.
- `.github/workflows/release.yml` handles image and Python releases on `X.Y.Z` tag pushes. It runs `scripts/check.sh`, builds and checks wheel/sdist metadata, smoke-tests the installed wheel outside the checkout, and verifies its version matches the tag. It then retags the existing `latest` image and creates the GitHub release with the build artifacts.
- The dependent PyPI publish job runs after the image/GitHub release succeeds and any `pypi` environment approval. It receives only the tested distributions and OIDC permission, and does not rebuild the package.
- For a local distribution smoke test, run from a temporary directory: `uv run --no-project --python 3.13 --with /absolute/path/to/dist/package.whl python -I /absolute/path/to/tests/smoke_distribution.py`. An optional final argument checks the expected release version. No live LLM or lookup service is required.

To configure automated PyPI releases:

1. Use the PyPI project `taranis-llm-bot` owned by the `taranis-ai` organization. The distribution name in `pyproject.toml` must match this project; the Python import package remains `llm_bot`. PyPI's `llm-bot` is an unrelated project.
2. Register a [trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) for project `taranis-llm-bot`, owner `taranis-ai`, repository `llm-bot`, workflow `release.yml`, environment `pypi`.
3. Create the GitHub `pypi` environment with required reviewers and release-tag restrictions.

Local uploads use `UV_PUBLISH_TOKEN` from secure storage; never commit the token.

An upload error saying the OIDC token is not valid for project `llm-bot` indicates
that the distribution was built with the old name. Rebuild from a commit with
the corrected package metadata into an empty output directory. Rerunning the old
tag's publish job reuses the old artifacts and cannot fix the name mismatch;
create the next release tag from the corrected commit instead.

## Change Discipline

- Keep changes focused and preserve unrelated work in a dirty worktree.
- Prefer the nearest existing pattern over introducing a new abstraction for one endpoint.
- Do not log API keys or authorization headers. Treat request text and model reasoning as potentially sensitive; DEBUG logging is opt-in for that reason.
- Do not manually edit generated build outputs such as `dist/`, `*.egg-info`, caches, or bytecode.
