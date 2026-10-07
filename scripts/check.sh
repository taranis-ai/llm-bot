#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
uv sync --locked --extra dev --extra server
uv run ruff check .
uv run ruff format --check .
uv run pytest
