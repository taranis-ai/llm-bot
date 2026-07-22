# AGENTS Instructions

Project guidance for coding agents working on llm-bot.

## Project Overview

llm-bot is a small asynchronous Quart service that exposes LLM-backed text-processing endpoints for Taranis AI. See [README.md](README.md) for the public API and configuration reference.

## Required Reading

- Before editing code, configuration, tests, packaging, or CI, read [Development Workflow](docs/agents/development-workflow.md).
- Before changing application structure, HTTP or upstream API contracts, schemas, prompts, or task behavior, also read [Architecture and Boundaries](docs/agents/architecture-and-boundaries.md).
- Before adding or changing an LLM-backed endpoint, read [LLM Task Development](docs/agents/llm-task-development.md).

Keep these documents accurate when a change alters the repository layout, development commands, data flow, or the expected way to add a task.
