# Build and publish

[Back to README](../README.md)

Run these commands from the repository root.

From a clean `X.Y.Z` release tag and an empty `dist/` directory:

```bash
./scripts/check.sh
uv build --no-sources
uvx twine check --strict dist/*
uv publish dist/*
```

Authenticate with `UV_PUBLISH_TOKEN`; use `--publish-url <upload-endpoint>` for
another registry. The [release workflow](../.github/workflows/release.yml) releases
images and the PyPI package on tag pushes; see [PyPI setup](agents/development-workflow.md#packaging-and-release).
