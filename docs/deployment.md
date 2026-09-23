# Embedded runtime deployment

Laya 0.3.11 is a required, uv-locked Python dependency, including PyTorch.
Lingua ships its language models in its wheel. Neither task uses a separate
inference service. Existing generative tasks still require `LLM_*` configuration.

## Model storage and startup

`LAYA_MODEL_REVISION` must be a full 40-character Hugging Face commit SHA. The
default is `aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8` of
`convaiinnovations/laya`; only the English root checkpoint is used.
Only its weights, tokenizer and encoder configuration are downloaded. Models
never download at import. The runtime resolves the pinned snapshot before giving
Laya an absolute local path; it never lets Laya resolve an unpinned model name.

Prepare the cache using the same configuration and user as the service:

```sh
uv run python -m llm_bot.local_inference --download
```

`LAYA_ALLOW_DOWNLOAD=false` is the default. Missing/incomplete weights fail
startup when `LAYA_PRELOAD=true` (default). `LAYA_ALLOW_DOWNLOAD=true` explicitly
permits first-use downloads; prefer prefetching for production. Cache directory
defaults to `~/.cache/llm-bot/models`, or `/app/models` in the container. Keep it
writable: Laya may repair tokenizer metadata in the cached snapshot.

`/health` is liveness. `/ready` is public, returns 200 only when the English Laya
checkpoint is resident, and otherwise returns 503. Startup preloads it before
serving. With `LAYA_PRELOAD=false`, it loads lazily on the first accepted English
analysis request. Library callers can explicitly
`await llm_bot.local_inference.runtime.preload()`; importing tasks does not load weights.

## Resources and concurrency

Use one application worker initially (`GRANIAN_WORKERS=1`). Each worker has its
own English model; two workers roughly double model memory. The checkpoint
contains about 421 million parameters, approximately 1.6 GiB of float32 parameters
alone. Allow at least 4 GiB RAM per worker for
loading, activations, tokenizers and runtime overhead, and several GiB of model
storage. This is a planning allowance, not a measured maximum. The container's
former 512 MiB worker recycle threshold is now 8192 MiB; size both container
limits and the recycle threshold using your workload. See
[evaluation results](../evaluation/README.md) for actual measurements on the development host.

`LAYA_DEVICE=cpu` is explicit and predictable; `cuda` and `mps` are opt-in.
Unavailable devices or Laya falling back to CPU fail initialization. The base
image does not configure GPU passthrough; CUDA requires a compatible PyTorch
build, driver and container device access. CUDA/MPS execution has not been
validated by this change. `LAYA_CPU_THREADS=4` bounds PyTorch intra-op threads.

One local inference runs per process. Work executes in a thread, off Quart's
event loop. Concurrent attempts return 503 (retry with backoff); there is no
unbounded model queue. Cancelling an HTTP request does not release a running
model or its lock. Model errors release the lock and permit later retries.
Language detection also runs off the event loop, using reusable local Lingua
models. It does not acquire the Laya inference lock.

## Published-image Compose example

Set `LLM_BOT_IMAGE` to a **published image tag or digest containing this change**
(the build pipeline publishes `ghcr.io/taranis-ai/taranis-llm-bot`). Do not use an
older release that predates these settings. Store `API_KEY` in the deployment's
secret environment, not in version-controlled YAML.

```yaml
services:
  llm-bot:
    image: ${LLM_BOT_IMAGE:?Set a published image tag or digest}
    ports:
      - "127.0.0.1:8000:8000"
    environment:
      API_KEY: ${API_KEY:?Set the incoming API key}
      LAYA_CACHE_DIR: /app/models
      LAYA_DEVICE: cpu
      LAYA_PRELOAD: "true"
      LAYA_ALLOW_DOWNLOAD: "false"
      GRANIAN_WORKERS: "1"
      GRANIAN_WORKERS_MAX_RSS: "8192"
    volumes:
      - laya-models:/app/models
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=5)"]
      interval: 30s
      timeout: 10s
      start_period: 120s
      retries: 3
volumes:
  laya-models:
```

Pull, prepare storage with the same image, restart, and verify:

```sh
podman compose pull
podman compose run --rm llm-bot python -m llm_bot.local_inference --download
podman compose up -d
podman compose ps
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail http://127.0.0.1:8000/info
```

Verify a protected inference request and inspect container memory before routing
traffic. The image runs as a non-root user; mounted storage must be writable by
that user. Keep the prior image digest and its cached model revision for rollback.

To temporarily restore the old sentiment/cybersecurity backend, set
`TEXT_ANALYSIS_BACKEND=llm`, configure `LLM_*`, and restart. `/classify` still uses
Laya and `/language` still uses Lingua. All three analysis endpoints require English
with either backend; non-English or inconclusive input returns 400 before inference.
Call `/translate` with `target_language="en"` first and pass its returned text to
analysis. Translation needs `LLM_*` settings and is never invoked automatically.
There is no automatic remote fallback:
an unavailable local model returns 503, invalid output returns a generic 502,
and blank/overlong input returns 400. For a full rollback, restore the previous
image digest and configuration, restart and verify its `/health` endpoint.
