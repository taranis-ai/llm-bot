"""One reusable English Laya checkpoint per process; no inference service."""

import asyncio
from pathlib import Path
from threading import Lock

from llm_bot.config import Config
from llm_bot.log import logger


class LocalInputError(ValueError):
    pass


class LocalInferenceUnavailable(RuntimeError):
    pass


def validate_local_text(text: str) -> None:
    if not text.strip():
        raise LocalInputError("Text must not be blank")
    if len(text) > Config.LOCAL_MAX_INPUT_CHARS:
        raise LocalInputError(f"Text exceeds LOCAL_MAX_INPUT_CHARS ({Config.LOCAL_MAX_INPUT_CHARS}); no text was analyzed")


class LayaRuntime:
    def __init__(self):
        self._model = None
        self._lock = Lock()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _load_agent(self):
        import laya
        import torch
        from huggingface_hub import snapshot_download

        torch.set_num_threads(Config.LAYA_CPU_THREADS)
        snapshot = snapshot_download(
            "convaiinnovations/laya",
            revision=Config.LAYA_MODEL_REVISION,
            cache_dir=str(Path(Config.LAYA_CACHE_DIR).expanduser()),
            local_files_only=not Config.LAYA_ALLOW_DOWNLOAD,
            allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
        )
        model_dir = Path(snapshot)
        # Require bundled configuration/tokenizers, so Laya cannot fall back to hub downloads.
        for name in ("rl_agent_config.json", "model.safetensors", "tokenizer/tokenizer.json", "encoder/config.json"):
            if not (model_dir / name).is_file():
                raise FileNotFoundError(f"Incomplete Laya checkpoint: missing {name}")
        agent = laya.load(str(model_dir.resolve()), device=Config.LAYA_DEVICE)
        if agent.device.type != Config.LAYA_DEVICE:
            raise RuntimeError("Laya could not use the configured device")
        logger.info("Loaded English Laya revision %s on %s", Config.LAYA_MODEL_REVISION, agent.device)
        return agent

    def _agent(self):
        if self._model is None:
            self._model = self._load_agent()
        return self._model

    async def _run(self, operation, *args):
        # The worker owns the lock: cancelling an HTTP request cannot release a running model.
        def run():
            if not self._lock.acquire(blocking=False):
                raise LocalInferenceUnavailable("Local inference is busy; retry later")
            try:
                return operation(*args)
            except LocalInputError:
                raise
            except Exception as exc:
                logger.error("Local inference failed (%s)", type(exc).__name__)
                raise LocalInferenceUnavailable("Local inference unavailable; check model storage and device configuration") from exc
            finally:
                self._lock.release()

        return await asyncio.to_thread(run)

    async def preload(self):
        await self._run(self._agent)

    async def predict(self, text: str, questions: dict) -> dict:
        """Run English text; task entry points enforce language before dispatch."""
        validate_local_text(text)
        return await self._run(self._predict, text, questions)

    def _predict(self, text: str, questions: dict) -> dict:
        from laya.common import build_sequence

        agent = self._agent()
        max_len = agent.cfg.get("max_len", 512)
        head_max_len = agent.cfg.get("head_max_len", 192)
        tokens = agent.tok(text.replace(agent.tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        for question in questions.values():
            # Use the pinned SDK's actual prompt construction, including final separator.
            header, _ = build_sequence(agent.tok, "", agent._to_internal(question), max_len, head_max_len)
            if len(header) + len(tokens) > max_len:
                raise LocalInputError("Text exceeds the selected model's token budget; shorten it explicitly; no text was analyzed")
        return agent.predict(text, questions)


runtime = LayaRuntime()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare and verify the pinned English Laya checkpoint")
    parser.add_argument("--download", action="store_true", help="Allow downloading pinned weights into LAYA_CACHE_DIR")
    args = parser.parse_args()
    Config.LAYA_ALLOW_DOWNLOAD = args.download
    asyncio.run(runtime.preload())
