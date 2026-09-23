import asyncio
from threading import Event

import pytest

from llm_bot.config import Config
from llm_bot.local_inference import LayaRuntime, LocalInferenceUnavailable, LocalInputError
from llm_bot.schemas import CybersecClassificationRequest, LocalTextRequest, SentimentRequest, TranslateRequest
from llm_bot.tasks.classify import TOPICS, classify_text
from llm_bot.tasks.cybersec_classification import classify_cybersecurity_text
from llm_bot.tasks.language import detect_language
from llm_bot.tasks.local_analysis import choice_probabilities
from llm_bot.tasks.sentiment import analyze_sentiment
from llm_bot.tasks.translate import translate_text
from tests.test_helpers import StubLLMClient


class StubInference:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    async def predict(self, text, questions):
        self.calls.append((text, questions))
        return {
            "answers": {
                name: {"choice": max(scores, key=scores.get), "probabilities": scores, "confidence": 0.01}
                for name, scores in self.scores.items()
            }
        }


async def test_topic_and_relevance_are_independent():
    topic = StubInference({"topic": {key: 0.8 if key == "politics" else 0.04 for key in TOPICS}})
    relevance = StubInference({"relevance": {"cybersecurity": 0.9, "non-cybersecurity": 0.1}})
    text = "Parliament approved new cybersecurity regulations."
    result = await classify_text(LocalTextRequest(text=text), inference=topic)
    cyber = await classify_cybersecurity_text(CybersecClassificationRequest(text=text), inference=relevance)
    assert result.category == "politics"
    assert sum(result.scores.values()) == pytest.approx(1)
    assert cyber.model_dump() == {"cybersecurity": 0.9, "non-cybersecurity": 0.1}
    assert "Hospital reporting a ransomware breach: incidents" in topic.calls[0][1]["topic"]["instructions"]


@pytest.mark.parametrize(("include_emotions", "present"), [(False, False), (True, False), (True, True)])
async def test_local_sentiment_uses_selected_probability_and_preserves_emotions(include_emotions, present):
    scores = {"sentiment": {"positive": 0.1333, "negative": 0.1333, "neutral": 0.7333}}
    if include_emotions:
        scores.update(
            {
                name: {"absent": 0.1 if present else 0.9, "present": 0.9 if present else 0.1}
                for name in ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")
            }
        )
    inference = StubInference(scores)
    result = await analyze_sentiment(
        SentimentRequest(
            text="A hospital reported a breach.", include_emotions=include_emotions, reasoning_effort="high", thinking_budget_tokens=512
        ),
        inference=inference,
    )
    sentiment = result.model_dump()["sentiment"]
    assert sentiment["label"] == "neutral"
    assert sentiment["score"] == pytest.approx(0.7333 / 0.9999)
    if include_emotions:
        assert sentiment["emotions"] == (["surprise", "anticipation"] if present else [])
    else:
        assert "emotions" not in sentiment
        assert set(inference.calls[0][1]) == {"sentiment"}


@pytest.mark.parametrize("scores", [{"a": 0.9, "b": 0.9}, {"a": float("nan"), "b": 1}, {"a": 1}, {"a": -0.1, "b": 1.1}])
def test_invalid_local_probabilities_are_not_hidden(scores):
    with pytest.raises(ValueError):
        choice_probabilities({"answers": {"test": {"choice": "a", "probabilities": scores}}}, "test", ("a", "b"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The hospital reported a ransomware breach affecting patient records. Investigators are assessing the incident.", "en"),
        ("Das Krankenhaus meldete einen Angriff auf seine Computersysteme. Die Polizei untersucht den Vorfall.", "de"),
        ("Le gouvernement a adopté une nouvelle loi pour renforcer la sécurité des réseaux informatiques.", "fr"),
        ("El gobierno anunció nuevas medidas para proteger los sistemas informáticos de los hospitales.", "es"),
        ("The hospital reported a breach affecting patient records. Das Krankenhaus meldete einen Angriff auf seine Computersysteme.", "und"),
        ("https://example.com/security CVE-2026-12345 192.168.1.1 SHA256:abcdef1234", "und"),
        ("Bonjour", "und"),
        ("病院は患者の記録に影響するサイバー攻撃を報告しました。警察は事件を調査しています。", "ja"),
    ],
)
async def test_local_language_policy(text, expected):
    assert (await detect_language(LocalTextRequest(text=text))).language == expected


async def test_translation_supplies_english_analysis_without_mutating_request():
    client = StubLLMClient({"output_text": '{"translation":"The hospital reported an attack."}'})
    request = TranslateRequest(text="Das Krankenhaus meldete einen Angriff auf seine Computersysteme.", target_language="en")
    translated = await translate_text(request, client=client)
    assert "The source language is de." in client.calls[0]["system_input"]
    assert request.source_language is None
    inference = StubInference({"topic": {key: 1.0 if key == "incidents" else 0.0 for key in TOPICS}})
    result = await classify_text(LocalTextRequest(text=translated.translation), inference=inference)
    assert result.category == "incidents"
    assert inference.calls[0][0] == "The hospital reported an attack."


@pytest.mark.parametrize(
    ("task", "request_model"),
    [(classify_text, LocalTextRequest), (analyze_sentiment, SentimentRequest), (classify_cybersecurity_text, CybersecClassificationRequest)],
)
@pytest.mark.parametrize(
    "text",
    [
        "Das Krankenhaus meldete einen Angriff auf seine Computersysteme.",
        "The hospital reported a breach affecting patient records. Das Krankenhaus meldete einen Angriff auf seine Computersysteme.",
        "Bonjour",
    ],
)
async def test_analysis_tasks_require_english_before_inference(task, request_model, text):
    inference = StubInference({})
    with pytest.raises(LocalInputError, match="English text is required"):
        await task(request_model(text=text), inference=inference)
    assert inference.calls == []


@pytest.mark.parametrize("path", ["/classify", "/sentiment", "/cybersec-classification"])
@pytest.mark.parametrize("backend", ["laya", "llm"])
async def test_analysis_http_requires_english_for_either_backend(app, monkeypatch, path, backend):
    monkeypatch.setattr(Config, "TEXT_ANALYSIS_BACKEND", backend)
    response = await app.test_client().post(path, json={"text": "Das Krankenhaus meldete einen Angriff auf seine Computersysteme."})
    assert response.status_code == 400
    error = (await response.get_json())["error"]
    assert "English text is required" in error
    assert "/translate" in error
    assert "target_language='en'" in error


async def test_runtime_reuses_model_and_keeps_lock_after_cancellation(monkeypatch):
    runtime = LayaRuntime()
    started, release, finished = Event(), Event(), Event()
    loaded = []

    def load():
        loaded.append(True)
        started.set()
        assert release.wait(5)
        finished.set()
        return object()

    monkeypatch.setattr(runtime, "_load_agent", load)
    first = asyncio.create_task(runtime.preload())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(LocalInferenceUnavailable, match="busy"):
            await runtime.preload()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
    await asyncio.to_thread(runtime._lock.acquire)
    runtime._lock.release()
    await runtime.preload()
    assert loaded == [True]
    assert runtime.ready


async def test_failed_loading_can_retry(monkeypatch):
    runtime = LayaRuntime()

    def fail():
        raise OSError("private local path")

    monkeypatch.setattr(runtime, "_load_agent", fail)
    with pytest.raises(LocalInferenceUnavailable, match="model storage") as error:
        await runtime.preload()
    assert "private" not in str(error.value)
    assert not runtime.ready
    monkeypatch.setattr(runtime, "_load_agent", lambda: object())
    await runtime.preload()
    assert runtime.ready


async def test_token_limit_uses_actual_laya_prompt_and_never_calls_truncated_inference(monkeypatch):
    from laya import Agent

    class Tokenizer:
        mask_token = "[MASK]"
        mask_token_id, cls_token_id, sep_token_id = 1, 2, 3

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}

    class AgentBoundary:
        cfg = {"max_len": 32, "head_max_len": 16}
        tok = Tokenizer()
        _to_internal = staticmethod(Agent._to_internal)
        calls = 0

        def predict(self, text, questions):
            self.calls += 1
            return {"answers": {}}

    agent = AgentBoundary()
    runtime = LayaRuntime()
    monkeypatch.setattr(runtime, "_load_agent", lambda: agent)

    questions = {"test": {"type": "choice", "instructions": "Topic?", "criteria": {"a": "", "b": ""}}}
    await runtime.predict("The hospital reported an attack on its computers.", questions)
    await runtime.predict("The police are investigating the attack on the hospital.", questions)
    with pytest.raises(LocalInputError, match="token budget"):
        await runtime.predict("The hospital reported an attack. " * 20, questions)
    assert agent.calls == 2


@pytest.mark.parametrize("path", ["/classify", "/language", "/sentiment", "/cybersec-classification", "/translate"])
async def test_local_routes_validate_and_authenticate(app, monkeypatch, path):
    monkeypatch.setattr(Config, "API_KEY", "test-key")
    client = app.test_client()
    assert (await client.post(path, json={"text": "hello"})).status_code == 401
    headers = {"Authorization": "Bearer test-key"}
    for text in ("", "   ", "a" * (Config.LOCAL_MAX_INPUT_CHARS + 1)):
        payload = {"text": text}
        if path == "/translate":
            payload["target_language"] = "en"
        assert (await client.post(path, headers=headers, json=payload)).status_code == 400


async def test_local_http_success_failure_and_readiness(app, monkeypatch):
    from llm_bot import routes
    from llm_bot.tasks import classify, sentiment

    inference = StubInference({"topic": {key: 1.0 if key == "incidents" else 0.0 for key in TOPICS}})
    monkeypatch.setattr(classify, "runtime", inference)
    client = app.test_client()
    response = await client.post("/classify", json={"text": "A hospital reported a breach."})
    assert response.status_code == 200
    assert (await response.get_json())["category"] == "incidents"
    language = await client.post("/language", json={"text": "Bonjour"})
    assert await language.get_json() == {"language": "und"}
    monkeypatch.setattr(sentiment, "runtime", StubInference({"sentiment": {"positive": 0.1, "negative": 0.1, "neutral": 0.8}}))
    response = await client.post("/sentiment", json={"text": "A hospital reported a breach."})
    assert await response.get_json() == {"sentiment": {"label": "neutral", "score": 0.8}}
    inference.scores = {"topic": {"other": 8.0}}
    assert (await client.post("/classify", json={"text": "The hospital reported an attack on its computers."})).status_code == 502
    fresh = LayaRuntime()
    monkeypatch.setattr(routes, "runtime", fresh)
    assert (await client.get("/ready")).status_code == 503
    fresh._model = object()
    assert (await client.get("/ready")).status_code == 200
    info = await client.get("/info")
    assert (await info.get_json())["current"]["text_analysis_languages"] == ["en"]


async def test_local_http_unavailable(app, monkeypatch):
    from llm_bot.tasks import classify

    runtime = LayaRuntime()
    monkeypatch.setattr(classify, "runtime", runtime)
    runtime._lock.acquire()
    try:
        response = await app.test_client().post("/classify", json={"text": "A hospital reported a breach."})
    finally:
        runtime._lock.release()
    assert response.status_code == 503


async def test_startup_preloads_or_fails(monkeypatch):
    from quart.testing.app import LifespanError

    from llm_bot.app import create_app

    runtime = LayaRuntime()
    monkeypatch.setattr("llm_bot.app.runtime", runtime)
    monkeypatch.setattr(runtime, "_load_agent", lambda: object())
    async with create_app().test_app():
        assert runtime.ready

    def fail():
        raise OSError("Unavailable checkpoint")

    runtime = LayaRuntime()
    monkeypatch.setattr("llm_bot.app.runtime", runtime)
    monkeypatch.setattr(runtime, "_load_agent", fail)
    with pytest.raises(LifespanError, match="Local inference unavailable"):
        async with create_app().test_app():
            pass


def test_loader_pins_revision_and_stays_offline(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []
    for name in ("rl_agent_config.json", "model.safetensors", "tokenizer/tokenizer.json", "encoder/config.json"):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()

    def snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)

    def load(path, **kwargs):
        assert path == str(tmp_path)
        return SimpleNamespace(device=SimpleNamespace(type="cpu"))

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setattr("laya.load", load)
    monkeypatch.setattr(Config, "LAYA_ALLOW_DOWNLOAD", False)
    monkeypatch.setattr(Config, "LAYA_DEVICE", "cpu")
    LayaRuntime()._load_agent()
    assert calls[0][0] == "convaiinnovations/laya"
    assert calls[0][1]["revision"] == Config.LAYA_MODEL_REVISION
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["allow_patterns"] == ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
    monkeypatch.setattr(Config, "LAYA_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="configured device"):
        LayaRuntime()._load_agent()


def test_evaluation_metrics_count_errors_and_calibration():
    from evaluation.evaluate import metrics

    result = metrics(
        [
            {"expected": "a", "predicted": "a", "confidence": 0.8},
            {"expected": "b", "predicted": "a", "confidence": 0.8},
        ],
        ["a", "b"],
    )
    assert result["accuracy"] == 0.5
    assert result["per_label"]["a"] == {"precision": 0.5, "recall": 1.0, "support": 1}
    assert result["per_label"]["b"]["recall"] == 0
    assert result["ece_10_bins"] == pytest.approx(0.3)
    assert result["confidence_brier"] == pytest.approx(0.34)
