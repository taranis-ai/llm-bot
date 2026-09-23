"""Run with uv run python -m evaluation.evaluate; weights must already be cached."""

import argparse
import asyncio
import json
import platform
import resource
import statistics
import sys
from pathlib import Path
from time import perf_counter, process_time

from laya.lang import analyse

from llm_bot.client import LLMClient
from llm_bot.config import Config
from llm_bot.local_inference import LocalInputError, runtime
from llm_bot.schemas import CybersecClassificationRequest, LocalTextRequest, SentimentRequest
from llm_bot.tasks.classify import TOPICS, classify_text
from llm_bot.tasks.cybersec_classification import classify_cybersecurity_text
from llm_bot.tasks.language import EnglishTextRequiredError, detect_language
from llm_bot.tasks.sentiment import analyze_sentiment


def metrics(rows, labels):
    if not rows:
        return {"count": 0}
    correct = [row["predicted"] == row["expected"] for row in rows]
    result = {"count": len(rows), "accuracy": statistics.mean(correct), "per_label": {}}
    for label in labels:
        tp = sum(row["predicted"] == row["expected"] == label for row in rows)
        predicted = sum(row["predicted"] == label for row in rows)
        support = sum(row["expected"] == label for row in rows)
        result["per_label"][label] = {
            "precision": tp / predicted if predicted else 0,
            "recall": tp / support if support else 0,
            "support": support,
        }
    scored = [row for row in rows if "confidence" in row]
    if scored:
        result["confidence_brier"] = statistics.mean((row["confidence"] - (row["expected"] == row["predicted"])) ** 2 for row in scored)
        ece = 0
        for bucket in range(10):
            group = [row for row in scored if min(int(row["confidence"] * 10), 9) == bucket]
            if group:
                ece += (
                    len(group)
                    / len(scored)
                    * abs(
                        statistics.mean(row["confidence"] for row in group)
                        - statistics.mean(row["expected"] == row["predicted"] for row in group)
                    )
                )
        result["ece_10_bins"] = ece
    return result


async def evaluate(args):
    dataset = json.loads(args.dataset.read_text())
    baseline = args.backend == "llm"
    start, cpu_start = perf_counter(), process_time()
    if not baseline:
        await runtime.preload()
    cold_seconds = perf_counter() - start
    predictions, errors, timings = [], [], {}
    heuristic = []
    for case in dataset["cases"]:
        text = case["text"] * case.get("repeat", 1)
        heuristic.append({"expected": case["language"], "predicted": analyse(text).get("language") or "und"})
        calls = {
            "language": lambda text=text: detect_language(LocalTextRequest(text=text)),
            "sentiment": lambda text=text: analyze_sentiment(
                SentimentRequest(text=text, include_emotions=True),
                client=LLMClient() if baseline else None,
                inference=None if baseline else runtime,
            ),
            "cybersecurity": lambda text=text: classify_cybersecurity_text(
                CybersecClassificationRequest(text=text), client=LLMClient() if baseline else None, inference=None if baseline else runtime
            ),
        }
        if not baseline:
            calls["topic"] = lambda text=text: classify_text(LocalTextRequest(text=text))
        for task, call in calls.items():
            before = perf_counter()
            try:
                response = await call()
                if task != "language" and (case["language"] != "en" or (case.get("local_rejection") and not baseline)):
                    raise AssertionError("Expected input rejection, but analysis succeeded")
                elapsed = perf_counter() - before
                row = {"id": case["id"], "task": task, "expected": case[task], "seconds": elapsed}
                if task == "language":
                    row["predicted"] = response.language
                elif task == "topic":
                    row.update(predicted=response.category, confidence=response.scores[response.category], scores=response.scores)
                elif task == "sentiment":
                    row.update(
                        predicted=response.sentiment.label,
                        confidence=response.sentiment.score,
                        emotions=response.sentiment.emotions,
                        expected_emotions=case["emotions"],
                    )
                else:
                    label = response.cybersecurity >= response.non_cybersecurity
                    row.update(predicted=label, confidence=response.cybersecurity if label else response.non_cybersecurity)
                predictions.append(row)
                timings.setdefault(task, []).append(elapsed)
            except Exception as exc:
                errors.append(
                    {
                        "id": case["id"],
                        "task": task,
                        "error": type(exc).__name__,
                        "expected_rejection": bool(
                            task != "language"
                            and (
                                (case["language"] != "en" and isinstance(exc, EnglishTextRequiredError))
                                or (case.get("local_rejection") and isinstance(exc, LocalInputError) and not baseline)
                            )
                        ),
                    }
                )
    summary = {}
    for task, labels in {
        "topic": list(TOPICS),
        "sentiment": ["positive", "negative", "neutral"],
        "language": sorted({case["language"] for case in dataset["cases"]}),
        "cybersecurity": [True, False],
    }.items():
        rows = [row for row in predictions if row["task"] == task]
        summary[task] = metrics(rows, labels)
        samples = sorted(timings.get(task, []))
        if samples:
            summary[task]["latency_seconds"] = {
                "p50": statistics.median(samples),
                "p95": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
            }
    emotion_rows = [row for row in predictions if row["task"] == "sentiment"]
    tp = sum(len(set(row["emotions"]) & set(row["expected_emotions"])) for row in emotion_rows)
    predicted = sum(len(row["emotions"]) for row in emotion_rows)
    expected = sum(len(row["expected_emotions"]) for row in emotion_rows)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {
        "provenance": dataset["provenance"],
        "backend": args.backend,
        "analysis_languages": ["en"],
        "model_revision": Config.LAYA_MODEL_REVISION,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "device": Config.LAYA_DEVICE,
        "cpu_threads": Config.LAYA_CPU_THREADS,
        "dataset_cases": len(dataset["cases"]),
        "cold_load_seconds": cold_seconds,
        "wall_seconds": perf_counter() - start,
        "cpu_seconds": process_time() - cpu_start,
        "peak_rss_mib": usage.ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024),
        "metrics": summary,
        "laya_language_heuristic": metrics(heuristic, sorted({case["language"] for case in dataset["cases"]})),
        "emotions_micro_f1": 2 * tp / (predicted + expected) if predicted + expected else 1,
        "errors": errors,
        "predictions": predictions,
    }
    if not baseline:
        import torch

        report["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}: {len(predictions)} predictions, {len(errors)} rejected/failed calls")
    return any(not error["expected_rejection"] for error in errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/taranis_examples.json"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/laya-report.json"))
    parser.add_argument(
        "--backend", choices=("laya", "llm"), default="laya", help="llm explicitly sends fixture text to the configured LLM backend"
    )
    raise SystemExit(asyncio.run(evaluate(parser.parse_args())))
