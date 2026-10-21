"""Opt-in live evaluation: uv run python -m scripts.evaluate_classify --output /tmp/classify-results.json."""

import argparse
import asyncio
import json
import logging
from datetime import UTC, datetime
from math import isclose
from pathlib import Path
from time import perf_counter

from llm_bot.app import create_app
from llm_bot.config import Config
from llm_bot.schemas import ClassifyResponse

EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "classify_examples.json"


async def evaluate(output: Path) -> bool:
    if not Config.DECISIONS_URL:
        raise SystemExit("Configure DECISIONS_URL and the other DECISIONS_* settings before running.")
    Config.DEBUG = False
    app = create_app()
    # Provider exceptions may contain connection details; record only HTTP status.
    logging.disable(logging.CRITICAL)
    headers = {"Authorization": f"Bearer {Config.API_KEY}"} if Config.API_KEY else {}
    results = []
    async with app.test_client() as http:
        for example in json.loads(EXAMPLES_PATH.read_text(encoding="utf-8")):
            started = perf_counter()
            response = await http.post("/classify", json={"text": example["text"]}, headers=headers)
            result = {**example, "status": response.status_code, "seconds": round(perf_counter() - started, 3), "matched": False}
            if response.status_code == 200:
                probabilities = ClassifyResponse.model_validate(await response.get_json()).model_dump()
                assert isclose(sum(probabilities.values()), 1, abs_tol=1e-5, rel_tol=0)
                predicted = max(probabilities, key=probabilities.get)
                result.update(probabilities=probabilities, predicted=predicted, matched=predicted == example["expected"])
            results.append(result)
            print(
                f"{example['id']}: expected={example['expected']}; predicted={result.get('predicted', 'HTTP failure')}; "
                f"status={response.status_code}; seconds={result['seconds']}",
                flush=True,
            )
    report = {
        "completed_at": datetime.now(UTC).isoformat(),
        "model": Config.DECISIONS_MODEL,
        "api_mode": Config.DECISIONS_API_MODE,
        "matched": sum(result["matched"] for result in results),
        "total": len(results),
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Matched {report['matched']}/{report['total']}; report: {output}")
    return report["matched"] == report["total"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.output)) else 1)
