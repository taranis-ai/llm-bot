# English-only analysis evaluation

This is an engineering smoke evaluation on **24 synthetic Taranis-style cases**,
not an independently labeled Taranis production dataset. The fixture includes
overlapping topics, neutral incident reports, explicit emotions, other languages,
mixed text, short input, identifiers and two long articles. Labels were authored
with the implementation; wording was revised using this fixture. These are
development results, not held-out estimates. No calibration was fitted to it.

`/classify`, `/sentiment` and `/cybersec-classification` now require English with
either backend. The evaluation retains non-English and inconclusive examples to
verify rejection, rather than dropping them from the dataset. `/language` remains
multilingual. Only the English Laya checkpoint is loaded.

## Reproduce

```sh
uv run python -m llm_bot.local_inference --download
uv run python -m evaluation.evaluate
# Optional comparison; explicitly sends accepted English texts to your configured LLM provider:
uv run python -m evaluation.evaluate --backend llm --output evaluation/llm-report.json
```

Use `--dataset path.json --output report.json` with the format in
[taranis_examples.json](taranis_examples.json). Each row has `id`, `text`, `topic`,
`cybersecurity`, `sentiment`, `emotions`, and `language`. `repeat` expands long
inputs; `local_rejection` marks expected local length-limit rejections. Reference
`language` values other than `en` must raise `EnglishTextRequiredError` for
analysis. An unexpected failure still makes the runner exit nonzero.

The report saves IDs and predictions, not article text. Keep private corpora and
results outside the repository. To evaluate translated articles, call `/translate`
with `target_language="en"` and create a separately labeled English dataset from
the returned text. This harness does not translate automatically; translation
requires a configured LLM backend and may affect the expressed tone.

## Recorded CPU result

Environment: macOS-27.0-arm64-arm-64bit-Mach-O, Python 3.13.15, Laya 0.3.11,
4 intra-op threads, `cpu`. English model revision:
`aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8`. See [raw measurements](laya-report.json).
Latency includes local English validation and tokenization in one sequential pass
on a development host; it is not an isolated throughput benchmark. No GPU
measurements were taken.

| Task | Evaluated | Accuracy | ECE (10 bins) | Confidence Brier | p50 / p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| topic | 15 | 93.3% | 0.205 | 0.066 | 96.9 / 99.6 |
| sentiment | 15 | 80.0% | 0.228 | 0.203 | 356.8 / 443.4 |
| language | 24 | 100.0% | n/a | n/a | 13.0 / 198.1 |
| cybersecurity | 15 | 100.0% | 0.193 | 0.052 | 64.7 / 75.6 |

Emotion micro-F1: **0.294**. Sentiment includes emotion extraction
(nine questions per call). Emotions use P(present) > 0.5, filtered through the
existing compatibility rules. The threshold is not calibrated.

| Primary topic | Support | Precision | Recall |
| --- | ---: | ---: | ---: |
| vulnerabilities | 2 | 66.7% | 100.0% |
| attacks | 2 | 100.0% | 50.0% |
| incidents | 4 | 100.0% | 100.0% |
| politics | 2 | 100.0% | 100.0% |
| business | 3 | 100.0% | 100.0% |
| other | 2 | 100.0% | 100.0% |

Cold English model loading from cache: **1.83 s**.
Total wall/CPU time: **12.11 / 24.15 s**; peak RSS:
**2851 MiB**. Downloads are excluded. This host measurement is not
a deployment memory limit. Each additional worker loads a separate model.

There were **27 expected rejection responses**: eight non-English or inconclusive
cases (including the long German article) rejected by each of the three analysis
tasks, plus the long English article rejected by all three local token-limit
checks. The remaining 15 cases were analyzed. Language detection processed all
24 cases. There were no unexpected failures.

**The higher aggregate accuracy reflects the English-only scope, not improved
model predictions.** Earlier figures included 22 accepted cases across languages
(72.7% topic, 77.3% sentiment, 86.4% relevance). The old English subset already had
14/15 correct topic predictions, 12/15 sentiment and 15/15 relevance. Rejections
are excluded from accuracy and calibration, never counted as correct predictions.

Calibration uses selected-label probability, normalizing only four-decimal
rounding drift. The SDK entropy-based confidence is not used. ECE compares
accuracy and selected probability in ten equal-width bins. Confidence Brier is
binary squared error between selected probability and correctness, not multiclass
Brier. These tiny samples cannot establish calibration.

## Remaining limits

The English hospital breach, retailer breach and earnings report still receive
negative sentiment despite neutral labels. Emotion false positives remain common.
Restricting language does not fix these errors. Neither the experimental native
yes/no emotion questions nor the subjectivity check has been adopted in this change.

Language checks use Lingua in normal accuracy mode. Its low-accuracy mode had
misidentified an English cybersecurity sentence mentioning Mimikatz as Basque;
the existing client test now covers that sentence under English validation.
The detector still abstains on short, technical or substantially mixed inputs.
Its 100% fixture accuracy does not establish general language-identification quality.

The upstream [Laya routing heuristic](https://github.com/NandhaKishorM/laya/blob/main/laya/lang.py) scored 95.8%
on these cases. The [upstream benchmarks](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)
also describe uneven emotion performance and task-dependent miscalibration.

No production corpus or configured LLM backend was available, so **no comparison
against the previous LLM implementation or an actual translation backend was run**.
The runner supports an English sentiment/relevance comparison; primary-topic
classification has no previous endpoint. Translation-to-analysis flow is covered
by a network-free test that substitutes the external translation response.
