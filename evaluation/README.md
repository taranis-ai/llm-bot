# Local analysis evaluation

This is an engineering smoke evaluation on **24 synthetic Taranis-style cases**,
not an independently labeled Taranis production dataset. The cases include overlapping
topics, neutral incident reports, explicit emotions, English/German/French/Spanish,
mixed text, short input, identifiers and two long articles. Labels were authored
with the implementation; wording was revised using this fixture, so these numbers
are development results, not held-out estimates. No calibration was fitted to it.

The implementation defaults to Laya as requested. No approval threshold is imposed.
Production quality and multilingual/emotion performance remain open concerns.

## Reproduce

```sh
uv run python -m llm_bot.local_inference --download
uv run python -m evaluation.evaluate
# Optional comparison; explicitly sends these texts to your configured LLM provider:
uv run python -m evaluation.evaluate --backend llm --output evaluation/llm-report.json
```

Use `--dataset path.json --output report.json` for separately annotated data with the
same format as [taranis_examples.json](taranis_examples.json). Every row includes
`id`, `text`, `topic`, `cybersecurity`, `sentiment`, `emotions`, and `language`.
`repeat` expands the long-input cases; `local_rejection` marks expected token-limit
rejections. The report saves IDs and predictions, not article text. Keep private
corpora and their results outside the repository.

## Recorded CPU result

Environment: macOS-27.0-arm64-arm-64bit-Mach-O, Python 3.13.15, Laya 0.3.11,
4 intra-op threads, device `cpu`. Model revision
`aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8`. See [raw measurements](laya-report.json).
Latency is one sequential pass on a development host, includes language routing and
tokenization, and is not an isolated throughput benchmark. No GPU measurements were taken.

| Task | Evaluated | Accuracy | ECE (10 bins) | Confidence Brier | p50 / p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| topic | 22 | 72.7% | 0.225 | 0.194 | 91.9 / 100.7 |
| sentiment | 22 | 77.3% | 0.175 | 0.176 | 349.3 / 370.6 |
| language | 24 | 100.0% | n/a | n/a | 5.5 / 160.4 |
| cybersecurity | 22 | 86.4% | 0.101 | 0.099 | 65.1 / 94.5 |

Emotion micro-F1: **0.244**. Sentiment was measured with emotion extraction
enabled (nine questions per call). Emotions are selected with P(present) > 0.5 and
filtered using the existing compatibility rules; this threshold is not calibrated.

| Primary topic | Support | Precision | Recall |
| --- | ---: | ---: | ---: |
| vulnerabilities | 3 | 50.0% | 100.0% |
| attacks | 2 | 33.3% | 50.0% |
| incidents | 6 | 80.0% | 66.7% |
| politics | 5 | 100.0% | 60.0% |
| business | 3 | 100.0% | 100.0% |
| other | 3 | 100.0% | 66.7% |

Cold model loading from the prepared local cache: **4.12 s**.
Total wall/CPU time: **15.17 / 29.40 s**; peak process RSS:
**3069 MiB**. Downloads are excluded. RSS is a host measurement, not a
deployment memory limit. Each additional worker loads separate models.

All six long-input model calls were rejected as intended (two articles × three
tasks); language detection analyzed all 24 texts. Model accuracy/calibration uses
the 22 accepted cases per task. Rejections are recorded separately and excluded
from those scores, not counted as correct predictions. All other calls succeeded.

Calibration uses the probability of the selected label. Laya rounds probabilities
to four decimals; only that rounding drift is normalized. Its entropy-based
`confidence` is never exposed as selected-label confidence. ECE compares accuracy
and selected probability in ten equal-width bins. Confidence Brier is the binary
squared error between selected probability and whether the label is correct, not
a multiclass Brier score. These tiny samples cannot establish calibration.

## Limits and comparison

The English hospital breach and retailer breach still receive negative sentiment
despite neutral labels. German and mixed-language incident reports also fail in
multiple tasks. Emotion false positives are frequent. These errors are recorded
in the raw report; no rule-based incident or severity override hides them.

Laya's built-in language heuristic scored 95.8% on these 24 cases, versus
100% for the conservative Lingua policy. This small four-language set does not
establish general language identification quality. The upstream detector is a
[checkpoint-routing heuristic](https://github.com/NandhaKishorM/laya/blob/main/laya/lang.py),
including script and function-word rules. Lingua supports actual language
identification and explicit abstention; mixed-language detection here works at
sentence boundaries, not arbitrary code-switches.

The upstream [benchmarks](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md)
also report uneven emotion performance and task-dependent miscalibration.
No representative production corpus or configured LLM backend was available,
so **no comparison with the previous LLM implementation was run**. The runner
supports that comparison for sentiment and cybersecurity; primary-topic
classification has no previous endpoint, and language detection is local in both
runs. Obtain independently annotated production articles and run the same harness
before drawing deployment-quality conclusions. This is a follow-up, not a default-switch gate.
