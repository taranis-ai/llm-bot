# Batch processing

[Back to README](../README.md) · [Configuration](configuration.md)

## NER for multiple stories

For 20 Taranis stories, prepare **one NER task per story** and submit all 20
tasks as one provider batch job. Each story has its own text and entity result.
This is available through the Python library; `POST /ner` still accepts only
one text, and llm-bot does not expose a batch HTTP endpoint or manage batch jobs.

Your provider must support a Batch API and the task's output schema; supporting
`/responses` or `/chat/completions` alone does not imply batch support. The
example below assumes an OpenAI-style Files/Batch API under `LLM_BASE_URL`
(including `/v1`) and a configured `LLM_MODEL` that supports batching. Adapt
submission and retrieval to your provider's API. See the
[Batch API guide](https://developers.openai.com/api/docs/guides/batch) for the
JSONL envelope and job lifecycle.

Run these snippets in the same Python session, with the library configuration
set before importing it. This example sends two different story texts; replace
`stories` with all 20 stories to process them together:

```python
import json

from niquests import Session

from llm_bot.client import LLMClient
from llm_bot.schemas import NerRequest
from llm_bot.tasks.ner import prepare_ner

stories = {
    "story-1": "APT29 used Mimikatz against government systems in Vienna.",
    "story-2": "Microsoft warned that Emotet targeted hospitals in Berlin.",
}
client = LLMClient(api_mode="chat_completions")
endpoint = "/v1/chat/completions"
tasks = {
    story_id: prepare_ner(NerRequest(text=text, cybersecurity=True))
    for story_id, text in stories.items()
}
batch_input = "".join(
    json.dumps({
        "custom_id": story_id,
        "method": "POST",
        "url": endpoint,
        "body": task.build_request(client),
    }) + "\n"
    for story_id, task in tasks.items()
)

session = Session()
if client.api_key:
    session.headers["Authorization"] = f"Bearer {client.api_key}"
upload = session.post(
    f"{client.base_url}/files",
    data={"purpose": "batch"},
    files={"file": ("ner-batch.jsonl", batch_input, "application/jsonl")},
    timeout=client.timeout,
)
upload.raise_for_status()
submission = session.post(
    f"{client.base_url}/batches",
    json={"input_file_id": upload.json()["id"], "endpoint": endpoint, "completion_window": "24h"},
    timeout=client.timeout,
)
submission.raise_for_status()
batch_id = submission.json()["id"]
print(batch_id)
```

The upload contains one JSONL line per story. The `/batches` request schedules
all of them as one asynchronous job; it does not return entities immediately.
Check its status later, and rerun this block while the job is pending:

```python
status = session.get(f"{client.base_url}/batches/{batch_id}", timeout=client.timeout)
status.raise_for_status()
batch = status.json()
print(batch["status"])
```

Once the status is `completed`, download and parse the results using the
original prepared tasks. Results can arrive in any order, so match them by
`custom_id`, not by line position:

```python
output = session.get(f"{client.base_url}/files/{batch['output_file_id']}/content", timeout=client.timeout)
output.raise_for_status()
entities_by_story = {}
for line in output.text.splitlines():
    item = json.loads(line)
    response = item.get("response")
    if item.get("error") or not response or response["status_code"] != 200:
        raise RuntimeError(f"Batch request failed for {item['custom_id']}: {item}")
    story_id = item["custom_id"]
    entities_by_story[story_id] = tasks[story_id].parse_result(response["body"], client).model_dump()
print(entities_by_story)
session.close()
```

For example, the result might include `{"story-1": {"APT29": "GROUP",
"Mimikatz": "TOOL", "Vienna": "GPE"}, "story-2": {"Microsoft": "ORG",
"Emotet": "MALWARE", "Berlin": "GPE"}}`. Inspect the job's `error_file_id`
for failed requests and verify that every input story has a result before
saving them. Failed or expired jobs may have only partial results. Keep the
original story inputs so tasks can be recreated if your process restarts.
`parse_result()` validates model output locally and raises on invalid output;
it does not send a repair request or recover truncated NER output.
