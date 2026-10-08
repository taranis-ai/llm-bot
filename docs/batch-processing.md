# Batch processing

[Back to README](../README.md) · [Configuration](configuration.md)

## NER for multiple stories

For 20 Taranis stories, prepare **one NER task per story** and submit all 20
tasks as one provider batch job. Each story has its own text and entity result.
This is available through the Python library; `POST /ner` still accepts only
one text, and llm-bot does not expose a batch HTTP endpoint or manage batch jobs.

This is the workflow described in the
[OpenRouter Batch API quickstart](https://openrouter.ai/docs/batch-quickstart):
submit requests together, then retrieve results asynchronously. The library's
`build_request()` supplies each request body; `parse_result()` validates each
returned body. Your application owns submission, polling, and story IDs.

The example below uses OpenRouter's inline `requests` array and inline results.
Set `LLM_API_KEY` to your OpenRouter key and `LLM_MODEL` to a model slug with
batch support. The selected provider must also accept the NER schema, which
uses dynamic entity-name keys. OpenAI's Files/Batch API uses a different
submission and retrieval format; adapt these steps when using another provider.

Verified on 2026-10-09 with `deepseek/deepseek-v4.1-flash:batch`: one live
batch of 20 distinct synthetic story texts completed successfully, and all
20 results passed the NER parser with the expected entities and types.

Run these snippets in the same Python session, with the library configuration
set before importing it. This example sends two different story texts; replace
`stories` with all 20 stories to process them together:

```python
from niquests import Session

from llm_bot.client import LLMClient
from llm_bot.schemas import NerRequest
from llm_bot.tasks.ner import prepare_ner

stories = {
    "story-1": "APT29 used Mimikatz against government systems in Vienna.",
    "story-2": "Microsoft warned that Emotet targeted hospitals in Berlin.",
}
client = LLMClient(base_url="https://openrouter.ai/api/v1", api_mode="chat_completions")
if not client.model:
    raise ValueError("Set LLM_MODEL to an OpenRouter model slug with batch support")
endpoint = "/v1/chat/completions"
tasks = {story_id: prepare_ner(NerRequest(text=text, cybersecurity=True)) for story_id, text in stories.items()}
session = Session()
if client.api_key:
    session.headers["Authorization"] = f"Bearer {client.api_key}"
submission = session.post(
    f"{client.base_url}/batches",
    json={
        "endpoint": endpoint,
        "model": client.model,
        "completion_window": "24h",
        "requests": [{"custom_id": story_id, "body": task.build_request(client)} for story_id, task in tasks.items()],
    },
    timeout=client.timeout,
)
submission.raise_for_status()
batch_id = submission.json()["id"]
print(batch_id)
```

This sends all story texts in one submission. Keep `requests` after the batch
settings in the JSON object, as OpenRouter requires. Check the job later:

```python
status = session.get(f"{client.base_url}/batches/{batch_id}", timeout=client.timeout)
status.raise_for_status()
batch = status.json()
print(batch["status"])
```

Rerun the status check until the job reaches a terminal status. Once it is
`completed`, parse `batch["results"]` with the original prepared tasks and
match stories by `custom_id`:

```python
if batch["status"] != "completed":
    raise RuntimeError(f"Batch is not complete: {batch['status']}; error: {batch.get('error')}")
entities_by_story = {}
for item in batch["results"]:
    response = item.get("response")
    if item.get("error") or not response or response["status_code"] != 200:
        raise RuntimeError(f"Batch request failed for {item['custom_id']}: {item}")
    story_id = item["custom_id"]
    entities_by_story[story_id] = tasks[story_id].parse_result(response["body"], client).model_dump()
if entities_by_story.keys() != tasks.keys():
    raise RuntimeError("Batch did not return results for every story")
print(entities_by_story)
session.close()
```

For example, the result might include `{"story-1": {"APT29": "GROUP",
"Mimikatz": "TOOL", "Vienna": "GPE"}, "story-2": {"Microsoft": "ORG",
"Emotet": "MALWARE", "Berlin": "GPE"}}`. Handle job-level and per-request
errors before saving results. Keep the original story inputs so tasks can be
recreated if your process restarts.
`parse_result()` validates model output locally and raises on invalid output;
it does not send a repair request or recover truncated NER output.
