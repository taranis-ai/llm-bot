from llm_bot.schemas import ClusterIds, ClusterResponse, NerResponse, SummarizeResponse
from llm_bot.app import create_app
from llm_bot.tasks.llm_utils import MissingOutputTextError
from llm_bot.tasks.ner import UnsupportedEntityTypesError


async def test_health_endpoint(app):
    test_client = app.test_client()

    response = await test_client.get("/health")
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"status": "ok"}


async def test_info_endpoint(app):
    test_client = app.test_client()

    response = await test_client.get("/info")
    body = await response.get_json()

    assert response.status_code == 200
    assert body["package_name"] == "llm_bot"
    assert "llm_base_url" in body
    assert "llm_model" in body
    assert body["summary_route_path"] == "/summarize"
    assert body["ner_route_path"] == "/ner"


async def test_summarize_endpoint(app, monkeypatch):
    async def fake_summarize(request_model):
        assert request_model.text == "Story text"
        assert request_model.max_words == 25
        return SummarizeResponse(summary="Condensed summary")

    monkeypatch.setattr("llm_bot.routes.summarize", fake_summarize)

    test_client = app.test_client()
    response = await test_client.post("/summarize", json={"text": "Story text", "max_words": 25})
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"summary": "Condensed summary"}


async def test_summarize_endpoint_rejects_missing_api_key(app, monkeypatch):
    app.config["TESTING"] = True

    async def fake_summarize(request_model):
        return SummarizeResponse(summary="Condensed summary")

    monkeypatch.setattr("llm_bot.routes.summarize", fake_summarize)
    monkeypatch.setattr("llm_bot.routes.Config.API_KEY", "secret")

    test_client = app.test_client()
    response = await test_client.post("/summarize", json={"text": "Story text"})
    body = await response.get_json()

    assert response.status_code == 401
    assert body == {"error": "Unauthorized"}


async def test_summarize_endpoint_accepts_valid_api_key(app, monkeypatch):
    async def fake_summarize(request_model):
        return SummarizeResponse(summary="Condensed summary")

    monkeypatch.setattr("llm_bot.routes.summarize", fake_summarize)
    monkeypatch.setattr("llm_bot.routes.Config.API_KEY", "secret")

    test_client = app.test_client()
    response = await test_client.post(
        "/summarize",
        json={"text": "Story text"},
        headers={"Authorization": "Bearer secret"},
    )
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"summary": "Condensed summary"}


async def test_summarize_endpoint_rejects_invalid_payload(app):
    test_client = app.test_client()

    response = await test_client.post("/summarize", json={"max_words": 25})
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Invalid summarize request payload"}


async def test_summarize_endpoint_returns_upstream_error(app, monkeypatch):
    async def failing_summarize(request_model):
        raise ValueError("malformed upstream response")

    monkeypatch.setattr("llm_bot.routes.summarize", failing_summarize)

    test_client = app.test_client()
    response = await test_client.post("/summarize", json={"text": "Story text"})
    body = await response.get_json()

    assert response.status_code == 502
    assert body == {"error": "Failed to generate summary"}


async def test_ner_endpoint(app, monkeypatch):
    async def fake_extract_entities(request_model):
        assert request_model.text == "APT29 used Mimikatz."
        assert request_model.cybersecurity is True
        return NerResponse({"APT29": "GROUP", "Mimikatz": "TOOL"})

    monkeypatch.setattr("llm_bot.routes.extract_entities", fake_extract_entities)

    test_client = app.test_client()
    response = await test_client.post("/ner", json={"text": "APT29 used Mimikatz.", "cybersecurity": True})
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"APT29": "GROUP", "Mimikatz": "TOOL"}


async def test_ner_endpoint_rejects_invalid_payload(app):
    test_client = app.test_client()

    response = await test_client.post("/ner", json={"cybersecurity": True})
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Invalid NER request payload"}


async def test_ner_endpoint_returns_upstream_error(app, monkeypatch):
    async def failing_extract_entities(request_model):
        raise RuntimeError("malformed upstream response")

    monkeypatch.setattr("llm_bot.routes.extract_entities", failing_extract_entities)

    test_client = app.test_client()
    response = await test_client.post("/ner", json={"text": "APT29 used Mimikatz."})
    body = await response.get_json()

    assert response.status_code == 502
    assert body == {"error": "Failed to extract entities"}


async def test_ner_endpoint_rejects_invalid_entity_types(app, monkeypatch):
    async def failing_extract_entities(request_model):
        raise UnsupportedEntityTypesError("Unsupported entity types requested: AlienType")

    monkeypatch.setattr("llm_bot.routes.extract_entities", failing_extract_entities)

    test_client = app.test_client()
    response = await test_client.post(
        "/ner",
        json={"text": "APT29 used Mimikatz.", "entity_types": ["GROUP", "AlienType"]},
    )
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Unsupported entity types requested: AlienType"}


async def test_ner_endpoint_returns_upstream_error_for_missing_output_text(app, monkeypatch):
    async def failing_extract_entities(request_model):
        raise MissingOutputTextError("Responses API payload did not contain output text")

    monkeypatch.setattr("llm_bot.routes.extract_entities", failing_extract_entities)

    test_client = app.test_client()
    response = await test_client.post("/ner", json={"text": "APT29 used Mimikatz."})
    body = await response.get_json()

    assert response.status_code == 502
    assert body == {"error": "Failed to extract entities"}

async def test_ner_endpoint_path_is_configurable(monkeypatch):
    async def fake_extract_entities(request_model):
        return NerResponse({"APT29": "GROUP"})

    monkeypatch.setattr("llm_bot.routes.extract_entities", fake_extract_entities)
    monkeypatch.setattr("llm_bot.routes.Config.NER_ROUTE_PATH", "/entities")

    app = create_app()
    app.config.update(TESTING=True)
    test_client = app.test_client()

    response = await test_client.post("/entities", json={"text": "APT29 used Mimikatz."})
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"APT29": "GROUP"}

async def test_cluster_endpoint(app, monkeypatch):
    async def fake_cluster_stories(request_model):
        assert len(request_model.stories) == 2
        return ClusterResponse(
            cluster_ids=ClusterIds(event_clusters=[["s1", "s2"]]),
            message="Clustering completed",
        )

    monkeypatch.setattr("llm_bot.routes.cluster_stories", fake_cluster_stories)

    test_client = app.test_client()
    response = await test_client.post(
        "/cluster",
        json={
            "stories": [
                {
                    "id": "s1",
                    "tags": {"APT29": {"tag_type": "APT"}},
                    "news_items": [{"title": "A", "content": "A", "language": "en"}],
                },
                {
                    "id": "s2",
                    "tags": {"APT28": {"tag_type": "APT"}},
                    "news_items": [{"title": "B", "content": "B", "language": "en"}],
                },
            ]
        },
    )
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {
        "cluster_ids": {"event_clusters": [["s1", "s2"]]},
        "message": "Clustering completed",
    }


async def test_cluster_endpoint_rejects_invalid_payload(app):
    test_client = app.test_client()

    response = await test_client.post("/cluster", json={"stories": []})
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Invalid Cluster request payload"}


async def test_cluster_endpoint_returns_upstream_error(app, monkeypatch):
    async def failing_cluster_stories(request_model):
        raise RuntimeError("malformed upstream response")

    monkeypatch.setattr("llm_bot.routes.cluster_stories", failing_cluster_stories)

    test_client = app.test_client()
    response = await test_client.post(
        "/cluster",
        json={
            "stories": [
                {
                    "id": "s1",
                    "tags": {"APT29": {"tag_type": "APT"}},
                    "news_items": [{"title": "A", "content": "A", "language": "en"}],
                }
            ]
        },
    )
    body = await response.get_json()

    assert response.status_code == 502
    assert body == {"error": "Failed to cluster stories"}
