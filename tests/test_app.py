from llm_bot.client import UpstreamLLMError
from llm_bot.schemas import (
    ClusterIds,
    ClusterResponse,
    LinkedNerResponse,
    NerResponse,
    SentimentResponse,
    SummarizeResponse,
    TitleResponse,
    TranslateResponse,
)
from llm_bot.app import create_app
from llm_bot.tasks.entity_linking import UnsupportedLinkingModeError
from llm_bot.tasks.ner import UnsupportedEntityTypesError


async def test_health_endpoint(app):
    test_client = app.test_client()

    response = await test_client.get("/health")
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"status": "ok"}


async def test_info_endpoint(app, monkeypatch):
    monkeypatch.setattr("llm_bot.routes.Config.LLM_REASONING_PROFILE", "gemma")
    monkeypatch.setattr("llm_bot.routes.Config.LOOKUP_BASE_URL", "https://lookup.example")
    monkeypatch.setattr("llm_bot.routes.Config.NER_LINKING_ENABLED", True)

    test_client = app.test_client()
    response = await test_client.get("/info")
    body = await response.get_json()

    assert response.status_code == 200
    assert body["package_name"] == "llm_bot"
    assert body["reasoning_profiles"]["gemma"]["description"] == "Prefixes the system prompt with <|think|>"
    assert body["linking_modes"] == ["deterministic", "llm"]
    assert body["endpoints"]["sentiment"] == "/sentiment"
    assert body["endpoints"]["title"] == "/title"
    assert body["endpoints"]["translate"] == "/translate"
    assert body["current"]["llm_reasoning_profile"] == "gemma"
    assert "llm_reasoning_effort" not in body["current"]
    assert body["current"]["lookup_base_url_configured"] is True
    assert body["current"]["ner_linking_enabled"] is True




async def test_title_endpoint(app, monkeypatch):
    async def fake_generate_title(request_model):
        assert request_model.news_items[0].title == "Story title"
        assert request_model.news_items[0].content == "Story text"
        assert request_model.max_chars == 55
        assert request_model.reasoning_effort == "high"
        assert request_model.thinking_budget_tokens == 128
        return TitleResponse(title="Concise story title")

    monkeypatch.setattr("llm_bot.routes.generate_title", fake_generate_title)

    test_client = app.test_client()
    response = await test_client.post(
        "/title",
        json={
            "news_items": [{"title": "Story title", "content": "Story text"}],
            "max_chars": 55,
            "reasoning_effort": "high",
            "thinking_budget_tokens": 128,
        },
    )
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"title": "Concise story title"}


async def test_title_endpoint_rejects_invalid_payload(app):
    test_client = app.test_client()

    response = await test_client.post("/title", json={})
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Invalid title request payload"}


async def test_translate_endpoint(app, monkeypatch):
    async def fake_translate_text(request_model):
        assert request_model.text == "Guten Morgen"
        assert request_model.target_language == "en"
        assert request_model.source_language == "de"
        return TranslateResponse(translation="Good morning")

    monkeypatch.setattr("llm_bot.routes.translate_text", fake_translate_text)

    test_client = app.test_client()
    response = await test_client.post(
        "/translate",
        json={"text": "Guten Morgen", "target_language": "en", "source_language": "de"},
    )
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"translation": "Good morning"}


async def test_translate_endpoint_rejects_invalid_payload(app):
    test_client = app.test_client()

    response = await test_client.post("/translate", json={"text": "Bonjour"})
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Invalid translate request payload"}


async def test_translate_endpoint_returns_upstream_error(app, monkeypatch):
    async def failing_translate_text(request_model):
        raise UpstreamLLMError("Unsupported parameter: text.format")

    monkeypatch.setattr("llm_bot.routes.translate_text", failing_translate_text)

    test_client = app.test_client()
    response = await test_client.post(
        "/translate",
        json={"text": "Guten Morgen", "target_language": "en"},
    )
    body = await response.get_json()

    assert response.status_code == 502
    assert body == {"error": "Failed to translate text: Unsupported parameter: text.format"}

async def test_api_key_required_rejects_missing_api_key(app, monkeypatch):
    async def fake_translate_text(request_model):
        return TranslateResponse(translation="Good morning")

    monkeypatch.setattr("llm_bot.routes.translate_text", fake_translate_text)
    monkeypatch.setattr("llm_bot.routes.Config.API_KEY", "secret")

    test_client = app.test_client()
    response = await test_client.post(
        "/translate",
        json={"text": "Guten Morgen", "target_language": "en"},
    )
    body = await response.get_json()

    assert response.status_code == 401
    assert body == {"error": "Unauthorized"}


async def test_api_key_required_accepts_valid_api_key(app, monkeypatch):
    async def fake_translate_text(request_model):
        return TranslateResponse(translation="Good morning")

    monkeypatch.setattr("llm_bot.routes.translate_text", fake_translate_text)
    monkeypatch.setattr("llm_bot.routes.Config.API_KEY", "secret")

    test_client = app.test_client()
    response = await test_client.post(
        "/translate",
        json={"text": "Guten Morgen", "target_language": "en"},
        headers={"Authorization": "Bearer secret"},
    )
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"translation": "Good morning"}

async def test_summarize_endpoint(app, monkeypatch):
    async def fake_summarize(request_model):
        assert request_model.news_items[0].title == "Story title"
        assert request_model.news_items[0].content == "Story text"
        assert request_model.max_words == 25
        assert request_model.thinking_budget_tokens == 256
        return SummarizeResponse(summary="Condensed summary")

    monkeypatch.setattr("llm_bot.routes.summarize", fake_summarize)

    test_client = app.test_client()
    response = await test_client.post(
        "/summarize",
        json={
            "news_items": [{"title": "Story title", "content": "Story text"}],
            "max_words": 25,
            "thinking_budget_tokens": 256,
        },
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
        raise UpstreamLLMError("Unsupported parameter: text.format")

    monkeypatch.setattr("llm_bot.routes.summarize", failing_summarize)

    test_client = app.test_client()
    response = await test_client.post(
        "/summarize",
        json={"news_items": [{"title": "Story title", "content": "Story text"}]},
    )
    body = await response.get_json()

    assert response.status_code == 502
    assert body == {"error": "Failed to generate summary: Unsupported parameter: text.format"}


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


async def test_ner_link_endpoint_returns_linked_response_in_deterministic_mode(app, monkeypatch):
    async def fake_extract_and_link(request_model):
        assert request_model.linking_mode == "deterministic"
        return LinkedNerResponse(
            entities=[
                {
                    "mention": "Apple",
                    "type": "ORG",
                    "wikidata_qid": "Q312",
                    "wikidata_label": "Apple Inc.",
                    "wikidata_description": "American technology company",
                    "matched_alias": "Apple",
                    "match_type": "alias",
                    "score": 0.98,
                    "candidate_count": 1,
                }
            ]
        )

    monkeypatch.setattr("llm_bot.routes.extract_and_link", fake_extract_and_link)

    test_client = app.test_client()
    response = await test_client.post(
        "/ner-link",
        json={"text": "Apple released a new device.", "linking_mode": "deterministic"},
    )
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {
        "entities": [
            {
                "mention": "Apple",
                "type": "ORG",
                "wikidata_qid": "Q312",
                "wikidata_label": "Apple Inc.",
                "wikidata_description": "American technology company",
                "matched_alias": "Apple",
                "match_type": "alias",
                "score": 0.98,
                "candidate_count": 1,
            }
        ]
    }


async def test_link_endpoint_returns_linked_response(app, monkeypatch):
    async def fake_link_entities(request_model):
        assert request_model.text == "Apple announced a new device in Cupertino."
        assert request_model.linking_mode == "deterministic"
        return LinkedNerResponse(
            entities=[
                {
                    "mention": "Apple",
                    "type": "ORG",
                    "wikidata_qid": "Q312",
                    "wikidata_label": "Apple Inc.",
                    "wikidata_description": "American technology company",
                    "matched_alias": "Apple",
                    "match_type": "alias",
                    "score": 0.98,
                    "candidate_count": 1,
                },
                {
                    "mention": "Cupertino",
                    "type": "GPE",
                    "wikidata_qid": "Q189471",
                    "wikidata_label": "Cupertino",
                    "wikidata_description": "city in California",
                    "matched_alias": "Cupertino",
                    "match_type": "label",
                    "score": 0.97,
                    "candidate_count": 1,
                },
            ]
        )

    monkeypatch.setattr("llm_bot.routes.link_entities", fake_link_entities)

    test_client = app.test_client()
    response = await test_client.post(
        "/link",
        json={
            "text": "Apple announced a new device in Cupertino.",
            "language": "en",
            "linking_mode": "deterministic",
            "entities": [
                {"mention": "Apple", "type": "ORG"},
                {"mention": "Cupertino", "type": "GPE"},
            ],
        },
    )
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {
        "entities": [
            {
                "mention": "Apple",
                "type": "ORG",
                "wikidata_qid": "Q312",
                "wikidata_label": "Apple Inc.",
                "wikidata_description": "American technology company",
                "matched_alias": "Apple",
                "match_type": "alias",
                "score": 0.98,
                "candidate_count": 1,
            },
            {
                "mention": "Cupertino",
                "type": "GPE",
                "wikidata_qid": "Q189471",
                "wikidata_label": "Cupertino",
                "wikidata_description": "city in California",
                "matched_alias": "Cupertino",
                "match_type": "label",
                "score": 0.97,
                "candidate_count": 1,
            },
        ]
    }


async def test_ner_endpoint_rejects_invalid_payload(app):
    test_client = app.test_client()

    response = await test_client.post("/ner", json={"cybersecurity": True, "linking_mode": "deterministic"})
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Invalid NER request payload"}

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


async def test_ner_link_endpoint_rejects_invalid_linking_mode(app, monkeypatch):
    async def failing_extract_and_link(request_model):
        raise UnsupportedLinkingModeError(
            "Unsupported linking mode requested: magic. Allowed linking modes: deterministic, llm"
        )

    monkeypatch.setattr("llm_bot.routes.extract_and_link", failing_extract_and_link)

    test_client = app.test_client()
    response = await test_client.post(
        "/ner-link",
        json={"text": "Apple released a new device.", "linking_mode": "magic"},
    )
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Unsupported linking mode requested: magic. Allowed linking modes: deterministic, llm"}


async def test_link_endpoint_rejects_invalid_payload(app):
    test_client = app.test_client()

    response = await test_client.post("/link", json={"text": "Apple announced a new device."})
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Invalid link request payload"}

async def test_ner_link_endpoint_rejects_invalid_payload(app):
    test_client = app.test_client()

    response = await test_client.post("/ner-link", json={"cybersecurity": True})
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Invalid NER link request payload"}

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

async def test_sentiment_endpoint(app, monkeypatch):
    async def fake_analyze_sentiment(request_model):
        assert request_model.text == "The launch was a success."
        assert request_model.include_emotions is False
        return SentimentResponse.model_validate({"sentiment": {"label": "positive", "score": 0.88}})

    monkeypatch.setattr("llm_bot.routes.analyze_sentiment", fake_analyze_sentiment)

    test_client = app.test_client()
    response = await test_client.post("/sentiment", json={"text": "The launch was a success."})
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"sentiment": {"label": "positive", "score": 0.88}}


async def test_sentiment_endpoint_rejects_invalid_payload(app):
    test_client = app.test_client()

    response = await test_client.post("/sentiment", json={"include_emotions": True})
    body = await response.get_json()

    assert response.status_code == 400
    assert body == {"error": "Invalid sentiment request payload"}
