from llm_bot.schemas import SummarizeResponse


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


async def test_summarize_endpoint(app, monkeypatch):
    async def fake_summarize(request_model):
        assert request_model.text == "Story text"
        assert request_model.max_words == 25
        return SummarizeResponse(summary="Condensed summary")

    monkeypatch.setattr("app.summarize", fake_summarize)

    test_client = app.test_client()
    response = await test_client.post("/summarize", json={"text": "Story text", "max_words": 25})
    body = await response.get_json()

    assert response.status_code == 200
    assert body == {"summary": "Condensed summary"}
