from llm_bot.schemas import LinkedNerResponse, NerLinkRequest


def test_ner_link_request_accepts_linking_fields():
    request = NerLinkRequest.model_validate(
        {
            "text": "Apple released a new device.",
            "language": "en",
            "linking_mode": "llm",
        }
    )

    assert request.text == "Apple released a new device."
    assert request.language == "en"
    assert request.linking_mode == "llm"


def test_linked_ner_response_validates_entity_list():
    response = LinkedNerResponse.model_validate(
        {
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
                    "candidate_count": 5,
                }
            ]
        }
    )

    assert response.entities[0].mention == "Apple"
    assert response.entities[0].wikidata_qid == "Q312"
