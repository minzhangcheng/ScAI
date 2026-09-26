"""Focused contracts for the PI-directed direction-literature coordinator."""

from unittest.mock import MagicMock

import pytest

from backend.agents.literature.direction_novelty import (
    PI_DIRECTION_SEARCH_POLICY_VERSION,
)
from backend.agents.literature_agent import LiteratureAgent
from backend.models.schemas import (
    AnalysisState,
    DirectionLiteratureExtraction,
)
from backend.utils.llm_client import UpstreamExhaustedError


def _agent() -> LiteratureAgent:
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.1
    provider.get_llm_client.return_value = MagicMock()
    provider.get_config_revision.return_value = 0
    provider.get_context_window.return_value = 8192
    provider.get_vision_setting.return_value = False
    agent = LiteratureAgent(settings_provider=provider)
    agent.set_session("direction-test")
    return agent


def _task(direction_id="d1", task_id="t1"):
    return {
        "task_id": task_id,
        "direction_id": direction_id,
        "evidence_question": "Does the proposed link have prior evidence?",
        "concept_groups": [
            {"group_id": "mechanism", "terms": [
                {"text": "Gene A", "field": "tiab"},
                {"text": "Gene B", "field": "title"},
            ]},
            {"group_id": "context", "terms": [{"text": "liver", "field": "tiab"}]},
        ],
        "exclusions": [{
            "exclusion_id": "excluded",
            "terms": [{"text": "review", "field": "pt"}],
            "rationale": "exclude reviews",
        }],
    }


@pytest.mark.asyncio
async def test_one_pi_followup_keeps_raw_inventory_and_links_tasks(monkeypatch):
    agent = _agent()
    calls = []

    async def search(query, max_results=20):
        calls.append(query)
        return [{"pmid": "1", "title": "Study", "abstract": "Exact finding passage."}]

    async def extract(candidate, task, articles):
        return [DirectionLiteratureExtraction(
            pmid="1", task_ids=[task.task_id], finding="Finding",
            source_passage="Exact finding passage.", provisional_classification="linked",
        )], []

    review_calls = []

    async def review(state, packet, phase="exploration"):
        review_calls.append(packet)
        if len(review_calls) == 1:
            return {"reviews": [{
                "direction_id": "d1", "action": "search_more", "rationale": "Test alternative",
                "follow_up_tasks": [_task(task_id="t2")],
            }]}
        return {"reviews": [{
            "direction_id": "d1", "action": "finish", "rationale": "Reviewed",
            "assessment_status": "assessed", "cited_article_ids": ["1"],
            "final_scientific_classifications": {"literature_novelty": "unknown"},
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    monkeypatch.setattr(agent, "_extract_direction_search_articles", extract)
    state = AnalysisState(user_id="u", initial_instruction="objective")
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}], state,
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
        pi_review_fn=review,
    )
    assert len(calls) == 2
    assert len(review_calls) == 2
    assert result[0]["assessment_status"] == "assessed"
    assert result[0]["article_inventory"][0]["task_ids"] == ["t1", "t2"]
    assert {item["task_id"] for item in result[0]["task_executions"]} == {"t1", "t2"}
    assert result[0]["policy_version"] == PI_DIRECTION_SEARCH_POLICY_VERSION
    assert result[0]["literature_novelty"] == "unknown"


@pytest.mark.parametrize(
    ("articles", "classification", "pi_novelty"),
    [
        ([], "unknown", "unknown"),
        ([{"pmid": "11", "title": "Background", "abstract": "Background context."}], "background", "unknown"),
        ([{"pmid": "12", "title": "Mixed", "abstract": "Mixed direct and conflicting context."}], "conflicting", "low"),
    ],
    ids=["narrow-zero", "broadened-background", "mixed-evidence"],
)
@pytest.mark.asyncio
async def test_offline_replay_patterns_keep_pi_scientific_ownership(
    monkeypatch, articles, classification, pi_novelty
):
    agent = _agent()

    async def search(query, max_results=20):
        return list(articles)

    async def extract(candidate, task, raw_articles):
        if not raw_articles:
            return [], []
        return [DirectionLiteratureExtraction(
            pmid=raw_articles[0]["pmid"],
            task_ids=[task.task_id],
            finding="Replay observation",
            source_passage=raw_articles[0]["abstract"],
            provisional_classification=classification,
        )], []

    async def review(state, packet, phase="exploration"):
        cited = [item["pmid"] for item in packet["article_index"]]
        return {"reviews": [{
            "direction_id": "d1",
            "action": "finish",
            "assessment_status": "uncertain" if pi_novelty == "unknown" else "assessed",
            "cited_article_ids": cited,
            "final_scientific_classifications": {"literature_novelty": pi_novelty},
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    monkeypatch.setattr(agent, "_extract_direction_search_articles", extract)
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}],
        AnalysisState(user_id="u"),
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
        pi_review_fn=review,
    )

    assert result[0]["literature_novelty"] == pi_novelty
    assert result[0]["negative_validation_confidence"] == "none"
    assert result[0]["negative_validation_axes"] == {}
    if articles and pi_novelty == "unknown":
        assert result[0]["article_inventory"]
        assert result[0]["literature_novelty"] == "unknown"


@pytest.mark.parametrize("interrupt_batch", ["initial", "followup"])
@pytest.mark.asyncio
async def test_serialized_partial_extraction_resumes_from_raw_without_cache(
    monkeypatch, interrupt_batch
):
    agent = _agent()
    state = AnalysisState(user_id="u")
    candidates = [{"direction_id": "d1", "direction": "Direction"}]
    searched = []
    extracted = []
    review_calls = 0
    interrupt_task = "t1" if interrupt_batch == "initial" else "t3"
    interrupted = False

    async def search(query, max_results=20):
        searched.append(query)
        return [{"pmid": "1", "title": "Study", "abstract": "Retained raw passage."}] * 2

    async def extract(candidate, task, articles):
        nonlocal interrupted
        extracted.append(task.task_id)
        assert articles[0]["abstract"] == "Retained raw passage."
        if task.task_id == interrupt_task and not interrupted:
            interrupted = True
            raise UpstreamExhaustedError("provider exhausted", agent_name="literature")
        return [DirectionLiteratureExtraction(
            pmid="1", task_ids=[task.task_id], source_passage="Retained raw passage.",
        )], []

    async def review(state, packet, phase="exploration"):
        nonlocal review_calls
        review_calls += 1
        if interrupt_batch == "followup" and review_calls == 1:
            return {"reviews": [{
                "direction_id": "d1", "action": "search_more",
                "follow_up_tasks": [_task(task_id="t2"), _task(task_id="t3")],
            }]}
        return {"reviews": [{
            "direction_id": "d1", "action": "finish", "assessment_status": "assessed",
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    monkeypatch.setattr(agent, "_extract_direction_search_articles", extract)
    with pytest.raises(UpstreamExhaustedError):
        await agent.evaluate_direction_literature_search(
            candidates, state,
            pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
            pi_review_fn=review,
        )
    state.literature_search_cache.clear()
    restored = AnalysisState.model_validate_json(state.model_dump_json())
    result = await agent.evaluate_direction_literature_search(candidates, restored, pi_review_fn=review)
    assert len(searched) == (1 if interrupt_batch == "initial" else 3)
    assert extracted == (["t1", "t1"] if interrupt_batch == "initial" else ["t1", "t2", "t3", "t3"])
    assert review_calls == (1 if interrupt_batch == "initial" else 2)
    assert result[0]["assessment_status"] == "assessed"
    executions = result[0]["task_executions"]
    assert all(item["status"] == "completed" for item in executions)
    assert all(item["returned_count"] == 2 for item in executions)
    assert all(item["duplicate_count"] == 1 for item in executions)
    assert all(item["retrieved_pmids"] == ["1", "1"] for item in executions)
    assert len({item["task_id"] for item in executions}) == len(executions)
    assert executions[-1]["batch"] == interrupt_batch


@pytest.mark.parametrize(
    "passage,expected_status",
    [
        ("Activity (p less than 0.01) decreased", "verified"),
        ("Activity decreased", "unverified"),
        ("Activity (p less than 0.01) decreased.", "unverified"),
    ],
)
@pytest.mark.asyncio
async def test_quote_mismatch_keeps_finding_and_shows_source_to_pi(monkeypatch, passage, expected_status):
    import json

    agent = _agent()
    abstract = "Activity (p less than 0.01) decreased, with further context."
    packets = []

    async def search(query, max_results=20):
        return [{"pmid": "1", "title": "Study", "abstract": abstract}]

    async def chat(*args, **kwargs):
        return json.dumps({"items": [
            {"pmid": "1", "source_passage": passage, "finding": "Reported activity change", "provisional_classification": "linked"},
            {"pmid": "999", "source_passage": abstract, "finding": "Unknown article must not enter evidence"},
        ]})

    async def review(state, packet, phase="exploration"):
        packets.append(packet)
        return {"reviews": [{"direction_id": "d1", "action": "finish", "assessment_status": "assessed"}]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    monkeypatch.setattr(agent, "chat_without_history", chat)
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}], AnalysisState(user_id="u"),
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]}, pi_review_fn=review,
    )
    extracted = result[0]["extractions"]
    assert len(extracted) == 1
    assert extracted[0]["pmid"] == "1"
    assert extracted[0]["finding"] == "Reported activity change"
    assert extracted[0]["source_passage"] == passage
    assert extracted[0]["source_passage_status"] == expected_status
    assert packets[0]["candidates"][0]["extractions"] == extracted
    if expected_status == "unverified":
        assert "not found" in extracted[0]["limitation"]
        assert packets[0]["requested_article_details"][0]["abstract"] == abstract
    else:
        assert packets[0]["requested_article_details"] == []
    assert any("999" in text for text in result[0]["limitations"])
    assert result[0]["assessment_status"] == "assessed"
