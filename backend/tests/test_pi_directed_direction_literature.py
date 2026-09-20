"""Focused contracts for the PI-directed direction-literature coordinator."""

import asyncio
from unittest.mock import MagicMock

import pytest

from backend.agents.literature.direction_novelty import (
    PI_DIRECTION_SEARCH_POLICY_VERSION,
)
from backend.agents.literature.query_builders import (
    DirectionSearchCompilationError,
    compile_direction_search_task,
)
from backend.agents.literature_agent import LiteratureAgent
from backend.models.schemas import (
    AnalysisState,
    DirectionLiteratureEvidence,
    DirectionLiteratureExtraction,
    DirectionSearchTask,
    DirectionSelectionRecord,
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


def test_compiler_preserves_pi_relationships_and_does_not_add_scope():
    query = compile_direction_search_task(_task(), candidate_ids={"d1"})
    assert query == '("Gene A"[tiab] OR "Gene B"[title]) AND "liver"[tiab] AND NOT "review"[pt]'
    assert "organism" not in query
    assert "mouse" not in query


@pytest.mark.parametrize(
    ("response", "expected_fragment"),
    [
        ("not json", "not valid JSON"),
        ('{"unexpected": []}', "omitted the items field"),
        ('{"items": []}', "returned no items"),
    ],
)
@pytest.mark.asyncio
async def test_malformed_or_empty_extraction_is_partial_and_retains_inventory(
    monkeypatch, response, expected_fragment
):
    agent = _agent()

    async def search(query, max_results=20):
        return [{"pmid": "10", "title": "Retained", "abstract": "Source text."}]

    async def chat(prompt, temperature=0.1):
        return response

    async def review(state, packet, phase="exploration"):
        return {"reviews": [{
            "direction_id": "d1", "action": "finish", "assessment_status": "uncertain"
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    monkeypatch.setattr(agent, "chat_without_history", chat)
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}],
        AnalysisState(user_id="u"),
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
        pi_review_fn=review,
    )

    assert result[0]["article_inventory"][0]["pmid"] == "10"
    assert result[0]["task_executions"][0]["status"] == "partial"
    assert any(expected_fragment in item for item in result[0]["assessment_limitations"])


@pytest.mark.asyncio
async def test_planning_receives_only_relevant_prior_direction_search_records(monkeypatch):
    agent = _agent()
    captured = []

    async def plan(state, candidates, prior_records=None, phase="exploration"):
        captured.append(prior_records)
        return {"tasks": [_task()]}

    async def review(state, packet, phase="exploration"):
        return {"reviews": [{
            "direction_id": "d1", "action": "finish", "assessment_status": "uncertain"
        }]}

    async def search(query, max_results=20):
        return []

    monkeypatch.setattr(agent, "_pubmed_search", search)
    empty_state = AnalysisState(user_id="u")
    await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Current direction"}], empty_state,
        pi_plan_fn=plan, pi_review_fn=review,
    )
    assert captured.pop() == []

    state = AnalysisState(
        user_id="u",
        direction_selection_records=[
            DirectionSelectionRecord(
                round_number=1,
                phase="exploration",
                candidates=[
                    {"direction_id": "d1", "direction": "Different reused ID"},
                    {"direction_id": "d2", "direction": "Unrelated"},
                ],
                literature_evidence=[{
                    "direction_id": "d1", "direction": "Different reused ID",
                    "queries": [{"category": "old", "query": "wrong"}],
                    "limitations": ["wrong record"],
                }],
            ),
            DirectionSelectionRecord(
                round_number=2,
                phase="integration",
                integration_cycle=1,
                candidates=[{"direction_id": "d1", "direction": "Current direction"}],
                literature_evidence=[{
                    "direction_id": "d1", "direction": "Current direction",
                    "policy_version": PI_DIRECTION_SEARCH_POLICY_VERSION,
                    "queries": [{"category": "pi_directed", "query": "retained query"}],
                    "task_executions": [{"task_id": "old-task", "status": "timeout"}],
                    "limitations": ["retained limitation"],
                }],
            ),
        ],
    )
    await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Current direction"}], state,
        pi_plan_fn=plan, pi_review_fn=review,
    )
    prior = captured.pop()
    assert len(prior) == 1
    assert prior[0]["phase"] == "integration"
    evidence = prior[0]["literature_evidence"][0]
    assert evidence["queries"][0]["query"] == "retained query"
    assert evidence["task_executions"][0]["status"] == "timeout"
    assert evidence["limitations"] == ["retained limitation"]
    assert "wrong record" not in str(prior)


@pytest.mark.parametrize("mismatch", ["policy", "plan"])
@pytest.mark.asyncio
async def test_stale_direction_search_cache_provenance_is_not_reused(monkeypatch, mismatch):
    agent = _agent()
    task = DirectionSearchTask.model_validate(_task())
    query = compile_direction_search_task(task, candidate_ids={"d1"})
    plan_hash = agent._direction_plan_hash([task])
    key = (
        f"direction_search:{PI_DIRECTION_SEARCH_POLICY_VERSION}:{plan_hash}:d1:t1:"
        + __import__("hashlib").sha256(query.encode()).hexdigest()
    )
    cached = {
        "policy_version": PI_DIRECTION_SEARCH_POLICY_VERSION,
        "plan_hash": plan_hash,
        "candidate_question": task.evidence_question,
        "compiled_query": query,
        "result": [{"pmid": "stale", "title": "Stale", "abstract": "Stale."}],
    }
    cached["policy_version" if mismatch == "policy" else "plan_hash"] = "mismatch"
    state = AnalysisState(user_id="u", literature_search_cache={key: cached})
    searches = []

    async def search(compiled_query, max_results=20):
        searches.append(compiled_query)
        return []

    monkeypatch.setattr(agent, "_pubmed_search", search)
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}], state,
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
        pi_review_fn=lambda *args, **kwargs: {"reviews": [{
            "direction_id": "d1", "action": "finish", "assessment_status": "uncertain"
        }]},
    )

    assert searches == [query]
    assert result[0]["article_inventory"] == []


@pytest.mark.parametrize("term", ["A OR B", "A AND B", "A [tiab]", 'A "quoted"'])
def test_compiler_rejects_embedded_control_syntax_without_dispatch(term):
    task = _task()
    task["concept_groups"][0]["terms"] = [{"text": term}]
    with pytest.raises(DirectionSearchCompilationError):
        compile_direction_search_task(task, candidate_ids={"d1"})


@pytest.mark.asyncio
async def test_zero_result_reaches_pi_and_remains_unknown(monkeypatch):
    agent = _agent()
    searched = []

    async def search(query, max_results=20):
        searched.append(query)
        return []

    async def plan(state, candidates, prior_records=None, phase="exploration"):
        return {"tasks": [_task()]}

    seen_packets = []

    async def review(state, packet, phase="exploration"):
        seen_packets.append(packet)
        return {"reviews": [{
            "direction_id": "d1", "action": "finish", "rationale": "No result is unresolved.",
            "assessment_status": "uncertain", "uncertainty": ["zero-result task"],
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    state = AnalysisState(user_id="u", initial_instruction="objective")
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}], state,
        pi_plan_fn=plan, pi_review_fn=review,
    )
    assert searched
    assert result[0]["task_executions"][0]["status"] == "zero_result"
    assert result[0]["assessment_status"] == "uncertain"
    assert result[0]["literature_novelty"] == "unknown"
    assert result[0]["assessment_uncertainty"] == ["zero-result task"]
    assert seen_packets[0]["candidates"][0]["tasks"][0]["returned_count"] == 0
    DirectionLiteratureEvidence.model_validate(result[0])


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


@pytest.mark.asyncio
async def test_invalid_plan_keeps_candidate_and_does_not_dispatch(monkeypatch):
    agent = _agent()
    searched = []

    async def search(query, max_results=20):
        searched.append(query)
        return [{"pmid": "9", "title": "Should not run", "abstract": ""}]

    async def review(state, packet, phase="exploration"):
        return {"reviews": [{"direction_id": "d1", "action": "finish", "assessment_status": "unavailable"}]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    invalid = _task()
    invalid["concept_groups"][0]["terms"] = [{"text": "bad OR syntax"}]
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}], AnalysisState(user_id="u"),
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [invalid]},
        pi_review_fn=review,
    )
    assert searched == []
    assert result[0]["unexecuted_task_ids"] == ["t1"]
    assert result[0]["article_inventory"] == []


@pytest.mark.asyncio
async def test_pi_named_detail_request_receives_only_requested_raw_article(monkeypatch):
    agent = _agent()

    async def search(query, max_results=20):
        return [{
            "pmid": "1",
            "title": "Retained study",
            "abstract": "The complete retained abstract.",
        }]

    async def extract(candidate, task, articles):
        return [DirectionLiteratureExtraction(
            pmid="1",
            task_ids=[task.task_id],
            finding="Bounded observation",
            source_passage="complete retained abstract",
            provisional_classification="background",
        )], []

    packets = []

    async def review(state, packet, phase="exploration"):
        packets.append(packet)
        if len(packets) == 1:
            return {"reviews": [{
                "direction_id": "d1",
                "action": "finish",
                "requested_detail_pmids": ["1"],
                "assessment_status": "unassessed",
            }]}
        return {"reviews": [{
            "direction_id": "d1",
            "action": "finish",
            "assessment_status": "uncertain",
            "cited_article_ids": ["1"],
            "uncertainty": ["bounded search"],
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    monkeypatch.setattr(agent, "_extract_direction_search_articles", extract)
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}],
        AnalysisState(user_id="u"),
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
        pi_review_fn=review,
    )

    assert len(packets) == 2
    assert "article_inventory" not in packets[0]
    assert packets[0]["article_index"][0]["pmid"] == "1"
    assert "abstract" not in packets[0]["article_index"][0]
    assert packets[0]["requested_article_details"] == []
    assert [item["pmid"] for item in packets[1]["requested_article_details"]] == ["1"]
    assert packets[1]["requested_article_details"][0]["abstract"] == "The complete retained abstract."
    assert result[0]["assessment_status"] == "uncertain"
    assert len(result[0]["pi_reviews"]) == 2


@pytest.mark.asyncio
async def test_upstream_retry_resumes_incomplete_extraction_without_retrieval(monkeypatch):
    agent = _agent()
    search_calls = 0
    extraction_calls = 0

    async def search(query, max_results=20):
        nonlocal search_calls
        search_calls += 1
        return [{"pmid": "1", "title": "Study", "abstract": "Exact passage."}]

    async def extract(candidate, task, articles):
        nonlocal extraction_calls
        extraction_calls += 1
        if extraction_calls == 1:
            raise UpstreamExhaustedError("stream incomplete", agent_name="literature")
        return [DirectionLiteratureExtraction(
            pmid="1",
            task_ids=[task.task_id],
            source_passage="Exact passage.",
        )], []

    async def intervene(error, node, state):
        await asyncio.sleep(0.08)
        return "retry"

    async def review(state, packet, phase="exploration"):
        return {"reviews": [{
            "direction_id": "d1", "action": "finish",
            "assessment_status": "assessed", "cited_article_ids": ["1"],
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    monkeypatch.setattr(agent, "_extract_direction_search_articles", extract)
    state = AnalysisState(user_id="u")
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}],
        state,
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
        pi_review_fn=review,
        intervention_fn=intervene,
        direction_timeout_seconds=0.06,
        query_timeout_seconds=0.02,
    )

    assert search_calls == 1
    assert extraction_calls == 2
    assert result[0]["assessment_status"] == "assessed"
    checkpoint = next(iter(state.direction_literature_in_progress.values()))
    assert checkpoint["task_executions"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_concurrent_upstream_failure_settles_sibling_before_skip(monkeypatch):
    agent = _agent()
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def search(query, max_results=20):
        pmid = "1" if "Gene A" in query else "2"
        return [{"pmid": pmid, "title": "Study", "abstract": "Abstract."}]

    async def extract(candidate, task, articles):
        if task.direction_id == "d1":
            await sibling_started.wait()
            raise UpstreamExhaustedError("provider exhausted", agent_name="literature")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise

    async def intervene(error, node, state):
        assert sibling_cancelled.is_set()
        return "skip"

    task2 = _task(direction_id="d2", task_id="t2")
    task2["concept_groups"][0]["terms"] = [{"text": "Gene C"}]
    monkeypatch.setattr(agent, "_pubmed_search", search)
    monkeypatch.setattr(agent, "_extract_direction_search_articles", extract)
    result = await agent.evaluate_direction_literature_search(
        [
            {"direction_id": "d1", "direction": "Direction 1"},
            {"direction_id": "d2", "direction": "Direction 2"},
        ],
        AnalysisState(user_id="u"),
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task(), task2]},
        pi_review_fn=lambda *args, **kwargs: pytest.fail("review must not run after skip"),
        intervention_fn=intervene,
    )

    assert sibling_cancelled.is_set()
    assert [item["direction_id"] for item in result] == ["d1", "d2"]
    assert all(item["assessment_status"] == "unavailable" for item in result)
    assert all(item["article_inventory"] for item in result)


@pytest.mark.asyncio
async def test_followup_and_final_review_share_original_deadline(monkeypatch):
    agent = _agent()
    review_calls = 0

    async def search(query, max_results=20):
        await asyncio.sleep(0.01)
        return []

    async def review(state, packet, phase="exploration"):
        nonlocal review_calls
        review_calls += 1
        if review_calls == 1:
            await asyncio.sleep(0.01)
            return {"reviews": [{
                "direction_id": "d1", "action": "search_more",
                "follow_up_tasks": [_task(task_id="t2")],
            }]}
        await asyncio.sleep(0.2)
        return {"reviews": [{
            "direction_id": "d1", "action": "finish", "assessment_status": "assessed",
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}],
        AnalysisState(user_id="u"),
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
        pi_review_fn=review,
        direction_timeout_seconds=0.16,
        query_timeout_seconds=0.03,
    )

    assert review_calls == 2
    assert {item["task_id"] for item in result[0]["task_executions"]} == {"t1", "t2"}
    assert result[0]["assessment_status"] == "unavailable"
    assert any("shared candidate deadline" in item for item in result[0]["assessment_limitations"])


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


def test_gate_identity_changes_when_candidate_question_changes():
    agent = _agent()
    state = AnalysisState(user_id="u", current_round=1)
    first = agent._direction_gate_id(
        [{"direction_id": "d1", "direction": "Direction", "main_hypothesis": "A"}],
        state,
        "exploration",
    )
    second = agent._direction_gate_id(
        [{"direction_id": "d1", "direction": "Direction", "main_hypothesis": "B"}],
        state,
        "exploration",
    )
    assert first != second


@pytest.mark.asyncio
async def test_short_window_candidate_does_not_cap_full_candidate_pi_review(monkeypatch):
    agent = _agent()
    agent.MIN_ATTEMPT_DIRECTION_TIMEOUT = 0.05
    candidates = [
        {"direction_id": f"d{index}", "direction": f"Direction {index}"}
        for index in range(1, 10)
    ]
    tasks = [
        _task(direction_id=candidate["direction_id"], task_id=f"t{index}")
        for index, candidate in enumerate(candidates, start=1)
    ]
    review_batches = []

    async def execute(*args, **kwargs):
        return None

    async def review(state, packet, phase="exploration"):
        ids = [item["direction_id"] for item in packet["candidates"]]
        review_batches.append(ids)
        await asyncio.sleep(0.03 if ids == ["d9"] else 0.08)
        return {"reviews": [
            {
                "direction_id": direction_id,
                "action": "finish",
                "assessment_status": "assessed",
            }
            for direction_id in ids
        ]}

    monkeypatch.setattr(agent, "_execute_direction_task", execute)
    result = await agent.evaluate_direction_literature_search(
        candidates,
        AnalysisState(user_id="u"),
        pi_plan_fn=lambda *args, **kwargs: {"tasks": tasks},
        pi_review_fn=review,
        direction_timeout_seconds=0.2,
        query_timeout_seconds=0.01,
        max_full_evidence_candidates=8,
    )

    assert review_batches[0] == ["d9"]
    assert set(review_batches[1]) == {f"d{index}" for index in range(1, 9)}
    assert all(item["assessment_status"] == "assessed" for item in result)


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
@pytest.mark.asyncio
async def test_serialized_finished_followup_survives_expired_deadline_without_replay(
    monkeypatch, legacy_checkpoint
):
    agent = _agent()
    state = AnalysisState(user_id="u")
    candidates = [{"direction_id": "d1", "direction": "Direction"}]
    search_calls = 0
    review_calls = 0

    async def search(query, max_results=20):
        nonlocal search_calls
        search_calls += 1
        return []

    async def review(state, packet, phase="exploration"):
        nonlocal review_calls
        review_calls += 1
        if review_calls == 1:
            return {"reviews": [{
                "direction_id": "d1", "action": "search_more",
                "follow_up_tasks": [_task(task_id="t2")],
            }]}
        return {"reviews": [{
            "direction_id": "d1", "action": "finish", "assessment_status": "assessed",
            "rationale": "Retained final assessment",
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    first = await agent.evaluate_direction_literature_search(
        candidates, state,
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
        pi_review_fn=review,
    )
    checkpoint = next(iter(state.direction_literature_in_progress.values()))
    checkpoint["candidate_deadline_epoch"]["d1"] = 1.0
    if legacy_checkpoint:
        checkpoint.pop("task_batches")
        checkpoint.pop("provided_detail_pmids_by_candidate")
    restored = AnalysisState.model_validate_json(state.model_dump_json())
    result = await agent.evaluate_direction_literature_search(
        candidates, restored,
        pi_plan_fn=lambda *args, **kwargs: pytest.fail("must not replan"),
        pi_review_fn=review,
    )
    assert search_calls == 2
    assert review_calls == 2
    assert result[0]["assessment_status"] == "assessed"
    assert result[0]["summary"] == "Retained final assessment"
    assert result[0]["task_executions"] == first[0]["task_executions"]
    assert result[0]["pi_reviews"] == first[0]["pi_reviews"]
    assert result[0]["assessment_limitations"] == []


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
@pytest.mark.asyncio
async def test_serialized_second_followup_stays_unexecuted(monkeypatch, legacy_checkpoint):
    agent = _agent()
    state = AnalysisState(user_id="u")
    candidates = [{"direction_id": "d1", "direction": "Direction"}]
    search_calls = 0
    review_calls = 0

    async def search(query, max_results=20):
        nonlocal search_calls
        search_calls += 1
        return []

    async def review(state, packet, phase="exploration"):
        nonlocal review_calls
        review_calls += 1
        return {"reviews": [{
            "direction_id": "d1", "action": "search_more",
            "follow_up_tasks": [_task(task_id=f"t{review_calls + 1}")],
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    await agent.evaluate_direction_literature_search(
        candidates, state,
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
        pi_review_fn=review,
    )
    checkpoint = next(iter(state.direction_literature_in_progress.values()))
    if legacy_checkpoint:
        checkpoint.pop("task_batches")
        checkpoint.pop("provided_detail_pmids_by_candidate")
    restored = AnalysisState.model_validate_json(state.model_dump_json())
    result = await agent.evaluate_direction_literature_search(candidates, restored, pi_review_fn=review)
    assert search_calls == 2
    assert review_calls == 2
    assert [(item["task_id"], item["batch"]) for item in result[0]["task_executions"]] == [
        ("t1", "initial"), ("t2", "followup"),
    ]
    assert [item["task_id"] for item in result[0]["tasks"]] == ["t1", "t2", "t3"]
    assert result[0]["unexecuted_task_ids"] == ["t3"]


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


@pytest.mark.asyncio
async def test_six_slow_queries_leave_reserved_time_for_pi_review(monkeypatch):
    from backend.agents.literature import direction_novelty

    agent = _agent()
    search_calls = 0
    review_budgets = []
    monkeypatch.setattr(direction_novelty, "PI_DIRECTION_REVIEW_RESERVE_SECONDS", 0.09)

    async def search(query, max_results=20):
        nonlocal search_calls
        search_calls += 1
        await asyncio.sleep(0.07)
        return []

    async def review(state, packet, phase="exploration"):
        review_budgets.append(packet["candidates"][0]["remaining_budget_seconds"])
        await asyncio.sleep(0.04)
        return {"reviews": [{
            "direction_id": "d1", "action": "finish", "assessment_status": "assessed",
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    result = await agent.evaluate_direction_literature_search(
        [{"direction_id": "d1", "direction": "Direction"}], AnalysisState(user_id="u"),
        pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task(task_id=f"t{i}") for i in range(6)]},
        pi_review_fn=review,
        direction_timeout_seconds=0.42,
        query_timeout_seconds=0.09,
    )
    assert search_calls < 6
    assert len(review_budgets) == 1
    assert review_budgets[0] >= 0.07
    assert result[0]["assessment_status"] == "assessed"
    assert result[0]["unexecuted_task_ids"]


@pytest.mark.asyncio
async def test_serialized_pending_detail_review_resumes_named_sources_once(monkeypatch):
    agent = _agent()
    state = AnalysisState(user_id="u")
    candidates = [{"direction_id": "d1", "direction": "Direction"}]
    packets = []

    async def search(query, max_results=20):
        return [{"pmid": "1", "title": "Study", "abstract": "Retained raw passage."}]

    async def extract(candidate, task, articles):
        return [DirectionLiteratureExtraction(pmid="1", task_ids=[task.task_id])], []

    async def review(state, packet, phase="exploration"):
        packets.append(packet)
        if len(packets) == 2:
            raise UpstreamExhaustedError("provider exhausted", agent_name="pi")
        return {"reviews": [{
            "direction_id": "d1", "action": "finish", "requested_detail_pmids": ["1"],
            "assessment_status": "assessed" if len(packets) > 2 else "unassessed",
        }]}

    monkeypatch.setattr(agent, "_pubmed_search", search)
    monkeypatch.setattr(agent, "_extract_direction_search_articles", extract)
    with pytest.raises(UpstreamExhaustedError):
        await agent.evaluate_direction_literature_search(
            candidates, state,
            pi_plan_fn=lambda *args, **kwargs: {"tasks": [_task()]},
            pi_review_fn=review,
        )
    restored = AnalysisState.model_validate_json(state.model_dump_json())
    result = await agent.evaluate_direction_literature_search(candidates, restored, pi_review_fn=review)
    assert result[0]["assessment_status"] == "assessed"
    assert len(packets) == 3
    assert packets[2]["requested_article_details"][0]["abstract"] == "Retained raw passage."
    checkpoint = next(iter(restored.direction_literature_in_progress.values()))
    checkpoint["candidate_deadline_epoch"]["d1"] = 1.0
    again = await agent.evaluate_direction_literature_search(
        candidates, AnalysisState.model_validate_json(restored.model_dump_json()), pi_review_fn=review,
    )
    assert len(packets) == 3
    assert again[0]["assessment_status"] == "assessed"
