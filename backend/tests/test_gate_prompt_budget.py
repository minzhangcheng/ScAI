"""Every candidate has to reach the model that judges it.

The integration scorer built its prompt with
``json.dumps(raw_candidates)[:8000]``. A normalised candidate serialises to
7-10 KB, so the cut landed inside the first or second entry and every later
candidate was removed from the prompt without a trace. The model cannot keep,
score, or rank an entry it never received, so the truncation silently decided
the outcome for everything past the cut -- and because the list is in generation
order, the same position always survived.

The block below gives every entry a slot and shortens from inside it instead.
Losing a candidate's least decisive fields is recoverable; losing the candidate
is not.
"""
import inspect
import json

import pytest

from backend.agents.pi.direction_candidates import (
    SCORER_CANDIDATE_BUDGET,
    SCORER_EVIDENCE_BUDGET,
)
from backend.utils.direction_payload import (
    CANDIDATE_PROMPT_FIELDS,
    EVIDENCE_PROMPT_FIELDS,
    budgeted_prompt_block,
)


def _fat_candidates(n, size=9000):
    return [
        {
            "direction_id": f"NEW-{i:02d}",
            "direction": f"Direction {i}",
            "core_observation": "o" * size,
            "main_hypothesis": "h" * size,
            "rationale": "r" * size,
        }
        for i in range(1, n + 1)
    ]


def test_every_candidate_appears_however_tight_the_budget():
    candidates = _fat_candidates(6)

    block = budgeted_prompt_block(candidates, CANDIDATE_PROMPT_FIELDS, 500)

    for candidate in candidates:
        assert candidate["direction_id"] in block
        assert candidate["direction"] in block


def test_truncating_the_serialised_list_is_what_loses_candidates():
    # The behaviour this module exists to prevent, stated as a fact about the
    # old expression rather than as a claim about the fix.
    candidates = _fat_candidates(3)

    old_style = json.dumps(candidates, ensure_ascii=False)[:8000]

    assert "NEW-01" in old_style
    assert "NEW-02" not in old_style
    assert "NEW-03" not in old_style


def test_block_is_valid_json_and_keeps_every_entry():
    parsed = json.loads(budgeted_prompt_block(_fat_candidates(5), CANDIDATE_PROMPT_FIELDS, 2000))

    assert len(parsed) == 5
    assert [c["direction_id"] for c in parsed] == [f"NEW-{i:02d}" for i in range(1, 6)]


def test_shortened_entries_say_that_they_were_shortened():
    parsed = json.loads(budgeted_prompt_block(_fat_candidates(4), CANDIDATE_PROMPT_FIELDS, 1200))

    assert any(c.get("fields_omitted_for_prompt_budget") for c in parsed)


def test_a_roomy_budget_keeps_the_decisive_fields_whole():
    candidate = {
        "direction_id": "NEW-01",
        "direction": "D",
        "core_observation": "observation text",
        "key_genes": ["Gene1", "Gene2"],
    }

    parsed = json.loads(budgeted_prompt_block([candidate], CANDIDATE_PROMPT_FIELDS, 4000))

    assert parsed[0]["core_observation"] == "observation text"
    assert parsed[0]["key_genes"] == ["Gene1", "Gene2"]
    assert "fields_omitted_for_prompt_budget" not in parsed[0]


def test_empty_input_renders_an_empty_array():
    assert budgeted_prompt_block([], CANDIDATE_PROMPT_FIELDS, 1000) == "[]"
    assert budgeted_prompt_block(None, CANDIDATE_PROMPT_FIELDS, 1000) == "[]"


def test_non_dict_entries_are_ignored_not_crashed_on():
    block = budgeted_prompt_block(
        ["junk", None, {"direction_id": "NEW-01", "direction": "D"}],
        CANDIDATE_PROMPT_FIELDS,
        1000,
    )

    assert json.loads(block) == [{"direction_id": "NEW-01", "direction": "D"}]


def test_evidence_projection_drops_the_bulk_and_keeps_the_verdict():
    record = {
        "direction_id": "NEW-01",
        "novelty_status": "unknown_due_to_timeout",
        "prior_art_level": "high",
        "direct_evidence_count": 4,
        "timed_out": True,
        # The bulk of a real record, none of which changes a scoring decision.
        "direct_evidence": [{"pmid": str(i), "title": "t" * 400} for i in range(30)],
        "queries": [{"category": "c", "query": "q" * 2000} for _ in range(7)],
    }

    parsed = json.loads(budgeted_prompt_block([record], EVIDENCE_PROMPT_FIELDS, SCORER_EVIDENCE_BUDGET))

    assert parsed[0]["novelty_status"] == "unknown_due_to_timeout"
    assert parsed[0]["prior_art_level"] == "high"
    assert parsed[0]["direct_evidence_count"] == 4
    assert "direct_evidence" not in parsed[0]
    assert "queries" not in parsed[0]


def test_every_novelty_record_survives_the_evidence_budget():
    records = [
        {
            "direction_id": f"NEW-{i:02d}",
            "novelty_status": "known_area",
            "summary": "s" * 5000,
            "direct_evidence": [{"pmid": str(j), "title": "t" * 400} for j in range(30)],
        }
        for i in range(1, 6)
    ]

    block = budgeted_prompt_block(records, EVIDENCE_PROMPT_FIELDS, SCORER_EVIDENCE_BUDGET)

    for record in records:
        assert record["direction_id"] in block


def test_new_policy_evidence_discloses_list_truncation_counts():
    record = {
        "direction_id": "NEW-01",
        "policy_version": "direction_search_v1",
        "article_inventory": [
            {"pmid": str(i), "title": "title" * 80} for i in range(100)
        ],
        "extractions": [
            {"pmid": str(i), "source_passage": "passage" * 80} for i in range(100)
        ],
        "task_executions": [
            {"task_id": f"task-{i}", "status": "completed", "compiled_query": "q" * 400}
            for i in range(100)
        ],
    }

    parsed = json.loads(
        budgeted_prompt_block([record], EVIDENCE_PROMPT_FIELDS, 350)
    )[0]

    assert parsed["direction_id"] == "NEW-01"
    for field in ("article_inventory", "extractions", "task_executions"):
        assert parsed[f"{field}_source_count"] == 100
        assert parsed[f"{field}_retained_count"] < 100
        assert parsed[f"{field}_omitted_count"] == (
            100 - parsed[f"{field}_retained_count"]
        )
        identity = "task_id" if field == "task_executions" else "pmid"
        assert all(item.get(identity) for item in parsed.get(field, []))
        assert parsed[f"{field}_retained_count"] == len(parsed.get(field, []))


@pytest.mark.parametrize(
    ("field", "items"),
    [
        (
            "article_inventory",
            [{"pmid": str(i), "title": "title" * 80} for i in range(8)],
        ),
        (
            "extractions",
            [{"pmid": str(i), "source_passage": "passage" * 80} for i in range(8)],
        ),
        (
            "task_executions",
            [
                {
                    "task_id": f"task-{i}",
                    "status": "completed",
                    "compiled_query": "query" * 80,
                }
                for i in range(8)
            ],
        ),
    ],
)
def test_new_policy_evidence_discloses_retained_detail_truncation(field, items):
    record = {
        "direction_id": "NEW-01",
        "policy_version": "direction_search_v1",
        field: items,
    }

    parsed = json.loads(
        budgeted_prompt_block([record], EVIDENCE_PROMPT_FIELDS, 350)
    )[0]

    identity = "task_id" if field == "task_executions" else "pmid"
    retained = parsed[field]
    originals_by_identity = {item[identity]: item for item in items}
    changed_details = sum(
        item != originals_by_identity[item[identity]] for item in retained
    )
    assert retained
    assert all(item.get(identity) for item in retained)
    assert parsed[f"{field}_source_count"] == 8
    assert parsed[f"{field}_retained_count"] == len(retained)
    assert changed_details > 0
    assert parsed[f"{field}_detail_truncated_count"] == changed_details
    assert parsed[f"{field}_detail_truncated"] is True


# --- the prompts themselves ---------------------------------------------------


def _scorer_source():
    from backend.agents.pi_agent import PIAgent
    return inspect.getsource(PIAgent.score_integration_directions_with_evidence)


def test_scorer_prompt_no_longer_truncates_its_candidate_list():
    src = _scorer_source()
    assert "[:8000]" not in src
    assert "[:6000]" not in src
    assert "budgeted_prompt_block(raw_candidates" in src


def test_scorer_prompt_states_how_many_candidates_there_are():
    # A count the model can check itself is what makes "return all of them"
    # verifiable rather than aspirational.
    assert "candidate_count" in _scorer_source()


def test_selection_prompt_does_not_dump_an_unbounded_candidate_list():
    from backend.agents.pi_agent import PIAgent

    src = inspect.getsource(PIAgent.select_integration_directions)
    assert "json.dumps(scored_candidates" not in src
    assert "budgeted_prompt_block(scored_candidates" in src


@pytest.mark.asyncio
async def test_real_sized_candidates_all_reach_the_scoring_prompt(monkeypatch):
    from backend.tests.test_integration_gate_no_silent_drop import _make_pi_agent

    agent = _make_pi_agent()
    captured = {}

    async def capture(prompt, temperature=0.3):
        captured["prompt"] = prompt
        return "[]"

    monkeypatch.setattr(agent, "chat", capture)
    candidates = _fat_candidates(4)

    from types import SimpleNamespace
    await agent.score_integration_directions_with_evidence(
        SimpleNamespace(name="s", initial_instruction="objective"), candidates, []
    )

    for candidate in candidates:
        assert candidate["direction_id"] in captured["prompt"]
