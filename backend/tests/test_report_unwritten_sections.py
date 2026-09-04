"""A report body that ends early must not silently lose its remaining chapters.

The observed failure: the writer hit its output cap mid-chapter, so the outline's
tail was never emitted. ``align_sections_to_outline`` keeps a skipped section only
when it owns a figure, so a text-only integration gate and a direction whose
figures were attributed elsewhere both disappeared -- with their analysis rounds --
and nothing in the report data said a chapter was gone.

The remedy is advisory throughout: hand the unwritten chapters back to the report
writer once, then disclose whatever is still absent. Nothing here blocks or
retries the report.
"""
from types import SimpleNamespace

import pytest

from backend.agents.supervisor import report_content, report_prompt
from backend.agents.supervisor_agent import SupervisorAgent
from backend.services.report.quality import build_report_partial_fields
from backend.services.report.report_outline import (
    build_report_outline,
    missing_outline_section_ids,
)


def _round(number, phase, direction=""):
    return SimpleNamespace(round_number=number, phase=phase, direction=direction)


def _outline():
    """Two directions separated by a gate, closed by a final integration round."""
    return build_report_outline(SimpleNamespace(
        round_results=[
            _round(1, "exploration"),
            _round(2, "innovation", "direction A"),
            _round(3, "integration"),
            _round(4, "innovation", "direction B"),
            _round(5, "integration"),
        ],
        publication_figures=[],
    ))


# The body the writer managed to emit before it ran out of budget: the outline
# tail (gate, second direction, closing synthesis, mechanisms, conclusions) is
# missing and the last chapter stops mid-sentence.
_TRUNCATED_BODY = """## [abstract] Abstract of the study
Copper handling was profiled across two modalities.

## [methods] Methods and processing
Nuclei were called with the standard pipeline.

## [exploration_findings] What the first pass found
Eighteen states were adjudicated across nine lineages.

## [direction_1] Innovation Round 2: cuproenzyme footprint
The COX assembly genes shifted together, while complex-I genes"""

_CONTINUATION_BODY = """## [integration_gate_1] Integration Cycle 1
Provisionally, the two directions agree on the arrest program.

## [direction_2] Innovation Round 4: paracrine axis
The ligand was detected in the arrest state at transcript level.

## [integrated_findings] Integrated Findings Across All Phases
The directions converge on stress-driven arrest.

## [mechanism_candidates] Data-Driven Mechanism Candidates
Copper depletion remains the highest-ranked candidate.

## [conclusions] Conclusions and Future Directions
Tier-1 assays remain outstanding.

**Limitations** Transcript-level evidence only.
"""

_UNWRITTEN = [
    "integration_gate_1",
    "direction_2",
    "integrated_findings",
    "mechanism_candidates",
    "conclusions",
]


def _agent(chat):
    agent = SupervisorAgent.__new__(SupervisorAgent)
    agent.session_id = "s"
    agent._set_llm_context = lambda: None
    prompt_calls = []

    def _build_prompt(narrative, stats, state, figure_map, language, **kwargs):
        prompt_calls.append(kwargs)
        return "CONTINUATION PROMPT BODY"

    async def _budgeted(method, **kwargs):
        return await method(**kwargs)

    agent._build_report_prompt = _build_prompt
    agent._budgeted_text_call = _budgeted
    agent.prompt_calls = prompt_calls
    report_prompt.bind_facade(report_content)
    return agent


def _ctx(client, body=_TRUNCATED_BODY):
    async def progress(*_args):
        return None

    return {
        "_progress": progress,
        "report_content": body,
        "language": "en",
        "outline": _outline(),
        "used_fallback": False,
        "attribution": {},
        "figure_map": {},
        "figure_metadata_by_stem": {},
        "narrative": "narrative",
        "stats": {},
        "table_list": [],
        "client": client,
    }


def _client(responder):
    calls = []

    async def chat(**kwargs):
        calls.append(kwargs)
        return responder(kwargs)

    return SimpleNamespace(
        base_url="", set_context=lambda *_args: None, chat=chat, calls=calls,
    )


async def _run(agent, ctx):
    await report_prompt.parse_and_place_report(
        agent,
        ctx,
        SimpleNamespace(),
        workspace_dir=None,
        chat_messages=None,
        progress_callback=None,
        report_model="test-model",
    )


def _ids(sections):
    return [str(section.get("id") or "") for section in sections]


def test_missing_outline_section_ids_counts_absent_and_prose_free_sections():
    outline = _outline()
    sections = [
        {"id": "abstract", "content": "Written."},
        # Materialised for its figures but carrying no narrative at all.
        {"id": "direction_1", "content": "   "},
    ]

    missing = missing_outline_section_ids(sections, outline)

    assert "direction_1" in missing
    assert "abstract" not in missing
    # Order follows the outline, so the report reads them in reading order.
    assert missing == [
        osec.id for osec in outline if osec.id in set(missing)
    ]


def test_absent_sections_are_disclosed_without_withholding_the_report():
    status, current_step, primary, reasons = build_report_partial_fields(
        missing_sections=["direction_2", "integrated_findings"],
    )

    assert status == "partial"
    assert primary == "section_coverage"
    assert "section_coverage" in reasons
    assert "direction_2" in current_step and "integrated_findings" in current_step


def test_no_absent_section_leaves_the_report_complete():
    status, _step, primary, reasons = build_report_partial_fields(
        missing_sections=[],
    )

    assert (status, primary, reasons) == ("completed", None, [])


@pytest.mark.asyncio
async def test_truncated_body_is_completed_by_the_report_writer():
    client = _client(lambda _kwargs: SimpleNamespace(content=_CONTINUATION_BODY))
    agent = _agent(client)
    ctx = _ctx(client)

    await _run(agent, ctx)

    # Exactly one completion request, carrying only the unwritten sections.
    assert len(client.calls) == 1
    assert [osec.id for osec in agent.prompt_calls[0]["outline"]] == _UNWRITTEN
    # An output cap is what truncated the first pass; the retry must not carry one.
    assert client.calls[0]["max_tokens"] is None

    ids = _ids(ctx["sections"])
    for section_id in _UNWRITTEN:
        assert section_id in ids
    direction = next(s for s in ctx["sections"] if s["id"] == "direction_2")
    assert "paracrine" in direction["title"] or "ligand" in direction["content"]
    assert ctx["missing_sections"] == []


@pytest.mark.asyncio
async def test_a_failed_completion_request_still_publishes_and_discloses():
    def _boom(_kwargs):
        raise RuntimeError("upstream refused")

    client = _client(_boom)
    agent = _agent(client)
    ctx = _ctx(client)

    await _run(agent, ctx)

    # The report is still assembled from what the writer did deliver ...
    assert "direction_1" in _ids(ctx["sections"])
    # ... and every chapter it never wrote is named rather than lost.
    assert ctx["missing_sections"] == _UNWRITTEN


@pytest.mark.asyncio
async def test_a_partially_answered_completion_request_discloses_the_remainder():
    partial = """## [integration_gate_1] Integration Cycle 1
Provisionally, the two directions agree.

## [direction_2] Innovation Round 4: paracrine axis
The ligand was detected at transcript level.
"""
    client = _client(lambda _kwargs: SimpleNamespace(content=partial))
    agent = _agent(client)
    ctx = _ctx(client)

    await _run(agent, ctx)

    ids = _ids(ctx["sections"])
    assert "integration_gate_1" in ids and "direction_2" in ids
    assert ctx["missing_sections"] == [
        "integrated_findings", "mechanism_candidates", "conclusions",
    ]


@pytest.mark.asyncio
async def test_a_complete_body_asks_the_writer_for_nothing():
    body = _TRUNCATED_BODY + " fell.\n\n" + _CONTINUATION_BODY
    client = _client(lambda _kwargs: SimpleNamespace(content=""))
    agent = _agent(client)
    ctx = _ctx(client, body=body)

    await _run(agent, ctx)

    assert client.calls == []
    assert ctx["missing_sections"] == []
