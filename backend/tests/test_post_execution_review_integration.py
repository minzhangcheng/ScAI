"""End-to-end post-execution review with the REAL PI and bioinformatics agents.

The pieces were built against a written contract, so the risk this covers is
contract drift: a PI prompt that cannot be built, a JSON shape the orchestrator
does not read, or an executor result whose keys the merge step drops. Only the
LLM and the kernel are stubbed.
"""
import base64
import hashlib
import itertools
import json
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.pi_agent import PIAgent
from backend.agents.pi.instructions import build_out_of_scope_context
from backend.models.schemas import (
    AnalysisState,
    CodeExecution,
    FigureReviewStatus,
    PublicationFigure,
)
from backend.models.schema_parts.session import AnalysisPlan, PlanStep
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes
from backend.workflow.post_execution import PostExecutionReviewMixin

_SIGNATURE = {
    "id": "adata-1", "n_obs": 1000, "n_vars": 2000, "obs_columns": ["leiden", "group"],
    "cluster_counts": {"0": 600, "1": 400}, "layers": ["counts"], "x_id": "x-1",
    "x_sum": 12345.0, "x_nnz": 999, "obs_hash": "oh", "var_hash": "vh",
    "layers_fingerprint": "lf",
}

_ARTIFACT_ROOT: Path | None = None
_ARTIFACT_SEQUENCE = itertools.count()


@pytest.fixture(scope="module", autouse=True)
def _frozen_figure_artifact_root(tmp_path_factory):
    global _ARTIFACT_ROOT
    _ARTIFACT_ROOT = tmp_path_factory.mktemp("post-execution-integration-artifacts")
    yield
    _ARTIFACT_ROOT = None

_REVIEW_JSON = json.dumps({
    "gaps": ["The approved pathway-activity step produced no result."],
    # Plan completion is PI-owned since 0.49.8: the host audit observes and
    # never writes PlanStep.status, so the reviewed step verdict has to come
    # from the PI response itself.
    "step_decisions": [{"plan_step_id": 1, "status": "completed"}],
    "supplementary_tasks": ["Run decoupler run_ulm on the existing object and report the top regulons."],
    "out_of_scope": ["Measure protein abundance — needs new laboratory work."],
    "verdict": "supplement",
    "rationale": "Repairing the one approved step that left no evidence.",
})

_ASSESS_JSON = json.dumps({
    "verdict": "resolved",
    "summary": "The pathway-activity gap is now closed by an executed analysis.",
    "resolved_gaps": ["The approved pathway-activity step produced no result."],
    "remaining_gaps": [],
})


def _provider():
    provider = MagicMock()
    provider.get_model.return_value = "test/model"
    provider.get_system_prompt.return_value = ""
    provider.get_temperature.return_value = 0.2
    return provider


class _Harness(PostExecutionReviewMixin):
    def __init__(self, pi, bio):
        self.session_id = "sess-int"
        self.pi_agent = pi
        self.bioinformatics_agent = bio
        self.config = SimpleNamespace(
            enable_post_execution_review=True,
            post_execution_review_max_tasks=5,
            post_execution_review_max_iterations=3,
            post_execution_review_timeout_seconds=60,
            max_rounds=3,
        )
        self.sent = []
        async def _save(_state, force=False):
            return True
        self.state_callback = _save

    async def _send_message(self, kind, payload, agent=None):
        self.sent.append(kind)


def _pi_agent(replies):
    agent = PIAgent(settings_provider=_provider())
    agent.set_session("sess-int")
    calls = []

    async def _chat(prompt, **kwargs):
        calls.append(prompt)
        reply = replies[min(len(calls) - 1, len(replies) - 1)]
        return reply(prompt) if callable(reply) else reply

    agent.chat = _chat
    agent.refresh_runtime_config_if_needed = lambda *a, **k: None
    agent.prompts_seen = calls
    return agent


def _bio_agent(
    replies,
    exec_output="top regulons: A, B",
    exec_error=None,
    exec_figures=None,
    exec_results=None,
):
    sandbox = MagicMock()
    executed = []
    result_index = 0

    async def _execute(code, **kwargs):
        nonlocal result_index
        executed.append(code)
        if exec_results is not None and result_index < len(exec_results):
            output, error = exec_results[result_index]
            result_index += 1
            return SimpleNamespace(output=output, error=error, figures=[])
        return SimpleNamespace(
            output=exec_output,
            error=exec_error,
            figures=list(exec_figures or []),
        )

    sandbox.execute = _execute
    agent = BioinformaticsAgent(settings_provider=_provider(), sandbox=sandbox)
    agent.set_session("sess-int")
    calls = []

    async def _chat(prompt, **kwargs):
        calls.append(prompt)
        reply = replies[min(len(calls) - 1, len(replies) - 1)]
        return reply(prompt) if callable(reply) else reply

    agent.chat = _chat
    agent.refresh_runtime_config_if_needed = lambda *a, **k: None

    async def _introspect(_hint, **kwargs):
        return {"var": "adata", "cluster_key": "leiden", "artifact_dir": "supplementary"}

    async def _capture(var_name, cluster_key):
        return dict(_SIGNATURE)

    agent.introspect_validation_metadata = _introspect
    agent.capture_main_object_signature = _capture
    agent.executed_code = executed
    agent.prompts_seen = calls
    return agent


def _state_and_plan():
    """A round whose first approved step ran and whose second left no evidence."""
    state = AnalysisState(
        id="sess-int",
        user_id="u1",
        name="Project",
        initial_instruction="analyze the comparison",
    )
    state.current_round = 2
    state.max_rounds = 3
    state.current_round_direction = "a test direction"
    state.current_plan = AnalysisPlan(
        phase="innovation",
        steps=[
            PlanStep(id=1, title="Marker detection",
                     method="Run sc.tl.rank_genes_groups per cluster"),
            PlanStep(id=2, title="Pathway activity",
                     method="Run decoupler run_ulm on the collected regulons"),
        ],
    )
    state.code_history.append(
        CodeExecution(code="sc.tl.rank_genes_groups(adata, 'leiden')", round_number=2)
    )
    # Marker detection is a result-bearing step and therefore needs a reviewed
    # visual deliverable. Seed that completed evidence so this fixture isolates
    # the genuinely missing pathway-activity computation in step 2.
    assert _ARTIFACT_ROOT is not None
    artifact_id = next(_ARTIFACT_SEQUENCE)
    png_path = _ARTIFACT_ROOT / f"{artifact_id}-marker-result.png"
    pdf_path = _ARTIFACT_ROOT / f"{artifact_id}-marker-result.pdf"
    render = valid_image_bytes("PNG", min_bytes=2200)
    pdf = valid_pdf_bytes()
    png_path.write_bytes(render)
    pdf_path.write_bytes(pdf)
    state.publication_figures.append(PublicationFigure(
        name="marker_result",
        description="Marker result overview",
        figure_type="dotplot",
        png_path=str(png_path),
        pdf_path=str(pdf_path),
        png_base64=base64.b64encode(render).decode(),
        render_sha256=hashlib.sha256(render).hexdigest(),
        preview_sha256=hashlib.sha256(render).hexdigest(),
        pdf_sha256=hashlib.sha256(pdf).hexdigest(),
        review_status=FigureReviewStatus.APPROVED,
        plan_step_id=1,
        plan_id=state.current_plan.id,
        purpose="result",
        round_number=2,
    ))
    return state


def _result():
    return {"findings": ["Cluster 0 differs from cluster 1."], "figures": [],
            "status": "completed", "analysis_log": [], "code_history": [],
            "evidence_execution_ids": ["e1"]}


@pytest.mark.parametrize("unavailable", [[], ["cell_communication"]])
def test_execution_audit_distinguishes_missing_execution_from_import_failure(unavailable):
    rendered = PIAgent._format_execution_audit({
        "unmet_families": ["cell_communication", "pathway_activity"],
        "unavailable_families": unavailable,
    }, [])
    assert "Planned method families without execution evidence" in rendered
    assert ("Method families with recorded import failures" in rendered) == bool(unavailable)
    assert ("No method import failures are recorded" in rendered) == (not unavailable)
    assert "Methods this kernel cannot run" not in rendered


@pytest.mark.asyncio
async def test_supplementary_deferrals_remain_scoped_in_later_main_planning():
    notes = [
        "LIANA and decoupler are unavailable in this restricted pass.",
        "Measure protein abundance — needs new laboratory work.",
    ]
    review_response = {"out_of_scope": notes, "verdict": "no_action"}
    pi = _pi_agent([
        json.dumps(review_response),
        json.dumps({"overview": "Evaluate existing data", "bioinformatics_tasks": []}),
        json.dumps({"decision": "approved", "feedback": "Review current scope.",
                    "modifications": []}),
    ])
    state = AnalysisState(user_id="u1", current_round=1)
    audit = {
        "unmet_families": ["cell_communication", "pathway_activity"],
        "unavailable_families": [], "executed_families": ["enrichment"],
        "step_observations": [{"plan_step_id": 1, "technical_linkage": "not_nominated"}],
    }
    review = await pi.review_round_execution(
        state=state, plan_audit=audit, admissions=[], findings=[], agent_suggestions=[],
        direction="", result_status="completed", is_terminal_round=False,
        max_tasks=2, trigger_reason="missing execution evidence",
    )
    # Even an old, inaccurate PI note is retained, rather than silently rewritten.
    assert review == review_response
    review_prompt = " ".join(pi.prompts_seen[0].split())
    assert "No method import failures are recorded" in review_prompt
    assert "do not establish dependency unavailability" in review_prompt
    assert "do not label it environment unavailability" in review_prompt
    state.post_execution_reviews = [{
        "round_number": 1, "review_attempt_id": "review-1",
        "review_plan": review, "plan_audit": audit,
    }, {
        # Legacy records may retain notes without an execution audit.
        "round_number": 2, "review_plan": {"out_of_scope": [notes[0]]},
    }]
    before = json.dumps(state.post_execution_reviews)
    context = build_out_of_scope_context(state)
    records = [json.loads(line[2:]) for line in context.splitlines() if line.startswith("- {")]
    assert [record["round_number"] for record in records] == [1, 1, 2]
    assert [record["historical_note"] for record in records] == notes + [notes[0]]
    assert records[0]["audit_unavailable_families"] == []
    assert records[0]["audit_executed_families"] == ["enrichment"]
    assert records[0]["review_attempt_id"] == "review-1"
    assert records[2]["audit_unavailable_families"] is None
    assert json.dumps(state.post_execution_reviews) == before

    state.current_round = 3
    await pi.innovation_instruction(state, "Compare communication and pathway responses")
    plan = AnalysisPlan(
        phase="innovation", stage="full", steps=[
            PlanStep(id=1, title="Communication", method="LIANA"),
            PlanStep(id=2, title="Pathway activity", method="decoupler"),
        ],
    )
    result = await pi.review_bioinfo_plan(state, plan)
    for prompt in pi.prompts_seen[1:]:
        normalized = " ".join(prompt.split())
        assert context in prompt
        assert "not session-wide capability rulings" in normalized
        assert "supersedes any earlier" in normalized
        assert "## Already ruled out in previous rounds" not in prompt
        assert "Do NOT plan a step whose purpose" not in normalized
    assert "main bioinformatics executor in the declared stage" in pi.prompts_seen[-1]
    assert result["decision"] == "approved" and result["modifications"] == []
    assert [step.method for step in plan.steps] == ["LIANA", "decoupler"]


@pytest.mark.asyncio
async def test_real_agents_complete_the_review_and_merge_back():
    state = _state_and_plan()
    marker_receipt = state.code_history[0]
    marker_receipt.id = "marker-exec"
    marker_receipt.evidence_schema_version = 2
    marker_receipt.plan_id = state.current_plan.id
    marker_receipt.plan_step_ids = [1]
    marker_receipt.execution_purpose = "planned"
    marker_receipt.plan_binding_status = "candidate"
    pathway_setup = CodeExecution(
        id="pathway-setup-exec",
        code="net = load_regulatory_network()",
        round_number=2,
        evidence_schema_version=2,
        plan_id=state.current_plan.id,
        plan_step_ids=[2],
        execution_purpose="planned",
        plan_binding_status="candidate",
    )
    state.code_history.append(pathway_setup)
    pi = _pi_agent([_REVIEW_JSON, _ASSESS_JSON])

    def _done_with_citation(prompt):
        execution_id = prompt.rsplit("[execution_id ", 1)[1].split("]", 1)[0]
        return json.dumps({"done": True, "result": {
            "summary": "Ran the pathway activity task.",
            "findings": [
                "Regulon A is elevated in the compared group "
                f"[execution_id {execution_id}]."
            ],
        }})

    bio = _bio_agent([
        json.dumps({"code": "import decoupler as dc\ndc.run_ulm(adata, net=net)"}),
        _done_with_citation,
    ])
    bio._current_analysis_state = state
    harness = _Harness(pi, bio)
    result = _result()

    reconciliation = {
        "plan_id": state.current_plan.id,
        "round_number": 2,
        "lineage_id": "main",
        "links": [
            {
                "plan_step_id": 1,
                "status": "evidenced",
                "execution_ids": [marker_receipt.id],
                "reason": "Marker detection and its reviewed output are complete.",
            },
            {
                "plan_step_id": 2,
                "status": "partial",
                "execution_ids": [pathway_setup.id],
                "reason": "The network was prepared, but activity was not computed.",
            },
        ],
    }
    audit_summary = harness._audit_round_plan_execution(
        state, result, reconciliation
    )
    assert audit_summary["n_unmet"] == 1

    meta, merge = await harness._run_post_execution_review(state, result, audit_summary, "full")

    # The PI actually saw the audit, and its prompt named the unmet step.
    assert "Pathway activity" in pi.prompts_seen[0]
    assert "auditing a FINISHED EXECUTION ATTEMPT" in pi.prompts_seen[0]
    assert "agent_nominated` is producer intent only" in pi.prompts_seen[0]
    assert "Host-validated technical facts" not in pi.prompts_seen[0]
    assert "SUPPLEMENTARY TECHNICAL CONTRACT v2" in pi.prompts_seen[0]
    assert "may mutate, remap, filter, normalize, scale, re-cluster" in pi.prompts_seen[0]
    assert "live-object mutation alone is not a reason to defer it" in pi.prompts_seen[0]
    assert meta["review_status"] in ("completed", "partial")
    assert meta["review_plan"]["supplementary_contract_version"] == 2
    assert meta["assessment"]["verdict"] == "resolved"
    assert bio.executed_code and "run_ulm" in bio.executed_code[0]

    # The executor's findings survive sanitizing and reach the merge payload.
    assert len(merge["findings"]) == 1
    assert merge["findings"][0].startswith(
        "Regulon A is elevated in the compared group [execution_id "
    )
    assert merge["code_history"], "executions must come back for evidence provenance"
    assert all(c.round_number == 2 for c in merge["code_history"])

    findings, _ = harness._merge_supplementary_into_round(
        state, result, result["findings"], result["figures"], merge)
    assert findings[-1] == "[Supplementary] " + merge["findings"][0]
    assert [c.code for c in state.code_history][-1] == bio.executed_code[-1]

    # Re-audit: the restricted pass closes the computational gap, but it cannot
    # mint an unreviewed image. The remaining result-figure gap is deliberately
    # returned to the main publication pipeline.
    repaired_execution_id = state.code_history[-1].id
    repaired_reconciliation = {
        "plan_id": state.current_plan.id,
        "round_number": 2,
        "lineage_id": "main",
        "links": [
            {
                "plan_step_id": 1,
                "status": "evidenced",
                "execution_ids": [marker_receipt.id],
                "reason": "Marker detection and its reviewed output are complete.",
            },
            {
                "plan_step_id": 2,
                "status": "evidenced",
                "execution_ids": [repaired_execution_id],
                "reason": "The supplementary execution completed pathway activity.",
            },
        ],
    }
    reaudit = harness._audit_round_plan_execution(
        state, result, repaired_reconciliation
    )
    assert reaudit["n_unmet"] == 1
    assert reaudit["unmet_figure_kinds"] == ["result"]
    assert [s.status for s in state.current_plan.steps] == ["completed", "pending"]


@pytest.mark.asyncio
async def test_scientific_preprocessing_is_not_vetoed_by_supplementary_execution():
    """A supplement may re-cluster when the approved task calls for it.

    The Host keeps objective file-write/resource boundaries only; it does not
    decide that a scientific mutation makes the execution fail.
    """
    state = _state_and_plan()
    pi = _pi_agent([_REVIEW_JSON, _ASSESS_JSON])
    bio = _bio_agent([
        json.dumps({"code": "sc.tl.leiden(adata, resolution=2.0)"}),
        json.dumps({"done": True, "result": {"summary": "Re-clustering completed.", "findings": []}}),
    ])
    bio._current_analysis_state = state
    harness = _Harness(pi, bio)
    result = _result()

    audit_summary = harness._audit_round_plan_execution(state, result)
    meta, merge = await harness._run_post_execution_review(state, result, audit_summary, "full")

    assert bio.executed_code == ["sc.tl.leiden(adata, resolution=2.0)"]
    assert meta["review_status"] in ("completed", "partial", "failed")
    assert merge["findings"] == []
    assert not any("rejected" in p for p in bio.prompts_seen)


@pytest.mark.asyncio
async def test_a_slow_step_is_interrupted_and_the_pass_continues(monkeypatch):
    """A step that overruns its budget must not abandon the pass: the kernel is
    interrupted, the executor is told, and the remaining steps still run."""
    import asyncio

    original_wait_for = asyncio.wait_for

    async def short_test_wait(awaitable, *, timeout):
        assert timeout >= 600
        return await original_wait_for(awaitable, timeout=0.01)

    monkeypatch.setattr(asyncio, "wait_for", short_test_wait)
    state = _state_and_plan()
    interrupts = []
    executed = []

    async def _slow_then_fast(code, **kwargs):
        executed.append(code)
        if len(executed) == 1:
            await asyncio.sleep(5)  # never completes within the step budget
        return SimpleNamespace(output="ok", error=None, figures=[])

    bio = _bio_agent([
        json.dumps({"code": "run_the_expensive_thing(adata)"}),
        json.dumps({"code": "dc.run_ulm(adata, net=net)"}),
        json.dumps({"done": True, "result": {
            "summary": "Second route worked.",
            "findings": ["Regulon A is elevated in the compared group."],
        }}),
    ])
    bio.sandbox.execute = _slow_then_fast
    bio.sandbox.km = SimpleNamespace(interrupt_kernel=lambda: interrupts.append(1))
    bio._current_analysis_state = state

    result = await bio.run_supplementary_tasks(
        tasks=["Run decoupler run_ulm and report the top regulons"],
        context_summary="", direction="d",
        max_iterations=4, timeout_seconds=1, total_timeout_seconds=60,
    )

    assert interrupts == [1], "the overrunning cell must be interrupted"
    assert len(executed) == 2, "the pass must keep going after a slow step"
    assert result["status"] in ("completed", "partial")
    assert any("exceeded its" in w for w in result["warnings"])
    # The executor is told what happened so it can choose a cheaper route.
    assert any("[timeout]" in p for p in bio.prompts_seen[1:])
    # Only the step that actually finished is recorded as evidence.
    assert [c.code for c in result["code_history"]] == ["dc.run_ulm(adata, net=net)"]


@pytest.mark.asyncio
async def test_supplementary_uses_exact_preparation_and_retains_source_receipt():
    from backend.models.schema_parts.adata_checkpoints import AdataSourcePreparation, AdataSourceSelection
    state = _state_and_plan()
    prep = AdataSourcePreparation(
        owner_id=state.current_plan.id, preparation_id="prep_test", binding_name="scai_work_test",
        selection=AdataSourceSelection(mode="checkpoint_copy", checkpoint_id="checkpoint_test"),
        anchor_name="scai_anchor_test", binding_id="binding_test", binding_identity_status="exact",
        adata_lineage_id="source_lineage", loaded_checkpoint_id="checkpoint_test",
    )
    state.adata_source_preparations.append(prep)
    bio = _bio_agent([
        json.dumps({"code": "print(scai_work_test.n_obs)"}),
        json.dumps({"done": True, "result": {"summary": "Source checked", "findings": []}}),
    ])
    bio._current_analysis_state = state
    observed = []

    async def inspect_source(hint, **kwargs):
        observed.append((hint, kwargs))
        return {"var": "scai_work_test", "cluster_key": "leiden", "obs_columns": ["leiden", "condition"], "partition_digest": "final"}

    bio.introspect_validation_metadata = inspect_source
    source = {"preparation_id": "prep_test", "cluster_key": "leiden"}
    names = await _Harness(_pi_agent([]), bio)._collect_session_names(
        state, {"resolved_adata_source_context": source}, True,
    )
    rendered = PIAgent._format_session_names(names)
    assert '"cluster_key": "leiden"' in rendered and "scai_work_test" in rendered
    result = await bio.run_supplementary_tasks(
        tasks=["Inspect final leiden cluster counts"], context_summary="", direction="d",
        source_context=source, max_iterations=2,
    )
    assert observed[-1] == ("leiden", {
        "preferred_var_name": "scai_work_test", "require_preferred": True,
        "expected_anchor_name": "scai_anchor_test",
        "allow_managed_preferred": False,
        "require_cluster_key": False,
    })
    assert result["active_object_selection"]["name"] == "scai_work_test"
    assert result["code_history"][0].adata_source_preparation_id == "prep_test"
    assert result["code_history"][0].adata_input_binding_id == "binding_test"
