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
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.agents.pi_agent import PIAgent
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

    async def _introspect(_hint):
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
    state = AnalysisState(user_id="u1", name="Project", initial_instruction="analyze the comparison")
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
    assert meta["review_status"] in ("completed", "partial")
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
async def test_denylisted_code_is_rejected_without_touching_the_kernel():
    """A supplement that tries to re-cluster must be refused, not executed."""
    state = _state_and_plan()
    pi = _pi_agent([_REVIEW_JSON, _ASSESS_JSON])
    bio = _bio_agent([
        json.dumps({"code": "sc.tl.leiden(adata, resolution=2.0)"}),
        json.dumps({"done": True, "result": {"summary": "Could not complete.", "findings": []}}),
    ])
    bio._current_analysis_state = state
    harness = _Harness(pi, bio)
    result = _result()

    audit_summary = harness._audit_round_plan_execution(state, result)
    meta, merge = await harness._run_post_execution_review(state, result, audit_summary, "full")

    assert bio.executed_code == [], "re-clustering must never reach the sandbox"
    assert meta["review_status"] in ("completed", "partial", "failed")
    assert merge["findings"] == []
    # The rejection is fed back to the model so it can correct itself.
    assert any("rejected" in p for p in bio.prompts_seen[1:])


@pytest.mark.parametrize("code", [
    "plt.savefig('unreviewed.png')",
    "fig.savefig ('unreviewed.png')",
    "from matplotlib.pyplot import savefig\nsavefig('unreviewed.png')",
    "sc.pl.umap(adata, color='group', save='_unreviewed.png')",
    "image.save('unreviewed.png')",
    "fig.canvas.print_png('unreviewed.png')",
    "open('unreviewed.png', 'wb').write(payload)",
    "mode = 'wb'\nopen('unreviewed.png', mode).write(payload)",
    "mode = choose_mode()\nopen('result.txt', mode=mode)",
])
def test_restricted_supplement_rejects_figure_file_writes(code):
    """A disk image would be rediscovered as a legacy report artifact despite
    having no PublicationFigure review record, so it must never reach the kernel."""
    bio = _bio_agent([])

    reason = bio._supplementary_code_is_safe(code, "adata")

    assert reason and "disallowed" in reason


def test_restricted_supplement_resolves_imported_write_aliases():
    bio = _bio_agent([])
    code = (
        "from os import open as o, write as w, utime as u\n"
        "fd = o('unreviewed.png', 65)\n"
        "w(fd, payload)\n"
        "u('unreviewed.png', ns=(original, original))"
    )

    reason = bio._supplementary_code_is_safe(code, "adata")

    assert reason and "os.open" in reason


def test_restricted_supplement_resolves_dotted_import_without_duplicate_parts():
    bio = _bio_agent([])
    code = (
        "import scanpy.pl\n"
        "scanpy.pl.umap(adata, color='group', save='_unreviewed.png')"
    )

    reason = bio._supplementary_code_is_safe(code, "adata")

    assert reason and "Scanpy" in reason


def test_restricted_supplement_quarantines_an_image_written_indirectly(tmp_path):
    """Runtime discovery is the backstop for a writer the static AST gate misses."""
    bio = _bio_agent([])
    bio.sandbox.working_dir = str(tmp_path)
    state = _state_and_plan()
    before = bio._supplementary_image_versions(tmp_path)

    image = tmp_path / "supplementary_outputs" / "figures" / "indirect.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-a-reportable-render")

    changed = bio._quarantine_supplementary_images(state, before)

    assert changed == {"indirect"}
    assert "indirect" in state.invalidated_artifact_stems


def test_supplementary_quarantine_detects_content_change_with_restored_mtime(
    tmp_path,
):
    from backend.tests.figure_image_helpers import valid_image_bytes

    bio = _bio_agent([])
    bio.sandbox.working_dir = str(tmp_path)
    state = _state_and_plan()
    image = tmp_path / "figures" / "restored_time.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(valid_image_bytes("PNG", color=(10, 20, 30)))
    original = image.stat().st_mtime_ns
    before = bio._supplementary_image_versions(tmp_path)

    image.write_bytes(valid_image_bytes("PNG", color=(200, 30, 20)))
    image.touch()
    import os
    os.utime(image, ns=(original, original))

    changed = bio._quarantine_supplementary_images(state, before)

    assert changed == {"restored_time"}
    assert "restored_time" in state.invalidated_artifact_stems


def test_supplementary_quarantine_only_invalidates_current_round_once(tmp_path):
    bio = _bio_agent([])
    bio.sandbox.working_dir = str(tmp_path)
    state = _state_and_plan()
    old_round = PublicationFigure(
        name="indirect",
        description="old round",
        figure_type="other",
        png_path=str(tmp_path / "indirect.png"),
        pdf_path=str(tmp_path / "indirect.pdf"),
        round_number=1,
    )
    current_round = old_round.model_copy(
        update={"id": "current-round", "round_number": state.current_round}
    )
    invalidated_at = datetime(2020, 1, 1)
    already_invalid = old_round.model_copy(
        update={
            "id": "already-invalid",
            "round_number": state.current_round,
            "is_current": False,
            "invalidated_at": invalidated_at,
            "invalidation_reason": "original reason",
        }
    )
    state.publication_figures = [old_round, current_round, already_invalid]
    before = bio._supplementary_image_versions(tmp_path)
    (tmp_path / "indirect.png").write_bytes(b"changed")

    bio._quarantine_supplementary_images(state, before)

    assert old_round.is_current is True
    assert current_round.is_current is False
    assert current_round.invalidated_at is not None
    assert already_invalid.invalidated_at == invalidated_at
    assert already_invalid.invalidation_reason == "original reason"


def test_scanpy_save_false_is_not_mistaken_for_a_disk_write():
    bio = _bio_agent([])
    assert bio._supplementary_code_is_safe(
        "sc.pl.umap(adata, color='group', save=False)", "adata"
    ) is None


@pytest.mark.parametrize("code", [
    "open('input.txt')",
    "open('input.txt', 'rb')",
    "Path('input.txt').open(mode='r')",
])
def test_restricted_supplement_allows_statically_read_only_open(code):
    bio = _bio_agent([])
    assert bio._supplementary_code_is_safe(code, "adata") is None


@pytest.mark.asyncio
async def test_incidental_inline_figures_are_discarded_from_evidence_and_result():
    """Notebook capture is neither registered nor reviewed. It may happen when
    the model ignores the prompt, but it cannot leak through CodeExecution or the
    automatic supplementary result into the archival report."""
    state = _state_and_plan()
    bio = _bio_agent([
        json.dumps({"code": "values = [1, 2, 3]\nplt.plot(values)"}),
        json.dumps({"done": True, "result": {
            "summary": "Computed values.",
            "findings": ["The values increased (execution_id recorded above)."],
        }}),
    ], exec_output="values: 1, 2, 3", exec_figures=["raw-base64-image"])
    bio._current_analysis_state = state

    result = await bio.run_supplementary_tasks(
        tasks=["Compute a compact quantitative summary"],
        context_summary="", direction="d", max_iterations=3,
        timeout_seconds=60, total_timeout_seconds=120,
    )

    assert result["figures"] == []
    assert result["code_history"] and result["code_history"][0].figures == []
    assert any("inline figure" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_max_iterations_exposes_success_and_failure_execution_records():
    state = _state_and_plan()
    bio = _bio_agent(
        [
            json.dumps({"code": "print('success output')"}),
            json.dumps({"code": "raise RuntimeError('failure')"}),
        ],
        exec_results=[
            ("success output", None),
            ("partial output", "Traceback\nRuntimeError: terminal failure"),
        ],
    )
    bio._current_analysis_state = state

    result = await bio.run_supplementary_tasks(
        tasks=["Run the requested computation"],
        context_summary="",
        direction="d",
        max_iterations=2,
        timeout_seconds=60,
        total_timeout_seconds=120,
    )

    records = result["evidence_records"]
    raw_records = result["code_history"]
    assert result["status"] == "partial"
    assert result["findings"] == []
    assert len(records) == len(raw_records) == 2
    assert [record["execution_id"] for record in records] == [
        record.id for record in raw_records
    ]
    assert [record["status"] for record in records] == ["completed", "failed"]
    assert records[0]["stdout_excerpt"] == "success output"
    assert records[0]["error"] == ""
    assert records[1]["stdout_excerpt"] == "partial output"
    assert "RuntimeError: terminal failure" in records[1]["error"]
    assert all(len(record["execution_id"]) <= 100 for record in records)
    assert all(len(record["stdout_excerpt"]) <= 2000 for record in records)
    assert all(len(record["error"]) <= 1000 for record in records)
    json.dumps(result["evidence_records"])


@pytest.mark.asyncio
async def test_pi_declining_reaches_no_executor_call():
    state = _state_and_plan()
    pi = _pi_agent([json.dumps({
        "gaps": [], "supplementary_tasks": [], "out_of_scope": ["Needs new sequencing."],
        "verdict": "no_action", "rationale": "Nothing runnable on the loaded data.",
    })])
    bio = _bio_agent([json.dumps({"done": True, "result": {}})])
    bio._current_analysis_state = state
    harness = _Harness(pi, bio)
    result = _result()

    audit_summary = harness._audit_round_plan_execution(state, result)
    meta, merge = await harness._run_post_execution_review(state, result, audit_summary, "full")

    assert meta["review_status"] == "no_action"
    assert bio.executed_code == [] and bio.prompts_seen == []
    assert merge == {}


@pytest.mark.asyncio
async def test_unparseable_pi_answer_degrades_to_no_action():
    state = _state_and_plan()
    pi = _pi_agent(["I think we should probably look into a few more things."])
    bio = _bio_agent([json.dumps({"done": True, "result": {}})])
    bio._current_analysis_state = state
    harness = _Harness(pi, bio)
    result = _result()

    audit_summary = harness._audit_round_plan_execution(state, result)
    meta, merge = await harness._run_post_execution_review(state, result, audit_summary, "full")

    assert meta["review_status"] in ("no_action", "failed")
    assert bio.executed_code == [] and merge == {}


@pytest.mark.asyncio
async def test_a_slow_step_is_interrupted_and_the_pass_continues():
    """A step that overruns its budget must not abandon the pass: the kernel is
    interrupted, the executor is told, and the remaining steps still run."""
    import asyncio

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
async def test_total_budget_still_backstops_a_non_converging_pass():
    """The total ceiling remains, for a pass that keeps producing cheap steps
    forever rather than one expensive one."""
    import asyncio

    state = _state_and_plan()
    interrupts = []

    async def _always_slow(code, **kwargs):
        await asyncio.sleep(5)
        return SimpleNamespace(output="ok", error=None, figures=[])

    bio = _bio_agent([json.dumps({"code": "keep_going(adata)"})])
    bio.sandbox.execute = _always_slow
    bio.sandbox.km = SimpleNamespace(interrupt_kernel=lambda: interrupts.append(1))
    bio._current_analysis_state = state

    result = await bio.run_supplementary_tasks(
        tasks=["Do the thing"], context_summary="", direction="d",
        max_iterations=8, timeout_seconds=1, total_timeout_seconds=2,
    )

    assert result["status"] == "partial"
    assert any("timeout" in w for w in result["warnings"])
    assert interrupts, "the total ceiling must interrupt the kernel too"


@pytest.mark.asyncio
async def test_no_active_object_fails_the_pass_not_the_round():
    state = _state_and_plan()
    pi = _pi_agent([_REVIEW_JSON, _ASSESS_JSON])
    bio = _bio_agent([json.dumps({"done": True, "result": {}})])
    bio._current_analysis_state = state

    async def _no_object(_hint):
        return {}

    bio.introspect_validation_metadata = _no_object
    harness = _Harness(pi, bio)
    result = _result()

    audit_summary = harness._audit_round_plan_execution(state, result)
    meta, merge = await harness._run_post_execution_review(state, result, audit_summary, "full")

    assert meta["review_status"] == "failed"
    assert bio.executed_code == []
    assert merge["findings"] == []
