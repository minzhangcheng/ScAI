import base64
import hashlib
from datetime import datetime
from pathlib import Path

from backend.models.schemas import (
    AnalysisState,
    CodeExecution,
    FigureReviewStatus,
    PublicationFigure,
)
from backend.services.report.artifacts import (
    description_with_data_drift_warning,
    is_gallery_eligible,
    is_report_eligible,
    resolve_archival_render,
)
from backend.tests.figure_image_helpers import valid_image_bytes, write_valid_image


def _figure(name, status, **kwargs):
    png_path = kwargs.pop("png_path", f"{name}.jpg")
    pdf_path = kwargs.pop("pdf_path", f"{name}.pdf")
    png_base64 = kwargs.get("png_base64")
    candidate = Path(png_path)
    if "render_sha256" not in kwargs:
        kwargs["render_sha256"] = (
            hashlib.sha256(candidate.read_bytes()).hexdigest()
            if candidate.is_file() else "1" * 64
        )
    if "pdf_sha256" not in kwargs:
        kwargs["pdf_sha256"] = "2" * 64
    if "preview_sha256" not in kwargs and png_base64:
        kwargs["preview_sha256"] = hashlib.sha256(
            base64.b64decode(png_base64)
        ).hexdigest()
    return PublicationFigure(
        name=name, description="", figure_type="other",
        png_path=png_path, pdf_path=pdf_path,
        review_status=status, **kwargs,
    )


def test_schema_v2_figure_selection_revalidates_durable_source_receipt():
    figure = _figure(
        "linked-result",
        FigureReviewStatus.APPROVED,
        plan_id="plan-current",
        plan_step_id=4,
        round_number=2,
        lineage_id="main",
        provenance_schema_version=2,
        producer_run_kind="ordinary",
        source_execution_id="execution-current",
        source_execution_run_kind="ordinary",
        source_execution_validated=True,
    )
    receipt = CodeExecution(
        id="execution-current",
        code="compute_declared_result()",
        round_number=2,
        plan_id="plan-current",
        plan_step_ids=[4],
        plan_binding_status="confirmed",
        evidence_schema_version=2,
        executor_run_kind="ordinary",
        execution_status="succeeded",
    )
    state = AnalysisState(
        id="figure-provenance",
        user_id="user",
        current_round=2,
        code_history=[receipt],
    )

    assert is_report_eligible(figure) is False
    assert "Analysis state is required" in figure.provenance_validation_reason
    assert is_report_eligible(figure, state=state) is True
    assert figure.provenance_validation_reason is None

    state.code_history = []
    assert is_report_eligible(figure, state=state) is False
    assert "missing from durable code history" in figure.provenance_validation_reason


def test_followup_figure_remains_reportable_after_many_scopes_and_disk_reload():
    from backend.models.schemas import AnalysisPlan, PlanStep
    from backend.services.figure_followup import (
        build_figure_followup_scope,
        get_figure_followup_scope,
        persist_figure_followup_scope,
    )
    from backend.services.session_store import load_session_from_disk, save_session_to_disk

    figure = _figure(
        "early-qc", FigureReviewStatus.APPROVED,
        plan_id="early-plan", plan_step_id=4, round_number=1,
        provenance_schema_version=2, producer_run_kind="figure_followup",
        producer_run_scope_id="scope-0", figure_followup_task_id="qc-task",
        plan_requirement_key="plan.4.1", source_execution_id="qc-source",
        source_execution_run_kind="ordinary", source_execution_validated=True,
    )
    state = AnalysisState(
        id="followup-scope-retention", user_id="user", current_round=1,
        current_plan=AnalysisPlan(
            id="early-plan", phase="exploration", steps=[
                PlanStep(id=4, title="Quality control", method="QC", status="completed"),
            ],
        ),
        code_history=[CodeExecution(
            id="qc-source", code="compute_qc()", round_number=1,
            plan_id="early-plan", plan_step_ids=[4], plan_binding_status="confirmed",
            evidence_schema_version=2, executor_run_kind="ordinary",
            execution_status="succeeded",
        )],
        publication_figures=[figure],
    )
    task = {"id": "qc-task", "plan_step_id": 4, "plan_requirement_key": "plan.4.1"}
    first_scope = build_figure_followup_scope(state, [task], run_scope_id="scope-0")
    persist_figure_followup_scope(state, first_scope)
    assert is_report_eligible(figure, state=state)

    for index in range(1, 16):
        state.current_round = index + 1
        state.current_plan.id = f"later-plan-{index}"
        persist_figure_followup_scope(state, build_figure_followup_scope(
            state, [], run_scope_id=f"scope-{index}",
        ))
    save_session_to_disk(state.id, state)
    loaded = load_session_from_disk(state.id)

    assert loaded is not None
    assert get_figure_followup_scope(loaded, "scope-0") == first_scope
    assert all(get_figure_followup_scope(loaded, f"scope-{i}") for i in range(16))
    assert is_report_eligible(loaded.publication_figures[0], state=loaded)


def test_historical_data_drift_invalidation_is_now_advisory():
    figure = _figure(
        "historical-drift", FigureReviewStatus.APPROVED,
        is_current=False,
        invalidated_at=datetime.now(),
        invalidation_reason=(
            "Expression representation changed in round 2; derived artifact "
            "generation 1 is stale."
        ),
    )

    assert is_gallery_eligible(figure)
    assert is_report_eligible(figure)
    assert "recorded analysis generation" in (
        description_with_data_drift_warning(figure)
    )


def test_record_hash_rejects_a_later_same_name_overwrite(tmp_path):
    workspace = tmp_path / "workspace"
    figures = workspace / "figures"
    figures.mkdir(parents=True)
    path = figures / "reviewed.jpg"
    original = write_valid_image(path, min_bytes=2200)
    preview = valid_image_bytes("JPEG")
    figure = _figure(
        "reviewed", FigureReviewStatus.APPROVED,
        png_path=str(path),
        png_base64=base64.b64encode(preview).decode(),
        render_sha256=hashlib.sha256(original).hexdigest(),
        preview_sha256=hashlib.sha256(preview).hexdigest(),
    )

    write_valid_image(path, color=(220, 10, 10), min_bytes=2200)
    resolved = resolve_archival_render(figure, workspace)

    assert resolved is not None
    assert resolved.read_bytes() == preview
    assert resolved != path
