"""Adoption of workspace renders into the plan figure contract.

See docs/HANDOVER.md for figure identity and review responsibilities.

The executor saves result figures with plt.savefig inside execute_code. Those
files never become PublicationFigure records, and _matches reads
``figure.plan_step_id`` -- never prose -- so the contract reports them missing
however plainly they sit on disk. Measured on session 3d675631: 37 jpg files,
one generate_publication_figure call all round, 16 requirements at 0/1 eligible,
every one of them present in workspace/figures.

Adoption closes that by constructing the record. A PI call decides which
candidate answers which requirement; this module validates that decision and
writes the fields the contract actually reads.
"""
import pytest

from backend.models.schemas import AnalysisStep, CodeExecution
from backend.services.figure_adoption import (
    FigureBinding,
    adopt_bindings,
    build_adoption_candidates,
    validate_bindings,
)
from backend.services.figure_contract import RequiredFigure
from backend.services.report.artifacts import (
    is_report_eligible,
)


def _render(path, *, pdf=True, seed=0):
    """A real image big enough for image_file_sha256's floors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(4, 3), dpi=120)
    rng = np.random.default_rng(seed)
    ax.imshow(rng.random((64, 64)))
    fig.savefig(path, format="jpg")
    if pdf:
        fig.savefig(str(path).replace(".jpg", ".pdf"), format="pdf")
    plt.close(fig)
    return path


@pytest.fixture
def workspace(tmp_path):
    figures = tmp_path / "figures"
    figures.mkdir()
    for stem in ("volcano_primary", "heatmap_primary",
                 "draft_render", "draft_render_final"):
        _render(figures / f"{stem}.jpg")
    return tmp_path


def _receipt(
    *,
    receipt_id="receipt-1",
    plan_id="plan-1",
    round_number=1,
    lineage_id="main",
    analysis_generation=4,
    cohort_generation=2,
    cohort_digest="c" * 64,
    **overrides,
):
    values = {
        "id": receipt_id,
        "code": "current_analysis()",
        "round_number": round_number,
        "evidence_schema_version": 2,
        "plan_id": plan_id,
        "execution_status": "succeeded",
        "lineage_id": lineage_id,
        "analysis_generation": analysis_generation,
        "cohort_generation": cohort_generation,
        "cohort_digest": cohort_digest,
    }
    values.update(overrides)
    return CodeExecution(**values)


def _state(tmp_path, *, plan_id="plan-1", round_number=1):
    from backend.models.schemas import AnalysisState

    state = AnalysisState(session_id="s", user_id="u", research_purpose="t")
    state.current_round = round_number
    state.publication_figures = []

    class _Plan:
        id = plan_id
        steps = []

    state.current_plan = _Plan()
    state.restore_branch_id = "main"
    state.sandbox_state = {
        "analysis_generation": 4,
        "cohort_identity_v1": {
            "cohort_generation": 2,
            "cohort_digest": "c" * 64,
        },
    }
    state.code_history = [_receipt(
        plan_id=plan_id, round_number=round_number,
    )]
    return state


def _req(step, figure_type, key=None):
    return RequiredFigure(
        key=key or f"plan.{step}.{figure_type}",
        figure_type=figure_type,
        description=f"{figure_type} for step {step}",
        minimum_count=1,
        plan_step_id=step,
        plan_id="plan-1",
        require_plan_link=True,
    )


def test_last_unverifiable_writer_does_not_fall_back_to_an_older_receipt(
    workspace, tmp_path
):
    state = _state(tmp_path)
    state.analysis_log = [
        AnalysisStep(
            step_number=7,
            step_type="execute_code",
            code_description="Current writer",
            execution_id="receipt-1",
            saved_figure_filenames=["volcano_primary.jpg"],
        ),
        AnalysisStep(
            step_number=8,
            step_type="execute_code",
            code_description="Stale writer",
            execution_id="receipt-old",
            saved_figure_filenames=["volcano_primary.jpg"],
        ),
    ]
    state.code_history.append(_receipt(
        receipt_id="receipt-old", round_number=0,
    ))

    candidates = build_adoption_candidates(state, workspace)
    candidate = next(item for item in candidates if item.name == "volcano_primary")
    assert candidate.source_execution_id is None
    assert "no current execution association" in candidate.provenance
    assert validate_bindings(
        [FigureBinding(3, "volcano", candidate.name, "c")],
        [_req(3, "volcano")],
        [candidate],
    )
    assert adopt_bindings(
        state,
        [FigureBinding(3, "volcano", candidate.name, "c")],
        [candidate],
    ) == []


def test_adopted_identity_uses_source_receipt_after_data_changes(
    workspace, tmp_path
):
    state = _state(tmp_path)
    # A later approved mutation must not erase this file's earlier writer.
    state.sandbox_state["analysis_generation"] = 5
    state.sandbox_state["cohort_identity_v1"] = {
        "cohort_generation": 3, "cohort_digest": "e" * 64,
    }
    state.analysis_log = [AnalysisStep(
        step_number=7,
        step_type="execute_code",
        code_description="Current writer",
        execution_id="receipt-1",
        saved_figure_filenames=["volcano_primary.jpg"],
    )]
    candidate = next(
        item for item in build_adoption_candidates(state, workspace)
        if item.name == "volcano_primary"
    )
    assert adopt_bindings(
        state,
        [FigureBinding(3, "volcano", candidate.name, "c")],
        [candidate],
    )
    figure = state.publication_figures[0]
    assert figure.source_execution_id == "receipt-1"
    assert figure.round_number == 1
    assert figure.plan_id == "plan-1"
    assert figure.lineage_id == "main"
    assert figure.analysis_generation == 4
    assert figure.cohort_generation == 2
    assert figure.cohort_digest == "c" * 64
    assert is_report_eligible(figure) is True
