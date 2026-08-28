"""Active-cohort identity and stale-artifact invalidation contracts."""

from backend.models.schemas import (
    AnalysisState,
    CodeExecution,
    PublicationFigure,
    PublicationTable,
    RoundResult,
)
from backend.services.cohort_identity import parse_cohort_probe, record_cohort_transition
from backend.services.figure_revision_tasks import (
    find_figure_revision_task,
    upsert_figure_revision_task,
)


def _probe(digest: str, n_obs: int = 10):
    return {
        "present": True,
        "canonical_object_name": "adata",
        "cohort_digest": digest,
        "n_obs": n_obs,
        "n_vars": 5,
    }


def test_parse_cohort_probe_accepts_only_structured_digest_marker():
    digest = "a" * 64
    output = (
        "ordinary output\n"
        '__SCAI_COHORT_IDENTITY_V1__:{"present": true, '
        f'"cohort_digest": "{digest}", "n_obs": 12, "n_vars": 7}}'
    )

    assert parse_cohort_probe(output) == {
        "present": True,
        "canonical_object_name": "adata",
        "cohort_digest": digest,
        "n_obs": 12,
        "n_vars": 7,
    }
    assert parse_cohort_probe("no marker") is None
    assert parse_cohort_probe(
        '__SCAI_COHORT_IDENTITY_V1__:{"present": true, "cohort_digest": "short"}'
    ) is None


def test_cohort_transition_invalidates_prior_current_artifacts():
    old_digest = "a" * 64
    new_digest = "b" * 64
    state = AnalysisState(id="cohort-session", user_id="user", current_round=2)
    state.sandbox_state = {
        "cohort_identity_v1": {
            **_probe(old_digest),
            "cohort_generation": 1,
            "lineage_id": "main",
            "round_number": 1,
        }
    }
    old_figure = PublicationFigure(
        name="old figure",
        description="old",
        figure_type="other",
        png_path="old.png",
        pdf_path="old.pdf",
        round_number=1,
        cohort_generation=1,
        cohort_digest=old_digest,
        revision_count=2,
    )
    state.publication_figures = [old_figure]
    state.publication_tables = [
        PublicationTable(
            name="old table",
            description="old",
            csv_path="old.csv",
            round_number=1,
            cohort_generation=1,
            cohort_digest=old_digest,
        )
    ]
    old_task = upsert_figure_revision_task(
        state,
        old_figure,
        reason="attempts_exhausted",
        must_fix=["Fix the old-cohort render"],
    )
    assert old_task.status == "exhausted_excluded"

    identity = record_cohort_transition(
        state, _probe(old_digest), _probe(new_digest, n_obs=8)
    )

    assert identity["cohort_generation"] == 2
    assert identity["cohort_digest"] == new_digest
    assert state.publication_figures[0].is_current is False
    assert state.publication_tables[0].is_current is False
    assert "Canonical analysis cohort changed" in (
        state.publication_figures[0].invalidation_reason or ""
    )
    assert old_task.status == "abandoned"
    assert old_task.reason == "cohort_changed"
    assert find_figure_revision_task(
        state, plan_id=None, plan_requirement_key=None
    ) is None


def test_current_cohort_report_excludes_legacy_unverified_findings():
    from backend.services.report.lineage import build_lineage_report_state

    digest = "d" * 64
    state = AnalysisState(
        id="cohort-report",
        user_id="user",
        sandbox_state={
            "cohort_identity_v1": {
                **_probe(digest),
                "cohort_generation": 1,
                "lineage_id": "main",
                "round_number": 1,
            }
        },
        round_results=[RoundResult(
            round_number=1,
            lineage_id="main",
            bioinformatics_findings=["Legacy claim without a semantic receipt."],
            bioinformatics_finding_records=[],
        )],
    )

    report_state = build_lineage_report_state(state, "main")

    result = report_state.round_results[0]
    assert result.bioinformatics_findings == []
    assert result.bioinformatics_evidence_verified is False
    assert any(
        "no current Supervisor-confirmed finding records" in note
        for note in result.evidence_notes
    )
    assert state.round_results[0].bioinformatics_findings == [
        "Legacy claim without a semantic receipt."
    ]


def test_report_requires_current_cohort_generation_even_for_same_digest():
    from backend.services.report.lineage import build_lineage_report_state

    digest = "e" * 64
    old_execution = CodeExecution(
        id="execution-generation-1",
        code="old_result()",
        round_number=1,
        evidence_schema_version=2,
        execution_status="succeeded",
        lineage_id="main",
        cohort_generation=1,
        cohort_digest=digest,
    )
    current_execution = old_execution.model_copy(update={
        "id": "execution-generation-2",
        "code": "current_result()",
        "cohort_generation": 2,
    })
    old_record = {
        "id": "finding-generation-1",
        "text": "Old generation finding.",
        "lineage_id": "main",
        "cohort_generation": 1,
        "cohort_digest": digest,
        "evidence_status": "verified",
        "semantic_status": "evidenced",
        "evidence_execution_ids": [old_execution.id],
    }
    current_record = {
        **old_record,
        "id": "finding-generation-2",
        "text": "Current generation finding.",
        "cohort_generation": 2,
        "evidence_execution_ids": [current_execution.id],
    }
    state = AnalysisState(
        id="cohort-report-generation",
        user_id="user",
        sandbox_state={
            "cohort_identity_v1": {
                **_probe(digest),
                "cohort_generation": 2,
                "lineage_id": "main",
                "round_number": 1,
            }
        },
        code_history=[old_execution, current_execution],
        round_results=[RoundResult(
            round_number=1,
            lineage_id="main",
            bioinformatics_findings=[
                old_record["text"], current_record["text"],
            ],
            bioinformatics_finding_records=[old_record, current_record],
        )],
    )

    report_state = build_lineage_report_state(state, "main")

    assert [item.id for item in report_state.code_history] == [
        current_execution.id
    ]
    assert report_state.round_results[0].bioinformatics_findings == [
        "Current generation finding."
    ]
    assert state.round_results[0].bioinformatics_findings == [
        "Old generation finding.", "Current generation finding.",
    ]


def test_unchanged_cohort_does_not_advance_generation():
    digest = "c" * 64
    state = AnalysisState(id="cohort-stable", user_id="user", current_round=1)
    state.sandbox_state = {
        "cohort_identity_v1": {
            **_probe(digest),
            "cohort_generation": 4,
            "lineage_id": "main",
            "round_number": 1,
        }
    }

    identity = record_cohort_transition(state, _probe(digest), _probe(digest))

    assert identity["cohort_generation"] == 4
    assert identity["cohort_digest"] == digest
