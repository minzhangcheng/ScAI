"""Active-cohort identity and stale-artifact invalidation contracts."""

from backend.models.schemas import (
    AnalysisStep,
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


def test_cohort_transition_retains_prior_artifacts_with_warning():
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
    assert state.publication_figures[0].is_current is True
    assert state.publication_tables[0].is_current is True
    assert "remains included" in state.publication_figures[0].description
    assert "remains included" in state.publication_tables[0].description
    assert old_task.status == "exhausted_excluded"
    assert old_task.reason == "attempts_exhausted"
    assert find_figure_revision_task(
        state, plan_id=None, plan_requirement_key=None
    ) is old_task


def test_current_cohort_report_retains_legacy_findings_as_pi_observations():
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
    assert result.bioinformatics_findings == [
        "Legacy claim without a semantic receipt."
    ]
    assert result.bioinformatics_evidence_verified is None
    assert any(
        "Execution linkage was not recorded" in note
        for note in result.evidence_notes
    )
    assert state.round_results[0].bioinformatics_findings == [
        "Legacy claim without a semantic receipt."
    ]


def test_current_cohort_report_retains_structured_unverified_findings():
    from backend.services.report.lineage import build_lineage_report_state

    digest = "f" * 64
    record = {
        "id": "unverified-current-finding",
        "text": "Current finding with unavailable semantic reconciliation.",
        "lineage_id": "main",
        "cohort_generation": 1,
        "cohort_digest": digest,
        "evidence_status": "unverified",
        "semantic_status": "unverified",
        "evidence_execution_ids": [],
    }
    state = AnalysisState(
        id="cohort-report-unverified",
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
            bioinformatics_findings=[record["text"]],
            bioinformatics_finding_records=[record],
        )],
    )

    result = build_lineage_report_state(state, "main").round_results[0]

    assert result.bioinformatics_findings == [record["text"]]
    assert result.bioinformatics_finding_records[0]["evidence_status"] == "unverified"
    assert result.bioinformatics_evidence_verified is False


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
        "semantic_status": "unverified",
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
        publication_figures=[PublicationFigure(
            name="old generation figure", description="historical render",
            figure_type="other", png_path="old.png", pdf_path="old.pdf",
            round_number=1, lineage_id="main", cohort_generation=1,
            cohort_digest=digest, source_execution_id=old_execution.id,
        )],
        publication_tables=[PublicationTable(
            name="old generation table", description="historical table",
            csv_path="old.csv", round_number=1, lineage_id="main",
            cohort_generation=1, cohort_digest=digest,
            source_execution_id=old_execution.id,
            is_current=False,
            invalidation_reason=(
                "Canonical analysis cohort changed; artifact belongs to an "
                "earlier cohort."
            ),
        )],
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
        old_execution.id, current_execution.id,
    ]
    assert report_state.round_results[0].bioinformatics_findings == [
        "Old generation finding.",
        "Current generation finding.",
    ]
    assert report_state.round_results[0].bioinformatics_evidence_verified is True
    assert report_state.round_results[0].evidence_execution_ids == [
        old_execution.id,
        current_execution.id,
    ]
    assert [figure.name for figure in report_state.publication_figures] == [
        "old generation figure"
    ]
    assert [table.name for table in report_state.publication_tables] == [
        "old generation table"
    ]
    assert state.round_results[0].bioinformatics_findings == [
        "Old generation finding.", "Current generation finding.",
    ]


def test_report_keeps_historical_figure_execution_without_a_table():
    from backend.services.report.lineage import build_lineage_report_state

    digest = "7" * 64
    old_execution = CodeExecution(
        id="old-figure-execution", code="draw_old_figure()", round_number=1,
        evidence_schema_version=2, execution_status="succeeded",
        lineage_id="main", analysis_generation=1, cohort_generation=1,
        cohort_digest=digest,
    )
    current_execution = old_execution.model_copy(update={
        "id": "current-execution", "code": "current_analysis()",
        "analysis_generation": 2, "cohort_generation": 2,
    })
    state = AnalysisState(
        id="historical-figure-receipt", user_id="user",
        sandbox_state={
            "analysis_generation": 2,
            "cohort_identity_v1": {
                **_probe(digest), "cohort_generation": 2,
                "lineage_id": "main",
            },
        },
        code_history=[old_execution, current_execution],
        analysis_log=[
            AnalysisStep(
                step_number=1, step_type="execute_code",
                execution_id=old_execution.id, round_number=1,
            ),
            AnalysisStep(
                step_number=2, step_type="execute_code",
                execution_id=current_execution.id, round_number=1,
            ),
        ],
        publication_figures=[PublicationFigure(
            name="historical figure", description="original caption",
            figure_type="other", png_path="old.png", pdf_path="old.pdf",
            round_number=1, lineage_id="main", analysis_generation=1,
            cohort_generation=1, cohort_digest=digest,
            source_execution_id=old_execution.id,
            is_current=False,
            invalidation_reason=(
                "Canonical analysis cohort changed; artifact belongs to an "
                "earlier cohort."
            ),
        )],
    )

    report_state = build_lineage_report_state(state, "main")

    assert [item.id for item in report_state.code_history] == [
        old_execution.id, current_execution.id,
    ]
    assert [item.execution_id for item in report_state.analysis_log] == [
        old_execution.id, current_execution.id,
    ]
    assert "recorded cohort identity" in (
        report_state.publication_figures[0].description
    )


def test_report_rejects_failed_and_cross_plan_execution_links():
    from backend.services.report.lineage import build_lineage_report_state

    digest = "9" * 64
    failed = CodeExecution(
        id="failed-current-execution",
        code="raise RuntimeError()",
        error="failed",
        round_number=1,
        evidence_schema_version=2,
        execution_status="failed",
        plan_id="plan-a",
        lineage_id="main",
        cohort_generation=1,
        cohort_digest=digest,
    )
    other_plan = failed.model_copy(update={
        "id": "other-plan-execution",
        "code": "successful_other_plan()",
        "error": None,
        "execution_status": "succeeded",
        "plan_id": "plan-b",
    })
    records = [
        {
            "id": "failed-link",
            "text": "Finding linked to failed execution.",
            "round_number": 1,
            "plan_id": "plan-a",
            "lineage_id": "main",
            "cohort_generation": 1,
            "cohort_digest": digest,
            "evidence_status": "verified",
            "evidence_execution_ids": [failed.id],
        },
        {
            "id": "cross-plan-link",
            "text": "Finding linked to another plan.",
            "round_number": 1,
            "plan_id": "plan-a",
            "lineage_id": "main",
            "cohort_generation": 1,
            "cohort_digest": digest,
            "evidence_status": "verified",
            "evidence_execution_ids": [other_plan.id],
        },
    ]
    state = AnalysisState(
        id="cohort-report-invalid-links",
        user_id="user",
        sandbox_state={"cohort_identity_v1": {
            **_probe(digest), "cohort_generation": 1, "lineage_id": "main",
        }},
        code_history=[failed, other_plan],
        publication_tables=[
            PublicationTable(
                name="failed table", description="failed source",
                csv_path="failed.csv", plan_id="plan-a", round_number=1,
                plan_binding_status="candidate", source_execution_id=failed.id,
                lineage_id="main", cohort_generation=1, cohort_digest=digest,
            ),
            PublicationTable(
                name="cross-plan table", description="wrong plan source",
                csv_path="cross.csv", plan_id="plan-a", round_number=1,
                plan_binding_status="candidate", source_execution_id=other_plan.id,
                lineage_id="main", cohort_generation=1, cohort_digest=digest,
            ),
        ],
        round_results=[RoundResult(
            round_number=1,
            lineage_id="main",
            bioinformatics_findings=[record["text"] for record in records],
            bioinformatics_finding_records=records,
        )],
    )

    report_state = build_lineage_report_state(state, "main")
    result = report_state.round_results[0]

    assert [execution.id for execution in report_state.code_history] == [other_plan.id]
    assert report_state.publication_tables == []
    assert result.bioinformatics_evidence_verified is False
    assert result.evidence_execution_ids == []
    assert all(
        record["evidence_status"] == "unverified"
        and record["evidence_execution_ids"] == []
        for record in result.bioinformatics_finding_records
    )


def test_report_rejects_stale_generation_and_failed_validation_output():
    from backend.services.report.lineage import build_lineage_report_state

    digest = "8" * 64
    stale = CodeExecution(
        id="stale-analysis-generation",
        code="old_analysis()",
        round_number=1,
        evidence_schema_version=2,
        execution_status="succeeded",
        plan_id="plan-a",
        lineage_id="main",
        analysis_generation=1,
        cohort_generation=1,
        cohort_digest=digest,
    )
    rejected = stale.model_copy(update={
        "id": "failed-validation-output",
        "output": "Validation failed: required result was not produced.",
        "analysis_generation": 2,
    })
    records = [
        {
            "id": f"finding-{execution.id}",
            "text": f"Finding linked to {execution.id}.",
            "round_number": 1,
            "plan_id": "plan-a",
            "lineage_id": "main",
            "analysis_generation": 2,
            "cohort_generation": 1,
            "cohort_digest": digest,
            "evidence_status": "verified",
            "evidence_execution_ids": [execution.id],
        }
        for execution in (stale, rejected)
    ]
    state = AnalysisState(
        id="cohort-report-analysis-generation",
        user_id="user",
        sandbox_state={
            "analysis_generation": 2,
            "cohort_identity_v1": {
                **_probe(digest),
                "cohort_generation": 1,
                "lineage_id": "main",
            },
        },
        code_history=[stale, rejected],
        round_results=[RoundResult(
            round_number=1,
            lineage_id="main",
            bioinformatics_findings=[record["text"] for record in records],
            bioinformatics_finding_records=records,
        )],
    )

    report_state = build_lineage_report_state(state, "main")
    result = report_state.round_results[0]

    assert [item.id for item in report_state.code_history] == [stale.id]
    assert result.bioinformatics_evidence_verified is False
    assert result.evidence_execution_ids == []
    assert [record["id"] for record in result.bioinformatics_finding_records] == [
        f"finding-{stale.id}",
        f"finding-{rejected.id}",
    ]
    assert all(
        record["evidence_status"] == "unverified"
        for record in result.bioinformatics_finding_records
    )


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
