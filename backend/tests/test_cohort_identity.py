"""Active-cohort identity and stale-artifact invalidation contracts."""

from backend.models.schemas import (
    AnalysisState,
    CodeExecution,
    PublicationFigure,
    PublicationTable,
    RoundResult,
)


def _probe(digest: str, n_obs: int = 10):
    return {
        "present": True,
        "canonical_object_name": "adata",
        "cohort_digest": digest,
        "n_obs": n_obs,
        "n_vars": 5,
    }


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
