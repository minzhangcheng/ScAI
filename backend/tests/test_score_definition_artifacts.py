from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.agents.bioinformatics_agent import TOOLS
from backend.agents.bioinformatics import executor_registration
from backend.models.schemas import (
    AnalysisState,
    AnalysisStep,
    FigureReviewStatus,
    PublicationTable,
    ScoreDefinition,
    ScoreDefinitionArtifactRef,
)
from backend.services.report.score_section import (
    build_score_definitions_section_html,
    ensure_score_definition_tables,
    write_score_definitions_csv,
)


def test_register_score_schema_supports_structured_definitions():
    tool = next(
        item for item in TOOLS
        if item["function"]["name"] == "register_score"
    )
    parameters = tool["function"]["parameters"]

    assert "definition_kind" in parameters["required"]
    assert parameters["properties"]["definition_kind"]["enum"] == [
        "gene_panel",
        "gene_set_collection",
        "regulatory_network",
    ]
    artifact = parameters["properties"]["definition_artifacts"]["items"]
    assert artifact["required"] == ["role", "table_id"]
    assert artifact["properties"]["role"]["enum"] == [
        "gene_set_membership",
        "regulatory_network",
        "ranked_input",
        "leading_edge",
    ]


def test_score_definition_artifact_round_trip_and_legacy_default():
    score = ScoreDefinition(
        score_name="hallmark_gsea",
        method="gseapy.prerank",
        description="Hallmark collection enrichment",
        definition_kind="gene_set_collection",
        definition_status="complete",
        definition_artifacts=[ScoreDefinitionArtifactRef(
            role="gene_set_membership",
            table_id="table-members",
        )],
    )

    restored = ScoreDefinition.model_validate(score.model_dump(mode="json"))
    assert restored.definition_kind == "gene_set_collection"
    assert restored.definition_status == "complete"
    assert restored.definition_artifacts[0].table_id == "table-members"

    legacy = ScoreDefinition(
        score_name="legacy",
        method="score_genes",
        description="Old saved record",
        gene_list=["G1"],
    )
    assert legacy.definition_kind == "unspecified"
    assert legacy.definition_status == "legacy"


def test_report_links_structured_definition_table_and_exports_identity(tmp_path):
    score = ScoreDefinition(
        score_name="hallmark_gsea",
        method="gseapy.prerank",
        description="Hallmark collection enrichment",
        gene_list_source="database:MSigDB:2024.1:mh.all",
        definition_kind="gene_set_collection",
        definition_status="complete",
        definition_artifacts=[{
            "role": "gene_set_membership",
            "table_id": "table-members",
        }],
    )
    tables = [{
        "table_id": "table-members",
        "filename": "hallmark_members.csv",
        "name": "Hallmark membership",
        "total_rows": 7341,
        "csv_sha256": "a" * 64,
    }]

    html = build_score_definitions_section_html([score], table_records=tables)
    assert 'href="tables/hallmark_members.csv"' in html
    assert "Gene-set membership" in html
    assert "7,341 rows" in html
    assert "Definition artifact not registered" not in html

    csv_path = write_score_definitions_csv([score], tmp_path)
    exported = Path(csv_path).read_text(encoding="utf-8")
    assert "Definition Kind" in exported
    assert "gene_set_collection" in exported
    assert "gene_set_membership:table-members" in exported


def test_report_makes_an_incomplete_structured_definition_explicit():
    score = ScoreDefinition(
        score_name="unlinked_network",
        method="decoupler.run_ulm",
        description="Network activity",
        definition_kind="regulatory_network",
        definition_status="incomplete",
    )

    html = build_score_definitions_section_html([score])
    assert "Definition artifact not registered" in html
    assert "N/A" not in html.split("unlinked_network", 1)[1].split("</tr>", 1)[0]


def test_referenced_definition_tables_are_included_deterministically():
    scores = [ScoreDefinition(
        score_name="network_activity",
        method="decoupler.run_ulm",
        description="Network activity",
        definition_kind="regulatory_network",
        definition_status="complete",
        definition_artifacts=[{
            "role": "regulatory_network",
            "table_id": "network-table",
        }],
    )]
    available = [
        {"table_id": "other", "filename": "other.csv"},
        {"table_id": "network-table", "filename": "network.csv"},
    ]
    generated = [{"table_id": "other", "filename": "other.csv"}]

    missing = ensure_score_definition_tables(scores, available, generated)

    assert missing == []
    assert [item["table_id"] for item in generated] == [
        "other", "network-table",
    ]


@pytest.mark.asyncio
async def test_non_panel_registration_uses_artifact_not_matched_gene_power(
    monkeypatch, tmp_path,
):
    table_path = tmp_path / "hallmark_members.csv"
    table_path.write_text("set_id,gene\nHALLMARK_A,G1\n", encoding="utf-8")
    table = PublicationTable(
        id="table-members",
        name="Hallmark membership",
        description="Exact set membership",
        csv_path=str(table_path),
        csv_sha256="a" * 64,
        columns=["set_id", "gene"],
        total_rows=1,
        review_status=FigureReviewStatus.APPROVED,
    )
    state = AnalysisState(
        id="session",
        user_id="owner",
        current_round=2,
        publication_tables=[table],
    )
    context = SimpleNamespace(
        agent=SimpleNamespace(_notify=AsyncMock()),
        state=state,
        messages=[],
        analysis_log=[],
        step_count=3,
        agent_reasoning="",
        evidence_gate_findings=[],
    )
    tool_call = SimpleNamespace(
        id="score-call",
        arguments={
            "score_name": "hallmark_gsea",
            "method": "gseapy.prerank",
            "description": "Hallmark collection enrichment",
            "gene_list_source": "database:MSigDB:2024.1:mh.all",
            "definition_kind": "gene_set_collection",
            "definition_artifacts": [{
                "role": "gene_set_membership",
                "table_id": table.id,
            }],
            # This was the historical bug: ranked-universe size was passed as
            # though it were the number of matched genes in one panel.
            "n_genes_matched": 19000,
        },
    )
    monkeypatch.setattr(executor_registration, "logger", MagicMock(), raising=False)
    monkeypatch.setattr(executor_registration, "AnalysisStep", AnalysisStep, raising=False)
    monkeypatch.setattr(
        executor_registration,
        "_method_rule_text",
        lambda _rule_id: "",
        raising=False,
    )
    monkeypatch.setattr(
        executor_registration,
        "_evidence_gate_record",
        lambda **kwargs: dict(kwargs),
        raising=False,
    )

    await executor_registration.handle_register_score(context, tool_call)

    score = state.score_definitions[0]
    assert score.definition_kind == "gene_set_collection"
    assert score.definition_status == "complete"
    assert score.definition_artifacts[0].table_id == table.id
    assert score.n_genes == 0
    assert score.n_genes_matched is None
    assert score.power == {}
    assert any(
        "n_genes_matched applies only to gene_panel" in note
        for note in score.plan_drift["pi_review_observations"]
    )
