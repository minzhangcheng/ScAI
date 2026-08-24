"""Focused A2 tests for authoritative executor context and API guidance."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.agents.bioinformatics_agent import BioinformaticsAgent
from backend.models.schemas import AnalysisState, DataFile


def test_main_and_supplementary_executors_receive_registered_nested_upload(tmp_path):
    bulk_path = tmp_path / "uploads" / "sequencing" / "_ungrouped" / "merged_counts.csv"
    bulk_path.parent.mkdir(parents=True)
    bulk_path.write_text("gene,Con1,DKO1\nA,1,2\n", encoding="utf-8")
    state = AnalysisState(
        user_id="owner",
        data_files=[
            DataFile(
                id="bulk-file-1",
                filename="merged_counts.csv",
                file_path=str(bulk_path),
                data_type="bulk RNA-seq",
                section="sequencing",
            ),
            DataFile(
                id="nested-note-1",
                filename="design.txt",
                file_path=str(tmp_path / "uploads" / "metadata" / "design.txt"),
                data_type="metadata",
                section="pathology",
            ),
        ],
    )
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)

    render_inventory = getattr(agent, "_render_registered_data_files", None)
    assert callable(render_inventory), "executor must expose registered upload inventory"
    inventory = render_inventory(state)
    main_context = agent._build_data_context(state)
    prompt = agent._build_supplementary_executor_prompt(
        ["Run replicate-aware bulk differential expression"],
        "Bulk analysis is pending.",
        "test",
        {"var": "adata", "n_obs": 10, "n_vars": 5},
        "",
        registered_data_files=inventory,
    )

    assert str(bulk_path) in inventory
    assert "bulk-file-1" in inventory
    assert "host_file_exists=true" in inventory
    assert str(bulk_path) in main_context
    assert "nested-note-1" in main_context
    assert "REGISTERED UPLOAD INVENTORY IS AUTHORITATIVE" in main_context
    assert str(bulk_path) in prompt
    assert "from pydeseq2.ds" in prompt


def test_executor_guidance_uses_current_pydeseq2_liana_and_index_contracts():
    try:
        evidence = importlib.import_module("backend.services.execution_evidence")
    except ModuleNotFoundError as exc:  # Product RED, not test collection noise.
        pytest.fail(f"A2 execution-evidence guidance service is missing: {exc}")

    pydeseq2 = evidence.PYDESEQ2_RUNTIME_GUIDANCE
    liana = evidence.LIANA_RUNTIME_GUIDANCE
    alignment = evidence.GENE_INDEX_ALIGNMENT_GUIDANCE
    assert "pydeseq2.dds" in pydeseq2
    assert "pydeseq2.ds" in pydeseq2
    assert "DeseqStats" in pydeseq2
    assert "rank_aggregate" in liana
    assert "mouseconsensus" in liana
    assert "groupby(level=0).sum()" in alignment
    assert "DataFrame.align(join='inner', axis=0)" in alignment
    assert "index.is_unique" in alignment


@pytest.mark.asyncio
async def test_supplementary_real_entry_receives_registered_inventory(tmp_path):
    upload = tmp_path / "uploads" / "sequencing" / "_ungrouped" / "counts.csv"
    upload.parent.mkdir(parents=True)
    upload.write_text("gene,ctrl\nA,1\n", encoding="utf-8")
    state = AnalysisState(
        user_id="owner",
        data_files=[DataFile(
            id="counts-1",
            filename="counts.csv",
            file_path=str(upload),
            data_type="bulk RNA-seq",
            section="sequencing",
        )],
    )
    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent._current_analysis_state = state
    agent.sandbox = SimpleNamespace(working_dir=str(tmp_path))
    agent.code_history = []
    agent.last_result = None
    agent.session_id = "supplementary-test"
    agent.introspect_validation_metadata = AsyncMock(
        return_value={"var": "adata", "cluster_key": "leiden"}
    )
    agent.capture_main_object_signature = AsyncMock(
        return_value={"n_obs": 1, "n_vars": 1, "obs_hash": "stable", "var_hash": "stable"}
    )
    agent.chat = AsyncMock(
        return_value='{"done": true, "result": {"status": "completed"}}'
    )

    result = await agent.run_supplementary_tasks(
        ["Summarize the registered input."],
        "The round is complete.",
        "test direction",
        max_iterations=1,
        timeout_seconds=1,
        total_timeout_seconds=1,
    )

    # No execution cell was produced, so the existing evidence policy correctly
    # downgrades the response; this test is about the real prompt wiring.
    assert result["status"] == "partial"
    prompt = agent.chat.await_args.args[0]
    assert "counts-1" in prompt
    assert str(upload) in prompt
    assert "Registered uploaded data files (authoritative)" in prompt
