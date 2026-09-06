"""S2 contracts for validated native gene-set collection binding."""

from __future__ import annotations

import asyncio
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


async def _run_fetch_list_sets_branch(tmp_path, monkeypatch, *, organism: str):
    from backend.agents.bioinformatics_agent import BioinformaticsAgent
    from backend.models.schemas import AgentType, AnalysisState
    from backend.tests.split_facade import setattr_facade
    from backend.utils.llm_client import LLMToolResponse, ToolCall
    import backend.services.gene_set_fetch as fetch_module

    events: list[str] = []
    client_messages: list[list[dict]] = []
    available = {"SET_A": ["GeneA", "GeneB"]}
    exact_path = tmp_path / "m5.go.bp.v2025.1.Mm.symbols.gmt"

    def fake_load(version, collection, browse_organism):
        events.append(f"load:{version}:{collection}:{browse_organism}")
        return available

    def fake_path(version, collection, requested_organism):
        events.append(
            f"path:{version}:{collection}:{requested_organism}:{exact_path}"
        )
        return exact_path

    monkeypatch.setattr(fetch_module, "load_collection", fake_load)
    monkeypatch.setattr(fetch_module, "collection_cache_path", fake_path)

    agent = BioinformaticsAgent.__new__(BioinformaticsAgent)
    agent.session_id = "gene-set-loop-test"
    agent.code_history = []
    agent.analysis_context = {}
    agent.last_result = None
    agent._recent_sup_feedback = set()
    agent._pause_event = None
    agent.literature_callback = None
    agent.pi_callback = None
    agent.annotate_clusters_callback = None
    agent.message_callback = None
    agent.figure_review_callback = None
    agent.supervisor_callback = None
    agent.species_validation_callback = None
    agent.marker_db_intervention_callback = None
    agent.technique_agent = None
    agent._provider = MagicMock()
    agent._provider.get_config_revision = MagicMock(return_value=0)
    agent._last_config_revision = 0
    agent.agent_type = AgentType.BIOINFORMATICS

    monkeypatch.setattr(BioinformaticsAgent, "model", property(lambda _self: "test-model"))
    monkeypatch.setattr(
        BioinformaticsAgent, "system_prompt", property(lambda _self: "test prompt")
    )
    monkeypatch.setattr(
        BioinformaticsAgent, "config", property(lambda _self: MagicMock(temperature=0.7))
    )
    monkeypatch.setattr(
        BioinformaticsAgent,
        "_static_system_prompt",
        property(lambda _self: "static test prompt"),
    )
    monkeypatch.setattr(
        BioinformaticsAgent, "_build_dynamic_context", lambda _self, _state: "dynamic"
    )
    monkeypatch.setattr(BioinformaticsAgent, "_set_llm_context", lambda _self: None)

    async def fake_bind(_self, collection, path):
        events.append(f"bind:{collection}:{path}")
        return "gene_set_collection_m5_go_bp_path"

    monkeypatch.setattr(BioinformaticsAgent, "_bind_gene_set_collection", fake_bind)

    fetch_call = ToolCall(
        id="fetch-1",
        name="fetch_gene_set",
        arguments={
            "version": "2025.1",
            "collection": "m5.go.bp",
            "organism": organism,
            "set_id": "SET_A",
            "list_sets": True,
        },
    )
    responses = [
        LLMToolResponse(
            content="",
            tool_calls=[fetch_call],
            model="test-model",
            usage={},
            raw_response=None,
        ),
        LLMToolResponse(
            content="done",
            tool_calls=[],
            model="test-model",
            usage={},
            raw_response=None,
        ),
    ]

    async def fake_chat_with_tools(*, messages, **_kwargs):
        client_messages.append([dict(message) for message in messages])
        events.append(f"llm{len(client_messages)}")
        return responses.pop(0)

    fake_client = MagicMock(base_url="https://fake.example.com/v1")
    fake_client.chat_with_tools = fake_chat_with_tools
    monkeypatch.setattr(
        BioinformaticsAgent, "llm_client", property(lambda _self: fake_client)
    )

    agent._notify = AsyncMock()
    agent.sandbox = MagicMock()
    agent.sandbox.state_dir = tmp_path / "states"
    agent.sandbox.working_dir = tmp_path / "workspace"
    agent.sandbox.km = MagicMock()
    agent.sandbox.start = AsyncMock()
    agent.sandbox.save_session = AsyncMock()
    agent.sandbox.sync_runtime_persistence_state = MagicMock(return_value={})

    class _NullCheckpoint:
        def __init__(self, *args, **kwargs):
            pass

        async def load(self):
            return None

        async def save(self, data):
            return None

        def clear(self):
            pass

        def should_checkpoint(self, step):
            return False

    setattr_facade(
        monkeypatch,
        "backend.agents.bioinformatics_agent",
        "CheckpointManager",
        _NullCheckpoint,
    )
    state = AnalysisState(
        id="gene-set-loop-test", user_id="test-user", species=organism
    )
    result = await agent.run_autonomous_analysis(
        state,
        initial_instruction="List the selected collection.",
        max_steps=2,
        resume_from_checkpoint=False,
    )
    return result, events, client_messages


def _fetch_schema() -> dict:
    from backend.agents.bioinformatics.tool_schemas import TOOLS

    return next(
        tool["function"]
        for tool in TOOLS
        if tool["function"]["name"] == "fetch_gene_set"
    )


@pytest.mark.parametrize(
    ("version", "collection", "organism"),
    [
        ("2025.1", "../escape", "mouse"),
        ("2025.1", "c2.cp.reactome", "danio rerio"),
        ("mh.all", "m5.go.bp", "mouse"),
        ("2025.1.Hs", "m5.go.bp", "mouse"),
    ],
)
def test_collection_cache_path_rejects_invalid_collection_owner_release_and_suffix(
    version: str, collection: str, organism: str,
):
    from backend.services.gene_set_fetch import collection_cache_path

    with pytest.raises(ValueError):
        collection_cache_path(version, collection, organism)


def test_collection_cache_path_normalizes_bare_and_stamped_release(tmp_path):
    from backend.services.gene_set_fetch import collection_cache_path

    bare = collection_cache_path("2025.1", "m5.go.bp", "mouse", tmp_path)
    stamped = collection_cache_path("2025.1.Mm", "m5.go.bp", "mouse", tmp_path)

    assert bare == stamped == tmp_path / "m5.go.bp.v2025.1.Mm.symbols.gmt"


def test_load_collection_uses_collection_cache_path(tmp_path, monkeypatch):
    from backend.services import gene_set_fetch

    canonical = tmp_path / "canonical.gmt"
    canonical.write_text("SET_A\tdescription\tGeneA\tGeneB\n", encoding="utf-8")
    calls: list[tuple[str, str, str, Path]] = []

    def fake_path(version, collection, organism, cache_dir=None):
        calls.append((version, collection, organism, Path(cache_dir)))
        return canonical

    monkeypatch.setattr(gene_set_fetch, "collection_cache_path", fake_path)
    loaded = gene_set_fetch.load_collection(
        "2025.1.Mm", "m5.go.bp", "mouse", cache_dir=tmp_path
    )

    assert loaded == {"SET_A": ["GeneA", "GeneB"]}
    assert calls == [("2025.1.Mm", "m5.go.bp", "mouse", tmp_path)]


@pytest.mark.asyncio
async def test_native_list_sets_loads_then_binds_exact_collection_path(tmp_path, monkeypatch):
    result, events, client_messages = await _run_fetch_list_sets_branch(
        tmp_path, monkeypatch, organism="mouse"
    )

    assert events == [
        "llm1",
        "load:2025.1:m5.go.bp:mouse",
        f"path:2025.1:m5.go.bp:mouse:{tmp_path / 'm5.go.bp.v2025.1.Mm.symbols.gmt'}",
        f"bind:m5.go.bp:{tmp_path / 'm5.go.bp.v2025.1.Mm.symbols.gmt'}",
        "llm2",
    ]
    tool_message = next(
        message for message in client_messages[1] if message.get("role") == "tool"
    )
    assert "gene_set_collection_m5_go_bp_path" in tool_message["content"]
    assert "gene_sets=" in tool_message["content"]
    assert result["status"] in {"max_steps_reached", "completed_with_fallback"}


@pytest.mark.asyncio
async def test_projected_discovery_uses_source_without_binding_source_gmt(tmp_path, monkeypatch):
    result, events, client_messages = await _run_fetch_list_sets_branch(
        tmp_path, monkeypatch, organism="danio rerio"
    )

    assert events == [
        "llm1",
        "load:2025.1:m5.go.bp:mouse",
        "llm2",
    ]
    tool_message = next(
        message for message in client_messages[1] if message.get("role") == "tool"
    )
    content = tool_message["content"]
    assert "mouse" in content
    assert "danio rerio" in content
    assert "GMT is NOT bound" in content
    assert "set_id" in content
    assert "mapping rate" in content
    assert result["status"] in {"max_steps_reached", "completed_with_fallback"}


def test_fetch_schema_explains_native_binding_and_projected_fetch():
    function = _fetch_schema()
    text = function["description"] + " " + function["parameters"]["properties"][
        "list_sets"
    ]["description"]

    assert "full" in text.lower()
    assert "GMT" in text
    assert "gene_sets=" in text
    assert "projection" in text.lower() or "projected" in text.lower()
    assert "not bound" in text.lower() or "source-organism" in text.lower()


@pytest.mark.skipif(
    not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
    reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
)
@pytest.mark.asyncio
async def test_live_native_collection_binding_runs_gseapy_prerank(tmp_path):
    from backend.agents.bioinformatics_agent import BioinformaticsAgent
    from backend.services.gene_set_fetch import collection_cache_path
    from backend.utils.sandbox import LocalJupyterSandbox

    cache_dir = tmp_path / "gene_sets"
    cache_dir.mkdir()
    path = collection_cache_path("2025.1", "m5.go.bp", "mouse", cache_dir)
    path.write_text("SET_A\tdescription\tGeneA\tGeneB\n", encoding="utf-8")

    sandbox = LocalJupyterSandbox(
        session_id="s2-native-gene-set",
        state_dir=str(tmp_path / "states"),
        working_dir=str(tmp_path / "workspace"),
    )
    await sandbox.start()
    try:
        agent = SimpleNamespace(session_id="s2-native-gene-set", sandbox=sandbox)
        variable = await BioinformaticsAgent._bind_gene_set_collection(
            agent, "m5.go.bp", path
        )
        assert variable == "gene_set_collection_m5_go_bp_path"

        result = await sandbox.execute(
            "import pandas as pd\n"
            "import gseapy\n"
            "ranking = pd.Series([3.0, 2.0, 1.0], index=['GeneA', 'GeneB', 'GeneC'])\n"
            f"enrichment = gseapy.prerank(rnk=ranking, gene_sets={variable}, "
            "permutation_num=2, min_size=1, max_size=10, outdir=None, "
            "no_plot=True, seed=0)\n"
            "assert 'SET_A' in enrichment.res2d['Term'].tolist()\n",
            records_history=False,
            skip_safety_check=True,
        )
        assert result.success, result.error
    finally:
        await sandbox.stop(save=False)
