"""S2 contracts for validated native gene-set collection binding."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


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
