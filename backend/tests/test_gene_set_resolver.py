"""Gene sets come from a versioned resource, not from the model's memory.

Sandbox code cannot reach the network, so before this the agent had no way to
fetch a curated set and every panel it scored was remembered. These tests cover
the pure resolution logic offline; the network path is exercised separately.

The cross-organism behaviour is the part worth pinning: the upstream resource
publishes two organisms natively, everything else is projected, and a projection
that loses most of the set must say so rather than passing as equivalent.
"""


import backend.agents.bioinformatics_agent as bio_mod
from backend.services.gene_set_resolver import (
    validate_set_id,
)


def test_validate_set_id_rejects_injection_shaped_input():
    assert validate_set_id("../../etc/passwd") is not None


def test_a_bind_the_kernel_refused_reports_no_variable():
    """The kernel says no in the result far more often than it raises.

    A dead or restarting kernel, an execution timeout and a MemoryError on a
    large panel all come back as an ordinary result carrying .error. Returning
    the variable name anyway makes the caller suppress the inline gene list, so
    the next cell dies on NameError or the model retypes the panel from memory
    -- the fetched-panel-becomes-remembered-panel failure this helper exists to
    prevent.
    """
    import asyncio
    from types import SimpleNamespace

    from backend.utils.sandbox_parts.shared import CodeExecutionResult

    class _Kernel:
        def __init__(self, result):
            self.result = result

        async def execute(self, code, **kwargs):
            return self.result

    refused = CodeExecutionResult(
        success=False, output="",
        error="Kernel died during execution and restart failed.")
    bind = bio_mod.BioinformaticsAgent._bind_gene_set

    agent = SimpleNamespace(session_id="t", sandbox=_Kernel(refused))
    assert asyncio.run(bind(agent, "HALLMARK_X", ["TP53", "MYC"], "gene set")) is None

    agent = SimpleNamespace(
        session_id="t", sandbox=_Kernel(CodeExecutionResult(success=True)))
    assert asyncio.run(
        bind(agent, "HALLMARK_X", ["TP53", "MYC"], "gene set")
    ) == "gene_set_HALLMARK_X"
