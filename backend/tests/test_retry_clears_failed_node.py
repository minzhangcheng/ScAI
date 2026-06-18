"""Node 成功执行后，failed_node 必须被清为 None。"""
import pytest
from backend.workflow.retry import with_retry


class _Fake:
    session_id = "s"
    _pause_event = None
    async def _send_message(self, *a, **k): pass

    @with_retry("biochemistry_analysis")
    async def node(self, state):
        return {**state, "biochemistry_result": {"findings": ["ok"], "status": "completed"}}


@pytest.mark.asyncio
async def test_success_clears_failed_node():
    obj = _Fake()
    state = {"skip_nodes": [], "failed_node": "pathology_analysis"}
    out = await obj.node(state)
    assert out["failed_node"] is None
