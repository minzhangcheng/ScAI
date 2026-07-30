"""_create_auto_snapshot honors a one-shot skip marker.

When the user chose "Skip snapshot & continue", the next required workflow-boundary
capture for that phase must be a no-op (no SnapshotService construction / capture),
so the graph resumes past the boundary that previously failed.
"""
import pytest

from backend.models.schemas import AnalysisState, TaskStatus
from backend.workflow.core import SingleCellAnalysisWorkflow


# Certifies Snapshot V2-enabled behaviour. The operator gate ships off, so pin it
# on here instead of inheriting it from the environment: these assertions describe
# the enabled lifecycle and are meaningless (and were failing) with it off.
pytestmark = pytest.mark.requires_snapshot_v2

def _bare_workflow():
    wf = object.__new__(SingleCellAnalysisWorkflow)
    wf.session_id = "sess-skip"
    wf._current_graph_state = {}
    wf.sandbox = None
    wf.snapshot_service = None
    sent = []

    async def _send_message(kind, content, *a, **kw):
        sent.append((kind, content))

    wf._send_message = _send_message
    wf._sent = sent
    return wf


def _state():
    return AnalysisState(id="sess-skip", name="T", user_id="u1",
                         initial_instruction="analyze")


@pytest.mark.asyncio
async def test_armed_skip_makes_capture_a_noop():
    wf = _bare_workflow()
    st = _state()
    st.skip_required_snapshot_phase = "pre_annotation"

    result = await wf._create_auto_snapshot(st, "pre_annotation")

    assert result is None
    # One-shot marker consumed.
    assert st.skip_required_snapshot_phase is None
    # No SnapshotService was created for a skipped boundary.
    assert wf.snapshot_service is None


@pytest.mark.asyncio
async def test_non_matching_phase_does_not_skip():
    wf = _bare_workflow()
    st = _state()
    st.skip_required_snapshot_phase = "pre_annotation"

    # A different phase must NOT consume the marker; it proceeds into capture.
    # We assert the marker is untouched (capture will fail later on the mock, but
    # the point is the skip did not apply). Capture construction may raise/return;
    # we only care the marker was not consumed for a non-matching phase.
    try:
        await wf._create_auto_snapshot(st, "pre_direction")
    except Exception:
        pass
    assert st.skip_required_snapshot_phase == "pre_annotation"
