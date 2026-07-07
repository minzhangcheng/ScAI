"""Integration-gate robustness: a generic continue (e.g. /human-input with no
direction selection) must still RESOLVE the gate instead of leaking an
unresolved INTEGRATION state into a brand-new round.

Repro of the field bug (session 7e0c5ae8...): user clicked the generic
"continue" at the integration gate without choosing a direction. The gate was
left unresolved (analysis_phase stayed INTEGRATION, integration_gate_candidates
stayed populated), routing then sent the graph to initialize_round and a fresh
round started with no integration direction selected.
"""
import asyncio
import types

from backend.constants import PI_DECIDE_SIGNAL
from backend.models.schemas import AnalysisState, AnalysisPhase, TaskStatus
from backend.workflow.synthesis import SynthesisMixin
from backend.workflow.direction_coverage import at_integration_gate


class _Harness(SynthesisMixin):
    def __init__(self):
        self.session_id = "sess"
        self.config = types.SimpleNamespace(max_rounds=5)
        self._continue_event = asyncio.Event()
        self._supplementary_queue = []
        self.sent = []

    async def _send_message(self, t, c, agent=None):
        self.sent.append((t, c))

    async def _send_state_sync(self, state, change_type):
        self.sent.append(("state_sync", change_type))

    def _set_status(self, state, status):
        state.status = status


def _integration_gate_state() -> AnalysisState:
    s = AnalysisState(name="t", user_id="u", initial_instruction="o")
    s.analysis_phase = AnalysisPhase.INTEGRATION
    s.status = TaskStatus.WAITING_INPUT
    s.round_wait_seconds = -1  # manual mode (matches the field session)
    s.current_round = 6
    s.integration_cycle = 1
    s.max_integration_cycles = 3
    s.integration_gate_candidates = [{"direction": "NEW", "direction_id": "int_001"}]
    return s


def test_generic_continue_resolves_integration_gate():
    """Signaling continue WITHOUT choosing a direction must resolve the gate:
    switch to INNOVATION + PI_DECIDE_SIGNAL so pi_planning selects directions.
    The gate must NOT be left open (phase INTEGRATION + candidates populated)."""
    h = _Harness()
    s = _integration_gate_state()
    h._continue_event.set()  # generic /human-input signalled continue, no direction

    out = asyncio.run(h._wait_for_user_input({"analysis_state": s}))

    assert out["should_continue"] is True
    # The gate must be resolved, not leaked into the next round.
    assert not at_integration_gate(s), (
        "integration gate left unresolved: phase="
        f"{s.analysis_phase} candidates={bool(s.integration_gate_candidates)}"
    )
    assert s.analysis_phase == AnalysisPhase.INNOVATION
    assert s.user_chosen_direction == PI_DECIDE_SIGNAL
