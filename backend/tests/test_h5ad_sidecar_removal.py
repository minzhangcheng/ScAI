"""The runtime h5ad sidecar is gone; a runtime generation is the pkl.

Spec 3.4 established the sidecar is dead code with the snapshot gate off: the
only producer is Snapshot V2's capture, and ``atomic_replace_h5ad`` is only ever
called from its restore. Spec 4.2 keeps backed-AnnData materialisation, which is
a different thing -- the pkl carries the AnnData itself.

Analysis-level ``.h5ad`` -- the files the user's code and the executor prompt
read and write -- is the platform's input format and is untouched by this batch.
"""
from __future__ import annotations

import pathlib


REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"


def test_the_pairing_probe_no_longer_treats_a_sidecar_manifest_as_a_runtime():
    """Once nothing writes one, its presence can only be a false negative."""
    source = (BACKEND / "utils" / "runtime_pairing.py").read_text()
    assert "H5AD" not in source.upper()


def test_a_session_with_neither_pkl_nor_sidecar_is_reported_unavailable(tmp_path, monkeypatch):
    from backend.models.schemas import AnalysisState
    from backend.utils import runtime_pairing

    states = tmp_path / "states"
    states.mkdir(parents=True)
    (states / "checkpoints.sqlite").write_bytes(b"")

    monkeypatch.setattr(
        "backend.config.get_session_states", lambda sid: states, raising=False
    )
    monkeypatch.setattr(
        runtime_pairing, "_pairing_states_dir", lambda sid: states, raising=False
    )

    state = AnalysisState(id="pairing-no-runtime", user_id="u1", current_round=2)
    runtime_pairing.refresh_runtime_availability(state)
    assert state.runtime_restore_status == "unavailable"


def test_the_analysis_input_format_is_untouched():
    """A diff that reaches the executor prompt or the agents is out of scope."""
    for rel in (
        "agents/bioinformatics/executor_prompt.py",
        "agents/bioinformatics_agent.py",
        "workflow/agent_nodes/shared.py",
    ):
        assert "h5ad" in (BACKEND / rel).read_text(), rel
