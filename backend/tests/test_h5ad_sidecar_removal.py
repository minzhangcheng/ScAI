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
def test_the_generation_manifest_still_records_which_names_the_pkl_carries():
    """The fact the h5ad manifest was recording has a surviving owner.

    ``runtime_save.py`` wrote ``runtime_h5ad_manifest.json`` with an **empty**
    sidecar dict and ``authoritative_pkl_names=saved_names``. It was not
    carrying sidecars; it was recording which names the pkl is authoritative
    for -- which is what A1's ``runtime_manifest.json`` already carries in
    ``saved_names``. Deleting the older record is only safe while this holds.
    """
    from backend.utils.runtime_manifest import build_runtime_manifest

    manifest = build_runtime_manifest(
        revision=3,
        state_revision=3,
        pkl_digest="a" * 64,
        saved_names=["adata", "results"],
        skipped_names=[],
        serializer_version="dill-zstd-1",
    )
    assert sorted(manifest["saved_names"]) == ["adata", "results"]


def test_the_ordinary_save_no_longer_writes_a_sidecar_manifest():
    """The producer side, which is separable from Snapshot V2.

    ``_commit_pkl_transaction`` is shared with Snapshot V2's restore, which
    still writes sidecars through its h5ad component kind, so the transaction's
    handling of that kind is deleted in A2-5 with the package. What A2-4 can
    remove is the *ordinary* save's own sidecar manifest, which recorded only
    which names the pkl is authoritative for -- the fact A1's
    ``runtime_manifest.json`` already carries in ``saved_names``.
    """
    source = (BACKEND / "utils" / "sandbox_parts" / "runtime_save.py").read_text()
    for gone in (
        "runtime_h5ad_manifest_path",
        "write_runtime_h5ad_manifest",
        "quarantine_unreferenced_runtime_h5ads",
    ):
        assert gone not in source, gone
