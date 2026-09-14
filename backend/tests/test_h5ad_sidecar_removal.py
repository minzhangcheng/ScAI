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


# ---------------------------------------------------------------------------
# A2-5 Task 5b: the sidecar itself, not just the manifest the save wrote
# ---------------------------------------------------------------------------


def test_no_producer_of_a_sidecar_survives():
    """Every writer went with the snapshot package.

    ``atomic_replace_h5ad`` installed one, ``write_runtime_h5ad_manifest``
    declared the set, and the two ``*_live_runtime_h5ad_set`` helpers staged and
    finalized it. All four were reachable only from the snapshot restore.
    """
    from backend.utils import runtime_apply

    for gone in (
        "atomic_replace_h5ad",
        "write_runtime_h5ad_manifest",
        "install_live_runtime_h5ad_set",
        "finalize_live_runtime_h5ad_set",
        "quarantine_unreferenced_runtime_h5ads",
    ):
        assert not hasattr(runtime_apply, gone), gone


def test_the_save_transaction_has_no_sidecar_half_left():
    from backend.tests.split_facade import split_source

    source = split_source("backend.utils.sandbox")
    for gone in (
        "_validate_runtime_h5ad_manifest",
        "_include_new_runtime_h5ads_in_rollback",
        "keep_h5ad_names",
        "retire_sidecars",
    ):
        assert gone not in source, gone


def test_a_journal_naming_a_retired_component_kind_still_recovers():
    """A crash under the previous build must not strand the pkl it was writing.

    Its journal can list ``h5ad`` components this build no longer installs.
    Resolution skips the kind it does not know rather than refusing the whole
    transaction, which would abandon the pkl the recovery is really about.
    """
    from backend.tests.split_facade import split_source

    source = split_source("backend.utils.sandbox")
    assert "Ignoring unknown component kind" in source


def test_a_runtime_that_lives_only_in_sidecars_is_refused_not_half_loaded(tmp_path):
    """The one case where deleting the reader would lose data silently.

    A session the previous build restored from a snapshot, and which has not
    saved since, holds its AnnData in sidecars and not in the pkl beside them.
    Loading that pkl alone brings the kernel up quietly missing ``adata``. The
    probe that decides this must fire on a manifest that names something, and
    must stay silent on the empty one the ordinary save wrote until A2-4 --
    that form claims nothing and refusing on it would strand healthy sessions.
    """
    import json

    from backend.utils.sandbox_parts.runtime_restore import (
        _names_only_in_retired_sidecars,
    )

    absent = tmp_path / "missing.json"
    assert _names_only_in_retired_sidecars(absent) == set()

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"version": 2, "entries": {}}))
    assert _names_only_in_retired_sidecars(empty) == set()

    named = tmp_path / "named.json"
    named.write_text(json.dumps({
        "version": 2,
        "entries": {"adata": {"path": "adata.h5ad", "sha256": "0" * 64}},
    }))
    assert _names_only_in_retired_sidecars(named) == {"adata"}

    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{not json")
    assert _names_only_in_retired_sidecars(unreadable) == set()


def test_the_generated_restore_script_no_longer_reads_sidecars():
    from backend.tests.split_facade import source_for_path

    # Explicit, PI-selected AnnData checkpoints have their own H5AD loader.
    # This invariant is specifically about ordinary runtime restoration no
    # longer reconstructing variables from the retired sidecar mechanism.
    source = source_for_path("backend/utils/sandbox_parts/runtime_restore.py")
    assert "_h5ad_manifest_path" not in source
    assert "read_h5ad" not in source


def test_retiring_a_runtime_still_sweeps_legacy_sidecars():
    """They are inert, but they name variables the retired generation held."""
    from backend.tests.split_facade import source_for_path

    source = source_for_path("backend/utils/runtime_apply.py")
    assert 'states_dir.glob("*.h5ad")' in source
