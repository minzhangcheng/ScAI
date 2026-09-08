"""Batch 1: four small defects, none of them data-integrity, all of them a
promise the code does not keep.

Each is verified here before it is fixed:

* the cross-instance re-key drops `carried_definitions`, the fourth hop of the
  same record that already had to be threaded through three;
* the export manifest's `runtime_saved_names` is always null, because nothing
  writes `saved_names` into the persistence record it reads;
* a results-only archive still ships the runtime manifest and its signature, so
  the destination reports "inconsistent" for a session that simply has no
  runtime;
* a brand-new session's first round is treated as completed work.
"""
from __future__ import annotations

# --- the carried record must survive a cross-instance re-key ----------------


def test_the_rekeyed_manifest_keeps_its_carried_definitions(tmp_path, monkeypatch):
    """Same defect as "the record never reached disk", one hop further on.

    An imported manifest is rebuilt and re-signed under this installation's key.
    Rebuilding it without `carried_definitions` loses, permanently, the only
    statement of which `__main__` definitions the dump carries and of the class
    caveat that rides with them.
    """
    from backend.config import get_session_states
    from backend.routes.export_import_parts.import_state import (
        _rekey_imported_runtime_manifest,
    )
    from backend.utils.runtime_manifest import (
        build_runtime_manifest,
        read_runtime_manifest,
        runtime_manifest_path,
        write_runtime_manifest,
    )
    from backend.utils.sandbox_parts.regular_save import _sha256_file

    sid = "rekey-carried"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    pkl = states / f"{sid}_session.pkl"
    pkl.write_bytes(b"DUMP")

    write_runtime_manifest(
        runtime_manifest_path(sid),
        build_runtime_manifest(
            revision=3,
            state_revision=3,
            pkl_digest=_sha256_file(pkl),
            saved_names=["adata", "helper"],
            skipped_names=[],
            serializer_version="dill-zstd-1",
            carried_definitions=[{"name": "helper", "kind": "function"}],
        ),
    )

    assert _rekey_imported_runtime_manifest(sid, pkl) is True

    rekeyed = read_runtime_manifest(runtime_manifest_path(sid))
    assert rekeyed.get("carried_definitions") == [
        {"name": "helper", "kind": "function"}
    ], "the re-key dropped the carried record"


# --- the archive must report the coverage it advertises ---------------------


def test_the_archive_manifest_reports_the_names_the_dump_holds(tmp_path):
    """`_generation_facts` promises a destination can learn coverage from the
    archive. It read `runtime_persistence["saved_names"]`, which nothing writes,
    so the field was always null. The runtime manifest beside the dump is the
    actual holder.
    """
    from backend.config import get_session_states
    from backend.routes.export_import_parts.export_routes import _generation_facts
    from backend.utils.runtime_manifest import (
        build_runtime_manifest,
        runtime_manifest_path,
        write_runtime_manifest,
    )

    sid = "facts-coverage"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    write_runtime_manifest(
        runtime_manifest_path(sid),
        build_runtime_manifest(
            revision=2,
            state_revision=2,
            pkl_digest="a" * 64,
            saved_names=["adata", "counts"],
            skipped_names=[{
                "name": "handle", "type": "TextIOWrapper",
                "reason": "excluded_volatile",
            }],
            serializer_version="dill-zstd-1",
        ),
    )

    facts = _generation_facts(sid, {"runtime_persistence": {}}, True)

    assert facts["runtime_saved_names"] == ["adata", "counts"]
    assert [s["name"] for s in facts["runtime_skipped_names"]] == ["handle"]


def test_a_results_only_archive_reports_no_coverage_rather_than_a_lie():
    """No runtime carried means no coverage to report -- not a stale reading of
    some other generation's."""
    from backend.routes.export_import_parts.export_routes import _generation_facts

    facts = _generation_facts("absent-session", {"runtime_persistence": {}}, False)

    assert facts["runtime_omitted_reason"] == "include_runtime=false"
    assert facts["runtime_saved_names"] is None


# --- a results-only archive must not ship the generation manifest -----------


def test_the_runtime_manifest_is_part_of_the_runtime():
    """It travels with the dump or not at all.

    A portable archive that carries the manifest lands a foreign-signed file in
    the destination, where `state_runtime_coherence` reads it, fails the HMAC and
    answers "inconsistent" -- for a session whose actual condition is that it has
    no runtime at all. The diagnosis is wrong and the file is litter.
    """
    from backend.routes.export_import_parts.core import is_runtime_generation_name

    for name in (
        "abc_session.runtime.json",
        "abc_session.runtime.json.sig",
        "abc_session.pkl",
        "abc_session.pkl.zst",
        "session.pkl",
    ):
        assert is_runtime_generation_name(name), name

    for name in (
        "abc.ipynb",
        "abc_state.json",
        "checkpoints.sqlite",
        "runtime_h5ad_manifest.json",
    ):
        assert not is_runtime_generation_name(name), name


def test_both_sides_drop_the_generation_together():
    """Export and import share the predicate, so the halves cannot separate."""
    from backend.tests.split_facade import source_for_path

    exporter = source_for_path("backend/routes/export_import_parts/tar_builders.py")
    assert "is_runtime_generation_name" in exporter

    importer = source_for_path("backend/routes/export_import_parts/import_tar.py")
    assert "is_runtime_generation_name" in importer


# --- a first round is not completed work ------------------------------------


def test_a_fresh_sessions_first_round_is_not_treated_as_completed_work():
    """`current_round` is 1 from the moment round 1 *starts*.

    Both availability checks used it as a proxy for "this session has completed
    rounds", so a brand-new session with no dump -- the normal state of every
    round 1 -- was marked `unavailable`. `prepare_sandbox` raises
    RuntimeUnavailable on exactly that status.
    """
    from backend.config import get_session_states
    from backend.models.schemas import AnalysisState
    from backend.utils.runtime_pairing import refresh_runtime_availability

    sid = "fresh-first-round"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    # A real round 1 has been persisted, which is what gets the check past its
    # "is there any record at all" guard. Without this the test passes for the
    # wrong reason.
    (states / f"{sid}_state.json").write_text("{}", encoding="utf-8")

    state = AnalysisState(id=sid, user_id="u", name="s")
    state.current_round = 1  # set by initialize_round at the START of round 1
    state.round_results = []

    refresh_runtime_availability(state)

    assert state.runtime_restore_status != "unavailable", (
        "a session that has completed no round was told its runtime was lost"
    )


def test_a_session_that_completed_a_round_and_lost_its_dump_is_unavailable():
    """The case the check exists for must survive the fix."""
    from backend.config import get_session_states
    from backend.models.schemas import AnalysisState
    from backend.utils.runtime_pairing import refresh_runtime_availability

    sid = "lost-after-a-round"
    states = get_session_states(sid)
    states.mkdir(parents=True, exist_ok=True)
    (states / f"{sid}_state.json").write_text("{}", encoding="utf-8")

    state = AnalysisState(id=sid, user_id="u", name="s")
    state.current_round = 2
    state.round_results = [{"round_number": 1}]

    refresh_runtime_availability(state)

    assert state.runtime_restore_status == "unavailable"
