"""The signed contract for one decision snapshot. Part B spec 4.4.

Per-file digests prove only what the manifest claims and say nothing about the
manifest, so the manifest is signed on its own and reconstructed through its
own builder on read. These tests are about the second half of that: a manifest
that merely parses could declare a bundle nobody can restore, and every such
shape has to be a hard error rather than a session that comes up missing.

The module under test is imported behind a guard. A module that raises while
being collected makes the runner report an incomplete collection and publish no
evidence at all, which is exactly the state a red-tree run is in.
"""
import json

import pytest

from backend.constants import (
    GATE_ANNOTATION_CONFIRMATION,
    GATE_DIRECTION_SELECTION,
    GATE_INTEGRATION_SELECTION,
    GATE_ROUND_WAIT,
)

try:
    from backend.services.round_snapshots.layout import (
        CONTINUATION_NAME,
        MANIFEST_NAME,
        PAIR_CHECKPOINT_RELPATH,
        PAIR_EXECUTOR_RELPATH,
        PAIR_STATE_RELPATH,
        RUNTIME_MANIFEST_RELPATH,
        RUNTIME_PKL_RELPATH,
        SESSION_STATE_NAME,
        parse_snapshot_id,
        snapshot_id,
    )
    from backend.services.round_snapshots.manifest import (
        SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        SnapshotManifestError,
        build_continuation,
        build_snapshot_manifest,
        canonical_manifest_bytes,
        manifest_sig_path,
        read_snapshot_manifest,
        validate_continuation,
        write_snapshot_manifest,
    )
except ImportError as exc:  # the modules under test do not exist yet
    _IMPORT_ERROR = exc
    # Bound to their literal values so this module still *collects* on a tree
    # that does not have the feature. Several of them are read by
    # ``@pytest.mark.parametrize`` at module scope, and a module that raises
    # while being collected makes the runner publish no evidence at all --
    # which is precisely the state a red-tree run has to read. On a patched
    # tree these are shadowed by the imports above, so a renamed constant still
    # breaks the import and is caught.
    CONTINUATION_NAME = "continuation.json"
    MANIFEST_NAME = "manifest.json"
    SESSION_STATE_NAME = "session.json"
    RUNTIME_PKL_RELPATH = "runtime/session.pkl"
    RUNTIME_MANIFEST_RELPATH = "runtime/session.runtime.json"
    PAIR_STATE_RELPATH = "runtime/pair/state.json"
    PAIR_CHECKPOINT_RELPATH = "runtime/pair/checkpoints.sqlite"
    PAIR_EXECUTOR_RELPATH = "runtime/pair/executor_checkpoint.json"
else:
    _IMPORT_ERROR = None


@pytest.fixture(autouse=True)
def _require_modules_under_test():
    if _IMPORT_ERROR is not None:
        pytest.fail(f"module under test is missing: {_IMPORT_ERROR}")


PKL_DIGEST = "b" * 64
CREATED_AT = "2026-09-08T14:03:11+00:00"


def _files(**overrides):
    table = {
        CONTINUATION_NAME: {"sha256": "0" * 64, "size": 220},
        SESSION_STATE_NAME: {"sha256": "1" * 64, "size": 4096},
        RUNTIME_PKL_RELPATH: {"sha256": PKL_DIGEST, "size": 199775703},
        RUNTIME_MANIFEST_RELPATH: {"sha256": "2" * 64, "size": 2490},
        PAIR_STATE_RELPATH: {"sha256": "3" * 64, "size": 1519},
        PAIR_CHECKPOINT_RELPATH: {"sha256": "4" * 64, "size": 10256384},
    }
    table.update(overrides)
    return table


def _manifest(**overrides):
    payload = dict(
        snapshot_id="r001-annotation-g57",
        source_session_id="source-session",
        source_session_name="Liver atlas",
        source_lifecycle_generation=1,
        scai_version="0.55.7",
        created_at=CREATED_AT,
        round_number=1,
        gate=GATE_ANNOTATION_CONFIRMATION,
        retention_rank=1,
        runtime_revision=57,
        pkl_digest=PKL_DIGEST,
        files=_files(),
        uploads={"matrix.h5ad": {"sha256": "5" * 64, "size": 1024}},
        coverage={
            "saved_names": 271,
            "skipped_names": [
                {"name": "ss", "type": "SingleSampleGSEA", "reason": "serialization_failed"},
            ],
            "carried_definitions": [{"name": "load_loom", "kind": "function"}],
        },
        executor_checkpoint_present=False,
        bytes_total=210_000_000,
        capture_seconds=41.2,
    )
    payload.update(overrides)
    return build_snapshot_manifest(**payload)


# --------------------------------------------------------------- identity


@pytest.mark.parametrize(
    "gate,word",
    [
        (GATE_ANNOTATION_CONFIRMATION, "annotation"),
        (GATE_DIRECTION_SELECTION, "direction"),
        (GATE_INTEGRATION_SELECTION, "integration"),
        (GATE_ROUND_WAIT, "round"),
    ],
)
def test_a_snapshot_id_names_its_round_decision_and_revision(gate, word):
    value = snapshot_id(round_number=3, gate=gate, revision=400)
    assert value == f"r003-{word}-g400"
    assert parse_snapshot_id(value) == (3, gate, 400)


def test_a_snapshot_id_survives_a_round_past_the_padding():
    value = snapshot_id(round_number=1000, gate=GATE_ROUND_WAIT, revision=7)
    assert parse_snapshot_id(value) == (1000, GATE_ROUND_WAIT, 7)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "r1-round-g4",            # unpadded round
        "r001-round",             # no revision
        "r001-plan_review-g4",    # a gate that is not a decision
        "r001-round-g4/../etc",   # a path, not an id
        "../r001-round-g4",
        "R001-ROUND-G4",
    ],
)
def test_an_identifier_that_does_not_parse_never_becomes_a_path(bad):
    with pytest.raises(ValueError):
        parse_snapshot_id(bad)


def test_an_identifier_cannot_name_a_gate_the_platform_does_not_park_on():
    with pytest.raises(ValueError):
        snapshot_id(round_number=1, gate="llm_failure", revision=1)


# --------------------------------------------------------------- signature


def test_a_manifest_round_trips_through_its_signature(tmp_path):
    manifest = _manifest()
    path = tmp_path / MANIFEST_NAME
    write_snapshot_manifest(path, manifest)
    assert read_snapshot_manifest(path) == manifest
    assert manifest["schema_version"] == SNAPSHOT_MANIFEST_SCHEMA_VERSION


def test_a_manifest_is_written_as_the_bytes_it_signs(tmp_path):
    """Canonical bytes, so a manifest that round-trips through JSON re-signs.

    Anything else and re-reading a manifest to add a field would produce a
    different payload for the same content, and the signature would stop
    matching for a reason that has nothing to do with tampering.
    """
    manifest = _manifest()
    path = tmp_path / MANIFEST_NAME
    write_snapshot_manifest(path, manifest)
    assert path.read_bytes() == canonical_manifest_bytes(manifest)
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


def test_a_tampered_manifest_is_refused(tmp_path):
    path = tmp_path / MANIFEST_NAME
    write_snapshot_manifest(path, _manifest())
    payload = path.read_text(encoding="utf-8").replace('"round":1', '"round":2')
    assert '"round":2' in payload
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(SnapshotManifestError):
        read_snapshot_manifest(path)


def test_a_tampered_signature_is_refused(tmp_path):
    path = tmp_path / MANIFEST_NAME
    write_snapshot_manifest(path, _manifest())
    signature = manifest_sig_path(path)
    text = signature.read_text(encoding="utf-8").strip()
    signature.write_text(("0" if text[0] != "0" else "1") + text[1:], encoding="utf-8")
    with pytest.raises(SnapshotManifestError):
        read_snapshot_manifest(path)


def test_an_unsigned_manifest_is_refused(tmp_path):
    """The signature is written second, so this is the interrupted state.

    A bundle whose manifest landed and whose signature did not has to fail
    closed; treating it as unsigned-but-fine is how a substituted manifest
    would get read.
    """
    path = tmp_path / MANIFEST_NAME
    write_snapshot_manifest(path, _manifest())
    manifest_sig_path(path).unlink()
    with pytest.raises(SnapshotManifestError):
        read_snapshot_manifest(path)


def test_an_absent_manifest_is_refused(tmp_path):
    with pytest.raises(SnapshotManifestError):
        read_snapshot_manifest(tmp_path / MANIFEST_NAME)


def test_a_signed_manifest_from_an_older_schema_is_refused(tmp_path):
    """Signed and well-formed is not enough; the shape has to be one we read."""
    manifest = dict(_manifest(), schema_version=SNAPSHOT_MANIFEST_SCHEMA_VERSION + 1)
    path = tmp_path / MANIFEST_NAME
    write_snapshot_manifest(path, manifest)
    with pytest.raises(SnapshotManifestError):
        read_snapshot_manifest(path)


# --------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "overrides",
    [
        {"round_number": 2},
        {"gate": GATE_ROUND_WAIT},
        {"runtime_revision": 58},
    ],
    ids=["round", "gate", "revision"],
)
def test_the_identifier_and_the_record_must_agree(overrides):
    """The id is what a path is built from; the record is what is read.

    If they can disagree, a bundle in directory ``r001-annotation-g57`` can
    declare itself to be round 2's direction gate, and the listing, the
    deletion order and the restore all act on a different snapshot from the one
    on disk.
    """
    with pytest.raises(ValueError):
        _manifest(**overrides)


@pytest.mark.parametrize(
    "bad",
    ["../escape", "/etc/passwd", "a//b", "a/../b", "./a", "a/", "", ".", "..",
     "a\\b", "a\x00b"],
)
def test_a_declared_path_cannot_escape_the_bundle(bad):
    with pytest.raises(ValueError):
        _manifest(files=_files(**{bad: {"sha256": "6" * 64, "size": 1}}))


@pytest.mark.parametrize("name", [MANIFEST_NAME, MANIFEST_NAME + ".sig"])
def test_a_manifest_cannot_declare_itself(name):
    with pytest.raises(ValueError):
        _manifest(files=_files(**{name: {"sha256": "6" * 64, "size": 1}}))


@pytest.mark.parametrize(
    "missing",
    [
        CONTINUATION_NAME,
        SESSION_STATE_NAME,
        RUNTIME_PKL_RELPATH,
        RUNTIME_MANIFEST_RELPATH,
        PAIR_STATE_RELPATH,
        PAIR_CHECKPOINT_RELPATH,
    ],
)
def test_a_bundle_must_declare_every_file_a_restore_reads(missing):
    files = _files()
    files.pop(missing)
    with pytest.raises(ValueError):
        _manifest(files=files)


def test_the_dump_digest_must_match_the_file_the_bundle_declares():
    """The generation's identity and the bundle's content, cross-checked.

    They are the same number written twice, and a bundle whose two copies
    disagree is describing a dump it does not carry.
    """
    with pytest.raises(ValueError):
        _manifest(files=_files(**{RUNTIME_PKL_RELPATH: {"sha256": "7" * 64, "size": 1}}))


def test_an_executor_checkpoint_must_be_declared_when_it_is_claimed():
    with pytest.raises(ValueError):
        _manifest(executor_checkpoint_present=True)


def test_an_executor_checkpoint_must_be_claimed_when_it_is_declared():
    """Absent is a fact about the decision, not a detail to be silent about.

    Only the annotation decision is taken while the executor holds a position,
    so a bundle carrying that file is claiming to be an annotation snapshot; a
    restore installs it on that basis and nothing else would catch a bundle
    that carried one without saying so.
    """
    with pytest.raises(ValueError):
        _manifest(
            files=_files(**{PAIR_EXECUTOR_RELPATH: {"sha256": "8" * 64, "size": 815}}),
            executor_checkpoint_present=False,
        )


def test_an_annotation_bundle_may_carry_the_executor_position():
    manifest = _manifest(
        files=_files(**{PAIR_EXECUTOR_RELPATH: {"sha256": "8" * 64, "size": 815}}),
        executor_checkpoint_present=True,
    )
    assert manifest["executor_checkpoint_present"] is True


def test_a_naive_timestamp_is_refused():
    """The listing shows this to a user who is rarely in the container's zone.

    A naive string is read as local time by the browser, which is the defect
    the repair spec's B1-4 removed from the runtime record; a new record that
    reintroduced it would be off by the viewer's offset.
    """
    with pytest.raises(ValueError):
        _manifest(created_at="2026-09-08T14:03:11")


@pytest.mark.parametrize("rank", [0, 5, -1])
def test_a_retention_rank_outside_the_four_decisions_is_refused(rank):
    with pytest.raises(ValueError):
        _manifest(retention_rank=rank)


def test_an_unknown_coverage_reason_is_refused():
    """The save's vocabulary, reused rather than restated.

    A reason the save can emit and the snapshot cannot record would make a
    bundle unreadable for a reason that has nothing to do with the bundle.
    """
    with pytest.raises(ValueError):
        _manifest(coverage={
            "saved_names": 1,
            "skipped_names": [{"name": "x", "type": "T", "reason": "because"}],
            "carried_definitions": [],
        })


def test_an_unknown_carried_definition_kind_is_refused():
    with pytest.raises(ValueError):
        _manifest(coverage={
            "saved_names": 1,
            "skipped_names": [],
            "carried_definitions": [{"name": "helper", "kind": "module"}],
        })


def test_coverage_is_ordered_so_two_captures_of_it_sign_the_same():
    manifest = _manifest(coverage={
        "saved_names": 2,
        "skipped_names": [
            {"name": "zeta", "type": "T", "reason": "excluded_volatile"},
            {"name": "alpha", "type": "T", "reason": "excluded_private"},
        ],
        "carried_definitions": [
            {"name": "second", "kind": "class"},
            {"name": "first", "kind": "function"},
        ],
    })
    assert [item["name"] for item in manifest["coverage"]["skipped_names"]] == [
        "alpha", "zeta",
    ]
    assert [item["name"] for item in manifest["coverage"]["carried_definitions"]] == [
        "first", "second",
    ]


def test_a_refused_workspace_entry_is_recorded_rather_than_dropped():
    """A capture that silently omits a file is the coverage gap in a new place."""
    manifest = _manifest(skipped_entries=[
        {"path": "workspace/tmp/link", "reason": "symlink"},
        {"path": "workspace/fifo", "reason": "not_a_regular_file"},
    ])
    assert [entry["path"] for entry in manifest["skipped_entries"]] == [
        "workspace/fifo", "workspace/tmp/link",
    ]


# --------------------------------------------------------------- continuation


@pytest.mark.parametrize(
    "gate,node,status",
    [
        (GATE_ANNOTATION_CONFIRMATION, "bioinformatics_analysis", "waiting_annotation"),
        (GATE_DIRECTION_SELECTION, "check_convergence", "waiting_input"),
        (GATE_INTEGRATION_SELECTION, "check_convergence", "waiting_input"),
        (GATE_ROUND_WAIT, "check_convergence", "waiting_input"),
    ],
)
def test_a_capsule_derives_its_node_and_status_from_its_gate(gate, node, status):
    capsule = build_continuation(
        gate=gate,
        analysis_phase="exploration",
        current_round=1,
        state_version=1189,
        workflow_state_keys=["planning_stage", "current_round"],
    )
    assert capsule["next_node"] == node
    assert capsule["restored_status"] == status
    assert capsule["workflow_state_keys"] == ["current_round", "planning_stage"]
    assert validate_continuation(capsule) == capsule


@pytest.mark.parametrize("field,value", [
    ("next_node", "initialize_round"),
    ("restored_status", "running"),
])
def test_a_capsule_that_disagrees_with_its_own_gate_is_refused(field, value):
    """Revision 4 of the Part A spec was overturned on exactly this point.

    Pinning continuation to ``initialize_round`` restored *past* the user's
    decision and past convergence. A capsule whose node does not follow from
    its gate can only have been edited or built by hand, and either way it
    would put a restored session somewhere the source run never was.
    """
    capsule = dict(
        build_continuation(
            gate=GATE_ROUND_WAIT,
            analysis_phase="innovation",
            current_round=3,
            state_version=10,
            workflow_state_keys=[],
        ),
        **{field: value},
    )
    with pytest.raises(SnapshotManifestError):
        validate_continuation(capsule)


def test_a_capsule_naming_an_incident_gate_is_refused():
    """Only the four decisions. An LLM failure is not something to roll back to."""
    with pytest.raises(SnapshotManifestError):
        validate_continuation({
            "schema_version": 1,
            "gate": "llm_failure",
            "next_node": "check_convergence",
            "restored_status": "waiting_input",
            "analysis_phase": "innovation",
            "current_round": 2,
            "state_version": 1,
            "workflow_state_keys": [],
        })


def test_a_capsule_from_an_older_schema_is_refused():
    capsule = dict(
        build_continuation(
            gate=GATE_ROUND_WAIT,
            analysis_phase="innovation",
            current_round=2,
            state_version=1,
            workflow_state_keys=[],
        ),
        schema_version=99,
    )
    with pytest.raises(SnapshotManifestError):
        validate_continuation(capsule)
