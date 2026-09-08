"""A2 ratchet: references from surviving code into snapshot_v2 may only fall."""
from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
BACKEND = REPO / "backend"

#: Files outside backend/services/snapshot_v2 and backend/tests that mention
#: ``snapshot_v2``, and how many lines of each do, measured at the A2-1 entry
#: SHA (the 0.52.11 merge). A2 drives every value to zero; A2-5 empties the
#: table. Lowering a value is the point of each batch. Raising one, or adding a
#: key, is a regression.
#:
#: A whitelist of "files pending deletion" was tried first and was wrong twice
#: over: most of these files survive with their snapshot references stripped
#: rather than being deleted, and a list assembled from ``import`` lines missed
#: 25 of the 42, because a reference is not always an import.
#: A2-5 emptied it. The three ratchets below stay: an empty table with
#: live tests is what keeps the surface clean, and a re-entry shows up
#: as a new key rather than as a silently raised count.
SNAPSHOT_V2_CENSUS: dict[str, int] = {}


def _observed() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(BACKEND.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(("backend/tests/", "backend/services/snapshot_v2/")):
            continue
        hits = sum(1 for line in path.read_text().splitlines()
                   if "snapshot_v2" in line)
        if hits:
            counts[rel] = hits
    return counts


def test_no_new_file_reaches_into_snapshot_v2():
    new = set(_observed()) - set(SNAPSHOT_V2_CENSUS)
    assert not new, (
        "these files newly reference snapshot_v2: " + ", ".join(sorted(new))
    )


def test_no_file_gained_snapshot_v2_references():
    observed = _observed()
    grew = {
        rel: (SNAPSHOT_V2_CENSUS[rel], observed[rel])
        for rel in observed
        if rel in SNAPSHOT_V2_CENSUS and observed[rel] > SNAPSHOT_V2_CENSUS[rel]
    }
    assert not grew, "reference counts rose: %r" % grew


def test_the_census_has_no_stale_entries():
    """A value that has fallen must be lowered in the table, not left stale."""
    observed = _observed()
    stale = {
        rel: (SNAPSHOT_V2_CENSUS[rel], observed.get(rel, 0))
        for rel in SNAPSHOT_V2_CENSUS
        if observed.get(rel, 0) != SNAPSHOT_V2_CENSUS[rel]
    }
    assert not stale, (
        "lower these census values to what the tree actually holds: %r" % stale
    )
def test_runtime_apply_does_not_use_the_snapshot_object_store():
    source = (BACKEND / "utils" / "runtime_apply.py").read_text()
    assert "object_store" not in source
    assert "_hash_file" not in source
def test_checkpoint_path_helpers_live_outside_snapshot_v2():
    from backend.utils.checkpoint_paths import (
        CHECKPOINT_SQLITE_FILENAME,
        SQLITE_COMPANION_SUFFIXES,
        checkpoint_sqlite_path,
        sqlite_companion_paths,
    )

    assert CHECKPOINT_SQLITE_FILENAME == "checkpoints.sqlite"
    assert SQLITE_COMPANION_SUFFIXES == ("-wal", "-shm")
    assert checkpoint_sqlite_path("/tmp/states").name == "checkpoints.sqlite"
    assert [c.name for c in sqlite_companion_paths("/tmp/s/checkpoints.sqlite")] == [
        "checkpoints.sqlite-wal",
        "checkpoints.sqlite-shm",
    ]




def test_no_surviving_module_imports_the_snapshot_checkpoint_paths():
    """A shim keeps them running; it does not sever them."""
    offenders = set()
    for path in sorted(BACKEND.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(("backend/tests/", "backend/services/snapshot_v2/")):
            continue
        if "snapshot_v2.checkpoint_paths" in path.read_text():
            offenders.add(rel)
    assert not offenders, sorted(offenders)
def test_runtime_name_validation_lives_outside_snapshot_v2():
    import pytest

    from backend.utils.runtime_names import validate_runtime_global_name

    assert validate_runtime_global_name("adata") == "adata"
    for bad in ("", "9lives", "a b", "../etc/passwd", "x" * 129):
        with pytest.raises(ValueError):
            validate_runtime_global_name(bad)


def test_the_runtime_persistence_chain_is_free_of_snapshot_v2():
    for rel in ("utils/runtime_apply.py", "utils/sandbox_parts/regular_save.py"):
        assert "snapshot_v2" not in (BACKEND / rel).read_text(), rel


def test_every_path_that_installs_a_pkl_writes_a_manifest():
    """A1 exit condition 1.

    The one installer that did not was ``runtime_apply.atomic_replace_pkl``:
    the bytes it wrote were a generation whose coverage it did not know, so it
    *retired* the previous manifest rather than writing one, and the dump then
    loaded as a legacy generation with no coverage and no pairing. That was
    correct while the snapshot restore existed. It has no caller now, so the
    condition is discharged by its absence -- and this asserts the absence
    rather than the promise.
    """
    from backend.utils import runtime_apply

    for gone in (
        "atomic_replace_pkl",
        "apply_runtime_pkl",
        "_retire_stale_runtime_manifest",
        "commit_runtime_persistence_metadata",
    ):
        assert not hasattr(runtime_apply, gone), gone


def test_a_committed_generation_always_has_a_manifest():
    """A1 exit condition 2, as an invariant rather than a claim.

    Every remaining path that commits a live pkl goes through
    ``_commit_pkl_transaction``, whose component set is built by
    ``_regular_component_paths``. The runtime manifest is one of those
    components, so it commits with the pkl or not at all -- there is no longer
    a second installer that can leave one without the other.
    """
    from backend.tests.split_facade import split_source

    source = split_source("backend.utils.sandbox")
    assert "def _commit_pkl_transaction" in source
    assert "runtime_manifest" in source

    callers = set()
    for path in sorted(BACKEND.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith("backend/tests/"):
            continue
        # Call sites, not mentions: a docstring naming the transaction is not a
        # second installer, and treating it as one would make this guard fire
        # on prose.
        if "_commit_pkl_transaction(" in path.read_text():
            callers.add(rel)
    # Its definition and the one call site; the two facade re-exports name it
    # without parentheses and so do not appear here.
    assert callers == {
        "backend/utils/sandbox_parts/regular_save.py",
        "backend/utils/sandbox_parts/runtime_save.py",
    }, sorted(callers)


def test_the_snapshot_v2_package_is_gone():
    """No source, and not importable.

    The directory itself is not the assertion: the test runner mounts this tree
    into a container that writes ``__pycache__`` as root, so an empty
    root-owned directory can outlive the sources and cannot be removed from
    here. What matters is that nothing is left to import.
    """
    import importlib.util

    package = BACKEND / "services" / "snapshot_v2"
    assert not sorted(package.rglob("*.py")), sorted(
        str(path.relative_to(REPO)) for path in package.rglob("*.py")
    )
    spec = importlib.util.find_spec("backend.services.snapshot_v2")
    # An empty directory still resolves as a namespace package, which is what a
    # leftover __pycache__ leaves behind. It carries no loader, so nothing can
    # be imported from it -- a real package would.
    assert spec is None or spec.loader is None, spec


def test_no_snapshot_route_is_registered():
    from backend.main import app

    paths = {getattr(route, "path", "") for route in app.routes}
    assert not [p for p in paths if "snapshot" in p.lower()], sorted(paths)


def test_no_snapshot_configuration_survives():
    source = (BACKEND / "config" / "_config.py").read_text()
    for gone in (
        "SNAPSHOT_V2_ENABLED",
        "is_snapshot_v2_enabled",
        "MAX_SESSION_SNAPSHOT_BYTES",
        "MAX_SNAPSHOT_KERNEL_PKL_BYTES",
        "get_integrity_dir",
        "SCAI_ENABLE_SNAPSHOT_V2",
    ):
        assert gone not in source, gone


def test_an_old_session_json_carrying_retired_fields_still_loads():
    """Dropping a field must not make an existing session unreadable.

    Spec 3.12: the risk is a stuck state, not a parse error. The gate fields
    stay declared precisely so the sanitizer can clear them; the record fields
    are gone and must be ignored rather than rejected.
    """
    from backend.models.schemas import AnalysisState, TaskStatus
    from backend.services.legacy_state_sanitizer import sanitize_legacy_state

    state = AnalysisState.model_validate({
        "id": "legacy-load",
        "name": "legacy",
        "user_id": "owner",
        "status": "waiting_input",
        "wait_input_reason": "snapshot_failure",
        "snapshot_failure": {"code": "QUOTA_EXCEEDED", "message": "no room"},
        "pending_required_snapshot": {"phase": "pre_restore"},
        "snapshots": [{"id": "s1", "name": "old", "phase": "manual"}],
        "created_from_snapshot_id": "s1",
        "last_snapshot_id": "s1",
    })

    assert sanitize_legacy_state(state) is True
    assert state.snapshot_failure is None
    assert state.pending_required_snapshot is None
    assert state.wait_input_reason is None
    assert state.status is TaskStatus.PAUSED
def test_the_restore_entry_plan_lives_in_the_workflow_package():
    import pydantic

    from backend.workflow.restore_entry import (
        RestoreEntryPlan,
        build_restore_entry_plan,
    )

    assert issubclass(RestoreEntryPlan, pydantic.BaseModel)

    class _Bare:
        pass

    plan = build_restore_entry_plan(_Bare())
    assert plan.intent == "normal"
    assert plan.target_node is None

    plan = build_restore_entry_plan(_Bare(), restore_start_intent="explicit_start")
    assert plan.intent == "explicit_start"
    assert plan.target_node == "initialize_round"
    assert plan.preserve_round_on_initialize is True


def test_workflow_core_does_not_import_the_snapshot_restore_entry():
    """Assert over the whole split, not over the re-export header.

    ``backend/workflow/core.py`` is a facade whose implementation lives in
    ``core_parts/``. A negative assertion read from the facade's own file would
    pass unconditionally the day this import moves into a part, which is what
    ``test_facade_patch_targets`` forbids.
    """
    from backend.tests.split_facade import split_source

    assert "snapshot_v2.restore_entry" not in split_source("backend.workflow.core")
def test_the_runtime_signature_root_is_not_the_snapshot_directory():
    """``get_integrity_dir`` is gone; the runtime chain never used it.

    Spec revision 4 4.5 claimed the ordinary runtime HMAC lived under
    ``<sid>/snapshot-v2``. It did not -- that was the snapshot subsystem's own
    root, and acting on the claim would have migrated a directory the runtime
    chain never reads. Corrected in revision 5; this guard keeps the correction
    from being undone by asserting the snapshot-only helper stays deleted and
    the real root stays where it is.
    """
    import backend.config._config as config_module
    from backend.utils.sandbox_parts.shared import _integrity_dir

    assert not hasattr(config_module, "get_integrity_dir")
    assert "snapshot" not in str(_integrity_dir("severance-guard"))

    callers = set()
    for path in sorted(BACKEND.rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith("backend/tests/"):
            continue
        if "get_integrity_dir" in path.read_text():
            callers.add(rel)
    assert not callers, sorted(callers)
def test_the_volatile_type_prefixes_live_outside_snapshot_v2():
    """Volatile-type exclusion is kept (spec 4.2); its list outlived V2.

    Order is behaviour, not style: the tuple's ``repr()`` is embedded verbatim
    into the generated kernel save script, so a reordering changes what that
    script excludes.
    """
    from backend.utils.runtime_types import VOLATILE_RUNTIME_TYPE_PREFIXES

    assert isinstance(VOLATILE_RUNTIME_TYPE_PREFIXES, tuple)
    assert VOLATILE_RUNTIME_TYPE_PREFIXES


def test_the_save_path_reads_the_prefixes_from_their_new_home():
    """Only the save script still needs them.

    ``state_tracking.py`` used them for the cell-time persistability probe,
    which A2-2 deleted; it must not reacquire them, and it must certainly not
    reacquire them from the snapshot package.
    """
    source = (BACKEND / "utils" / "sandbox_parts" / "runtime_save.py").read_text()
    assert "VOLATILE_RUNTIME_TYPE_PREFIXES" in source
    assert "backend.utils.runtime_types" in source
    assert "snapshot_v2" not in source

    tracking = (BACKEND / "utils" / "sandbox_parts" / "state_tracking.py").read_text()
    assert "VOLATILE_RUNTIME_TYPE_PREFIXES" not in tracking
    assert "snapshot_v2" not in tracking
