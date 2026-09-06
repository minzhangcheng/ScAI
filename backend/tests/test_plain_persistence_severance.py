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
SNAPSHOT_V2_CENSUS: dict[str, int] = {
    "backend/agents/bioinformatics/executor_loop.py": 1,
    "backend/config/__init__.py": 1,
    "backend/config/_config.py": 9,
    "backend/main.py": 4,
    "backend/routes/chat_parts/delegation_flow.py": 2,
    "backend/routes/chat_parts/endpoint_flow.py": 3,
    "backend/routes/chat_parts/snapshots.py": 10,
    "backend/routes/config.py": 3,
    "backend/routes/export_import.py": 5,
    "backend/routes/export_import_parts/__init__.py": 1,
    "backend/routes/export_import_parts/core.py": 5,
    "backend/routes/export_import_parts/export_routes.py": 8,
    "backend/routes/export_import_parts/import_archive_policy.py": 4,
    "backend/routes/export_import_parts/import_identity.py": 4,
    "backend/routes/export_import_parts/import_impl.py": 4,
    "backend/routes/export_import_parts/import_routes.py": 8,
    "backend/routes/export_import_parts/import_state.py": 5,
    "backend/routes/export_import_parts/import_tar.py": 9,
    "backend/routes/export_import_parts/import_zip.py": 4,
    "backend/routes/export_import_parts/tar_builders.py": 4,
    "backend/routes/interaction.py": 1,
    "backend/routes/lifecycle.py": 6,
    "backend/routes/lifecycle_parts/run_resume.py": 6,
    "backend/routes/lifecycle_parts/snapshot_failures.py": 6,
    "backend/routes/snapshots.py": 23,
    "backend/routes/websocket_parts/analysis_connection.py": 1,
    "backend/routes/websocket_parts/state_sync.py": 1,
    "backend/services/session_lifecycle.py": 1,
    "backend/services/session_store.py": 15,
    "backend/services/workflow_runner.py": 2,
    "backend/utils/sandbox_parts/runtime_save.py": 3,
    "backend/utils/sandbox_parts/state_tracking.py": 3,
    "backend/utils/snapshot.py": 1,
    "backend/workflow/core.py": 1,
    "backend/workflow/core_parts/execution.py": 18,
    "backend/workflow/core_parts/initialization.py": 11,
    "backend/workflow/core_parts/interventions.py": 2,
    "backend/workflow/core_parts/shutdown.py": 1,
    "backend/workflow/reporting.py": 8,
    "backend/workflow/retry.py": 1,
}


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


def test_the_snapshot_checkpoint_module_is_only_a_shim():
    """The V2 copy must not drift from the relocated one before A2-5."""
    from backend.services.snapshot_v2 import checkpoint_paths as shim
    from backend.utils import checkpoint_paths as real

    assert shim.checkpoint_sqlite_path is real.checkpoint_sqlite_path
    assert shim.sqlite_companion_paths is real.sqlite_companion_paths
    assert shim.CHECKPOINT_SQLITE_FILENAME is real.CHECKPOINT_SQLITE_FILENAME


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
    source = (BACKEND / "workflow" / "core.py").read_text()
    assert "snapshot_v2.restore_entry" not in source
