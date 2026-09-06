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
    "backend/routes/export_import_parts/import_archive_policy.py": 5,
    "backend/routes/export_import_parts/import_identity.py": 5,
    "backend/routes/export_import_parts/import_impl.py": 5,
    "backend/routes/export_import_parts/import_routes.py": 8,
    "backend/routes/export_import_parts/import_state.py": 6,
    "backend/routes/export_import_parts/import_tar.py": 10,
    "backend/routes/export_import_parts/import_zip.py": 5,
    "backend/routes/export_import_parts/tar_builders.py": 4,
    "backend/routes/interaction.py": 1,
    "backend/routes/lifecycle.py": 7,
    "backend/routes/lifecycle_parts/run_resume.py": 6,
    "backend/routes/lifecycle_parts/snapshot_failures.py": 6,
    "backend/routes/snapshots.py": 23,
    "backend/routes/websocket_parts/analysis_connection.py": 1,
    "backend/routes/websocket_parts/state_sync.py": 1,
    "backend/services/session_lifecycle.py": 1,
    "backend/services/session_store.py": 15,
    "backend/services/workflow_runner.py": 2,
    "backend/utils/runtime_apply.py": 4,
    "backend/utils/sandbox_parts/regular_save.py": 3,
    "backend/utils/sandbox_parts/runtime_save.py": 3,
    "backend/utils/sandbox_parts/state_tracking.py": 3,
    "backend/utils/snapshot.py": 1,
    "backend/workflow/core.py": 2,
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
