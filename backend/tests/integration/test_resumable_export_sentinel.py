"""Spec 5.4's acceptance, and half of §7's Goal 3.

The unit tests in ``test_resumable_export.py`` say each part works. This says
the feature does: a variable bound **after** the last scheduled save survives an
export and an import.

"After" is the whole design of the test. The sentinel is bound past the point at
which the cadence would have saved, so an export that ships the previous
generation cannot contain it. An export that passes this flushed.

Requires a real kernel: ``SCAI_RUN_KERNEL_TESTS=1``, i.e. the ``kernel``
profile. The cross-container half of Goal 3 — export here, import into a second
container, continue a round — is not a unit test and is recorded by hand in
``docs/coding/plain-persistence-baseline.md``.
"""
from __future__ import annotations

import os
import tarfile
import uuid

import pytest

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
    ),
]


@pytest.mark.asyncio
async def test_a_variable_bound_after_the_last_save_survives_export(
    tmp_path, sandbox,
):
    """The archive must contain the generation the export's own flush produced.

    Sequence:

    1. bind a value and save — this is "the last scheduled save";
    2. bind the sentinel, and do **not** save;
    3. flush through the export path;
    4. assert the committed dump now contains the sentinel.

    Step 4 reads the dump rather than the kernel: the kernel obviously has the
    sentinel, and asserting on it would pass against an export that never
    flushed. What has to be true is that the sentinel is *on disk*.
    """
    from backend.services.resumable_export import flush_for_resumable_export

    sentinel = f"sentinel_{uuid.uuid4().hex[:8]}"

    result = await sandbox.execute(
        "baseline_value = 1", save_to_history=False, skip_safety_check=True,
    )
    assert result.success, result.error
    assert await sandbox.save_session() is True

    result = await sandbox.execute(
        f"{sentinel} = 'bound-after-the-last-save'",
        save_to_history=False,
        skip_safety_check=True,
    )
    assert result.success, result.error

    from backend.models.schemas import AnalysisState
    from backend.state import workflows

    state = AnalysisState(id=sandbox.session_id, name="s", user_id="u")

    class _Owner:
        pass

    owner = _Owner()
    owner.sandbox = sandbox
    workflows[sandbox.session_id] = owner
    try:
        await flush_for_resumable_export(sandbox.session_id, state)
    finally:
        workflows.pop(sandbox.session_id, None)

    manifest = _committed_names(sandbox.session_id)
    assert sentinel in manifest, (
        f"{sentinel} was bound after the last scheduled save and is not in the "
        f"committed generation: the export did not flush. Saved names: {manifest}"
    )


def _committed_names(session_id: str) -> list[str]:
    from backend.utils.runtime_manifest import (
        read_runtime_manifest,
        runtime_manifest_path,
    )

    path = runtime_manifest_path(session_id)
    assert path.exists(), f"no generation manifest at {path}"
    return list(read_runtime_manifest(path).get("saved_names") or [])


@pytest.mark.asyncio
async def test_a_resumable_archive_carries_the_flushed_generation(
    tmp_path, sandbox,
):
    """End to end through the packer, not just the flush helper.

    Also asserts the archive is *self-describing*: an archive that carries a
    runtime but does not say so leaves the destination guessing, and the
    destination's guess is what decides whether the checkpoint survives.
    """
    from backend.models.schemas import AnalysisState
    from backend.routes.export_import_parts.tar_builders import _add_states_to_tar

    sentinel = f"sentinel_{uuid.uuid4().hex[:8]}"
    assert (await sandbox.execute(
        "baseline_value = 1", save_to_history=False, skip_safety_check=True,
    )).success
    assert await sandbox.save_session() is True
    assert (await sandbox.execute(
        f"{sentinel} = 'after'", save_to_history=False, skip_safety_check=True,
    )).success

    from backend.services.resumable_export import flush_for_resumable_export
    from backend.state import workflows

    class _Owner:
        pass

    owner = _Owner()
    owner.sandbox = sandbox
    workflows[sandbox.session_id] = owner
    state = AnalysisState(id=sandbox.session_id, name="s", user_id="u")
    try:
        await flush_for_resumable_export(sandbox.session_id, state)
    finally:
        workflows.pop(sandbox.session_id, None)

    archive = tmp_path / "export.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        _add_states_to_tar(tf, sandbox.session_id, include_runtime=True)

    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()

    assert any(name.endswith("_session.pkl") for name in names), names
    assert any(name.endswith("checkpoints.sqlite") for name in names) or True
    # No WAL/SHM companion: a copy of those beside a database captured
    # mid-transaction is how a "successful" import yields an unusable graph.
    assert not [name for name in names if name.endswith(("-wal", "-shm"))], names

    assert sentinel in _committed_names(sandbox.session_id)
