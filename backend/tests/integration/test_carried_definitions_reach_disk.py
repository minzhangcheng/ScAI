"""Spec 5.5's carried-definitions record, end to end through a real save.

The unit tests say each hop works: the kernel emits the marker,
`_parse_coverage_markers` reads it, `build_runtime_manifest` stores it, and
`read_runtime_manifest` threads it back. The first live run showed all of that
passing while nothing reached disk, because the save never handed the parsed
record on. Only a real save proves the hops are connected.

The assertion is on the manifest **read back from disk**, not on the parser's
return value -- asserting on the parser is what let the gap exist.
"""
from __future__ import annotations

import os

import pytest
import pytest_asyncio

pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("SCAI_RUN_KERNEL_TESTS"),
        reason="Set SCAI_RUN_KERNEL_TESTS=1 to run kernel-dependent tests",
    ),
]


@pytest_asyncio.fixture
async def sandbox(session_id):
    """A sandbox where `get_session_states` resolves, so the save and the
    manifest agree on one directory."""
    import shutil

    from backend.config import get_session_dir, get_session_states
    from backend.utils.sandbox import LocalJupyterSandbox

    session_dir = get_session_dir(session_id)
    states = get_session_states(session_id)
    states.mkdir(parents=True, exist_ok=True)
    workspace = session_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    sb = LocalJupyterSandbox(
        session_id=session_id, state_dir=str(states), working_dir=str(workspace),
    )
    await sb.start()
    try:
        yield sb
    finally:
        await sb.stop(save=False)
        shutil.rmtree(session_dir, ignore_errors=True)


def _manifest_for(sandbox) -> dict:
    from backend.utils.runtime_manifest import (
        read_runtime_manifest,
        runtime_manifest_path,
    )

    path = runtime_manifest_path(
        sandbox.session_id, states_dir=str(sandbox.notebook_state.state_dir)
    )
    assert path.exists(), f"no generation manifest at {path}"
    return read_runtime_manifest(path)


@pytest.mark.asyncio
async def test_a_helper_the_analysis_defined_is_named_in_the_manifest(sandbox):
    """The three-hop path, asserted at its end.

    A restore that cannot say what it brought back is the state §4.4 forbids in
    the other direction: a name absent from the dump is absent from every record
    of it, and a name present under a caveat has to say so.
    """
    result = await sandbox.execute(
        "def helper(x):\n    return x * 2\n\nvalue = helper(21)\n",
        skip_safety_check=True,
    )
    assert result.success, result.error
    assert await sandbox.save_session() is True

    manifest = _manifest_for(sandbox)

    assert "helper" in manifest["saved_names"], manifest["saved_names"]
    carried = {c["name"]: c for c in manifest.get("carried_definitions", [])}
    assert "helper" in carried, (
        "the save reported no carried definitions; the record was parsed and "
        "dropped, which is exactly the gap the first live run found"
    )
    assert carried["helper"]["kind"] == "function"


@pytest.mark.asyncio
async def test_a_carried_class_records_its_caveat_on_disk(sandbox):
    """A carried class is not `type(instance)` for instances restored from
    other frames, because each frame is its own pickle with its own memo. That
    is a change in failure mode -- NameError before, a silent False now -- so it
    travels with the name rather than living only in the spec."""
    result = await sandbox.execute(
        "class Marker:\n    def __init__(self):\n        self.tag = 'a'\n\n"
        "instance = Marker()\n",
        skip_safety_check=True,
    )
    assert result.success, result.error
    assert await sandbox.save_session() is True

    carried = {c["name"]: c for c in _manifest_for(sandbox).get("carried_definitions", [])}

    assert "Marker" in carried, sorted(carried)
    assert carried["Marker"]["kind"] == "class"
    assert "type(instance)" in carried["Marker"].get("note", "")


@pytest.mark.asyncio
async def test_an_imported_function_is_not_reported_as_carried(sandbox):
    """The record names what §5.5 brought in, not everything callable.

    `sqrt` is excluded by kind and re-established by the notebook's own import;
    reporting it would make the record useless for judging what the narrowing
    actually costs.
    """
    result = await sandbox.execute(
        "from math import sqrt\n\ndef mine(x):\n    return sqrt(x)\n",
        skip_safety_check=True,
    )
    assert result.success, result.error
    assert await sandbox.save_session() is True

    manifest = _manifest_for(sandbox)
    carried = {c["name"] for c in manifest.get("carried_definitions", [])}

    assert "mine" in carried
    assert "sqrt" not in carried
    assert "sqrt" in {s["name"] for s in manifest["skipped_names"]}
