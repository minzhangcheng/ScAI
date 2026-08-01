"""Runtime memory preflight for Snapshot V2 captures.

The preflight exists because the capture itself allocates: the h5ad template
deep-copies each AnnData before sanitizing and writing it. It deliberately does
NOT police ordinary analysis code — predicting arbitrary user code is
unreliable, whereas the size of a copy this process is about to make is a
measured fact.
"""

from types import SimpleNamespace

import pytest

from backend.services.snapshot_v2.errors import (
    SNAPSHOT_FAILURE_ACTIONS_MEMORY,
    SNAPSHOT_MEMORY_INSUFFICIENT_CODE,
    SnapshotCaptureError,
)
from backend.services.snapshot_v2.failure_gate import (
    SNAPSHOT_FAILURE_GATE_ACTIONS,
    SNAPSHOT_SKIPPABLE_GATE_ACTIONS,
    _actions_for_code,
    _actions_for_producer,
)
from backend.services.snapshot_v2.runtime_parts.capture import (
    _parse_memory_payload,
    _preflight_capture_memory,
)
from backend.services.snapshot_v2.runtime_parts.kernel_templates import _INTROSPECT_CODE

GIB = 1024 ** 3


def _meta(*, largest_gib: float, headroom_gib: float, name: str = "adata") -> dict:
    return {
        "bytes": {name: int(largest_gib * GIB)},
        "rss": int(6.4 * GIB),
        "headroom": int(headroom_gib * GIB),
    }


def test_ample_headroom_allows_the_capture():
    # The reference deployment: a 1.3 GiB adata inside a 6.4 GiB kernel on a
    # host with 151 GiB free. Nothing here should ever be refused.
    _preflight_capture_memory(_meta(largest_gib=1.3, headroom_gib=151.0), ["adata"])


def test_capture_is_refused_when_the_copy_would_not_fit():
    with pytest.raises(SnapshotCaptureError) as excinfo:
        _preflight_capture_memory(_meta(largest_gib=20.0, headroom_gib=10.0), ["adata"])
    assert excinfo.value.code == SNAPSHOT_MEMORY_INSUFFICIENT_CODE
    details = excinfo.value.details
    assert details["name"] == "adata"
    assert details["allowed_actions"] == list(SNAPSHOT_FAILURE_ACTIONS_MEMORY)
    assert details["required_bytes"] > details["budget_bytes"]


def test_budget_is_the_largest_object_not_their_sum():
    # _capture_adata runs once per name, so only one copy is ever live. Summing
    # would refuse captures that comfortably fit.
    meta = {
        "bytes": {"a": 3 * GIB, "b": 3 * GIB, "c": 3 * GIB},
        "headroom": 5 * GIB,
    }
    _preflight_capture_memory(meta, ["a", "b", "c"])


def test_safety_ratio_shrinks_the_usable_headroom():
    meta = _meta(largest_gib=7.0, headroom_gib=10.0)
    _preflight_capture_memory(meta, ["adata"], safety_ratio=1.0)
    with pytest.raises(SnapshotCaptureError):
        _preflight_capture_memory(meta, ["adata"], safety_ratio=0.5)


@pytest.mark.parametrize(
    "memory_meta, names",
    [
        (None, ["adata"]),                                   # probe absent
        ({"bytes": {"adata": 99 * GIB}, "headroom": None}, ["adata"]),  # unknown headroom
        ({"bytes": {"adata": 99 * GIB}, "headroom": 0}, ["adata"]),     # unreadable headroom
        ({"bytes": {"other": 99 * GIB}, "headroom": GIB}, ["adata"]),   # target unmeasured
        ({"bytes": "not-a-dict", "headroom": GIB}, ["adata"]),          # malformed probe
        (_meta(largest_gib=99.0, headroom_gib=1.0), []),                # nothing via h5ad
    ],
)
def test_missing_or_unusable_evidence_never_refuses(memory_meta, names):
    # A capture must not be blocked because the evidence for blocking it could
    # not be read; the fallback is the previous behaviour, not a refusal.
    _preflight_capture_memory(memory_meta, names)


def test_memory_payload_round_trips_from_kernel_output():
    output = (
        "__RUNTIME_GLOBALS_BEGIN__\n{}\n__RUNTIME_GLOBALS_END__\n"
        '__RUNTIME_MEMORY_BEGIN__\n{"bytes": {"adata": 7}, "rss": 8, "headroom": 9}\n'
        "__RUNTIME_MEMORY_END__\n"
    )
    assert _parse_memory_payload(output) == {
        "bytes": {"adata": 7}, "rss": 8, "headroom": 9,
    }


@pytest.mark.parametrize("output", [
    "no markers at all",
    "__RUNTIME_MEMORY_BEGIN__\nnot json\n__RUNTIME_MEMORY_END__",
    "__RUNTIME_MEMORY_BEGIN__\n[1, 2]\n__RUNTIME_MEMORY_END__",
    "__RUNTIME_MEMORY_BEGIN__\n{}",  # truncated, no end marker
])
def test_unparsable_memory_payload_yields_none(output):
    assert _parse_memory_payload(output) is None


def test_introspect_template_emits_both_payloads():
    # The globals payload is authoritative for the required set; the memory
    # payload feeds the preflight. Losing either marker silently disables a
    # feature, so pin both.
    assert "__RUNTIME_GLOBALS_BEGIN__" in _INTROSPECT_CODE
    assert "__RUNTIME_MEMORY_BEGIN__" in _INTROSPECT_CODE
    # Sparse matrices must be measured through data/indices/indptr, never via
    # the container's own meaningless nbytes.
    assert "indptr" in _INTROSPECT_CODE
    # The probe must not leave its helpers behind in the analysis namespace.
    assert "globals().pop(" in _INTROSPECT_CODE


def test_memory_gate_drops_retry_and_manage():
    # Retry would re-run a deterministic check against the same kernel and
    # refuse again — the gate could never clear. Manage reclaims disk, not RAM.
    actions = _actions_for_code(
        SNAPSHOT_MEMORY_INSUFFICIENT_CODE, SNAPSHOT_SKIPPABLE_GATE_ACTIONS
    )
    assert actions == ["proceed", "skip", "terminate"]
    assert "retry" not in actions and "manage" not in actions


def test_memory_gate_does_not_smuggle_skip_into_non_skippable_producers():
    # pre_restore's snapshot is the only safety point before a restore
    # overwrites live state; a memory refusal must not make it skippable.
    actions = _actions_for_code(
        SNAPSHOT_MEMORY_INSUFFICIENT_CODE, SNAPSHOT_FAILURE_GATE_ACTIONS
    )
    assert actions == ["proceed", "terminate"]


def test_non_memory_failures_keep_their_producer_actions():
    assert _actions_for_code("SOME_OTHER_CODE", SNAPSHOT_SKIPPABLE_GATE_ACTIONS) == list(
        SNAPSHOT_SKIPPABLE_GATE_ACTIONS
    )
    assert _actions_for_producer("workflow_boundary") == list(
        SNAPSHOT_SKIPPABLE_GATE_ACTIONS
    )
    assert _actions_for_producer("pre_restore") == list(SNAPSHOT_FAILURE_GATE_ACTIONS)


def test_actions_for_producer_honours_the_memory_code():
    assert _actions_for_producer(
        "workflow_boundary", SNAPSHOT_MEMORY_INSUFFICIENT_CODE
    ) == ["proceed", "skip", "terminate"]
    assert _actions_for_producer(
        "pre_restore", SNAPSHOT_MEMORY_INSUFFICIENT_CODE
    ) == ["proceed", "terminate"]


# --- A memory refusal must reach the dialog with its own action set ----------
#
# The memory action set lives inside ``details``. ``SnapshotFailureRecord``
# defaults ``allowed_actions`` to the base set, so the gate's "not in payload"
# guard saw the key present and left it: the dialog rendered Retry / Manage /
# Terminate with no "Capture anyway" and POST …/snapshot-failure/proceed
# answered 409. Retry re-runs a deterministic preflight and refuses identically
# every time, and pre_restore is not skippable by design, so Terminate — which
# stops the analysis — was the user's only exit.


def _memory_failure_record(**kwargs):
    from backend.services.snapshot_v2.models import SnapshotFailureRecord

    return SnapshotFailureRecord(
        snapshot_id="snap-1",
        session_id="s1",
        failed_at="2026-01-01T00:00:00",
        code=SNAPSHOT_MEMORY_INSUFFICIENT_CODE,
        message="not enough headroom",
        details={"allowed_actions": list(SNAPSHOT_FAILURE_ACTIONS_MEMORY)},
        **kwargs,
    )


def test_failure_record_derives_its_actions_from_the_code():
    from backend.services.snapshot_v2.models import SnapshotFailureRecord

    assert _memory_failure_record().allowed_actions == ["proceed", "terminate"]
    # An ordinary refusal is unchanged: the base set is still the right answer.
    ordinary = SnapshotFailureRecord(
        snapshot_id="snap-1",
        session_id="s1",
        failed_at="2026-01-01T00:00:00",
        code="required_runtime_unpersistable",
        message="boom",
    )
    assert ordinary.allowed_actions == list(SNAPSHOT_FAILURE_GATE_ACTIONS)


def test_an_explicit_action_set_and_a_round_trip_are_never_rewritten():
    from backend.services.snapshot_v2.models import SnapshotFailureRecord

    explicit = _memory_failure_record(allowed_actions=["retry", "terminate"])
    assert explicit.allowed_actions == ["retry", "terminate"]

    derived = _memory_failure_record()
    again = SnapshotFailureRecord.model_validate(derived.model_dump(mode="json"))
    assert again.allowed_actions == derived.allowed_actions


def test_pre_restore_memory_gate_offers_capture_anyway():
    from backend.services.snapshot_v2.failure_gate import enter_snapshot_failure_gate

    state = SimpleNamespace()
    # pre_restore.py hands the dumped record straight to the gate.
    enter_snapshot_failure_gate(
        state,
        _memory_failure_record().model_dump(mode="json"),
        retry_context={"producer": "pre_restore"},
    )

    actions = state.snapshot_failure["allowed_actions"]
    assert "proceed" in actions, (
        "the memory gate reached the dialog without its escape hatch; "
        "Terminate would be the only way out of a deterministic refusal"
    )
    assert "skip" not in actions, (
        "pre_restore's snapshot is the only safety point before a restore "
        "overwrites live state"
    )
    assert "retry" not in actions and "manage" not in actions


def test_a_stale_or_defaulted_action_set_is_repaired_by_the_gate():
    """The gate is the last line: a payload can also arrive from disk."""
    from backend.services.snapshot_v2.failure_gate import enter_snapshot_failure_gate

    state = SimpleNamespace()
    enter_snapshot_failure_gate(state, {
        "code": SNAPSHOT_MEMORY_INSUFFICIENT_CODE,
        "message": "not enough headroom",
        "allowed_actions": list(SNAPSHOT_FAILURE_GATE_ACTIONS),
    })
    assert state.snapshot_failure["allowed_actions"] == ["proceed", "terminate"]

    # A caller that does allow skipping keeps it.
    skippable = SimpleNamespace()
    enter_snapshot_failure_gate(skippable, {
        "code": SNAPSHOT_MEMORY_INSUFFICIENT_CODE,
        "message": "not enough headroom",
        "allowed_actions": list(SNAPSHOT_SKIPPABLE_GATE_ACTIONS),
    })
    assert skippable.snapshot_failure["allowed_actions"] == [
        "proceed", "skip", "terminate",
    ]

    # An empty list is "unspecified", not "no actions at all".
    empty = SimpleNamespace()
    enter_snapshot_failure_gate(empty, {
        "code": "required_runtime_unpersistable",
        "message": "boom",
        "allowed_actions": [],
    })
    assert empty.snapshot_failure["allowed_actions"] == list(
        SNAPSHOT_FAILURE_GATE_ACTIONS
    )


# --- "Capture anyway" is approval for ONE capture ---------------------------
#
# The flag is force-written to session.json the moment Proceed is pressed, and
# the re-dispatched capture has several ways never to run (409 for a missing
# retry context, SAFE_POINT_BUSY, the disk gate, a workflow_boundary hand-off
# waiting on a WebSocket reconnect that never comes). It used to be cleared only
# inside the h5ad branch of a capture, so an aborted attempt left the preflight
# permanently disabled and a later capture on a different, larger AnnData
# skipped it — accepting an OOM risk on the user's behalf.


@pytest.fixture
def override_session():
    """A session registered in ``backend.state`` with a sandbox pointing at it."""
    import backend.state as backend_state
    from backend.services.snapshot_v2.failure_gate import (
        clear_memory_preflight_override,
    )

    session_id = "override-session"
    state = SimpleNamespace(
        id=session_id,
        force_snapshot_memory_override=False,
        status=None,
        wait_input_reason=None,
        snapshot_failure=None,
        pending_required_snapshot=None,
        skip_required_snapshot_phase=None,
        termination_reason=None,
    )
    backend_state.sessions[session_id] = state
    try:
        yield state, SimpleNamespace(session_id=session_id)
    finally:
        clear_memory_preflight_override(state)
        backend_state.sessions.pop(session_id, None)


def test_the_approved_capture_still_gets_its_override(override_session):
    from backend.services.snapshot_v2.failure_gate import (
        arm_memory_preflight_override,
    )
    from backend.services.snapshot_v2.runtime_parts.capture import (
        _consume_memory_preflight_override,
    )

    state, sandbox = override_session
    arm_memory_preflight_override(state)

    assert _consume_memory_preflight_override(sandbox) is True
    assert state.force_snapshot_memory_override is False


def test_a_second_capture_never_inherits_the_override(override_session):
    from backend.services.snapshot_v2.failure_gate import (
        arm_memory_preflight_override,
    )
    from backend.services.snapshot_v2.runtime_parts.capture import (
        _consume_memory_preflight_override,
    )

    state, sandbox = override_session
    arm_memory_preflight_override(state)
    _consume_memory_preflight_override(sandbox)

    assert _consume_memory_preflight_override(sandbox) is False


def test_an_override_left_on_disk_is_inert_and_scrubbed(override_session):
    """No grant means the flag is debris from an abandoned attempt.

    A capture that aborts before the preflight, and a process that restarted
    since the approval, both leave exactly this: ``True`` in session.json with
    nothing recording which capture it was for.
    """
    from backend.services.snapshot_v2.runtime_parts.capture import (
        _consume_memory_preflight_override,
    )

    state, sandbox = override_session
    state.force_snapshot_memory_override = True  # as read back from session.json

    assert _consume_memory_preflight_override(sandbox) is False
    assert state.force_snapshot_memory_override is False


def test_re_entering_the_gate_spends_the_approval(override_session):
    from backend.services.snapshot_v2.failure_gate import (
        arm_memory_preflight_override,
        enter_snapshot_failure_gate,
    )
    from backend.services.snapshot_v2.runtime_parts.capture import (
        _consume_memory_preflight_override,
    )

    state, sandbox = override_session
    arm_memory_preflight_override(state)
    # The retried capture failed for some other reason and re-gated the session.
    enter_snapshot_failure_gate(state, {
        "code": "required_runtime_unpersistable", "message": "boom",
    })

    assert state.force_snapshot_memory_override is False
    assert _consume_memory_preflight_override(sandbox) is False


def test_terminate_abandons_the_approval(override_session):
    from backend.services.snapshot_v2.failure_gate import (
        arm_memory_preflight_override,
        terminate_snapshot_failure,
    )
    from backend.services.snapshot_v2.runtime_parts.capture import (
        _consume_memory_preflight_override,
    )

    state, sandbox = override_session
    arm_memory_preflight_override(state)
    terminate_snapshot_failure(state)

    assert state.force_snapshot_memory_override is False
    assert _consume_memory_preflight_override(sandbox) is False


def test_another_session_cannot_spend_this_approval(override_session):
    import backend.state as backend_state
    from backend.services.snapshot_v2.failure_gate import (
        arm_memory_preflight_override,
    )
    from backend.services.snapshot_v2.runtime_parts.capture import (
        _consume_memory_preflight_override,
    )

    state, _sandbox = override_session
    arm_memory_preflight_override(state)

    other = SimpleNamespace(id="other-session", force_snapshot_memory_override=True)
    backend_state.sessions["other-session"] = other
    try:
        assert _consume_memory_preflight_override(
            SimpleNamespace(session_id="other-session")
        ) is False
    finally:
        backend_state.sessions.pop("other-session", None)


@pytest.mark.asyncio
async def test_proceed_disarms_and_persists_when_the_retry_never_dispatches(
    monkeypatch, override_session,
):
    """The 409 path is where the override used to become durable debris."""
    from fastapi import HTTPException

    from backend.routes import lifecycle
    from backend.services.snapshot_v2.failure_gate import (
        has_memory_preflight_grant,
    )

    state, _sandbox = override_session
    state.wait_input_reason = "snapshot_failure"
    state.snapshot_failure = {
        "code": SNAPSHOT_MEMORY_INSUFFICIENT_CODE,
        "message": "not enough headroom",
        "allowed_actions": ["proceed", "terminate"],
        # No retry_context: _snapshot_failure_retry_locked answers 409.
    }
    saved: list[bool] = []

    async def _owner(session_id, user_id):
        return state

    async def _cache(*args, **kwargs):
        return None

    async def _save(session_id, st, force=False):
        saved.append(bool(st.force_snapshot_memory_override))

    monkeypatch.setattr(lifecycle, "require_session_ownership", _owner)
    monkeypatch.setattr(lifecycle, "_update_session_cache", _cache)
    monkeypatch.setattr(lifecycle, "async_save_session", _save)

    with pytest.raises(HTTPException) as excinfo:
        await lifecycle._snapshot_failure_proceed_locked(state.id, "u1")

    assert excinfo.value.status_code == 409
    assert state.force_snapshot_memory_override is False
    assert has_memory_preflight_grant(state.id) is False
    assert saved == [False], (
        "the disarmed override was not persisted; the next process load would "
        "read it back as an approval"
    )


# --- The size probe must measure what AnnData.copy() really duplicates -------


def _introspect_helpers():
    """The shipped template's helper defs, executed exactly as the kernel does."""
    namespace: dict = {}
    source = _INTROSPECT_CODE.split("for _name, _obj in list(globals().items()):")[0]
    exec(source, namespace)  # noqa: S102 - this is the shipped kernel template
    return namespace


def test_a_reused_address_does_not_zero_out_a_live_array():
    """``_seen`` must hold the objects, not just their ids.

    ``.X`` / ``.raw.X`` materialise a fresh temporary on every access for a
    view, a backed store, or a dask/CuPy-wrapped matrix. The temporary is freed
    when the call returns, CPython hands the same address to the next array, and
    that array was then measured as 0 bytes — the total feeds the preflight,
    which passes, and the deep copy OOM-kills the kernel.
    """
    import numpy as np

    matrix_bytes = _introspect_helpers()["_runtime_matrix_bytes"]

    seen = {}
    freed = np.zeros(1000, dtype=np.float64)
    freed_id = id(freed)
    assert matrix_bytes(freed, seen) == freed.nbytes
    del freed

    # With the object held in _seen the address cannot come back at all; if it
    # ever does, the second array must still be measured.
    for _ in range(4000):
        candidate = np.zeros(1000, dtype=np.float64)
        if id(candidate) == freed_id:
            assert matrix_bytes(candidate, seen) == candidate.nbytes
            break
        del candidate


def test_uns_and_the_frames_are_measured_because_copy_duplicates_them():
    anndata = pytest.importorskip("anndata")
    import numpy as np
    import pandas as pd

    adata_bytes = _introspect_helpers()["_runtime_adata_bytes"]

    adata = anndata.AnnData(X=np.zeros((200, 10), dtype=np.float64))
    # AnnData.copy() does `deepcopy(self._uns)`, so this is allocated twice.
    adata.uns["stored_graph"] = np.zeros(1_000_000, dtype=np.float64)
    adata.obs["label"] = pd.Categorical(["a"] * 200)

    measured = adata_bytes(adata)

    assert measured >= adata.uns["stored_graph"].nbytes + adata.X.nbytes, (
        "uns was not counted; any session whose bulk lives in a stored graph or "
        "a results table is undercounted on every run"
    )


def test_the_estimate_does_not_exceed_what_the_copy_actually_allocates():
    """Over-counting refuses captures that would have fitted."""
    anndata = pytest.importorskip("anndata")
    import tracemalloc

    import numpy as np
    import pandas as pd

    adata_bytes = _introspect_helpers()["_runtime_adata_bytes"]

    adata = anndata.AnnData(X=np.zeros((2000, 500), dtype=np.float64))
    adata.uns["stored_graph"] = np.zeros(1_000_000, dtype=np.float64)
    adata.obs["label"] = pd.Categorical(["a"] * 2000)
    estimate = adata_bytes(adata)

    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    duplicate = adata.copy()
    allocated = tracemalloc.get_traced_memory()[0] - before
    tracemalloc.stop()
    del duplicate

    assert estimate <= allocated * 1.05, (
        f"estimate {estimate} overstates the real copy {allocated}"
    )


def test_a_matrix_shared_between_slots_is_still_counted_once():
    anndata = pytest.importorskip("anndata")
    import numpy as np

    adata_bytes = _introspect_helpers()["_runtime_adata_bytes"]

    x = np.zeros((100, 100), dtype=np.float64)
    adata = anndata.AnnData(X=x)
    adata.layers["alias"] = adata.X

    # Frames and the empty uns add a little; the aliased layer must not add a
    # second full matrix.
    assert adata_bytes(adata) < 2 * x.nbytes
