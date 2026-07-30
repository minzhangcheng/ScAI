"""Runtime memory preflight for Snapshot V2 captures.

The preflight exists because the capture itself allocates: the h5ad template
deep-copies each AnnData before sanitizing and writing it. It deliberately does
NOT police ordinary analysis code — predicting arbitrary user code is
unreliable, whereas the size of a copy this process is about to make is a
measured fact.
"""

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
