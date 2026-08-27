# RuntimeCapture: drives kernel-side capture of the live analysis runtime.
# Moved verbatim out of backend/services/snapshot_v2/runtime.py.
"""Runtime capture for Snapshot V2.

Owns kernel introspection, required-set assembly via the canonical registry,
and authoritative serializer success (h5ad + dill).
"""

from __future__ import annotations

import sys

import json
import shutil
import uuid
from pathlib import Path

from backend.config import SNAPSHOT_MEMORY_SAFETY_RATIO
from backend.services.snapshot_v2.errors import (
    SNAPSHOT_FAILURE_ACTIONS_MEMORY,
    SNAPSHOT_MEMORY_INSUFFICIENT_CODE,
    SnapshotCaptureError,
)
from backend.utils.error_parsing import compact_traceback
from backend.services.snapshot_v2.models import (
    ObjectRef,
    RuntimeDependencyManifest,
    RuntimeManifest,
    RuntimeSkip,
    empty_complete_runtime,
)
from backend.services.snapshot_v2.object_store import ObjectStore
from backend.services.snapshot_v2.paths import SnapshotV2Paths
from backend.services.snapshot_v2.runtime_names import (
    runtime_h5ad_path,
    validate_runtime_global_name,
)
from backend.services.snapshot_v2.runtime_registry import (
    CANONICAL_RUNTIME_ARTIFACTS,
    build_required_runtime_manifest,
    is_known_unpersistable_residual,
)


# Re-export for existing imports / tests.

from .kernel_templates import _INTROSPECT_CODE, _SAVE_ANNDATA_TEMPLATE, _SAVE_DILL_TEMPLATE
from . import capture_steps as _capture_steps
from .proof import RUNTIME_CAPTURE_FREE_SPACE_RESERVE_BYTES, _parse_dill_name_line, _parse_dill_saved_names, _parse_dill_skipped, _parse_dill_verified_names, _validate_h5ad_file


#: Multiplier applied to the largest captured AnnData when estimating what the
#: capture will allocate. The h5ad template deep-copies the object before
#: sanitizing it, so one full copy is the floor; the surcharge covers the
#: sanitize pass, which can convert individual obsm/varm slots to fresh arrays.
_CAPTURE_COPY_OVERHEAD = 1.1


def _parse_memory_payload(output: str) -> dict | None:
    """Extract the optional memory probe from introspection output.

    Returns ``None`` whenever the probe is absent or unreadable. A capture must
    never be refused because the evidence for refusing it could not be read.
    """
    begin_marker = "__RUNTIME_MEMORY_BEGIN__"
    end_marker = "__RUNTIME_MEMORY_END__"
    start = output.find(begin_marker)
    if start == -1:
        return None
    start += len(begin_marker)
    end = output.find(end_marker, start)
    if end == -1:
        return None
    try:
        parsed = json.loads(output[start:end].strip())
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _consume_memory_preflight_override(sandbox) -> bool:
    """One-shot check-and-consume of the user's "proceed anyway" decision.

    Mirrors ``consume_snapshot_skip``: the flag is armed by the proceed endpoint
    when the user accepts the OOM risk, and is cleared here so that a later
    capture is preflighted afresh instead of silently inheriting the override.

    The persisted flag is not sufficient on its own. It is force-written to
    session.json the moment Proceed is pressed, and the re-dispatched capture has
    several ways to never run at all (409 on a missing retry context,
    SAFE_POINT_BUSY, the disk gate, or a workflow_boundary hand-off waiting on a
    WebSocket reconnect that never comes). An approval is for one capture in one
    running process, so it is honoured only alongside the in-process grant that
    ``arm_memory_preflight_override`` records; a flag found without one is debris
    from an abandoned attempt or from a previous process and is cleared instead
    of being spent on a capture nobody accepted the risk for.
    """
    session_id = getattr(sandbox, "session_id", None)
    if not session_id:
        return False
    try:
        from backend.state import sessions
    except Exception:
        return False
    state = sessions.get(session_id)
    if state is None or not getattr(state, "force_snapshot_memory_override", False):
        return False

    from backend.services.snapshot_v2.failure_gate import (
        clear_memory_preflight_override,
        has_memory_preflight_grant,
    )

    granted = has_memory_preflight_grant(str(session_id))
    # Consumed either way: an honoured approval is spent, and a stale flag is
    # scrubbed so the next capture cannot inherit it.
    clear_memory_preflight_override(state)
    return granted


def _preflight_capture_memory(
    memory_meta: dict | None,
    h5ad_names: list[str],
    *,
    safety_ratio: float = SNAPSHOT_MEMORY_SAFETY_RATIO,
) -> None:
    """Refuse a capture whose AnnData copy would not fit in measured headroom.

    This is the only memory check in the analysis path, and it exists because
    the capture itself allocates: ``_SAVE_ANNDATA_TEMPLATE`` deep-copies each
    AnnData before writing it. Ordinary analysis code is deliberately not
    checked — predicting arbitrary user code is unreliable, whereas the size of
    a copy this process is about to make is a measured fact.

    Objects are captured one at a time, so the peak is driven by the largest
    single object rather than by their sum.

    Raises ``SnapshotCaptureError`` with the memory action set; returns silently
    when the probe is missing, when nothing goes through h5ad, or when headroom
    is unknown.
    """
    if not memory_meta or not h5ad_names:
        return
    headroom = memory_meta.get("headroom")
    if not isinstance(headroom, int) or headroom <= 0:
        return
    sizes = memory_meta.get("bytes")
    if not isinstance(sizes, dict):
        return

    measured = {
        name: int(sizes[name])
        for name in h5ad_names
        if isinstance(sizes.get(name), int)
    }
    if not measured:
        return

    largest_name = max(measured, key=lambda n: measured[n])
    largest_bytes = measured[largest_name]
    required = int(largest_bytes * _CAPTURE_COPY_OVERHEAD)
    budget = int(headroom * safety_ratio)
    if required <= budget:
        return

    gib = 1024 ** 3
    raise SnapshotCaptureError(
        SNAPSHOT_MEMORY_INSUFFICIENT_CODE,
        (
            f"Snapshot capture needs about {required / gib:.1f} GiB to copy "
            f"'{largest_name}' ({largest_bytes / gib:.1f} GiB) before writing it, "
            f"but only {headroom / gib:.1f} GiB of memory headroom is available "
            f"({budget / gib:.1f} GiB usable at the configured "
            f"{safety_ratio:.0%} safety ratio). Capturing now risks an "
            f"out-of-memory kill that would lose the live analysis state."
        ),
        details={
            "name": largest_name,
            "largest_object_bytes": largest_bytes,
            "required_bytes": required,
            "headroom_bytes": headroom,
            "budget_bytes": budget,
            "safety_ratio": safety_ratio,
            "kernel_rss_bytes": memory_meta.get("rss"),
            "allowed_actions": list(SNAPSHOT_FAILURE_ACTIONS_MEMORY),
        },
    )


class RuntimeCapture:
    FREE_SPACE_RESERVE_BYTES: int = RUNTIME_CAPTURE_FREE_SPACE_RESERVE_BYTES

    def __init__(self, object_store: ObjectStore, paths: SnapshotV2Paths):
        self._object_store = object_store
        self._paths = paths

    def mark_runtime_committed(self, sandbox, runtime_revision: int | None = None) -> None:
        """Signals that the current runtime generation has been committed.

        Called after a successful snapshot capture transaction. Sets
        ``saved_runtime_revision`` on the sandbox to the frozen
        ``runtime_revision`` (or current if not provided) so subsequent
        captures can detect drift.
        """
        if runtime_revision is None:
            runtime_revision = sandbox.runtime_revision
        sandbox.saved_runtime_revision = runtime_revision

    async def _run_kernel(
        self,
        sandbox,
        code: str,
        *,
        exec_lock_held: bool = False,
    ):
        """Execute trusted capture code; use unlocked path when lock already held.

        Asserts that the kernel namespace did not change generation underneath
        the capture. `_execute_unlocked`'s pre-execute liveness guard restarts a
        dead kernel and then runs the caller's code on the *new* one, which
        `_restart_kernel` has rehydrated from the last saved pkl — an older
        generation than the one this capture already partially serialized. With
        no re-check, a kernel reaped between two capture cells produced a
        manifest whose variables never coexisted, and every object hash in it is
        internally consistent so nothing downstream could detect it.
        """
        epoch_before = getattr(sandbox, "kernel_epoch", None)
        result = await self._run_kernel_unchecked(
            sandbox, code, exec_lock_held=exec_lock_held,
        )
        epoch_after = getattr(sandbox, "kernel_epoch", None)
        if (
            epoch_before is not None
            and epoch_after is not None
            and epoch_after != epoch_before
        ):
            raise SnapshotCaptureError(
                "required_runtime_unpersistable",
                "The kernel restarted during runtime capture; the namespace is "
                "a different generation from the one already serialized",
                details={
                    "kernel_epoch_before": epoch_before,
                    "kernel_epoch_after": epoch_after,
                },
            )
        return result

    async def _run_kernel_unchecked(
        self,
        sandbox,
        code: str,
        *,
        exec_lock_held: bool = False,
    ):
        """Raw execution without the kernel-generation assertion."""
        if exec_lock_held:
            unlocked = getattr(sandbox, "_execute_unlocked", None)
            if unlocked is not None and callable(unlocked):
                # MagicMock auto-attrs are callable but not coroutine functions;
                # only use when the call returns an awaitable.
                try:
                    maybe = unlocked(
                        code, save_to_history=False, skip_safety_check=True
                    )
                except TypeError:
                    maybe = None
                if hasattr(maybe, "__await__"):
                    return await maybe
                # Non-awaitable (e.g. MagicMock default) — fall through to execute.
        return await sandbox.execute(
            code, save_to_history=False, skip_safety_check=True
        )


    async def _introspect_globals(
        self, sandbox, *, exec_lock_held: bool = False,
    ) -> tuple[dict[str, str], dict | None]:
        """Read kernel globals metadata and the memory probe in one round-trip.

        Returns ``(globals_meta, memory_meta)``. The globals payload is
        authoritative and its absence is fatal. The memory payload is advisory:
        a kernel that could not produce it (older template, unreadable /proc)
        yields ``None``, and the caller then skips the memory preflight rather
        than refusing a capture on missing evidence.
        """
        result = await self._run_kernel(
            sandbox, _INTROSPECT_CODE, exec_lock_held=exec_lock_held
        )
        if not result.success:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_FAILED",
                "Kernel globals introspection failed: "
                + compact_traceback(result.error or "", max_chars=3000),
            )
        output = result.output or ""
        begin_marker = "__RUNTIME_GLOBALS_BEGIN__"
        end_marker = "__RUNTIME_GLOBALS_END__"
        start = output.find(begin_marker)
        if start == -1:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_PARSE_ERROR",
                f"Cannot find globals marker in output: {output[:500]}",
            )
        start += len(begin_marker)
        end = output.find(end_marker, start)
        if end == -1:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_PARSE_ERROR",
                "Missing end marker in introspection output",
            )
        json_str = output[start:end].strip()
        if not json_str:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_PARSE_ERROR",
                "Empty globals payload between markers",
            )
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_PARSE_ERROR",
                f"Failed to parse globals JSON: {e}",
                details={"json_snippet": json_str[:500]},
            ) from e
        if not isinstance(parsed, dict):
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_PARSE_ERROR",
                f"Globals payload is not a dict: {type(parsed).__name__}",
            )
        return (
            {str(k): str(v) for k, v in parsed.items()},
            _parse_memory_payload(output),
        )

    async def _capture_adata(
        self, sandbox, var_name: str, type_str: str, capture_dir: Path,
        *, exec_lock_held: bool = False,
    ) -> ObjectRef:
        var_name = validate_runtime_global_name(var_name)
        h5ad_path = runtime_h5ad_path(capture_dir, var_name)
        code = _SAVE_ANNDATA_TEMPLATE.format(
            var_name_repr=var_name,
            h5ad_path_repr=str(h5ad_path),
        )
        result = await self._run_kernel(
            sandbox, code, exec_lock_held=exec_lock_held
        )
        if not result.success:
            raise SnapshotCaptureError(
                "required_runtime_unpersistable",
                (
                    f"AnnData variable '{var_name}' (type={type_str or 'unknown'}, "
                    "serializer=h5ad) write failed: "
                    + compact_traceback(result.error or "", max_chars=3000)
                ),
                details={
                    "name": var_name,
                    "type": type_str,
                    "serializer": "h5ad",
                },
            )
        if not h5ad_path.exists():
            raise SnapshotCaptureError(
                "required_runtime_unpersistable",
                (
                    f"AnnData variable '{var_name}' (type={type_str or 'unknown'}, "
                    f"serializer=h5ad) could not be serialized to h5ad"
                ),
                details={
                    "name": var_name,
                    "type": type_str,
                    "serializer": "h5ad",
                },
            )
        _validate_h5ad_file(h5ad_path)
        try:
            put_result = self._object_store.put_file(h5ad_path, "application/x-hdf5")
            ref = put_result.ref
        except SnapshotCaptureError:
            raise
        except Exception as e:
            compact_error = compact_traceback(str(e), max_chars=1500)
            raise SnapshotCaptureError(
                "required_runtime_unpersistable",
                (
                    f"AnnData variable '{var_name}' (type={type_str or 'unknown'}, "
                    f"serializer=h5ad) object store put failed: {compact_error}"
                ),
                details={
                    "name": var_name,
                    "type": type_str,
                    "serializer": "h5ad",
                },
            ) from e
        h5ad_path.unlink(missing_ok=True)
        return ref


    async def _capture_residual_dill(
        self,
        sandbox,
        var_names: list[str],
        type_by_name: dict[str, str],
        capture_dir: Path,
        *,
        exec_lock_held: bool = False,
    ) -> tuple[ObjectRef | None, list[str], list[str]]:
        """Return (ref, saved_names, skipped_names)."""
        if not var_names:
            return None, [], []
        var_names = [validate_runtime_global_name(name) for name in var_names]
        dill_path = capture_dir / "residual.pkl"
        code = _SAVE_DILL_TEMPLATE.format(
            names_repr=var_names,
            dill_path_repr=str(dill_path),
        )
        result = await self._run_kernel(
            sandbox, code, exec_lock_held=exec_lock_held
        )
        output = result.output or ""
        skipped = _parse_dill_skipped(output)
        saved = _parse_dill_saved_names(output)
        saved_proof = _parse_dill_name_line(output, "__DILL_SAVED_NAMES__:")
        verified = _parse_dill_verified_names(output)
        if not result.success:
            # Treat whole batch as unpersistable for required names.
            return None, saved, list(var_names)
        if not dill_path.exists():
            return None, saved, list(set(var_names) - set(saved)) or list(var_names)
        # Current capture requires structural output and an in-kernel readback
        # proof. Output loss must never be interpreted as successful capture.
        if saved_proof is None or verified is None or verified != saved:
            dill_path.unlink(missing_ok=True)
            return None, [], list(var_names)
        put_result = self._object_store.put_file(dill_path, "application/octet-stream")
        ref = put_result.ref
        dill_path.unlink(missing_ok=True)
        return ref, saved, skipped

    def _raise_unpersistable(
        self, name: str, type_str: str, serializer: str, reason: str
    ) -> None:
        raise SnapshotCaptureError(
            "required_runtime_unpersistable",
            (
                f"Required variable '{name}' (type={type_str or 'unknown'}, "
                f"serializer={serializer}) {reason}"
            ),
            details={
                "name": name,
                "type": type_str,
                "serializer": serializer,
            },
        )

    async def capture(
        self,
        sandbox,
        workflow_state: dict | None = None,
        declared_outputs: set[str] | None = None,
        checkpoint_dependencies: set[str] | None = None,
        kernel_present: bool | None = None,
        *,
        exec_lock_held: bool = False,
    ) -> RuntimeManifest:
        """Capture required runtime dependencies from a live sandbox."""
        return await _capture_steps.capture(
            self,
            sandbox,
            workflow_state=workflow_state,
            declared_outputs=declared_outputs,
            checkpoint_dependencies=checkpoint_dependencies,
            kernel_present=kernel_present,
            exec_lock_held=exec_lock_held,
            bindings=sys.modules[__name__],
        )
