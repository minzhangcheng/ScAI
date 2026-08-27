"""Runtime-capture traversal and completeness-proof construction."""
from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.services.snapshot_v2.errors import SnapshotCaptureError
from backend.services.snapshot_v2.models import (
    ObjectRef,
    RuntimeDependencyManifest,
    RuntimeManifest,
    RuntimeSkip,
    empty_complete_runtime,
)
from backend.services.snapshot_v2.runtime_registry import (
    CANONICAL_RUNTIME_ARTIFACTS,
    build_required_runtime_manifest,
    is_known_unpersistable_residual,
)


@dataclass
class _LiveCaptureState:
    dep_manifest: Any
    globals_meta: dict[str, str]
    memory_meta: dict | None
    memory_override: bool
    mandatory_names: set[str]
    pruned_required: dict[str, Any]
    captured_adata_refs: dict[str, ObjectRef] = field(default_factory=dict)
    captured_residual_vars: list[str] = field(default_factory=list)
    captured_residual_ref: ObjectRef | None = None
    incidental_skips: list[RuntimeSkip] = field(default_factory=list)
    dropped_present: set[str] = field(default_factory=set)

async def capture(
    runtime_capture,
    sandbox,
    workflow_state: dict | None = None,
    declared_outputs: set[str] | None = None,
    checkpoint_dependencies: set[str] | None = None,
    kernel_present: bool | None = None,
    *,
    exec_lock_held: bool = False,
    bindings,
) -> RuntimeManifest:
    """Capture required runtime dependencies from a live sandbox.

    Returns a :class:`RuntimeManifest` directly (no transitional wrapper).

    ``kernel_present``:
      - ``False``: empty required set is accepted as complete (no runtime refs).
      - ``True``/``None`` with live kernel: empty required set is fail-closed.

    ``exec_lock_held``:
      When True, caller already holds ``sandbox._exec_lock``; use
      ``_execute_unlocked`` to avoid non-reentrant asyncio.Lock deadlock.
    """
    if kernel_present is None:
        kernel_present = getattr(sandbox, "km", None) is not None

    # No live kernel: do not introspect; empty complete is valid.
    if kernel_present is False:
        # Still allow declared/checkpoint names to force a required set, but
        # without a kernel they cannot be captured.
        dep_manifest = build_required_runtime_manifest(
            globals_meta={},
            persistent_outputs=declared_outputs,
            checkpoint_dependencies=checkpoint_dependencies,
        )
        if dep_manifest.required:
            missing = sorted(dep_manifest.required)
            raise SnapshotCaptureError(
                "required_runtime_unpersistable",
                (
                    "Required runtime values declared but no live kernel "
                    f"present to capture them: {missing}"
                ),
                details={"names": missing},
            )
        return empty_complete_runtime()

    # Spend the user's "capture anyway" approval at the top of the attempt
    # it belongs to, not at the preflight call further down: everything
    # between the two can abort (introspection failure, an empty required
    # set, the kernel-generation assertion), and an attempt that dies there
    # must not hand the approval on to whatever capture comes next.
    memory_override = bindings._consume_memory_preflight_override(sandbox)

    globals_meta, memory_meta = await runtime_capture._introspect_globals(
        sandbox, exec_lock_held=exec_lock_held
    )

    dep_manifest = build_required_runtime_manifest(
        globals_meta=globals_meta,
        persistent_outputs=declared_outputs,
        checkpoint_dependencies=checkpoint_dependencies,
    )
    required = set(dep_manifest.required)

    if not required:
        # Live kernel + empty required set is fail-closed: registry and
        # introspection must be authoritative; empty is only valid when
        # kernel_present=False.
        raise SnapshotCaptureError(
            "NO_RUNTIME_VARIABLES",
            "No runtime variables found to capture from live kernel",
        )

    # Kernel writes must stay inside one of its explicit leaf roots. Keep
    # transient runtime material under the sandbox state directory (the
    # same storage device as regular persistence), then let the host-side
    # object store authenticate and commit it. Snapshot V2's shared
    # staging tree is intentionally not kernel-writable.
    kernel_staging = getattr(sandbox, "_kernel_staging_root", None)
    capture_root = (
        Path(kernel_staging)
        if isinstance(kernel_staging, (str, Path))
        else runtime_capture._paths.staging
    )
    if isinstance(kernel_staging, (str, Path)):
        if capture_root.is_symlink():
            raise SnapshotCaptureError(
                "RUNTIME_CAPTURE_STAGING_INVALID",
                "Kernel capture staging root must not be a symlink",
            )
        capture_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        capture_root.chmod(0o700)
        if capture_root.resolve().parent != capture_root.parent.resolve():
            raise SnapshotCaptureError(
                "RUNTIME_CAPTURE_STAGING_INVALID",
                "Kernel capture staging root escaped its state directory",
            )
    capture_dir = capture_root / f"rtcap_{uuid.uuid4().hex[:12]}"
    capture_dir.mkdir(parents=True, exist_ok=True)

    state = _LiveCaptureState(
        dep_manifest=dep_manifest,
        globals_meta=globals_meta,
        memory_meta=memory_meta,
        memory_override=memory_override,
        mandatory_names=set(checkpoint_dependencies or set()),
        pruned_required=dict(dep_manifest.required),
    )

    try:
        await _capture_components(
            runtime_capture,
            sandbox,
            capture_dir,
            state,
            exec_lock_held=exec_lock_held,
            bindings=bindings,
        )
        return _build_runtime_manifest(runtime_capture, state)
    finally:
        try:
            if capture_dir.exists():
                for p in capture_dir.iterdir():
                    p.unlink(missing_ok=True)
                capture_dir.rmdir()
        except OSError:
            pass

async def _capture_components(
    runtime_capture,
    sandbox,
    capture_dir: Path,
    state: _LiveCaptureState,
    *,
    exec_lock_held: bool,
    bindings,
) -> None:
    dep_manifest = state.dep_manifest
    globals_meta = state.globals_meta
    memory_meta = state.memory_meta
    memory_override = state.memory_override
    mandatory_names = state.mandatory_names
    pruned_required = state.pruned_required
    captured_adata_refs = state.captured_adata_refs
    captured_residual_vars = state.captured_residual_vars
    captured_residual_ref = state.captured_residual_ref
    incidental_skips = state.incidental_skips
    dropped_present = state.dropped_present

    def _soft_skip_absent(name: str, serializer: str) -> None:
        type_str = dep_manifest.required[name].type_name
        incidental_skips.append(
            RuntimeSkip(
                name=name,
                type_name=type_str,
                reason=(
                    f"declared runtime dependency absent from kernel globals "
                    f"at capture (serializer={serializer}); soft-skipped"
                ),
            )
        )
        pruned_required.pop(name, None)

    # Partition required names into h5ad vs dill based on dependency serializer.
    h5ad_names: list[str] = []
    dill_names: list[str] = []
    for name, dep in dep_manifest.required.items():
        if dep.serializer == "h5ad":
            h5ad_names.append(name)
        else:
            dill_names.append(name)

    # Refuse before allocating anything if the copy this capture is
    # about to make would not fit. Raised here rather than inside
    # _capture_adata so nothing has been written to the staging dir yet.
    if not memory_override:
        bindings._preflight_capture_memory(memory_meta, h5ad_names)

    for var_name in sorted(h5ad_names):
        type_str = globals_meta.get(var_name, dep_manifest.required[var_name].type_name)
        if var_name not in globals_meta:
            if var_name in mandatory_names:
                runtime_capture._raise_unpersistable(
                    var_name, type_str, "h5ad", "not found in kernel globals"
                )
            _soft_skip_absent(var_name, "h5ad")
            continue
        try:
            ref = await runtime_capture._capture_adata(
                sandbox, var_name, type_str, capture_dir,
                exec_lock_held=exec_lock_held,
            )
        except SnapshotCaptureError:
            raise
        except Exception as e:
            runtime_capture._raise_unpersistable(
                var_name, type_str, "h5ad", f"capture failed: {e}"
            )
        captured_adata_refs[var_name] = ref

    # Drop absent advisory dill names (soft-skip); keep absent mandatory
    # names so the fail-closed check below still raises for them.
    # Also pre-skip present advisory names with known-unpersistable types
    # (e.g. LoomConnection file handles) so they never brick capture.
    present_dill_names: list[str] = []
    for var_name in sorted(dill_names):
        type_str = globals_meta.get(
            var_name, dep_manifest.required[var_name].type_name
        )
        if var_name not in globals_meta and var_name not in mandatory_names:
            _soft_skip_absent(var_name, "dill")
            continue
        if (
            var_name not in mandatory_names
            and var_name in globals_meta
            and is_known_unpersistable_residual(type_str)
        ):
            incidental_skips.append(
                RuntimeSkip(
                    name=var_name,
                    type_name=type_str,
                    reason=(
                        "known unpersistable residual type; soft-skipped "
                        f"(serializer=dill): {type_str}"
                    ),
                )
            )
            pruned_required.pop(var_name, None)
            # The value IS live in kernel globals — this drop discards
            # real state, which is exactly what `dropped_present` marks
            # (see its comment above). Recording it only here, and not
            # for the absent-name soft skip, keeps the deliberate
            # "dropping an absent name loses nothing" behaviour while
            # letting the post-pruning guard below see that a live value
            # was dropped. Without it, a session whose only tracked
            # globals were unpersistable handles published a live-kernel
            # runtime marked "complete" with zero content — the state
            # the invariant at the top of this method declares valid
            # only when kernel_present is False.
            dropped_present.add(var_name)
            continue
        present_dill_names.append(var_name)

    residual_names = sorted(present_dill_names)
    if residual_names:
        for var_name in residual_names:
            if var_name not in globals_meta:
                type_str = dep_manifest.required[var_name].type_name
                runtime_capture._raise_unpersistable(
                    var_name, type_str, "dill", "not found in kernel globals"
                )
        type_by_name = {
            n: globals_meta.get(n, dep_manifest.required[n].type_name)
            for n in residual_names
        }
        try:
            ref, saved, dill_skipped = await runtime_capture._capture_residual_dill(
                sandbox, residual_names, type_by_name, capture_dir,
                exec_lock_held=exec_lock_held,
            )
        except SnapshotCaptureError:
            raise
        except Exception as e:
            raise SnapshotCaptureError(
                "required_runtime_unpersistable",
                f"Residual dill capture failed: {e}",
                details={"names": residual_names},
            ) from e

        required_skipped = set(residual_names) & set(dill_skipped)
        # Also treat required names that neither saved nor skipped as missing.
        if saved is not None:
            not_accounted = set(residual_names) - set(saved) - set(dill_skipped)
            required_skipped |= not_accounted

        # A canonical artifact that is LIVE in kernel globals and that
        # dill unexpectedly failed on is not the case soft-skip exists
        # for. Soft-skip's rationale — "the user already deleted it, so
        # dropping it loses nothing" — applies to absent names, and the
        # pre-skip above already covers types that are inherently
        # unpersistable. What is left here is real state that was there
        # and did not survive: dropping it silently published a runtime
        # marked "complete" that the restored kernel then hits as a
        # NameError. `checkpoint_dependencies` was meant to carry this
        # tier, but nothing in production writes it.
        strict_names = mandatory_names | {
            n for n in required_skipped
            if n in globals_meta and n in CANONICAL_RUNTIME_ARTIFACTS
        }
        advisory_skipped = {
            n for n in required_skipped if n not in strict_names
        }
        mandatory_skipped = {
            n for n in required_skipped if n in strict_names
        }
        for n in sorted(advisory_skipped):
            type_str = type_by_name.get(n, "unknown")
            incidental_skips.append(
                RuntimeSkip(
                    name=n,
                    type_name=type_str,
                    reason=(
                        "dill serialization failed for advisory residual; "
                        "soft-skipped"
                    ),
                )
            )
            pruned_required.pop(n, None)
            dropped_present.add(n)
        if mandatory_skipped:
            first = sorted(mandatory_skipped)[0]
            runtime_capture._raise_unpersistable(
                first,
                type_by_name.get(first, "unknown"),
                "dill",
                f"could not be serialized via dill "
                f"(also missing: {sorted(mandatory_skipped)})",
            )

        # Only treat successfully saved residual names as captured
        residual_names_saved = [
            n for n in residual_names if n in set(saved or [])
        ]
        if not residual_names_saved:
            if residual_names and not mandatory_names.intersection(
                residual_names
            ):
                # all advisory — ok to have no residual pickle
                captured_residual_vars = []
                captured_residual_ref = None
            elif residual_names:
                # mandatory names remained but none saved (should have raised)
                first = sorted(
                    mandatory_names.intersection(residual_names)
                )[0]
                runtime_capture._raise_unpersistable(
                    first,
                    type_by_name.get(first, "unknown"),
                    "dill",
                    "residual pickle was not produced",
                )
        else:
            if ref is None:
                runtime_capture._raise_unpersistable(
                    residual_names_saved[0],
                    type_by_name.get(residual_names_saved[0], "unknown"),
                    "dill",
                    "residual pickle was not produced",
                )
            captured_residual_vars = list(residual_names_saved)
            captured_residual_ref = ref

    # Fail closed when a live kernel had a required value PRESENT in
    # globals, it could not be serialized, and NOTHING survived capture.
    # Advisory soft-skip exists so one unpersistable value cannot brick a
    # snapshot that still holds real runtime — it must not publish a
    # runtime marked "complete" with zero content. This is the
    # post-pruning form of the rule already enforced above (a live kernel
    # with an empty required set is fail-closed; empty is only valid when
    # kernel_present=False), and it fails closed for the same reason.
    if (
        dropped_present
        and not pruned_required
        and not captured_adata_refs
        and not captured_residual_vars
    ):
        first = sorted(dropped_present)[0]
        runtime_capture._raise_unpersistable(
            first,
            dep_manifest.required[first].type_name,
            dep_manifest.required[first].serializer,
            (
                "could not be serialized and no other required runtime "
                "value survived capture (dropped live values: "
                f"{sorted(dropped_present)})"
            ),
        )


    state.captured_residual_vars = captured_residual_vars
    state.captured_residual_ref = captured_residual_ref


def _build_runtime_manifest(
    runtime_capture,
    state: _LiveCaptureState,
) -> RuntimeManifest:
    captured_adata_refs = state.captured_adata_refs
    captured_residual_vars = state.captured_residual_vars
    captured_residual_ref = state.captured_residual_ref
    pruned_required = state.pruned_required
    incidental_skips = state.incidental_skips
    if runtime_capture.FREE_SPACE_RESERVE_BYTES > 0:
        total_size = sum(ref.size for ref in captured_adata_refs.values())
        if captured_residual_ref is not None:
            total_size += captured_residual_ref.size
        probe = (
            getattr(runtime_capture._paths, "staging", None)
            or getattr(runtime_capture._paths, "root", None)
        )
        if probe is None:
            raise SnapshotCaptureError(
                "RUNTIME_CAPTURE_DISK_FULL",
                "Cannot verify free disk space for runtime capture: "
                "no staging/root path available",
                details={"total_size": total_size},
            )
        probe_path = Path(probe)
        while not probe_path.exists() and probe_path != probe_path.parent:
            probe_path = probe_path.parent
        if not probe_path.exists():
            raise SnapshotCaptureError(
                "RUNTIME_CAPTURE_DISK_FULL",
                "Cannot verify free disk space for runtime capture: "
                f"no existing path from {probe}",
                details={"total_size": total_size, "probe": str(probe)},
            )
        try:
            usage = shutil.disk_usage(probe_path)
        except OSError as exc:
            raise SnapshotCaptureError(
                "RUNTIME_CAPTURE_DISK_FULL",
                f"Failed to check disk usage for runtime capture: {exc}",
                details={"total_size": total_size},
            ) from exc
        if usage.free < runtime_capture.FREE_SPACE_RESERVE_BYTES:
            raise SnapshotCaptureError(
                "RUNTIME_CAPTURE_DISK_FULL",
                (
                    f"Insufficient disk space for runtime capture: "
                    f"{usage.free} free after write, "
                    f"{total_size} captured, "
                    f"{runtime_capture.FREE_SPACE_RESERVE_BYTES} reserve"
                ),
                details={
                    "free": usage.free,
                    "total_size": total_size,
                    "reserve": runtime_capture.FREE_SPACE_RESERVE_BYTES,
                },
            )

    # Build complete RuntimeManifest with dependency proof.
    # Required is the registry-declared set minus soft-skipped absent
    # advisory names, so the completeness proof holds against what was
    # actually captured.
    runtime = RuntimeManifest(
        status="complete",
        dependencies=RuntimeDependencyManifest(
            required=pruned_required,
            incidental_skips=incidental_skips,
        ),
        residual_ref=captured_residual_ref,
        captured_residual_names=list(captured_residual_vars),
        adata_refs=dict(captured_adata_refs),
    )

    # Return RuntimeManifest directly.
    return runtime
