"""Runtime capture for Snapshot V2.

Owns kernel introspection, required-set assembly via the canonical registry,
and authoritative serializer success (h5ad + dill).
"""

from __future__ import annotations

import ast
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from backend.services.snapshot_v2.errors import SnapshotCaptureError, SnapshotIntegrityError
from backend.services.snapshot_v2.models import (
    ObjectRef,
    RuntimeDependency,
    RuntimeDependencyManifest,
    RuntimeGenerationManifest,
    RuntimeManifest,
    RuntimeSkip,
    empty_complete_runtime,
)
from backend.services.snapshot_v2.object_store import ObjectStore
from backend.services.snapshot_v2.paths import SnapshotV2Paths
from backend.services.snapshot_v2.runtime_registry import (
    CANONICAL_RUNTIME_ARTIFACTS,
    build_required_runtime_manifest,
    build_required_runtime_set,
)


# Re-export for existing imports / tests.
__all__ = [
    "CANONICAL_RUNTIME_ARTIFACTS",
    "RuntimeCapture",
    "RUNTIME_CAPTURE_QUOTA_BYTES",
    "build_required_runtime_manifest",
    "build_required_runtime_set",
    "validate_runtime_dependency_proof",
    "_parse_dill_skipped",
]


_INTROSPECT_CODE = """
import anndata as _runtime_ad
import json as _runtime_json
_globals = {}
for _name, _obj in list(globals().items()):
    if _name.startswith('_'):
        continue
    if isinstance(_obj, _runtime_ad.AnnData):
        _shape = str(getattr(_obj, 'shape', '?'))
        _globals[_name] = 'AnnData(' + _shape + ')'
    else:
        _globals[_name] = type(_obj).__module__ + '.' + type(_obj).__qualname__
print('__RUNTIME_GLOBALS_BEGIN__')
print(_runtime_json.dumps(_globals))
print('__RUNTIME_GLOBALS_END__')
"""

_SAVE_ANNDATA_TEMPLATE = """
import anndata as _runtime_ad
_adata = globals().get({var_name_repr!r})
if _adata is not None:
    _adata.write({h5ad_path_repr!r})
    import os as _runtime_os
    _size = _runtime_os.path.getsize({h5ad_path_repr!r})
    print('__H5AD_SAVED__:' + str(_size))
else:
    print('__H5AD_NOT_FOUND__:' + {var_name_repr!r})
"""

_SAVE_DILL_TEMPLATE = """
import dill as _runtime_dill
import json as _runtime_json
_names = {names_repr!r}
import os as _runtime_os
_saved_count = 0
_skipped = []
_saved = []
with open({dill_path_repr!r}, 'wb') as _f:
    for _n in _names:
        _obj = globals().get(_n)
        if _obj is None and _n not in globals():
            _skipped.append(_n)
            continue
        try:
            _runtime_dill.dump((_n, _obj), _f)
            _saved_count += 1
            _saved.append(_n)
        except Exception as _e:
            _skipped.append(_n)
_size = _runtime_os.path.getsize({dill_path_repr!r})
print('__DILL_SAVED__:' + str(_saved_count) + '/' + str(len(_names)) + ' size=' + str(_size))
print('__DILL_SAVED_NAMES__:' + _runtime_json.dumps(_saved))
if _skipped:
    print('__DILL_SKIPPED__:' + _runtime_json.dumps(_skipped))
"""


# Default quota: 2 GB total captured runtime size
RUNTIME_CAPTURE_QUOTA_BYTES: int = 2 * 1024 * 1024 * 1024


def _parse_dill_skipped(output: str) -> list[str]:
    """Structurally parse ``__DILL_SKIPPED__`` lines (JSON or Python literal)."""
    skipped: list[str] = []
    for line in (output or "").splitlines():
        if not line.startswith("__DILL_SKIPPED__:"):
            continue
        raw = line[len("__DILL_SKIPPED__:"):].strip()
        parsed = _parse_name_list(raw)
        if parsed is not None:
            skipped = parsed
    return skipped


def _parse_dill_saved_names(output: str) -> list[str]:
    for line in (output or "").splitlines():
        if line.startswith("__DILL_SAVED_NAMES__:"):
            raw = line[len("__DILL_SAVED_NAMES__:"):].strip()
            parsed = _parse_name_list(raw)
            if parsed is not None:
                return parsed
    return []


def _parse_name_list(raw: str) -> list[str] | None:
    """Parse a list of names from JSON or a Python literal (ast.literal_eval)."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def validate_runtime_dependency_proof(runtime: RuntimeManifest) -> RuntimeManifest:
    """Re-validate a staged runtime's completeness dependency proof.

    Ensures required names are covered by adata_refs / captured residual names,
    residual_ref is present when residual names are declared, and every
    ObjectRef in the runtime is self-consistent with status.
    """
    try:
        validated = RuntimeManifest.model_validate(runtime.model_dump(mode="json"))
    except Exception as exc:
        raise SnapshotIntegrityError(
            "RUNTIME_PROOF_INVALID",
            f"Runtime dependency proof failed validation: {exc}",
            details={"error": str(exc)},
        ) from exc

    if validated.status == "complete":
        captured = set(validated.adata_refs) | set(validated.captured_residual_names)
        missing = set(validated.dependencies.required) - captured
        if missing:
            raise SnapshotIntegrityError(
                "RUNTIME_PROOF_INCOMPLETE",
                f"complete runtime missing required values: {sorted(missing)}",
                details={"missing": sorted(missing)},
            )
        if validated.captured_residual_names and validated.residual_ref is None:
            raise SnapshotIntegrityError(
                "RUNTIME_PROOF_INCOMPLETE",
                "captured residual names require residual_ref",
            )
    return validated


def _validate_h5ad_file(path: Path) -> None:
    """Ensure *path* is a readable, structurally valid h5ad before put_file.

    Uses ``backed="r"`` mode for memory-efficient structural validation.
    Raises SnapshotCaptureError with code ``required_runtime_unpersistable`` on failure.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise SnapshotCaptureError(
            "required_runtime_unpersistable",
            f"h5ad file missing or empty: {path.name}",
            details={"path": str(path), "serializer": "h5ad"},
        )
    try:
        import anndata as ad

        # backed="r" validates structure without loading data into memory
        _ = ad.read_h5ad(path, backed="r")
    except SnapshotCaptureError:
        raise
    except Exception as e:
        raise SnapshotCaptureError(
            "required_runtime_unpersistable",
            f"h5ad validation failed for {path.name}: {e}",
            details={"path": str(path), "serializer": "h5ad", "error": str(e)},
        ) from e


class RuntimeCapture:
    QUOTA_BYTES: int = RUNTIME_CAPTURE_QUOTA_BYTES

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
        """Execute trusted capture code; use unlocked path when lock already held."""
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



    async def _introspect_globals(self, sandbox, *, exec_lock_held: bool = False) -> dict[str, str]:
        result = await self._run_kernel(
            sandbox, _INTROSPECT_CODE, exec_lock_held=exec_lock_held
        )
        if not result.success:
            raise SnapshotCaptureError(
                "RUNTIME_INTROSPECT_FAILED",
                f"Kernel globals introspection failed: {result.error}",
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
        return {str(k): str(v) for k, v in parsed.items()}

    async def _capture_adata(
        self, sandbox, var_name: str, type_str: str, capture_dir: Path,
        *, exec_lock_held: bool = False,
    ) -> ObjectRef:
        h5ad_path = capture_dir / f"{var_name}.h5ad"
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
                    f"serializer=h5ad) write failed: {result.error}"
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
            raise SnapshotCaptureError(
                "required_runtime_unpersistable",
                (
                    f"AnnData variable '{var_name}' (type={type_str or 'unknown'}, "
                    f"serializer=h5ad) object store put failed: {e}"
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
        if not result.success:
            # Treat whole batch as unpersistable for required names.
            return None, saved, list(var_names)
        if not dill_path.exists():
            return None, saved, list(set(var_names) - set(saved)) or list(var_names)
        if not saved and not skipped:
            # Older kernel output without structural saved list: infer from skip.
            skipped = skipped or []
            saved = [n for n in var_names if n not in skipped]
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

        globals_meta = await self._introspect_globals(
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

        capture_dir = self._paths.staging / f"rtcap_{uuid.uuid4().hex[:12]}"
        capture_dir.mkdir(parents=True, exist_ok=True)

        captured_adata_vars: list[str] = []
        captured_adata_refs: dict[str, ObjectRef] = {}
        captured_residual_vars: list[str] = []
        captured_residual_ref: ObjectRef | None = None
        incidental_skips: list[RuntimeSkip] = []
        manifest_warnings: list[str] = []

        # Names that are continuation-critical (checkpoint dependencies) must
        # fail closed when absent from globals. Advisory declared names (e.g. an
        # LLM-declared persistent output that was later `del`-eted) cannot be
        # captured if they are gone, and failing closed on them would permanently
        # brick every snapshot. Those are soft-skipped instead.
        mandatory_names = set(checkpoint_dependencies or set())

        # Pruned required set drives the completeness proof; soft-skipped absent
        # advisory names are removed from it and recorded as incidental skips.
        pruned_required = dict(dep_manifest.required)

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

        try:
            # Partition required names into h5ad vs dill based on dependency serializer.
            h5ad_names: list[str] = []
            dill_names: list[str] = []
            for name, dep in dep_manifest.required.items():
                if dep.serializer == "h5ad":
                    h5ad_names.append(name)
                else:
                    dill_names.append(name)

            for var_name in sorted(h5ad_names):
                type_str = globals_meta.get(var_name, dep_manifest.required[var_name].type_name)
                if var_name not in globals_meta:
                    if var_name in mandatory_names:
                        self._raise_unpersistable(
                            var_name, type_str, "h5ad", "not found in kernel globals"
                        )
                    _soft_skip_absent(var_name, "h5ad")
                    continue
                try:
                    ref = await self._capture_adata(
                        sandbox, var_name, type_str, capture_dir,
                        exec_lock_held=exec_lock_held,
                    )
                except SnapshotCaptureError:
                    raise
                except Exception as e:
                    self._raise_unpersistable(
                        var_name, type_str, "h5ad", f"capture failed: {e}"
                    )
                captured_adata_vars.append(var_name)
                captured_adata_refs[var_name] = ref

            # Drop absent advisory dill names (soft-skip); keep absent mandatory
            # names so the fail-closed check below still raises for them.
            present_dill_names: list[str] = []
            for var_name in sorted(dill_names):
                if var_name not in globals_meta and var_name not in mandatory_names:
                    _soft_skip_absent(var_name, "dill")
                    continue
                present_dill_names.append(var_name)

            residual_names = sorted(present_dill_names)
            if residual_names:
                for var_name in residual_names:
                    if var_name not in globals_meta:
                        type_str = dep_manifest.required[var_name].type_name
                        self._raise_unpersistable(
                            var_name, type_str, "dill", "not found in kernel globals"
                        )
                type_by_name = {
                    n: globals_meta.get(n, dep_manifest.required[n].type_name)
                    for n in residual_names
                }
                try:
                    ref, saved, dill_skipped = await self._capture_residual_dill(
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
                if required_skipped:
                    # Raise with name/type/serializer for the first missing value;
                    # include full set in details.
                    first = sorted(required_skipped)[0]
                    self._raise_unpersistable(
                        first,
                        type_by_name.get(first, "unknown"),
                        "dill",
                        f"could not be serialized via dill "
                        f"(also missing: {sorted(required_skipped)})",
                    )

                if ref is None:
                    self._raise_unpersistable(
                        residual_names[0],
                        type_by_name.get(residual_names[0], "unknown"),
                        "dill",
                        "residual pickle was not produced",
                    )

                captured_residual_vars = list(residual_names)
                captured_residual_ref = ref

            # Quota check counts physical captured object sizes.
            total_size = sum(ref.size for ref in captured_adata_refs.values())
            if captured_residual_ref is not None:
                total_size += captured_residual_ref.size
            if total_size > self.QUOTA_BYTES:
                raise SnapshotCaptureError(
                    "RUNTIME_CAPTURE_QUOTA_EXCEEDED",
                    (
                        f"Total captured runtime size ({total_size} bytes) exceeds "
                        f"quota ({self.QUOTA_BYTES} bytes)"
                    ),
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

            # Also attach a legacy generation record on a private attribute for
            # transitional callers that still read RuntimeGenerationManifest fields.
            generation = RuntimeGenerationManifest(
                capture_command="runtime_capture_v2",
                capture_timestamp=datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                kernel_id=getattr(sandbox, "session_id", "") or "",
                captured_adata_vars=captured_adata_vars,
                captured_residual_vars=captured_residual_vars,
                captured_adata_refs=captured_adata_refs,
                captured_residual_ref=captured_residual_ref,
                warnings=manifest_warnings,
            )
            # Return RuntimeManifest directly.
            return runtime
        finally:
            try:
                if capture_dir.exists():
                    for p in capture_dir.iterdir():
                        p.unlink(missing_ok=True)
                    capture_dir.rmdir()
            except OSError:
                pass
