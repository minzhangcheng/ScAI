# Parsing and validation of the dill/h5ad capture proof emitted by the kernel.
# Moved verbatim out of backend/services/snapshot_v2/runtime.py.
"""Runtime capture for Snapshot V2.

Owns kernel introspection, required-set assembly via the canonical registry,
and authoritative serializer success (h5ad + dill).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from backend.services.snapshot_v2.errors import SnapshotCaptureError, SnapshotIntegrityError
from backend.services.snapshot_v2.models import RuntimeManifest


# Re-export for existing imports / tests.


# Production runtime capture has no disk quota or free-space reserve. The class
# attribute remains injectable for focused lower-level tests and callers.
RUNTIME_CAPTURE_FREE_SPACE_RESERVE_BYTES: int = 0

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

def _parse_dill_name_line(output: str, prefix: str) -> list[str] | None:
    for line in (output or "").splitlines():
        if line.startswith(prefix):
            raw = line[len(prefix):].strip()
            parsed = _parse_name_list(raw)
            if parsed is not None:
                return parsed
    return None

def _parse_dill_saved_names(output: str) -> list[str]:
    parsed = _parse_dill_name_line(output, "__DILL_SAVED_NAMES__:")
    if parsed is not None:
        return parsed
    return []

def _parse_dill_verified_names(output: str) -> list[str] | None:
    return _parse_dill_name_line(output, "__DILL_VERIFIED_NAMES__:")

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
