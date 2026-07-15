"""Shared live restore verification contract for Snapshot V2.

Used by both:
- ``SnapshotService._verify_live_restore`` (pre-commit, journal still APPLYING)
- ``_RecoveryRestorer.verify_target_complete`` (COMMITTED recovery re-verify)

Contract (aligned deliberately):
- session.json presence (NOT raw analysis_state_ref hash equality — post-activate
  session.json is continuation-mutated)
- current_round vs target_manifest.round_number when available
- scoped workspace path size + content hashes
- complete runtime with residual_ref: live ``{session_id}_session.pkl`` +
  integrity signature (+ hash match). reset/keep policy must NOT substitute
  for a missing residual on complete targets
- non-complete / unavailable runtime: reset/keep policy may authorize empty
  runtime when residual is not required
- notebook / checkpoint content hashes when refs exist
- branch metadata + continuation target when available
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path
from backend.services.snapshot_v2.workspace import WorkspaceScope

logger = logging.getLogger(__name__)


def live_session_pkl_path(states_root: Path | str, session_id: str) -> Path:
    """Production residual path written by ``atomic_replace_pkl``.

    Authoritative location is ``get_session_states(session_id)/{session_id}_session.pkl``
    — the same path ``atomic_replace_pkl`` / ``_live_pkl`` use, with HMAC under
    ``_integrity_dir(session_id)``. Fall back to ``states_root`` only when the
    production path is absent (tests that fully mock session layout).
    """
    try:
        from backend.config._config import get_session_states

        prod = get_session_states(session_id) / f"{session_id}_session.pkl"
        if prod.is_file():
            return prod
        # Prefer production path even when not yet present so callers that
        # install residual after resolve land on the same location.
        fallback = Path(states_root) / f"{session_id}_session.pkl"
        if fallback.is_file():
            return fallback
        return prod
    except Exception:
        return Path(states_root) / f"{session_id}_session.pkl"


def live_session_integrity_dir(session_id: str) -> Path:
    """Host-side integrity dir used by sandbox / runtime_apply signatures."""
    from backend.utils.sandbox import _integrity_dir

    return _integrity_dir(session_id)


def live_session_sig_path(states_root: Path | str, session_id: str) -> Path:
    from backend.utils.sandbox import _sig_path

    pkl = live_session_pkl_path(states_root, session_id)
    return _sig_path(pkl, live_session_integrity_dir(session_id))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_policy_from_context(
    *,
    runtime_choice: str | None,
    journal: dict | None,
) -> str | None:
    """Return reset/keep policy if declared on request or applied ledger."""
    choice = (runtime_choice or "").strip() or None
    if choice in ("reset", "keep"):
        return choice
    if not journal:
        return None
    ledger = journal.get("applied_step_ledger") or []
    steps = {s.get("step") for s in ledger if s.get("completed")}
    if "runtime_reset" in steps:
        return "reset"
    if "runtime_keep" in steps:
        return "keep"
    # Journal may also store the original choice.
    j_choice = (journal.get("runtime_choice") or "").strip() or None
    if j_choice in ("reset", "keep"):
        return j_choice
    return None


def _resolve_current_round(
    *,
    restored_state: Any | None,
    session_json_path: Path,
) -> int | None:
    if restored_state is not None:
        try:
            return int(getattr(restored_state, "current_round", None))
        except (TypeError, ValueError):
            pass
    if session_json_path.is_file():
        try:
            data = json.loads(session_json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("current_round") is not None:
                return int(data["current_round"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None
    return None


def verify_live_restore_components(
    *,
    session_id: str,
    target_manifest: Any,
    session_dir: Path | str,
    workspace_root: Path | str,
    states_root: Path | str,
    runtime_choice: str | None = None,
    restored_state: Any | None = None,
    staged_checkpoint: Any | None = None,
    journal: dict | None = None,
    require_analysis_state: bool = False,
) -> dict[str, Any]:
    """Verify live restore side effects against the target snapshot contract.

    Returns a verification proof dict::
        {
            "verified_live_root_hash": str,
            "component_hashes": dict[str, str],
            "verified_at": str,
            "restore_journal_version": 1,
        }

    Raises ``ValueError`` with a multi-line message of failures (callers map
    to LIVE_VERIFY_FAILED / TARGET_VERIFY_FAILED).
    """
    session_dir_p = Path(session_dir)
    workspace_root_p = Path(workspace_root)
    states_root_p = Path(states_root)
    scope = WorkspaceScope()
    failures: list[str] = []
    component_hashes: dict[str, str] = {}

    # --- session.json: presence only (post-activate state is mutated) ---
    session_json = session_dir_p / "session.json"
    if not session_json.is_file():
        failures.append("session.json missing after apply")
    else:
        try:
            component_hashes["session_json"] = _sha256_file(session_json)
        except OSError as exc:
            failures.append(f"session.json unreadable: {exc}")

    # --- current_round ---
    if getattr(target_manifest, "round_number", None) is not None:
        live_round = _resolve_current_round(
            restored_state=restored_state,
            session_json_path=session_json,
        )
        expected_round = int(target_manifest.round_number)
        if live_round is None:
            failures.append(
                f"current_round unavailable; expected={expected_round}"
            )
        elif live_round != expected_round:
            failures.append(
                f"current_round mismatch: live={live_round} "
                f"expected={expected_round}"
            )
        else:
            component_hashes["current_round"] = str(live_round)

    # Optional: require analysis_state_ref object still exists in store
    # (does NOT demand live session.json hash equality).
    if require_analysis_state and getattr(target_manifest, "analysis_state_ref", None) is not None:
        ref = target_manifest.analysis_state_ref
        # Presence of ref is enough; hash equality is intentionally not checked.
        if not getattr(ref, "sha256", None):
            failures.append("analysis_state_ref missing sha256")

    # --- scoped workspace ---
    workspace_hashes: dict[str, str] = {}
    for we in target_manifest.workspace or []:
        if not scope.is_in_scope(we.path):
            continue
        live_path = workspace_root_p / we.path
        if not live_path.is_file():
            failures.append(f"workspace missing: {we.path}")
            continue
        try:
            if live_path.stat().st_size != we.size:
                failures.append(f"workspace size mismatch: {we.path}")
                continue
            live_hash = _sha256_file(live_path)
        except OSError as exc:
            failures.append(f"workspace unreadable: {we.path}: {exc}")
            continue
        if live_hash != we.object_hash:
            failures.append(
                f"workspace hash mismatch: {we.path} "
                f"(live={live_hash} expected={we.object_hash})"
            )
        else:
            workspace_hashes[we.path] = live_hash
    if workspace_hashes:
        component_hashes["workspace"] = hashlib.sha256(
            json.dumps(workspace_hashes, sort_keys=True).encode()
        ).hexdigest()

    # --- runtime pkl+sig or explicit reset/keep policy ---
    runtime = getattr(target_manifest, "runtime", None)
    runtime_status = getattr(runtime, "status", None) if runtime is not None else None
    residual_ref = getattr(runtime, "residual_ref", None) if runtime is not None else None
    policy = _runtime_policy_from_context(
        runtime_choice=runtime_choice, journal=journal,
    )

    if runtime_status == "complete" and residual_ref is not None:
        # Complete residual targets always require the production residual pair.
        # reset/keep may not substitute for a missing residual on complete targets.
        pkl_path = live_session_pkl_path(states_root_p, session_id)
        expected = getattr(residual_ref, "sha256", None)
        if not pkl_path.is_file():
            failures.append(
                "runtime residual missing for complete target "
                f"(expected {session_id}_session.pkl + integrity signature; "
                "reset/keep policy cannot substitute)"
            )
        else:
            try:
                from backend.utils.sandbox import _verify_pkl_integrity

                idir = live_session_integrity_dir(session_id)
                if not _verify_pkl_integrity(pkl_path, idir):
                    failures.append(
                        "runtime pkl signature missing or invalid "
                        f"for {pkl_path.name}"
                    )
                else:
                    pkl_hash = _sha256_file(pkl_path)
                    component_hashes["runtime_pkl"] = pkl_hash
                    if expected and pkl_hash != expected:
                        failures.append(
                            f"runtime pkl hash mismatch: live={pkl_hash} "
                            f"expected={expected}"
                        )
                    # Record sig presence as part of the pair proof.
                    sig_path = live_session_sig_path(states_root_p, session_id)
                    if sig_path.is_file():
                        component_hashes["runtime_sig"] = _sha256_file(sig_path)
            except Exception as exc:
                failures.append(f"runtime residual verify failed: {exc}")
    elif policy in ("reset", "keep"):
        # Policy may authorize empty/unavailable runtime only when the target
        # is NOT a complete residual (unavailable / no residual_ref / non-complete).
        component_hashes["runtime_policy"] = str(policy)

    # --- live h5ad sidecars + manifest (complete runtime with adata_refs) ---
    if runtime_status == "complete" and runtime is not None:
        adata_refs = getattr(runtime, "adata_refs", None) or {}
        if adata_refs:
            from backend.services.snapshot_v2.errors import SnapshotIntegrityError
            from backend.services.snapshot_v2.runtime_names import runtime_h5ad_path
            from backend.utils.runtime_apply import runtime_h5ad_manifest_path

            try:
                from backend.config._config import get_session_states

                prod_states = get_session_states(session_id)
            except Exception:
                prod_states = states_root_p

            try:
                manifest_path = runtime_h5ad_manifest_path(session_id)
                if not manifest_path.exists():
                    alt = states_root_p / "runtime_h5ad_manifest.json"
                    if alt.exists():
                        manifest_path = alt
                    else:
                        raise SnapshotIntegrityError(
                            "LIVE_RUNTIME_H5AD_MANIFEST_MISSING",
                            "live runtime h5ad manifest missing after restore",
                        )
                payload = json.loads(manifest_path.read_text())
                entries = payload.get("entries") or {}
                for var_name, ref in adata_refs.items():
                    live_h5ad = runtime_h5ad_path(prod_states, var_name)
                    if not live_h5ad.exists():
                        live_h5ad = runtime_h5ad_path(states_root_p, var_name)
                    if not live_h5ad.exists():
                        raise SnapshotIntegrityError(
                            "LIVE_RUNTIME_H5AD_MISSING",
                            f"LIVE_RUNTIME_H5AD_MISSING:{var_name}",
                        )
                    actual = hashlib.sha256(live_h5ad.read_bytes()).hexdigest()
                    if actual != ref.sha256:
                        raise SnapshotIntegrityError(
                            "LIVE_RUNTIME_H5AD_HASH_MISMATCH",
                            f"LIVE_RUNTIME_H5AD_HASH_MISMATCH:{var_name}",
                        )
                    if entries.get(var_name, {}).get("sha256") != ref.sha256:
                        raise SnapshotIntegrityError(
                            "LIVE_RUNTIME_H5AD_MANIFEST_MISMATCH",
                            f"LIVE_RUNTIME_H5AD_MANIFEST_MISMATCH:{var_name}",
                        )
                    component_hashes[f"runtime_h5ad:{var_name}"] = actual
            except SnapshotIntegrityError as exc:
                failures.append(str(exc))

    # --- notebook ---
    notebook_ref = getattr(target_manifest, "notebook_ref", None)
    if notebook_ref is not None:
        nb_path = states_root_p / "notebook.ipynb"
        if not nb_path.is_file():
            failures.append("notebook missing after apply")
        else:
            try:
                nb_hash = _sha256_file(nb_path)
                component_hashes["notebook"] = nb_hash
                expected = getattr(notebook_ref, "sha256", None)
                if expected and nb_hash != expected:
                    failures.append(
                        f"notebook hash mismatch: live={nb_hash} "
                        f"expected={expected}"
                    )
            except OSError as exc:
                failures.append(f"notebook unreadable: {exc}")

    # --- checkpoint ---
    checkpoint_ref = getattr(target_manifest, "checkpoint_ref", None)
    if checkpoint_ref is not None:
        ckpt_path = checkpoint_sqlite_path(states_root_p)
        if not ckpt_path.is_file():
            failures.append("checkpoint missing after apply")
        else:
            try:
                ckpt_hash = _sha256_file(ckpt_path)
                component_hashes["checkpoint"] = ckpt_hash
                expected = getattr(checkpoint_ref, "sha256", None)
                if expected and ckpt_hash != expected:
                    failures.append(
                        f"checkpoint hash mismatch: live={ckpt_hash} "
                        f"expected={expected}"
                    )
            except OSError as exc:
                failures.append(f"checkpoint unreadable: {exc}")

    # --- continuation ---
    if staged_checkpoint is not None and restored_state is not None:
        kind = getattr(staged_checkpoint, "kind", None)
        if kind == "graph_node":
            if restored_state.restore_resume_node != staged_checkpoint.next_node:
                failures.append("continuation next_node mismatch")
            component_hashes["continuation"] = (
                f"graph_node:{staged_checkpoint.next_node}"
            )
        elif kind == "semantic_gate":
            if restored_state.restore_resume_gate != staged_checkpoint.gate:
                failures.append("continuation gate mismatch")
            component_hashes["continuation"] = (
                f"semantic_gate:{staged_checkpoint.gate}"
            )
        elif kind in ("idle", "terminal"):
            component_hashes["continuation"] = str(kind)
    elif journal is not None:
        # Recovery path: record journal continuation kind; when v1 proof is
        # present, prefer stored value for stable re-verify.
        stored = (journal.get("component_hashes") or {}).get("continuation")
        if stored:
            component_hashes["continuation"] = str(stored)
        else:
            component_hashes["continuation"] = str(
                journal.get("continuation_kind") or "idle"
            )

    # --- branch metadata ---
    branch_meta = {
        "snapshot_id": getattr(target_manifest, "snapshot_id", "")
        or (journal or {}).get("target_snapshot_id")
        or "",
        "root_hash": getattr(target_manifest, "root_hash", "") or "",
    }
    if journal is not None and journal.get("branch_id"):
        branch_meta["branch_id"] = journal.get("branch_id") or "main"
    component_hashes["branch"] = hashlib.sha256(
        json.dumps(branch_meta, sort_keys=True).encode()
    ).hexdigest()

    if failures:
        raise ValueError("; ".join(failures))

    verified_at = _now_iso()
    root_material = json.dumps(component_hashes, sort_keys=True).encode()
    return {
        "verified_live_root_hash": hashlib.sha256(root_material).hexdigest(),
        "component_hashes": component_hashes,
        "verified_at": verified_at,
        "restore_journal_version": 1,
    }
