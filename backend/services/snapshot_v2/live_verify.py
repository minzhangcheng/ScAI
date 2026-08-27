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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.services.snapshot_v2.checkpoint_paths import checkpoint_sqlite_path
from backend.services.snapshot_v2.workspace import WorkspaceScope

logger = logging.getLogger(__name__)

from . import live_verify_steps
from .live_verify_steps import _LiveVerifyContext


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
    objects_root: Path | str | None = None,
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
    context = _LiveVerifyContext(
        session_id=session_id,
        target_manifest=target_manifest,
        session_dir=Path(session_dir),
        workspace_root=Path(workspace_root),
        states_root=Path(states_root),
        runtime_choice=runtime_choice,
        restored_state=restored_state,
        staged_checkpoint=staged_checkpoint,
        journal=journal,
        require_analysis_state=require_analysis_state,
        objects_root=Path(objects_root) if objects_root is not None else None,
    )
    live_verify_steps.verify_state_round_workspace(context, bindings=sys.modules[__name__])
    live_verify_steps.verify_runtime_components(context, bindings=sys.modules[__name__])
    live_verify_steps.verify_notebook_checkpoint(context, bindings=sys.modules[__name__])
    live_verify_steps.verify_continuation_branch(context, bindings=sys.modules[__name__])

    if context.failures:
        raise ValueError("; ".join(context.failures))

    verified_at = _now_iso()
    root_material = json.dumps(context.component_hashes, sort_keys=True).encode()
    return {
        "verified_live_root_hash": hashlib.sha256(root_material).hexdigest(),
        "component_hashes": context.component_hashes,
        "verified_at": verified_at,
        "restore_journal_version": 1,
    }
