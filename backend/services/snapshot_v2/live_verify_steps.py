"""Component-specific live restore verification steps."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class _LiveVerifyContext:
    session_id: str
    target_manifest: Any
    session_dir: Path
    workspace_root: Path
    states_root: Path
    runtime_choice: str | None
    restored_state: Any | None
    staged_checkpoint: Any | None
    journal: dict | None
    require_analysis_state: bool
    objects_root: Path | None
    failures: list[str] = field(default_factory=list)
    component_hashes: dict[str, str] = field(default_factory=dict)

def verify_state_round_workspace(context: _LiveVerifyContext, *, bindings) -> None:
    session_id = context.session_id
    target_manifest = context.target_manifest
    session_dir_p = context.session_dir
    workspace_root_p = context.workspace_root
    states_root_p = context.states_root
    runtime_choice = context.runtime_choice
    restored_state = context.restored_state
    staged_checkpoint = context.staged_checkpoint
    journal = context.journal
    require_analysis_state = context.require_analysis_state
    objects_root = context.objects_root
    failures = context.failures
    component_hashes = context.component_hashes
    Path = bindings.Path
    hashlib = bindings.hashlib
    json = bindings.json
    checkpoint_sqlite_path = bindings.checkpoint_sqlite_path
    WorkspaceScope = bindings.WorkspaceScope
    _sha256_file = bindings._sha256_file
    _resolve_current_round = bindings._resolve_current_round
    _runtime_policy_from_context = bindings._runtime_policy_from_context
    live_session_pkl_path = bindings.live_session_pkl_path
    live_session_integrity_dir = bindings.live_session_integrity_dir
    live_session_sig_path = bindings.live_session_sig_path
    scope = WorkspaceScope()
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

    # Optional: require the analysis_state_ref OBJECT to still exist in the
    # store (does NOT demand live session.json hash equality).
    #
    # This used to assert only that ref.sha256 was truthy — but sha256 is a
    # non-optional, pattern-validated field on ObjectRef, so it is always a
    # 64-hex string for any manifest the store could load. The check could not
    # fail, and nothing here ever looked at the store. Recovery would then
    # declare a COMMITTED restore verified and delete its journal while the
    # snapshot it "verified" could no longer be staged at all (stage() raises
    # OBJECT_NOT_FOUND). Verify the object is really there.
    if require_analysis_state and getattr(target_manifest, "analysis_state_ref", None) is not None:
        ref = target_manifest.analysis_state_ref
        sha = getattr(ref, "sha256", None)
        if not sha:
            failures.append("analysis_state_ref missing sha256")
        elif objects_root is not None:
            # Existence only: re-hashing a large state object on every recovery
            # pass would be a needless cost, and content integrity is already
            # the object store's own invariant.
            if not (Path(str(objects_root)) / str(sha)).is_file():
                failures.append(
                    f"analysis_state object missing from store: {sha}"
                )

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



def verify_runtime_components(context: _LiveVerifyContext, *, bindings) -> None:
    session_id = context.session_id
    target_manifest = context.target_manifest
    session_dir_p = context.session_dir
    workspace_root_p = context.workspace_root
    states_root_p = context.states_root
    runtime_choice = context.runtime_choice
    restored_state = context.restored_state
    staged_checkpoint = context.staged_checkpoint
    journal = context.journal
    require_analysis_state = context.require_analysis_state
    objects_root = context.objects_root
    failures = context.failures
    component_hashes = context.component_hashes
    Path = bindings.Path
    hashlib = bindings.hashlib
    json = bindings.json
    checkpoint_sqlite_path = bindings.checkpoint_sqlite_path
    WorkspaceScope = bindings.WorkspaceScope
    _sha256_file = bindings._sha256_file
    _resolve_current_round = bindings._resolve_current_round
    _runtime_policy_from_context = bindings._runtime_policy_from_context
    live_session_pkl_path = bindings.live_session_pkl_path
    live_session_integrity_dir = bindings.live_session_integrity_dir
    live_session_sig_path = bindings.live_session_sig_path
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
    elif runtime_status == "complete":
        # Empty-complete (or h5ad-only) targets require the old residual pair
        # to be absent; reset/keep cannot substitute for exact application.
        pkl_path = live_session_pkl_path(states_root_p, session_id)
        sig_path = live_session_sig_path(states_root_p, session_id)
        if pkl_path.exists():
            failures.append(
                f"unexpected runtime residual for empty complete target: {pkl_path.name}"
            )
        if sig_path.exists():
            failures.append(
                f"unexpected runtime signature for empty complete target: {sig_path.name}"
            )
        component_hashes["runtime_empty_residual"] = "verified"
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



def verify_notebook_checkpoint(context: _LiveVerifyContext, *, bindings) -> None:
    session_id = context.session_id
    target_manifest = context.target_manifest
    session_dir_p = context.session_dir
    workspace_root_p = context.workspace_root
    states_root_p = context.states_root
    runtime_choice = context.runtime_choice
    restored_state = context.restored_state
    staged_checkpoint = context.staged_checkpoint
    journal = context.journal
    require_analysis_state = context.require_analysis_state
    objects_root = context.objects_root
    failures = context.failures
    component_hashes = context.component_hashes
    Path = bindings.Path
    hashlib = bindings.hashlib
    json = bindings.json
    checkpoint_sqlite_path = bindings.checkpoint_sqlite_path
    WorkspaceScope = bindings.WorkspaceScope
    _sha256_file = bindings._sha256_file
    _resolve_current_round = bindings._resolve_current_round
    _runtime_policy_from_context = bindings._runtime_policy_from_context
    live_session_pkl_path = bindings.live_session_pkl_path
    live_session_integrity_dir = bindings.live_session_integrity_dir
    live_session_sig_path = bindings.live_session_sig_path
    # --- notebook ---
    notebook_ref = getattr(target_manifest, "notebook_ref", None)
    if notebook_ref is not None:
        from backend.utils.live_paths import live_notebook_path

        nb_path = live_notebook_path(states_root_p, session_id)
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



def verify_continuation_branch(context: _LiveVerifyContext, *, bindings) -> None:
    session_id = context.session_id
    target_manifest = context.target_manifest
    session_dir_p = context.session_dir
    workspace_root_p = context.workspace_root
    states_root_p = context.states_root
    runtime_choice = context.runtime_choice
    restored_state = context.restored_state
    staged_checkpoint = context.staged_checkpoint
    journal = context.journal
    require_analysis_state = context.require_analysis_state
    objects_root = context.objects_root
    failures = context.failures
    component_hashes = context.component_hashes
    Path = bindings.Path
    hashlib = bindings.hashlib
    json = bindings.json
    checkpoint_sqlite_path = bindings.checkpoint_sqlite_path
    WorkspaceScope = bindings.WorkspaceScope
    _sha256_file = bindings._sha256_file
    _resolve_current_round = bindings._resolve_current_round
    _runtime_policy_from_context = bindings._runtime_policy_from_context
    live_session_pkl_path = bindings.live_session_pkl_path
    live_session_integrity_dir = bindings.live_session_integrity_dir
    live_session_sig_path = bindings.live_session_sig_path
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
