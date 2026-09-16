"""Disposable runtime cloned from a verified committed analysis generation."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any


class FigureFollowupRuntime:
    """Start an isolated kernel from the main session's committed generation.

    The temporary session owns its workspace, state files, checkpoint directory,
    integrity namespace, kernel temp root, and namespace. No code path saves its
    kernel back to the owner session.
    """

    def __init__(self, main_sandbox: Any, state: Any) -> None:
        self.main_sandbox = main_sandbox
        self.state = state
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self.session_id = f"figure_render_{uuid.uuid4().hex}"
        self.sandbox = None
        self._integrity_root: Path | None = None

    async def __aenter__(self):
        try:
            self.sandbox = await self._start()
            return self.sandbox
        except BaseException:
            await self._cleanup()
            raise

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._cleanup()

    async def _start(self):
        from backend.utils.runtime_manifest import (
            read_runtime_manifest,
            runtime_manifest_path,
            runtime_manifest_sig_path,
        )
        from backend.utils.sandbox import LocalJupyterSandbox, _integrity_dir
        from backend.utils.sandbox_parts.regular_save import (
            _sha256_file,
            _verify_pkl_integrity,
        )
        from backend.utils.sandbox_parts.runtime_restore import plan_restore

        source_notebook = getattr(self.main_sandbox, "notebook_state", None)
        if source_notebook is None:
            raise RuntimeError("Main session has no committed runtime source")
        exec_lock = getattr(self.main_sandbox, "_exec_lock", None)
        save_locked = getattr(self.main_sandbox, "_save_session_locked", None)
        sync_persistence = getattr(
            self.main_sandbox, "sync_runtime_persistence_state", None
        )
        if exec_lock is None or not callable(save_locked) or not callable(sync_persistence):
            raise RuntimeError(
                "Main session cannot pin a committed generation under its execution lock"
            )

        source_pkl = Path(source_notebook.session_path)
        source_states = Path(source_notebook.state_dir)
        source_manifest_path = runtime_manifest_path(
            str(self.state.id), states_dir=source_states
        )
        self._tempdir = tempfile.TemporaryDirectory(prefix="scai-figure-render-")
        root = Path(self._tempdir.name)
        states_dir = root / "states"
        workspace = root / "workspace"
        pinned_dir = root / "pinned_generation"
        pinned_integrity_dir = pinned_dir / "integrity"
        states_dir.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(parents=True, exist_ok=True)
        pinned_dir.mkdir(parents=True, exist_ok=True)
        render_pkl = states_dir / f"{self.session_id}_session.pkl"
        render_manifest = runtime_manifest_path(
            self.session_id, states_dir=states_dir
        )
        self._integrity_root = Path(_integrity_dir(self.session_id))
        source_pkl_signature = _integrity_dir(str(self.state.id)) / (
            source_pkl.name + ".sig"
        )

        async with exec_lock:
            if await save_locked() is not True:
                raise RuntimeError(
                    "Main session runtime save did not commit a current generation"
                )
            persistence = sync_persistence(self.state)
            if not isinstance(persistence, dict):
                raise RuntimeError(
                    "Main session runtime persistence metadata could not be synchronized"
                )
            persistence = dict(persistence)
            if not source_notebook.has_session_dump():
                raise RuntimeError(
                    "Main session has no integrity-verified committed runtime dump"
                )

            pinned_pkl = pinned_dir / source_pkl.name
            pinned_manifest = pinned_dir / source_manifest_path.name
            pinned_manifest_signature = runtime_manifest_sig_path(pinned_manifest)
            pinned_pkl_signature = pinned_integrity_dir / (
                source_pkl.name + ".sig"
            )
            generation_files = (
                (source_pkl, pinned_pkl),
                (source_manifest_path, pinned_manifest),
                (
                    runtime_manifest_sig_path(source_manifest_path),
                    pinned_manifest_signature,
                ),
                (source_pkl_signature, pinned_pkl_signature),
            )
            for source, target in generation_files:
                if not source.is_file():
                    raise RuntimeError(
                        f"Committed runtime generation is missing {source.name}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)

            if not _verify_pkl_integrity(pinned_pkl, pinned_integrity_dir):
                raise RuntimeError("Committed runtime dump signature does not verify")
            manifest = read_runtime_manifest(pinned_manifest)
            if _sha256_file(pinned_pkl) != manifest.get("pkl_sha256"):
                raise RuntimeError(
                    "Committed runtime dump does not match its signed manifest"
                )
            plan_restore(pinned_manifest, pinned_pkl)
            committed_digest = str(persistence.get("committed_digest") or "")
            if committed_digest and committed_digest != manifest.get("pkl_sha256"):
                raise RuntimeError(
                    "Committed runtime metadata disagrees with the signed manifest"
                )
            committed_revision = int(
                persistence.get("committed_revision")
                or persistence.get("saved_revision")
                or manifest.get("revision")
                or 0
            )
            if int(manifest.get("revision") or 0) != committed_revision:
                raise RuntimeError(
                    "Committed runtime revision disagrees with the signed manifest"
                )

            # The main execution lock remains held until all four exact files
            # have been copied and checked in the disposable session. A save
            # cannot publish a newer pkl/manifest pair between these operations.
            self._integrity_root.mkdir(parents=True, exist_ok=True)
            render_pkl_signature = self._integrity_root / (render_pkl.name + ".sig")
            shutil.copy2(pinned_pkl, render_pkl)
            shutil.copy2(pinned_manifest, render_manifest)
            shutil.copy2(
                pinned_manifest_signature,
                runtime_manifest_sig_path(render_manifest),
            )
            shutil.copy2(pinned_pkl_signature, render_pkl_signature)

            render_manifest_record = read_runtime_manifest(render_manifest)
            if render_manifest_record != manifest:
                raise RuntimeError("Disposable runtime manifest changed while copying")
            if _sha256_file(render_pkl) != render_manifest_record.get("pkl_sha256"):
                raise RuntimeError("Disposable runtime dump does not match its manifest")
            if not _verify_pkl_integrity(render_pkl, self._integrity_root):
                raise RuntimeError("Disposable runtime dump signature does not verify")
            plan_restore(render_manifest, render_pkl)

        state_payload = {
            "session_id": self.session_id,
            "execution_count": 0,
            "runtime_persistence": {
                "status": "saved",
                "current_revision": committed_revision,
                "saved_revision": committed_revision,
                "committed_revision": committed_revision,
                "committed_source": persistence.get("committed_source") or "regular_save",
                "committed_digest": manifest["pkl_sha256"],
                "committed_pairing": manifest.get("pairing"),
                "last_attempt_revision": committed_revision,
                "last_attempt_status": "succeeded",
                "requires_recompute": False,
                "recompute_variables": [],
            },
        }
        (states_dir / f"{self.session_id}_state.json").write_text(
            json.dumps(state_payload, sort_keys=True), encoding="utf-8"
        )

        sandbox = LocalJupyterSandbox(
            session_id=self.session_id,
            state_dir=str(states_dir),
            working_dir=str(workspace),
            timeout=int(getattr(self.main_sandbox, "timeout", 1200) or 1200),
            runtime_budgets=dict(
                getattr(self.main_sandbox, "runtime_budgets", None) or {}
            ),
        )
        self.sandbox = sandbox
        started = await sandbox.start()
        if started.get("restore_failed") or not started.get("session_restored"):
            raise RuntimeError(
                "Disposable renderer did not restore the verified committed generation"
            )
        if not sandbox.notebook_state.has_session_dump():
            raise RuntimeError("Disposable renderer did not retain committed dump identity")
        return sandbox

    async def _cleanup(self) -> None:
        sandbox, self.sandbox = self.sandbox, None
        if sandbox is not None:
            try:
                await asyncio.shield(sandbox.stop(save=False))
            except BaseException:
                # Cleanup below still removes only this renderer's temporary files.
                pass
        if self._integrity_root is not None:
            try:
                shutil.rmtree(self._integrity_root)
            except FileNotFoundError:
                pass
            self._integrity_root = None
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
        temp_root = Path(tempfile.gettempdir()) / f"scai-{self.session_id}"
        try:
            if temp_root.is_symlink():
                temp_root.unlink()
            elif temp_root.exists():
                shutil.rmtree(temp_root)
        except FileNotFoundError:
            pass
