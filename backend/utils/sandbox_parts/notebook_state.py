"""Notebook state and execution-result models for the sandbox facade."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Final, List, Optional

import nbformat
from nbformat.v4 import new_code_cell, new_notebook, new_output

from .regular_save import _sha256_file, _verify_pkl_integrity

_NOTEBOOK_INLINE_IMAGE_LIMIT_BYTES: Final[int] = 64 * 1024
_NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS: Final[int] = 256 * 1024
_NOTEBOOK_HTML_OUTPUT_LIMIT_CHARS: Final[int] = 256 * 1024
_NOTEBOOK_TRACEBACK_LIMIT_LINES: Final[int] = 200
_NOTEBOOK_OUTPUT_OMISSION = "\n...[output omitted]...\n"
_NOTEBOOK_HTML_OMISSION = "[HTML output omitted: exceeds limit]"
_NOTEBOOK_TRACEBACK_OMISSION = "[... traceback omitted ...]"

logger = logging.getLogger("backend.utils.sandbox_parts.shared")


def _integrity_dir(sid: str) -> Path:
    # Preserve the historical patch seam on sandbox_parts.shared. NotebookState
    # moved into this module, but callers still override the shared helper.
    from . import shared

    return shared._integrity_dir(sid)


class CodeExecutionResult:
    """Result of code execution"""
    def __init__(
        self,
        success: bool,
        output: str = "",
        error: Optional[str] = None,
        figures: Optional[List[str]] = None,  # Base64 encoded images
        execution_count: int = 0,
        timed_out: bool = False,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.figures = figures or []
        self.execution_count = execution_count
        self.timed_out = bool(timed_out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "figures": self.figures,
            "execution_count": self.execution_count
        }


class NotebookState:
    """Represents the state of a Jupyter notebook session"""
    def __init__(
        self,
        session_id: str,
        state_dir: str = "states",
        lifecycle_generation: int = 0,
    ):
        self.session_id = session_id
        self.lifecycle_generation = max(0, int(lifecycle_generation or 0))
        self.state_dir = Path(state_dir).absolute()  # Use absolute path
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.working_dir = self.state_dir.parent / "workspace"

        self.cells: List[Dict[str, Any]] = []  # List of {code, outputs, execution_count}
        self.execution_count: int = 0
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = datetime.now().isoformat()
        self.runtime_persistence: Dict[str, Any] = {
            "status": "stale",
            "current_revision": 0,
            "saved_revision": 0,
            "dirty_since": None,
            "last_saved_at": None,
            "last_restored_at": None,
            "last_failed_at": None,
            "failed_variables": [],
            "error_summary": None,
            "requires_recompute": False,
            "recompute_variables": [],
            "recommended_action": None,
            "commit_source": None,
            "runtime_digest": None,
            # The committed generation remains independently addressable when
            # a later save/restore attempt fails. Attempt state must never make
            # the last verified pkl inaccessible.
            "committed_revision": 0,
            "committed_source": None,
            "committed_digest": None,
            "last_attempt_status": None,
            "last_attempt_at": None,
            "last_attempt_revision": 0,
            "last_attempt_error": None,
            "last_attempt_failed_variables": [],
            "degraded_recovery": False,
            "degraded_from_revision": None,
            "degraded_to_revision": None,
            "budget_violations": [],
            "top_offenders": [],
            "source": "notebook_state",
            "lifecycle_generation": self.lifecycle_generation,
            "metadata_updated_at": None,
        }
        self._load_runtime_persistence_metadata()
        persisted_generation = int(
            self.runtime_persistence.get("lifecycle_generation") or 0
        )
        if (
            self.lifecycle_generation
            and persisted_generation
            and persisted_generation != self.lifecycle_generation
        ):
            raise RuntimeError(
                "Runtime persistence metadata belongs to lifecycle generation "
                f"{persisted_generation}, not {self.lifecycle_generation}."
            )
        if self.lifecycle_generation:
            self.runtime_persistence["lifecycle_generation"] = (
                self.lifecycle_generation
            )

    @property
    def notebook_path(self) -> Path:
        from backend.utils.live_paths import live_notebook_path

        return live_notebook_path(self.state_dir, self.session_id)

    @property
    def state_path(self) -> Path:
        return self.state_dir / f"{self.session_id}_state.json"

    @property
    def session_path(self) -> Path:
        """Path for dill session dump"""
        return self.state_dir / f"{self.session_id}_session.pkl"

    @staticmethod
    def _bound_notebook_text(value: Any, limit: int) -> Any:
        if not isinstance(value, str) or len(value) <= limit:
            return value
        marker = _NOTEBOOK_OUTPUT_OMISSION
        if limit <= len(marker):
            return marker[:limit]
        remaining = limit - len(marker)
        head = (remaining + 1) // 2
        tail = remaining - head
        return value[:head] + marker + (value[-tail:] if tail else "")

    def _compact_outputs_for_state(self, outputs: List[Dict]) -> List[Dict]:
        """Compact outputs with this state's pinned workspace context."""
        compacted = copy.deepcopy(outputs or [])
        from backend.utils.notebook_media import (
            EXTERNAL_IMAGE_PLACEHOLDER,
            NOTEBOOK_EXTERNAL_IMAGE_MIME,
            externalize_notebook_image,
        )

        def estimated_size(encoded: Any) -> int:
            if not isinstance(encoded, str):
                return 0
            padding = len(encoded) - len(encoded.rstrip("="))
            return max(0, (len(encoded) * 3) // 4 - padding)

        for output in compacted:
            if not isinstance(output, dict):
                continue
            for field in ("text", "evalue", "error"):
                if isinstance(output.get(field), str):
                    output[field] = self._bound_notebook_text(
                        output[field], _NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS
                    )
            traceback = output.get("traceback")
            if isinstance(traceback, list):
                lines = [
                    self._bound_notebook_text(line, _NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS)
                    for line in traceback
                ]
                output["traceback"] = (
                    lines[:100] + [_NOTEBOOK_TRACEBACK_OMISSION] + lines[-100:]
                    if len(lines) > _NOTEBOOK_TRACEBACK_LIMIT_LINES
                    else lines
                )
            data = output.get("data")
            if not isinstance(data, dict):
                continue
            existing_external = data.pop(NOTEBOOK_EXTERNAL_IMAGE_MIME, None)
            external_references: List[Any] = []
            if isinstance(existing_external, list):
                external_references.extend(existing_external)
            elif existing_external is not None:
                external_references.append(existing_external)
            oversized_html = False
            for key, value in list(data.items()):
                if key == "text/plain":
                    data[key] = self._bound_notebook_text(
                        value, _NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS
                    )
                elif key == "text/html" and isinstance(value, str):
                    if len(value) > _NOTEBOOK_HTML_OUTPUT_LIMIT_CHARS:
                        data.pop(key, None)
                        oversized_html = True
                elif key in {"image/png", "image/jpeg"}:
                    if estimated_size(value) <= _NOTEBOOK_INLINE_IMAGE_LIMIT_BYTES:
                        continue
                    reference = externalize_notebook_image(
                        self.working_dir,
                        self.notebook_path,
                        value,
                        media_type=key,
                    )
                    data.pop(key, None)
                    if reference is None:
                        data.setdefault("text/plain", EXTERNAL_IMAGE_PLACEHOLDER)
                    else:
                        external_references.append(reference)
                        data.setdefault("text/plain", f"[external image: {reference['path']}]")
            if external_references:
                data[NOTEBOOK_EXTERNAL_IMAGE_MIME] = (
                    external_references[0]
                    if len(external_references) == 1
                    else external_references
                )
            if oversized_html:
                plain = data.get("text/plain")
                if isinstance(plain, str) and plain:
                    plain = f"{plain}\n{_NOTEBOOK_HTML_OMISSION}"
                else:
                    plain = _NOTEBOOK_HTML_OMISSION
                data["text/plain"] = self._bound_notebook_text(
                    plain,
                    _NOTEBOOK_TEXT_OUTPUT_LIMIT_CHARS,
                )
        return compacted

    def add_cell(self, code: str, outputs: List[Dict], execution_count: int):
        """Add an executed cell to the history"""
        self.cells.append({
            "code": code,
            "outputs": self._compact_outputs_for_state(outputs),
            "execution_count": execution_count,
            "executed_at": datetime.now().isoformat()
        })
        self.execution_count = execution_count
        self.updated_at = datetime.now().isoformat()

    def save_notebook(self):
        """Save as Jupyter notebook format"""
        nb = new_notebook()
        for index, cell in enumerate(self.cells):
            code_cell = new_code_cell(source=cell["code"])
            # nbformat generates a random cell id when one is omitted.  A
            # repeated save must be byte-stable so exports and checkpoints do
            # not look changed merely because notebook metadata was rebuilt.
            code_cell["id"] = hashlib.sha256(
                f"{index}:{cell['code']}".encode("utf-8")
            ).hexdigest()[:8]
            code_cell.execution_count = cell["execution_count"]

            # Convert outputs to notebook format
            cell_outputs = []
            for out in self._compact_outputs_for_state(cell["outputs"]):
                if out["type"] == "stream":
                    cell_outputs.append(new_output(
                        output_type="stream",
                        name=out.get("name", "stdout"),
                        text=out["text"]
                    ))
                elif out["type"] == "execute_result":
                    cell_outputs.append(new_output(
                        output_type="execute_result",
                        data=out.get("data", {}),
                        execution_count=cell["execution_count"]
                    ))
                elif out["type"] == "display_data":
                    cell_outputs.append(new_output(
                        output_type="display_data",
                        data=out.get("data", {})
                    ))
                elif out["type"] == "error":
                    cell_outputs.append(new_output(
                        output_type="error",
                        ename=out.get("ename", "Error"),
                        evalue=out.get("evalue", ""),
                        traceback=out.get("traceback", [])
                    ))

            code_cell.outputs = cell_outputs
            nb.cells.append(code_cell)

        # Write notebook file atomically. The live notebook is rewritten in
        # place on every save and is packed into every session export, so a
        # truncate-and-rewrite window let the exporter capture a half-written
        # file -- an archive that imports "successfully" with a corrupt notebook.
        tmp_path = self.notebook_path.with_name(
            f"{self.notebook_path.name}.{os.getpid()}.tmp"
        )
        from backend.services.session_lifecycle import (
            assert_session_generation_active,
            lifecycle_file_lock,
        )

        try:
            with lifecycle_file_lock(self.session_id):
                assert_session_generation_active(self.session_id)
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    nbformat.write(nb, f)
                    f.flush()
                    os.fsync(f.fileno())
                assert_session_generation_active(self.session_id)
                os.replace(tmp_path, self.notebook_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        logger.info(f"Saved notebook: {self.notebook_path}")

    def _load_runtime_persistence_metadata(self) -> None:
        """Load only persistence metadata, even when no notebook is present."""
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            persisted = payload.get("runtime_persistence")
            if isinstance(persisted, dict):
                self.runtime_persistence.update(persisted)
                self._migrate_committed_runtime_metadata()
        except Exception:
            logger.warning(
                "[%s] Could not read runtime persistence metadata",
                self.session_id,
                exc_info=True,
            )

    def _migrate_committed_runtime_metadata(self) -> None:
        """Derive explicit committed fields from pre-separation metadata."""
        persistence = self.runtime_persistence
        source = persistence.get("committed_source")
        digest = persistence.get("committed_digest")
        revision = int(persistence.get("committed_revision") or 0)
        if (
            not source
            and persistence.get("commit_source") in {
                "regular_save", "snapshot_apply"
            }
            and isinstance(persistence.get("runtime_digest"), str)
            and len(str(persistence.get("runtime_digest"))) == 64
        ):
            source = persistence.get("commit_source")
            digest = persistence.get("runtime_digest")
            revision = int(persistence.get("saved_revision") or 0)
        persistence["committed_source"] = source
        persistence["committed_digest"] = digest
        persistence["committed_revision"] = max(0, revision)

    def update_runtime_persistence(
        self,
        status: str,
        *,
        current_revision: int,
        saved_revision: int,
        failed_variables: Optional[List[str]] = None,
        error_summary: Optional[str] = None,
        requires_recompute: bool = False,
        recompute_variables: Optional[List[str]] = None,
        commit_source: Optional[str] = None,
        runtime_digest: Optional[str] = None,
        budget_violations: Optional[List[Dict[str, Any]]] = None,
        top_offenders: Optional[List[Dict[str, Any]]] = None,
        recommended_action: Optional[str] = None,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Persist the live runtime lifecycle without exposing raw tracebacks."""
        now = datetime.now().isoformat()
        previous = dict(self.runtime_persistence)
        from backend.utils.error_parsing import bounded_diagnostic, compact_traceback

        def _safe_runtime_text(value: Any, limit: int) -> str:
            # Redaction must precede slicing: a credential or private path near
            # the retained edge must never be persisted merely because a caller
            # supplied an already-long diagnostic string.
            rendered = compact_traceback(
                str(value or ""), max_chars=max(512, limit * 4),
            )
            rendered = rendered.replace(
                "[... traceback frames omitted ...]", "",
            )
            return " ".join(rendered.split())[:limit]

        safe_error = None
        if error_summary:
            safe_error = bounded_diagnostic(error_summary, 500)
        safe_variables = sorted({
            _safe_runtime_text(name, 128)
            for name in (failed_variables or [])
            if _safe_runtime_text(name, 128)
        })[:100]
        safe_budget_violations = []
        for item in list(budget_violations or [])[:20]:
            if not isinstance(item, dict):
                continue
            resource = _safe_runtime_text(item.get("resource"), 128)
            if not resource:
                continue
            safe_budget_violations.append({
                "resource": resource,
                "observed": item.get("observed")
                if isinstance(item.get("observed"), (int, float))
                and not isinstance(item.get("observed"), bool)
                else None,
                "limit": item.get("limit")
                if isinstance(item.get("limit"), (int, float))
                and not isinstance(item.get("limit"), bool)
                else None,
            })
        safe_top_offenders = []
        for item in list(top_offenders or [])[:20]:
            if not isinstance(item, dict):
                continue
            name = _safe_runtime_text(item.get("name"), 128)
            if not name:
                continue
            offender = {
                "name": name,
                "type": _safe_runtime_text(
                    item.get("type") or item.get("type_name"), 128,
                ),
            }
            raw_size = (
                item.get("bytes")
                if item.get("bytes") is not None
                else item.get("estimated_bytes")
            )
            if isinstance(raw_size, (int, float)) and not isinstance(raw_size, bool):
                offender["bytes"] = max(0, raw_size)
            safe_top_offenders.append(offender)
        self.runtime_persistence.update({
            "status": status,
            "current_revision": max(0, int(current_revision)),
            "saved_revision": max(0, int(saved_revision)),
            "failed_variables": safe_variables,
            "error_summary": safe_error,
            "requires_recompute": bool(requires_recompute),
            "recompute_variables": sorted({
                _safe_runtime_text(name, 128)
                for name in (recompute_variables or [])
                if _safe_runtime_text(name, 128)
            })[:100],
            "recommended_action": (
                _safe_runtime_text(recommended_action, 128)
                or (
                    "recompute_required_dependencies"
                    if requires_recompute else None
                )
            ),
            "commit_source": (
                str(commit_source)
                if commit_source is not None
                else previous.get("commit_source")
            ),
            "runtime_digest": (
                str(runtime_digest)
                if runtime_digest is not None
                else previous.get("runtime_digest")
            ),
            "budget_violations": safe_budget_violations,
            "top_offenders": safe_top_offenders,
            "source": "notebook_state",
            "lifecycle_generation": self.lifecycle_generation,
            "metadata_updated_at": now,
        })
        if status == "live_dirty" and not previous.get("dirty_since"):
            self.runtime_persistence["dirty_since"] = now
        elif status == "current":
            self.runtime_persistence.update({
                "dirty_since": None,
                "last_saved_at": now,
                "failed_variables": [],
                "error_summary": None,
                "requires_recompute": False,
                "recompute_variables": [],
                "recommended_action": None,
                "budget_violations": [],
                "top_offenders": [],
                "committed_revision": max(0, int(saved_revision)),
                "committed_source": (
                    str(commit_source)
                    if commit_source is not None
                    else previous.get("committed_source")
                ),
                "committed_digest": (
                    str(runtime_digest)
                    if runtime_digest is not None
                    else previous.get("committed_digest")
                ),
                "last_attempt_status": "succeeded",
                "last_attempt_at": now,
                "last_attempt_revision": max(0, int(current_revision)),
                "last_attempt_error": None,
                "last_attempt_failed_variables": [],
                "degraded_recovery": False,
                "degraded_from_revision": None,
                "degraded_to_revision": None,
            })
        elif status == "failed":
            self.runtime_persistence.update({
                "last_failed_at": now,
                "last_attempt_status": "failed",
                "last_attempt_at": now,
                "last_attempt_revision": max(0, int(current_revision)),
                "last_attempt_error": safe_error,
                "last_attempt_failed_variables": safe_variables,
            })
        if persist:
            self.save_metadata()
        return dict(self.runtime_persistence)

    def record_runtime_restored(
        self,
        *,
        current_revision: int,
        saved_revision: int,
        commit_source: Optional[str],
        degraded_recovery: bool = False,
        attempted_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Record rehydrate without fabricating a new persistence event."""
        restored_revision = max(0, int(saved_revision))
        attempted = max(
            restored_revision,
            int(
                attempted_revision
                if attempted_revision is not None
                else self.runtime_persistence.get("last_attempt_revision") or 0
            ),
        )
        self.runtime_persistence.update({
            "status": "degraded" if degraded_recovery else "current",
            "current_revision": restored_revision,
            "saved_revision": restored_revision,
            "commit_source": (
                str(commit_source)
                if commit_source is not None
                else self.runtime_persistence.get("commit_source")
            ),
            "last_restored_at": datetime.now().isoformat(),
            "degraded_recovery": bool(degraded_recovery),
            "degraded_from_revision": attempted if degraded_recovery else None,
            "degraded_to_revision": restored_revision if degraded_recovery else None,
            "requires_recompute": (
                True
                if degraded_recovery
                else bool(self.runtime_persistence.get("requires_recompute"))
            ),
            "recommended_action": (
                "recompute_or_reduce_runtime"
                if degraded_recovery
                else self.runtime_persistence.get("recommended_action")
            ),
            "source": "notebook_state",
            "lifecycle_generation": self.lifecycle_generation,
            "metadata_updated_at": datetime.now().isoformat(),
        })
        self._migrate_committed_runtime_metadata()
        self.save_metadata()
        return dict(self.runtime_persistence)

    def _metadata_payload(self) -> Dict[str, Any]:
        persistence = dict(self.runtime_persistence)
        self._migrate_committed_runtime_metadata()
        persistence = dict(self.runtime_persistence)
        runtime_digest = persistence.get("committed_digest")
        # Derived from the committed record, not by re-verifying the file. The
        # verification form ran an HMAC pass and a SHA-256 pass over the whole
        # dump every time metadata was written -- several gigabytes twice per
        # save -- and still could not answer the question during a commit, when
        # the recorded digest belongs to the generation being replaced.
        # ``has_session_dump()`` keeps the expensive form for the one caller
        # that needs proof: restore.
        has_current_session = bool(
            persistence.get("committed_source") in {
                "regular_save", "snapshot_apply"
            }
            and self.session_path.exists()
            and isinstance(runtime_digest, str)
            and len(runtime_digest) == 64
        )
        return {
            "session_id": self.session_id,
            "execution_count": self.execution_count,
            "cell_count": len(self.cells),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # Compatibility field, now derived from committed generation state.
            "has_session": has_current_session,
            "runtime_persistence": persistence,
        }

    def save_metadata(self):
        """Save state metadata atomically."""
        state_data = self._metadata_payload()
        tmp_path = self.state_path.with_name(
            f"{self.state_path.name}.{os.getpid()}.tmp"
        )
        from backend.services.session_lifecycle import (
            assert_session_generation_active,
            lifecycle_file_lock,
        )

        try:
            with lifecycle_file_lock(self.session_id):
                assert_session_generation_active(self.session_id)
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                assert_session_generation_active(self.session_id)
                os.replace(tmp_path, self.state_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def save(self):
        """Save notebook and metadata"""
        from backend.services.session_lifecycle import (
            assert_session_generation_active,
            lifecycle_file_lock,
        )

        with lifecycle_file_lock(self.session_id):
            assert_session_generation_active(self.session_id)
            self.save_notebook()
            self.save_metadata()

    def load(self) -> bool:
        """Load notebook state from disk"""
        if not self.notebook_path.exists():
            return False

        try:
            # Load notebook
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            self.cells = []
            for cell in nb.cells:
                if cell.cell_type == "code":
                    outputs = []
                    for out in cell.outputs:
                        if out.output_type == "stream":
                            outputs.append({
                                "type": "stream",
                                "name": out.get("name", "stdout"),
                                "text": out.text
                            })
                        elif out.output_type == "execute_result":
                            outputs.append({
                                "type": "execute_result",
                                "data": dict(out.data)
                            })
                        elif out.output_type == "display_data":
                            outputs.append({
                                "type": "display_data",
                                "data": dict(out.data)
                            })
                        elif out.output_type == "error":
                            outputs.append({
                                "type": "error",
                                "ename": out.ename,
                                "evalue": out.evalue,
                                "traceback": out.traceback
                            })

                    self.cells.append({
                        "code": cell.source,
                        "outputs": outputs,
                        "execution_count": cell.execution_count or 0,
                        "executed_at": None
                    })

            # Load state metadata
            if self.state_path.exists():
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                self.execution_count = state_data.get("execution_count", 0)
                self.created_at = state_data.get("created_at", self.created_at)
                self.updated_at = state_data.get("updated_at", self.updated_at)
                persisted = state_data.get("runtime_persistence")
                if isinstance(persisted, dict):
                    self.runtime_persistence.update(persisted)
                    self._migrate_committed_runtime_metadata()

            logger.info(f"Loaded notebook state: {len(self.cells)} cells")
            return True
        except Exception as e:
            logger.exception("Failed to load notebook state")
            return False

    def has_session_dump(self) -> bool:
        """Check for a committed, integrity-valid dump of the live generation."""
        self._migrate_committed_runtime_metadata()
        persistence = self.runtime_persistence
        runtime_digest = persistence.get("committed_digest")
        return bool(
            persistence.get("committed_source") in {
                "regular_save", "snapshot_apply"
            }
            and self.session_path.exists()
            and _verify_pkl_integrity(self.session_path, _integrity_dir(self.session_id))
            and isinstance(runtime_digest, str)
            and len(runtime_digest) == 64
            and _sha256_file(self.session_path) == runtime_digest
        )

CodeExecutionResult.__module__ = "backend.utils.sandbox_parts.shared"
NotebookState.__module__ = "backend.utils.sandbox_parts.shared"
