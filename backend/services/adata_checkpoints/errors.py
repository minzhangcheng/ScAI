"""Typed, operation-local failures for AnnData checkpoints and source loading."""
from __future__ import annotations

from typing import Any, Dict, Optional


class AdataCheckpointError(RuntimeError):
    """Base error for checkpoint storage/capture/load operations."""

    reason = "adata_checkpoint_error"

    def __init__(self, detail: str = "", *, context: Optional[Dict[str, Any]] = None):
        self.detail = str(detail or "")
        self.context: Dict[str, Any] = dict(context or {})
        super().__init__(self.detail or self.reason)

    def to_context(self) -> Dict[str, Any]:
        return {"reason": self.reason, **self.context}


class AdataCaptureError(AdataCheckpointError):
    reason = "adata_capture_failed"


class AdataStorageError(AdataCheckpointError):
    reason = "adata_checkpoint_storage_failed"


class AdataSourcePreparationError(AdataCheckpointError):
    reason = "adata_source_preparation_failed"

    def __init__(
        self,
        detail: str = "",
        *,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(detail, context=context)
        if reason:
            self.reason = str(reason)

    def to_context(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            **self.context,
            "action": (
                "choose another checkpoint/live binding or explicitly continue "
                "with current live data"
            ),
        }
