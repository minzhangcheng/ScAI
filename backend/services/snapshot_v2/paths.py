from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

from backend.config._config import get_session_dir, get_integrity_dir


def _compute_batches(root: Path) -> Path:
    return root / "batches"


@dataclass(frozen=True)
class SnapshotV2Paths:
    root: Path
    objects: Path
    staging: Path
    journals: Path
    integrity: Path
    committed: Path
    batches: Path = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.batches is None:
            object.__setattr__(self, "batches", _compute_batches(self.root))

    @classmethod
    def for_session(cls, session_id: str) -> "SnapshotV2Paths":
        base = get_session_dir(session_id)
        return cls(
            root=base / "v2",
            objects=base / "objects",
            staging=base / "staging",
            journals=base / "journals",
            integrity=get_integrity_dir(session_id),
            committed=base / "committed",
        )
