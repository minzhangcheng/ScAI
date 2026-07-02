"""One-shot migration: sign existing session pkls so C2 integrity verification
passes for sessions created before signing was introduced.

Run intentionally at upgrade time; running it asserts you trust the current
on-disk pkls. Usage: python -m backend.migrate_sign_session_pkls
"""
from pathlib import Path
from typing import Iterator, Tuple
import logging

from backend.config._config import DATA_BASE_DIR
from backend.utils.sandbox import _write_pkl_signature, _integrity_dir

logger = logging.getLogger(__name__)


def iter_session_pkls() -> Iterator[Tuple[str, Path]]:
    sessions_root = DATA_BASE_DIR / "sessions"
    if not sessions_root.exists():
        return
    for sid_dir in sessions_root.iterdir():
        if not sid_dir.is_dir():
            continue
        pkl = sid_dir / "states" / f"{sid_dir.name}_session.pkl"
        if pkl.exists():
            yield sid_dir.name, pkl


def sign_all() -> int:
    count = 0
    for sid, pkl in iter_session_pkls():
        try:
            _write_pkl_signature(pkl, _integrity_dir(sid))
            count += 1
        except Exception:
            logger.exception("Failed to sign pkl for session %s", sid)
    logger.info("Signed %d session pkl(s)", count)
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sign_all()
