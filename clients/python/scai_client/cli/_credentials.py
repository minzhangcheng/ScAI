"""Credential storage for the ``scai`` CLI."""
import os
from pathlib import Path


def key_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "scai" / "api_key"


def read_key() -> str:
    try:
        return key_path().read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


def write_key(key: str) -> Path:
    path = key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    return path
