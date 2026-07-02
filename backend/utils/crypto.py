"""Encryption utilities for per-user LLM configuration secrets.

Uses Fernet (AES-128-CBC + HMAC-SHA256) from the ``cryptography`` package.

Secret key resolution order:
  1. ``SCAI_CONFIG_SECRET`` environment variable (recommended for deployments)
  2. ``<DATA_BASE_DIR>/.config_secret`` file (auto-generated on first use, chmod 600)
  3. Neither available → ERROR logged, per-user config disabled (functions raise
     :exc:`ConfigEncryptionUnavailable`; the module does **not** crash or fall back
     silently).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import secrets
from typing import Optional

from cryptography.fernet import Fernet

from backend.config._config import DATA_BASE_DIR

logger = logging.getLogger(__name__)


class ConfigEncryptionUnavailable(Exception):
    """Raised when the encryption secret is unavailable.

    This happens when neither the ``SCAI_CONFIG_SECRET`` environment variable is
    set nor a secret file can be read or created under ``DATA_BASE_DIR``.
    """


# ---------------------------------------------------------------------------
# Secret management
# ---------------------------------------------------------------------------

def _get_or_create_secret() -> Optional[str]:
    """Return the raw secret string, or *None* if it cannot be obtained.

    The returned value is a URL-safe base64 string suitable for use as a
    Fernet key (after derivation; see :func:`_build_fernet`).

    Priority:
    1. ``SCAI_CONFIG_SECRET`` env var
    2. ``<DATA_BASE_DIR>/.config_secret`` file (auto-created with chmod 600)
    3. Returns ``None`` and logs an ERROR — callers must handle gracefully.
    """
    if env := os.environ.get("SCAI_CONFIG_SECRET"):
        return env

    secret_path = DATA_BASE_DIR / ".config_secret"
    if secret_path.exists():
        try:
            return secret_path.read_text().strip()
        except OSError as e:
            logger.error("Cannot read config secret file: %s; per-user config disabled", e)
            return None

    # Attempt to create a new persistent secret atomically (O_EXCL ensures the
    # file is created with 0o600 from the start — no TOCTOU window).
    secret = secrets.token_urlsafe(32)
    try:
        fd = os.open(str(secret_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, secret.encode())
        finally:
            os.close(fd)
        logger.info("Created new config secret at %s", secret_path)
        return secret
    except FileExistsError:
        # Another process created the file between the exists() check above and
        # our O_EXCL open — just read the file that is already there.
        try:
            return secret_path.read_text().strip()
        except OSError as e:
            logger.error("Cannot read config secret file: %s; per-user config disabled", e)
            return None
    except OSError as e:
        logger.error(
            "Cannot persist config secret: %s; per-user config disabled", e
        )
        return None


def _build_fernet(secret: str) -> Fernet:
    """Derive a 32-byte Fernet key from *secret* and return a :class:`Fernet` instance.

    Fernet requires exactly 32 bytes encoded as URL-safe base64 (44 chars).
    We SHA-256 hash the raw secret bytes so that secrets of arbitrary length
    all produce a valid 32-byte key.
    """
    key_bytes = hashlib.sha256(secret.encode()).digest()  # always 32 bytes
    fernet_key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(fernet_key)


# ---------------------------------------------------------------------------
# Module-level secret (resolved once at import time)
# ---------------------------------------------------------------------------

_SECRET: Optional[str] = _get_or_create_secret()
_fernet: Optional[Fernet] = _build_fernet(_SECRET) if _SECRET else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_integrity_secret() -> str:
    """Return the shared secret for HMAC-integrity checks (e.g. session pkl).

    Uses the same secret store as :func:`encrypt_value`. Raises
    :exc:`ConfigEncryptionUnavailable` if no secret is available.

    The returned value is the raw secret string — callers should feed it into
    ``hmac.new(..., digestmod=hashlib.sha256)``.
    """
    s = _get_or_create_secret()
    if not s:
        raise ConfigEncryptionUnavailable("No integrity secret available")
    return s


def encrypt_value(plaintext: str) -> str:
    """Encrypt *plaintext* and return a URL-safe base64 ciphertext string.

    Raises:
        ConfigEncryptionUnavailable: if the encryption secret is not available.
    """
    if _SECRET is None:
        raise ConfigEncryptionUnavailable(
            "Encryption secret is unavailable; per-user config is disabled. "
            "Set SCAI_CONFIG_SECRET or ensure DATA_BASE_DIR is writable."
        )
    fernet = _fernet
    return fernet.encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str) -> str:
    """Decrypt a ciphertext produced by :func:`encrypt_value`.

    Raises:
        ConfigEncryptionUnavailable: if the encryption secret is not available.
        cryptography.fernet.InvalidToken: if *ciphertext* is invalid or tampered.
            Callers must import this exception directly from
            ``cryptography.fernet`` if they wish to catch it.
    """
    if _SECRET is None:
        raise ConfigEncryptionUnavailable(
            "Encryption secret is unavailable; per-user config is disabled. "
            "Set SCAI_CONFIG_SECRET or ensure DATA_BASE_DIR is writable."
        )
    fernet = _fernet
    return fernet.decrypt(ciphertext.encode()).decode()
