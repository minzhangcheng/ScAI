"""SQLite database access layer for users and API keys."""
import aiosqlite
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timezone

from backend.models.user import User, ApiKey

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT,
    username TEXT,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    status TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    hashed_key TEXT UNIQUE NOT NULL,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(hashed_key);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email IS NOT NULL;

CREATE TABLE IF NOT EXISTS upload_sessions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    expected_sha256 TEXT NOT NULL,
    total_chunks INTEGER NOT NULL,
    chunk_size INTEGER NOT NULL,
    received_chunks INTEGER NOT NULL DEFAULT 0,
    received_mask TEXT NOT NULL DEFAULT '',
    placeholder_file_id TEXT,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_uploads_expires ON upload_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_uploads_session ON upload_sessions(session_id);

CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_llm_config (
    user_id TEXT PRIMARY KEY,
    encrypted_api_key TEXT NOT NULL DEFAULT '',
    llm_base_url TEXT NOT NULL DEFAULT '',
    encrypted_proxy_url TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""

# Throttle API key last_used_at updates (seconds)
_KEY_USE_THROTTLE = 60.0
_KEY_USE_TIMES_MAX = 10000
_key_use_times: Dict[str, float] = {}


class AppDB:
    def __init__(self, path: Path):
        self.path = Path(path)

    async def _connect(self) -> aiosqlite.Connection:
        """Open a connection with foreign keys enabled."""
        db = await aiosqlite.connect(self.path, timeout=30)
        await db.execute("PRAGMA foreign_keys = ON")
        return db

    async def init(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA foreign_keys = ON")
            await db.executescript(SCHEMA)
            await db.commit()

    # ── Users ──────────────────────────────────────────────────────────

    async def create_user(self, username: str, password_hash: str, email: Optional[str] = None, role: str = "user", status: str = "pending") -> str:
        uid = str(uuid.uuid4())
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute(
                "INSERT INTO users(id,email,username,password_hash,created_at,role,status) VALUES(?,?,?,?,?,?,?)",
                (uid, email, username, password_hash, datetime.now(timezone.utc).isoformat(), role, status),
            )
            await db.commit()
        return uid

    async def get_user_by_email(self, email: str) -> Optional[User]:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM users WHERE email=?", (email,))
            row = await cur.fetchone()
            if row is None:
                return None
            return User(id=row["id"], email=row["email"],
                        username=row["username"],
                        password_hash=row["password_hash"],
                        created_at=row["created_at"],
                        role=row["role"],
                        status=row["status"])

    async def get_user_by_id(self, uid: str) -> Optional[User]:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM users WHERE id=?", (uid,))
            row = await cur.fetchone()
            if row is None:
                return None
            return User(id=row["id"], email=row["email"],
                        username=row["username"],
                        password_hash=row["password_hash"],
                        created_at=row["created_at"],
                        role=row["role"],
                        status=row["status"])

    async def get_user_by_username(self, username: str) -> Optional[User]:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM users WHERE username=?", (username,))
            row = await cur.fetchone()
            if row is None:
                return None
            return User(id=row["id"], email=row["email"],
                        username=row["username"],
                        password_hash=row["password_hash"],
                        created_at=row["created_at"],
                        role=row["role"],
                        status=row["status"])

    async def list_users(self) -> List[User]:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM users ORDER BY created_at DESC")
            rows = await cur.fetchall()
            return [
                User(id=r["id"], email=r["email"],
                     username=r["username"],
                     password_hash=r["password_hash"],
                     created_at=r["created_at"],
                     role=r["role"],
                     status=r["status"])
                for r in rows
            ]

    async def update_user_password(self, uid: str, password_hash: str) -> bool:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            cur = await db.execute(
                "UPDATE users SET password_hash=? WHERE id=?",
                (password_hash, uid),
            )
            await db.commit()
            return cur.rowcount > 0

    async def update_user_email(self, uid: str, email: str) -> bool:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            cur = await db.execute(
                "UPDATE users SET email=? WHERE id=?",
                (email, uid),
            )
            await db.commit()
            return cur.rowcount > 0

    async def set_user_status(self, uid: str, status: str) -> bool:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            cur = await db.execute(
                "UPDATE users SET status=? WHERE id=?",
                (status, uid),
            )
            await db.commit()
            return cur.rowcount > 0

    async def delete_user(self, uid: str) -> bool:
        async with self._connect() as db:
            cur = await db.execute("DELETE FROM users WHERE id=?", (uid,))
            await db.commit()
            return cur.rowcount > 0

    async def delete_api_keys_by_user(self, uid: str) -> None:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute("DELETE FROM api_keys WHERE user_id=?", (uid,))
            await db.commit()

    async def count_admins(self) -> int:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT COUNT(*) as cnt FROM users WHERE role='admin'"
            )
            row = await cur.fetchone()
            return row["cnt"] if row else 0

    async def is_admin_initialized(self) -> bool:
        """Check if admin has been initialized (same check as setup route)."""
        meta = await self.get_meta("admin_initialized")
        return meta == "1"

    async def create_admin_if_missing(self, *, password_hash: str, email: Optional[str] = None) -> tuple[str, bool]:
        """
        Get existing admin or create new one if not exists.

        Returns (admin_id, created) where created is True if a new admin was inserted.
        If admin already exists and email is provided, updates the admin's email.
        """
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT id FROM users WHERE role='admin' LIMIT 1"
            )
            row = await cur.fetchone()
            if row is not None:
                if email:
                    await db.execute("UPDATE users SET email=? WHERE id=?", (email, row["id"]))
                    await db.commit()
                return row["id"], False
            uid = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO users(id,email,username,password_hash,created_at,role,status) VALUES(?,?,?,?,?,?,?)",
                (uid, email, "admin", password_hash, datetime.now(timezone.utc).isoformat(), "admin", "active"),
            )
            await db.commit()
            return uid, True

    async def get_meta(self, key: str) -> Optional[str]:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT value FROM app_meta WHERE key=?", (key,))
            row = await cur.fetchone()
            return row["value"] if row else None

    async def set_meta(self, key: str, value: str) -> None:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute(
                "INSERT OR REPLACE INTO app_meta(key,value) VALUES(?,?)",
                (key, value),
            )
            await db.commit()

    # ── User LLM Config ──────────────────────────────────────────────────

    async def get_user_llm_config(self, user_id: str) -> Optional[dict]:
        """Return the LLM config dict for *user_id* or None if not found."""
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM user_llm_config WHERE user_id=?", (user_id,)
            )
            row = await cur.fetchone()
            if row is None:
                return None
            return dict(row)

    async def set_user_llm_config(
        self,
        user_id: str,
        encrypted_api_key: str,
        llm_base_url: str,
        encrypted_proxy_url: str,
    ) -> None:
        """Upsert the LLM config for *user_id*.

        On INSERT, created_at is set to the current UTC time.
        On UPDATE, created_at is preserved; only the payload fields and
        updated_at are changed.
        """
        now = datetime.now(timezone.utc).isoformat()
        async with self._connect() as db:
            await db.execute(
                """INSERT INTO user_llm_config
                       (user_id, encrypted_api_key, llm_base_url, encrypted_proxy_url,
                        created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       encrypted_api_key=excluded.encrypted_api_key,
                       llm_base_url=excluded.llm_base_url,
                       encrypted_proxy_url=excluded.encrypted_proxy_url,
                       updated_at=excluded.updated_at""",
                (user_id, encrypted_api_key, llm_base_url, encrypted_proxy_url, now, now),
            )
            await db.commit()

    async def delete_user_llm_config(self, user_id: str) -> None:
        """Delete the LLM config for *user_id* (no-op if not found)."""
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute(
                "DELETE FROM user_llm_config WHERE user_id=?", (user_id,)
            )
            await db.commit()

    # ── API Keys ─────────────────────────────────────────────────────────

    async def create_api_key(self, user_id: str, hashed_key: str, label: str) -> tuple[str, str]:
        kid = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute(
                "INSERT INTO api_keys(id,user_id,hashed_key,label,created_at) VALUES(?,?,?,?,?)",
                (kid, user_id, hashed_key, label, created_at),
            )
            await db.commit()
        return kid, created_at

    async def lookup_user_by_hashed_key(self, hashed_key: str) -> Optional[str]:
        """Return user_id if a matching API key hash is found, else None.
        Throttles last_used_at updates to at most once per 60 seconds.
        """
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT user_id FROM api_keys WHERE hashed_key=?", (hashed_key,)
            )
            row = await cur.fetchone()
            if row is None:
                return None
            # Throttle last_used_at updates
            now = time.time()
            last = _key_use_times.get(hashed_key, 0)
            if now - last > _KEY_USE_THROTTLE:
                await db.execute(
                    "UPDATE api_keys SET last_used_at=? WHERE hashed_key=?",
                    (datetime.now(timezone.utc).isoformat(), hashed_key),
                )
                await db.commit()
                _key_use_times[hashed_key] = now
                # Periodic cleanup: prevent memory growth when many keys are used
                if len(_key_use_times) > _KEY_USE_TIMES_MAX:
                    oldest_keys = sorted(_key_use_times, key=_key_use_times.get)[:len(_key_use_times) // 2]
                    for k in oldest_keys:
                        _key_use_times.pop(k, None)
            return row["user_id"]

    async def list_api_keys(self, user_id: str) -> List[ApiKey]:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM api_keys WHERE user_id=? ORDER BY created_at DESC",
                (user_id,),
            )
            rows = await cur.fetchall()
            return [
                ApiKey(id=r["id"], user_id=r["user_id"], hashed_key=r["hashed_key"],
                       label=r["label"], created_at=r["created_at"],
                       last_used_at=r["last_used_at"])
                for r in rows
            ]

    # ── Upload Sessions ──────────────────────────────────────────────────

    async def create_upload_session(
        self, session_id: str, filename: str, expected_sha256: str,
        total_chunks: int, chunk_size: int, expires_at_iso: str,
        placeholder_file_id: str,
    ) -> str:
        uid = str(uuid.uuid4())
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute(
                """INSERT INTO upload_sessions
                (id, session_id, filename, expected_sha256, total_chunks,
                 chunk_size, received_chunks, received_mask, placeholder_file_id,
                 created_at, expires_at, completed)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uid, session_id, filename, expected_sha256, total_chunks,
                 chunk_size, 0, "", placeholder_file_id,
                 datetime.now(timezone.utc).isoformat(), expires_at_iso, 0),
            )
            await db.commit()
        return uid

    async def get_upload_session(self, upload_id: str) -> Optional[dict]:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM upload_sessions WHERE id=?", (upload_id,))
            row = await cur.fetchone()
            if row is None:
                return None
            return dict(row)

    async def mark_chunk_received(self, upload_id: str, chunk_index: int) -> Optional[dict]:
        """Mark a chunk as received. Returns dict with is_new, received_chunks, total_chunks or None if not found."""
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute("BEGIN IMMEDIATE")
            cur = await db.execute(
                "SELECT received_mask, received_chunks, total_chunks FROM upload_sessions WHERE id=?",
                (upload_id,),
            )
            row = await cur.fetchone()
            if row is None:
                await db.execute("COMMIT")
                return None
            mask_str = row[0] or ""
            total = row[2]
            # Check if chunk already received
            existing = set(mask_str.split(",")) - {""}
            if str(chunk_index) in existing:
                await db.execute("COMMIT")
                return {"is_new": False, "received_chunks": row[1], "total_chunks": total}
            # Append directly instead of rebuilding mask string (O(1) vs O(N))
            new_mask = f"{mask_str},{chunk_index}" if mask_str else str(chunk_index)
            new_count = row[1] + 1
            await db.execute(
                "UPDATE upload_sessions SET received_mask=?, received_chunks=? WHERE id=?",
                (new_mask, new_count, upload_id),
            )
            await db.commit()
            return {"is_new": True, "received_chunks": new_count, "total_chunks": total}

    async def complete_upload_session(self, upload_id: str) -> None:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute(
                "UPDATE upload_sessions SET completed=1 WHERE id=?", (upload_id,),
            )
            await db.commit()

    async def list_expired_uploads(self, now_iso: str) -> List[dict]:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM upload_sessions WHERE expires_at < ? AND completed=0",
                (now_iso,),
            )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def delete_upload_session(self, upload_id: str) -> None:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute("DELETE FROM upload_sessions WHERE id=?", (upload_id,))
            await db.commit()

    async def delete_api_key(self, key_id: str, user_id: str) -> bool:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            cur = await db.execute(
                "DELETE FROM api_keys WHERE id=? AND user_id=?", (key_id, user_id)
            )
            await db.commit()
            return cur.rowcount > 0

    async def list_uploads_for_session(self, session_id: str) -> list:
        """List all upload sessions for a given session_id."""
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM upload_sessions WHERE session_id=?",
                (session_id,),
            )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def delete_uploads_for_session(self, session_id: str) -> None:
        """Delete all upload session records for a given session_id."""
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute(
                "DELETE FROM upload_sessions WHERE session_id=?",
                (session_id,),
            )
            await db.commit()

    async def delete_session_record(self, session_id: str) -> None:
        """Delete upload session records from the database.

        File cleanup is handled by services/upload.py:cleanup_session_uploads.
        """
        await self.delete_uploads_for_session(session_id)
