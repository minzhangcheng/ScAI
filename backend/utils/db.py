"""SQLite database access layer for users and API keys."""
import logging
import aiosqlite
import uuid
import time
from pathlib import Path
from typing import Optional, List, Dict, Sequence
from datetime import datetime, timezone

from backend.models.user import User, ApiKey

logger = logging.getLogger(__name__)

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
    total_size INTEGER NOT NULL,
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

CREATE TABLE IF NOT EXISTS user_agent_default_secrets (
    user_id TEXT PRIMARY KEY,
    encrypted_payload TEXT NOT NULL DEFAULT '',
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

    async def init(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        async with aiosqlite.connect(self.path, timeout=30) as db:
            self.path.chmod(0o600)
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA foreign_keys = ON")
            await db.executescript(SCHEMA)
            columns = {
                row[1]
                for row in await (
                    await db.execute("PRAGMA table_info(upload_sessions)")
                ).fetchall()
            }
            if "total_size" not in columns:
                # Active sessions created before this migration cannot be
                # safely completed because their accepted byte budget was not
                # persisted.  NULL keeps them distinguishable and fail-closed.
                await db.execute(
                    "ALTER TABLE upload_sessions ADD COLUMN total_size INTEGER"
                )
            await db.commit()

    # ── Users ──────────────────────────────────────────────────────────

    async def create_user(
        self,
        username: str,
        password_hash: str,
        email: Optional[str] = None,
        role: str = "user",
        status: str = "pending",
        *,
        required_admin_id: Optional[str] = None,
    ) -> Optional[str]:
        uid = str(uuid.uuid4())
        async with aiosqlite.connect(self.path, timeout=30) as db:
            values = (
                uid,
                email,
                username,
                password_hash,
                datetime.now(timezone.utc).isoformat(),
                role,
                status,
            )
            if required_admin_id is None:
                cur = await db.execute(
                    "INSERT INTO users(id,email,username,password_hash,created_at,role,status) "
                    "VALUES(?,?,?,?,?,?,?)",
                    values,
                )
            else:
                cur = await db.execute(
                    "INSERT INTO users(id,email,username,password_hash,created_at,role,status) "
                    "SELECT ?,?,?,?,?,?,? WHERE EXISTS ("
                    "SELECT 1 FROM users AS actor WHERE actor.id=? "
                    "AND actor.role='admin' "
                    "AND actor.status IN ('active','readonly'))",
                    (*values, required_admin_id),
                )
            await db.commit()
        return uid if cur.rowcount > 0 else None

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

    async def update_user_password(
        self,
        uid: str,
        password_hash: str,
        *,
        unless_current_status: Optional[str] = None,
        required_admin_id: Optional[str] = None,
    ) -> bool:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            clauses = ["id=?"]
            params: list[str] = [password_hash, uid]
            if unless_current_status is not None:
                clauses.append("status<>?")
                params.append(unless_current_status)
            if required_admin_id is not None:
                clauses.append(
                    "EXISTS (SELECT 1 FROM users AS actor WHERE actor.id=? "
                    "AND actor.role='admin' "
                    "AND actor.status IN ('active','readonly'))"
                )
                params.append(required_admin_id)
            cur = await db.execute(
                "UPDATE users SET password_hash=? WHERE " + " AND ".join(clauses),
                tuple(params),
            )
            await db.commit()
            return cur.rowcount > 0

    async def compare_and_swap_user_password(
        self,
        uid: str,
        expected_hash: str,
        password_hash: str,
        *,
        allowed_statuses: Sequence[str],
    ) -> bool:
        """Replace a password only while the verified credential still owns it."""
        statuses = tuple(str(status) for status in allowed_statuses)
        if not statuses:
            return False
        placeholders = ",".join("?" for _ in statuses)
        async with aiosqlite.connect(self.path, timeout=30) as db:
            cur = await db.execute(
                "UPDATE users SET password_hash=? "
                "WHERE id=? AND password_hash=? "
                f"AND status IN ({placeholders})",
                (password_hash, uid, expected_hash, *statuses),
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

    async def update_user_username(
        self,
        uid: str,
        username: str,
        *,
        expected_username: str,
        allowed_statuses: Sequence[str],
        expected_password_hash: Optional[str] = None,
        required_admin_id: Optional[str] = None,
    ) -> bool:
        """Rename without changing identity, checking authority at the write boundary."""
        statuses = tuple(allowed_statuses)
        if not statuses:
            return False
        placeholders = ",".join("?" for _ in statuses)
        clauses = ["id=?", "username=?", f"status IN ({placeholders})"]
        params = [username, uid, expected_username, *statuses]
        if expected_password_hash is not None:
            clauses.append("password_hash=?")
            params.append(expected_password_hash)
        if required_admin_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM users AS actor WHERE actor.id=? "
                "AND actor.role='admin' AND actor.status IN ('active','readonly'))"
            )
            params.append(required_admin_id)
        async with aiosqlite.connect(self.path, timeout=30) as db:
            cur = await db.execute(
                "UPDATE users SET username=? WHERE " + " AND ".join(clauses),
                tuple(params),
            )
            await db.commit()
            return cur.rowcount > 0

    async def set_user_status(
        self,
        uid: str,
        status: str,
        *,
        unless_current_status: Optional[str] = None,
        forbid_current_role: Optional[str] = None,
        required_admin_id: Optional[str] = None,
    ) -> bool:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            clauses = ["id=?"]
            params: list[str] = [status, uid]
            if unless_current_status is not None:
                clauses.append("status<>?")
                params.append(unless_current_status)
            if forbid_current_role is not None:
                clauses.append("role<>?")
                params.append(forbid_current_role)
            if required_admin_id is not None:
                clauses.append(
                    "EXISTS (SELECT 1 FROM users AS actor WHERE actor.id=? "
                    "AND actor.role='admin' "
                    "AND actor.status IN ('active','readonly'))"
                )
                params.append(required_admin_id)
            cur = await db.execute(
                "UPDATE users SET status=? WHERE " + " AND ".join(clauses),
                tuple(params),
            )
            await db.commit()
            return cur.rowcount > 0

    async def set_user_role(
        self,
        uid: str,
        role: str,
        *,
        status: Optional[str] = None,
        unless_current_status: Optional[str] = None,
        activate_pending_admin: bool = False,
        required_admin_id: Optional[str] = None,
    ) -> bool:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            clauses = ["id=?"]
            where_params: list[str] = [uid]
            if unless_current_status is not None:
                clauses.append("status<>?")
                where_params.append(unless_current_status)
            if required_admin_id is not None:
                clauses.append(
                    "EXISTS (SELECT 1 FROM users AS actor WHERE actor.id=? "
                    "AND actor.role='admin' "
                    "AND actor.status IN ('active','readonly'))"
                )
                where_params.append(required_admin_id)
            if status is not None:
                statement = "UPDATE users SET role=?, status=? WHERE "
                params = [role, status, *where_params]
            elif activate_pending_admin:
                statement = (
                    "UPDATE users SET role=?, "
                    "status=CASE WHEN ?='admin' AND status='pending' "
                    "THEN 'active' ELSE status END WHERE "
                )
                params = [role, role, *where_params]
            else:
                statement = "UPDATE users SET role=? WHERE "
                params = [role, *where_params]
            cur = await db.execute(
                statement + " AND ".join(clauses),
                tuple(params),
            )
            await db.commit()
            return cur.rowcount > 0

    async def mark_user_deleting_if_not_last_admin(
        self,
        uid: str,
        *,
        required_admin_id: Optional[str] = None,
    ) -> str:
        """Atomically revoke one user without removing the last usable admin.

        Returns ``updated``, ``not_found``, ``last_admin``, or ``forbidden``.
        ``BEGIN IMMEDIATE`` serializes competing admin revocation/deletion
        before either request evaluates the authorization and last-admin
        invariants.
        """
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            try:
                if required_admin_id is not None:
                    actor = await (
                        await db.execute(
                            "SELECT 1 FROM users WHERE id=? AND role='admin' "
                            "AND status IN ('active','readonly')",
                            (required_admin_id,),
                        )
                    ).fetchone()
                    if actor is None:
                        await db.rollback()
                        return "forbidden"
                row = await (
                    await db.execute(
                        "SELECT role, status FROM users WHERE id=?", (uid,)
                    )
                ).fetchone()
                if row is None:
                    await db.rollback()
                    return "not_found"
                if row["status"] == "deleting":
                    await db.commit()
                    return "updated"
                if row["role"] == "admin" and row["status"] in (
                    "active", "readonly",
                ):
                    count_row = await (
                        await db.execute(
                            "SELECT COUNT(*) FROM users WHERE role='admin' "
                            "AND status IN ('active','readonly')"
                        )
                    ).fetchone()
                    if int(count_row[0] if count_row else 0) <= 1:
                        await db.rollback()
                        return "last_admin"
                await db.execute(
                    "UPDATE users SET status='deleting' WHERE id=?", (uid,)
                )
                await db.commit()
                return "updated"
            except BaseException:
                await db.rollback()
                raise

    async def delete_user(self, uid: str) -> bool:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            cur = await db.execute("DELETE FROM users WHERE id=?", (uid,))
            await db.commit()
            return cur.rowcount > 0

    async def delete_api_keys_by_user(self, uid: str) -> None:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute("DELETE FROM api_keys WHERE user_id=?", (uid,))
            await db.commit()

    async def demote_admin_if_not_last(
        self,
        uid: str,
        new_role: str,
        *,
        unless_current_status: Optional[str] = None,
        required_admin_id: Optional[str] = None,
    ) -> bool:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            clauses = [
                "id=?",
                "role='admin'",
                "(status NOT IN ('active','readonly') OR "
                "(SELECT COUNT(*) FROM users WHERE role='admin' "
                "AND status IN ('active','readonly')) > 1)",
            ]
            params: list[str] = [new_role, uid]
            if unless_current_status is not None:
                clauses.append("status<>?")
                params.append(unless_current_status)
            if required_admin_id is not None:
                clauses.append(
                    "EXISTS (SELECT 1 FROM users AS actor WHERE actor.id=? "
                    "AND actor.role='admin' "
                    "AND actor.status IN ('active','readonly'))"
                )
                params.append(required_admin_id)
            cur = await db.execute(
                "UPDATE users SET role=? WHERE " + " AND ".join(clauses),
                tuple(params),
            )
            await db.commit()
            return cur.rowcount > 0

    async def is_usable_admin(self, uid: str) -> bool:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            row = await (
                await db.execute(
                    "SELECT 1 FROM users WHERE id=? AND role='admin' "
                    "AND status IN ('active','readonly')",
                    (uid,),
                )
            ).fetchone()
            return row is not None

    async def count_admins(self) -> int:
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT COUNT(*) as cnt FROM users WHERE role='admin' "
                "AND status IN ('active','readonly')"
            )
            row = await cur.fetchone()
            return row["cnt"] if row else 0

    async def is_admin_initialized(self) -> bool:
        """Check if admin has been initialized (same check as setup route)."""
        meta = await self.get_meta("admin_initialized")
        return meta == "1"

    async def create_admin_if_missing(self, *, username: str, password_hash: str, email: Optional[str] = None) -> tuple[str, bool]:
        """
        Get existing admin or create a new one if none exists.

        Returns (admin_id, created) where created is True if a new admin was inserted.
        `username` is ONLY applied when creating a new admin; if an admin already
        exists and the supplied username differs, a warning is logged and the stored
        username is kept unchanged. When an admin already exists and a non-empty
        email is provided, the admin's email is updated.
        """
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT id FROM users WHERE role='admin' LIMIT 1"
            )
            row = await cur.fetchone()
            if row is not None:
                cur2 = await db.execute(
                    "SELECT username FROM users WHERE id=?", (row["id"],)
                )
                existing = await cur2.fetchone()
                if existing is not None and existing["username"] != username:
                    logger.warning(
                        "create_admin_if_missing called with username=%r but an "
                        "admin already exists as %r; keeping existing username",
                        username, existing["username"],
                    )
                if email:
                    await db.execute("UPDATE users SET email=? WHERE id=?", (email, row["id"]))
                    await db.commit()
                return row["id"], False
            uid = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO users(id,email,username,password_hash,created_at,role,status) VALUES(?,?,?,?,?,?,?)",
                (uid, email, username, password_hash, datetime.now(timezone.utc).isoformat(), "admin", "active"),
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
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute("PRAGMA foreign_keys = ON")
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

    async def get_user_agent_default_secrets(self, user_id: str) -> Optional[str]:
        """Return the encrypted per-user Agent-default secret payload."""
        async with aiosqlite.connect(self.path, timeout=30) as db:
            cur = await db.execute(
                "SELECT encrypted_payload FROM user_agent_default_secrets WHERE user_id=?",
                (user_id,),
            )
            row = await cur.fetchone()
            return row[0] if row is not None else None

    async def set_user_agent_default_secrets(
        self,
        user_id: str,
        encrypted_payload: str,
    ) -> None:
        """Upsert the encrypted per-user Agent-default secret payload."""
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                """INSERT INTO user_agent_default_secrets
                       (user_id, encrypted_payload, created_at, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       encrypted_payload=excluded.encrypted_payload,
                       updated_at=excluded.updated_at""",
                (user_id, encrypted_payload, now, now),
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
        total_size: int, total_chunks: int, chunk_size: int, expires_at_iso: str,
        placeholder_file_id: str,
    ) -> str:
        uid = str(uuid.uuid4())
        async with aiosqlite.connect(self.path, timeout=30) as db:
            await db.execute(
                """INSERT INTO upload_sessions
                (id, session_id, filename, expected_sha256, total_size, total_chunks,
                 chunk_size, received_chunks, received_mask, placeholder_file_id,
                 created_at, expires_at, completed)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uid, session_id, filename, expected_sha256, total_size, total_chunks,
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
        # Note: deliberately *not* filtering completed=0. A completed upload
        # whose finalize step crashed could leave chunk files on disk; those
        # records still need to be returned so ttl_cleanup can sweep them.
        async with aiosqlite.connect(self.path, timeout=30) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM upload_sessions WHERE expires_at < ?",
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
