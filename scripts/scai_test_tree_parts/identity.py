"""Git source identity and manifest verification."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from .discovery import (
    HEX64, SOURCE_STATES, ContractError, SourceState, _fail, _git, _relative_path,
    _sha256_bytes, _sha256_file,
)

def _mode_for_stat(st: os.stat_result) -> str:
    mode = stat.S_IMODE(st.st_mode)
    if stat.S_ISREG(st.st_mode):
        if mode & 0o111:
            return "100755"
        return "100644"
    if stat.S_ISLNK(st.st_mode):
        return "120000"
    _fail("directories and special files are not valid source entries")
    raise AssertionError


def _entry_key(entry: dict[str, Any]) -> tuple[str, str]:
    return (str(entry["path"]), str(entry["state"]))


def _canonical_entries(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            _fail("manifest entry must be an object")
        if not isinstance(entry.get("path"), str):
            _fail("manifest entry path is missing")
        result.append(dict(entry))
    paths = [str(entry["path"]) for entry in result]
    if paths != sorted(paths):
        _fail("manifest entries are not sorted")
    if len(paths) != len(set(paths)):
        _fail("manifest contains duplicate paths")
    return result


def _manifest_digest(entries: Sequence[dict[str, Any]]) -> str:
    payload = b"".join(
        json.dumps(entry, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
        for entry in entries
    )
    return _sha256_bytes(payload)


def _manifest(kind: str, base: str, entries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    canonical = _canonical_entries(entries)
    return {
        "schema": 1,
        "kind": kind,
        "base": base,
        "digest": _manifest_digest(canonical),
        "entries": canonical,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _validate_manifest_payload(payload: Any, expected_kind: str, expected_base: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        _fail("manifest must be an object")
    if set(payload) != {"schema", "kind", "base", "digest", "entries"}:
        _fail("manifest keys are not exact")
    if payload["schema"] != 1 or payload["kind"] != expected_kind:
        _fail("manifest schema or kind mismatch")
    if payload["base"] != expected_base:
        _fail("manifest base mismatch")
    digest = payload["digest"]
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        _fail("manifest digest must be a lowercase SHA-256")
    entries = payload["entries"]
    if not isinstance(entries, list):
        _fail("manifest entries must be a list")
    canonical = _canonical_entries(entries)
    for entry in canonical:
        if not isinstance(entry, dict):
            _fail("manifest entry must be an object")
        path_name = entry.get("path")
        state = entry.get("state")
        if not isinstance(path_name, str) or _relative_path(path_name) != path_name:
            _fail("manifest entry path is invalid")
        if state not in SOURCE_STATES:
            _fail(f"manifest entry state is invalid: {state!r}")
        if state == "deleted":
            if set(entry) != {"path", "state", "type", "mode", "marker"}:
                _fail("deleted entry keys are not exact")
            if entry["type"] != "deleted" or entry["mode"] is not None or entry["marker"] != "DELETED":
                _fail("deleted entry marker is invalid")
        else:
            if set(entry) != {"path", "state", "type", "mode", "sha256"}:
                _fail("present entry keys are not exact")
            if entry["type"] not in {"regular", "symlink"}:
                _fail("present entry type is invalid")
            if entry["mode"] not in {"100644", "100755", "120000"}:
                _fail("present entry mode is invalid")
            if entry["type"] == "symlink" and entry["mode"] != "120000":
                _fail("symlink must have mode 120000")
            if entry["type"] == "regular" and entry["mode"] not in {"100644", "100755"}:
                _fail("regular file mode is invalid")
            if not isinstance(entry["sha256"], str) or not HEX64.fullmatch(entry["sha256"]):
                _fail("present entry hash is invalid")
    if _manifest_digest(canonical) != digest:
        _fail("manifest digest does not match entries")
    return {"schema": 1, "kind": expected_kind, "base": expected_base, "digest": digest, "entries": canonical}


def load_manifest(path: Path, expected_kind: str, expected_base: str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _fail(f"cannot read manifest {path}: {exc}")
    return _validate_manifest_payload(payload, expected_kind, expected_base)


def _coerce_manifest(value: Any, expected_kind: str, expected_base: str | None = None) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if expected_base is None:
            try:
                expected_base = json.loads(path.read_text(encoding="utf-8"))["base"]
            except (OSError, KeyError, TypeError, ValueError) as exc:
                _fail(f"cannot determine manifest base: {exc}")
        return load_manifest(path, expected_kind, expected_base)
    if not isinstance(value, dict):
        _fail("manifest value must be a mapping or JSON path")
    if expected_base is None:
        expected_base = value.get("base")
    if not isinstance(expected_base, str):
        _fail("manifest base is missing")
    return _validate_manifest_payload(value, expected_kind, expected_base)


def _present_entry(repo: Path, relative: str, state: str) -> dict[str, Any]:
    relative = _relative_path(relative)
    path = repo / relative
    try:
        st = path.lstat()
    except FileNotFoundError:
        _fail(f"manifest path disappeared: {relative}")
    mode = _mode_for_stat(st)
    if stat.S_ISLNK(st.st_mode):
        data = os.readlink(path).encode("utf-8", "surrogateescape")
        kind = "symlink"
    else:
        data = path.read_bytes()
        kind = "regular"
    return {
        "path": relative,
        "state": state,
        "type": kind,
        "mode": mode,
        "sha256": _sha256_bytes(data),
    }


def _deleted_entry(relative: str) -> dict[str, Any]:
    return {"path": _relative_path(relative), "state": "deleted", "type": "deleted", "mode": None, "marker": "DELETED"}


def _parse_nul_records(data: bytes) -> list[str]:
    if not data:
        return []
    if not data.endswith(b"\x00"):
        _fail("git NUL output is not terminated")
    return [record.decode("utf-8", "surrogateescape") for record in data[:-1].split(b"\x00")]


def _worktree_changes(repo: Path, base: str) -> list[tuple[str, str]]:
    records = _parse_nul_records(_git(repo, "diff", "--name-status", "-z", "--no-renames", base, "--"))
    result: list[tuple[str, str]] = []
    index = 0
    while index < len(records):
        status = records[index]
        index += 1
        if index >= len(records):
            _fail(f"malformed git name-status record: {status!r}")
        code = status.split("\t", 1)[0]
        relative = records[index]
        index += 1
        relative = _relative_path(relative)
        if code == "D":
            result.append((relative, "deleted"))
        elif code in {"M", "A"}:
            result.append((relative, "modified" if code == "M" else "added"))
        else:
            _fail(f"unsupported git worktree status {code!r}")
    untracked = _parse_nul_records(_git(repo, "ls-files", "--others", "--exclude-standard", "-z"))
    for relative in untracked:
        result.append((_relative_path(relative), "untracked"))
    return result


def build_worktree_manifest(repo: Path, base: str) -> dict[str, Any]:
    changes = _worktree_changes(repo, base)
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    for relative, state in changes:
        if relative in seen:
            _fail(f"worktree path appears more than once: {relative}")
        seen.add(relative)
        entries.append(_deleted_entry(relative) if state == "deleted" else _present_entry(repo, relative, state))
    entries.sort(key=lambda entry: entry["path"])
    return _manifest("worktree", base, entries)


def _parse_raw_diff(data: bytes) -> list[tuple[str, str, str]]:
    records = _parse_nul_records(data)
    result: list[tuple[str, str, str]] = []
    for record in records:
        if not record.startswith(":"):
            _fail(f"malformed raw diff header: {record!r}")
        fields = record.split(" ")
        if len(fields) != 5:
            _fail(f"malformed raw diff fields: {record!r}")
        old_mode, new_mode, _old_oid, _new_oid, status = fields
        result.append((new_mode, status, record))
    return result


def _raw_paths(data: bytes) -> list[tuple[str, str, str]]:
    records = _parse_nul_records(data)
    result: list[tuple[str, str, str]] = []
    index = 0
    while index < len(records):
        header = records[index]
        index += 1
        if not header.startswith(":"):
            _fail(f"malformed raw diff header: {header!r}")
        fields = header.split(" ")
        if len(fields) != 5:
            _fail(f"malformed raw diff fields: {header!r}")
        new_mode, status = fields[1], fields[4]
        if index >= len(records):
            _fail("raw diff record has no path")
        relative = _relative_path(records[index])
        index += 1
        if status.startswith("R") or status.startswith("C"):
            if index >= len(records):
                _fail("rename/copy raw diff record has no second path")
            _fail("rename/copy entries are forbidden; use --no-renames")
        result.append((relative, new_mode, status))
    return result


def _git_object_entry(repo: Path, relative: str, state: str, ref: str, mode: str) -> dict[str, Any]:
    spec = f":{relative}" if ref == ":" else f"{ref}:{relative}"
    data = _git(repo, "show", spec)
    if mode == "120000":
        kind = "symlink"
    elif mode in {"100644", "100755"}:
        kind = "regular"
    else:
        _fail(f"unsupported git object mode {mode!r}")
    return {"path": relative, "state": state, "type": kind, "mode": mode, "sha256": _sha256_bytes(data)}


def _build_raw_manifest(repo: Path, base: str, kind: str, commit: str | None = None) -> dict[str, Any]:
    if kind == "index":
        raw = _git(repo, "diff", "--cached", "--raw", "-z", "--no-renames", base, "--")
        ref = ":"
    else:
        if not commit:
            _fail("commit manifest requires a commit")
        raw = _git(repo, "diff-tree", "-r", "--no-commit-id", "--raw", "-z", "--no-renames", base, commit, "--")
        ref = commit
    entries: list[dict[str, Any]] = []
    for relative, new_mode, status in _raw_paths(raw):
        if status == "D":
            entries.append(_deleted_entry(relative))
            continue
        if status not in {"M", "A"}:
            _fail(f"unsupported raw diff status {status!r}")
        state = "added" if status == "A" else "modified"
        entries.append(_git_object_entry(repo, relative, state, ref, new_mode))
    entries.sort(key=lambda entry: entry["path"])
    return _manifest(kind, base, entries)


def build_index_manifest(repo: Path, base: str) -> dict[str, Any]:
    return _build_raw_manifest(repo, base, "index")


def build_commit_manifest(repo: Path, base: str, commit: str) -> dict[str, Any]:
    return _build_raw_manifest(repo, base, "commit", commit)


def compare_post_summary(tested: Any, post: Any, summary_path: str | Path) -> None:
    tested_manifest = _coerce_manifest(tested, "worktree")
    post_manifest = _coerce_manifest(post, "worktree", tested_manifest["base"])
    summary = _relative_path(str(summary_path).replace("\\", "/"))
    tested_entries = {entry["path"]: entry for entry in tested_manifest["entries"]}
    post_entries = {entry["path"]: entry for entry in post_manifest["entries"]}
    if summary in tested_entries:
        _fail("summary path was already present in tested source manifest")
    if set(post_entries) != set(tested_entries) | {summary}:
        _fail("post-run source manifest does not contain exactly one summary entry")
    for path_name, entry in tested_entries.items():
        if post_entries[path_name] != entry:
            _fail(f"tested source changed after the run: {path_name}")
    if post_entries[summary]["state"] != "untracked":
        _fail("summary entry must be untracked")


def compare_index(post: Any, index: Any) -> None:
    post_manifest = _coerce_manifest(post, "worktree")
    index_manifest = _coerce_manifest(index, "index", post_manifest["base"])
    projected = []
    for entry in post_manifest["entries"]:
        projected.append({**entry, "state": "added"} if entry["state"] == "untracked" else entry)
    projected.sort(key=lambda entry: entry["path"])
    expected = _manifest("index", post_manifest["base"], projected)
    if expected["entries"] != index_manifest["entries"]:
        _fail("index does not exactly contain the validated post-run tree")


def _commit_parents(repo: Path, commit: str) -> list[str]:
    text = _git(repo, "show", "-s", "--format=%P", commit).decode().strip()
    return text.split() if text else []


def verify_final_commit(repo: Path, base: str, verified_index: Any, subject: str) -> str:
    index_manifest = _coerce_manifest(verified_index, "index", base)
    head = _git(repo, "rev-parse", "HEAD").decode().strip()
    if _commit_parents(repo, head) != [base]:
        _fail("final commit must have exactly the fixed base as its sole parent")
    commit_manifest = build_commit_manifest(repo, base, head)
    if commit_manifest["entries"] != index_manifest["entries"]:
        _fail("final commit tree does not match the verified index")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all").decode()
    if status:
        _fail("checkout is not clean after the final commit")
    format_text = _git(repo, "show", "-s", "--format=%B", head).decode("utf-8", "replace")
    lines = format_text.splitlines()
    if not lines or lines[0] != subject or any(line.strip() for line in lines[1:]):
        _fail("final commit message is not the exact requested subject")
    lowered = format_text.lower()
    forbidden = ("co-authored-by", "generated with", "written by", "codex", "claude", "openai", "agent", "bot")
    if any(token in lowered for token in forbidden):
        _fail("final commit contains forbidden attribution")
    return head


# ---------------------------------------------------------------------------
# Collection/outcome proofs
