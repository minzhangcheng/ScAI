#!/usr/bin/env python3
"""Immutable source-tree and test-outcome proofs for the ScAI runner.

This module deliberately uses only the Python standard library.  It is used by
the host-side runner before and after the container boundary and is also
importable by the contract tests.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence


class ContractError(ValueError):
    """A source, artifact, or outcome contract was not satisfied."""


@dataclass(frozen=True)
class ArtifactBytes:
    name: str
    payload: bytes
    size: int
    sha256: str


HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_STATES = {"modified", "untracked", "deleted", "added"}
SourceState = Literal["modified", "untracked", "deleted", "added"]


def _fail(message: str) -> None:
    raise ContractError(message)


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        _fail(
            "git command failed: "
            + " ".join(["git", *args])
            + ": "
            + proc.stderr.decode("utf-8", "replace").strip()
        )
    return proc.stdout


def _relative_path(path: str) -> str:
    if not path or path.startswith("/") or "\x00" in path:
        _fail(f"invalid repository-relative path: {path!r}")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail(f"non-canonical repository path: {path!r}")
    return path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_direct_children_nofollow(root: Path, expected_names: tuple[str, ...]) -> dict[str, ArtifactBytes]:
    """Read an exact direct-child file set once through no-follow descriptors."""
    if len(expected_names) != len(set(expected_names)) or any(
        not name or "/" in name or name in {".", ".."} for name in expected_names
    ):
        _fail("artifact child names are not exact direct names")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        _fail(f"artifact root is not a no-follow directory: {root}: {exc}")
    try:
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            _fail(f"artifact root is not a directory: {root}")
        actual_names = set(os.listdir(root_fd))
        if actual_names != set(expected_names):
            _fail(f"artifact child allowlist mismatch: {sorted(actual_names)}")
        result: dict[str, ArtifactBytes] = {}
        for name in expected_names:
            try:
                fd = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=root_fd,
                )
            except OSError as exc:
                _fail(f"unsafe artifact child {name}: {exc}")
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    _fail(f"unsafe artifact child {name}: not a regular single-link file")
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                while True:
                    block = os.read(fd, 1024 * 1024)
                    if not block:
                        break
                    chunks.append(block)
                    digest.update(block)
                payload = b"".join(chunks)
                result[name] = ArtifactBytes(name, payload, len(payload), digest.hexdigest())
            finally:
                os.close(fd)
        return result
    finally:
        os.close(root_fd)


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

def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _hash_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value) + b"\n")


def _nodeids_hash(nodeids: Sequence[str]) -> str:
    return _sha256_bytes("".join(nodeid + "\n" for nodeid in nodeids).encode("utf-8"))


def _normalize_detail(value: Any) -> str:
    if not isinstance(value, str):
        _fail("outcome detail must be a string")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"/(?:workspace/scai|tmp|private/tmp)(?:/[^\n ]*)?", "<path>", value)
    lines = value.split("\n")
    if lines:
        lines[0] = re.sub(r"[ \t]+", " ", lines[0]).strip()
    lines = [line.rstrip() for line in lines]
    return "\n".join(lines).rstrip()


def _phase_signature(phase: dict[str, Any]) -> str:
    content = {
        "when": phase["when"],
        "outcome": phase["outcome"],
        "wasxfail": phase["wasxfail"],
        "detail": phase["detail"],
    }
    return _sha256_bytes(_canonical_json(content))


def _record_signature(record: dict[str, Any]) -> str:
    content = {
        "nodeid": record["nodeid"],
        "outcome": record["outcome"],
        "detail": record["detail"],
        "xfail": record["xfail"],
        "xpass": record["xpass"],
        "phases": record["phases"],
    }
    return _sha256_bytes(_canonical_json(content))


def _validate_collection_nodeids(collection_nodeids: tuple[str, ...]) -> tuple[str, ...]:
    if any(not isinstance(nodeid, str) or not nodeid for nodeid in collection_nodeids):
        _fail("collection contains an empty or non-string nodeid")
    if tuple(sorted(collection_nodeids)) != collection_nodeids:
        _fail("collection nodeids must be sorted")
    if len(set(collection_nodeids)) != len(collection_nodeids):
        _fail("collection contains duplicate nodeids")
    return collection_nodeids


def _phase_result(phases: list[dict[str, Any]]) -> tuple[str, str, bool, bool]:
    """Apply the host-owned final phase state machine."""
    by_when = {phase["when"]: phase for phase in phases}
    setup = by_when.get("setup")
    call = by_when.get("call")
    teardown = by_when.get("teardown")
    xfail = any(phase["outcome"] == "skipped" and phase["wasxfail"] is not None for phase in phases)
    xpass = bool(call and call["wasxfail"] is not None and call["outcome"] in {"passed", "failed"})

    setup_failed = setup is not None and setup["outcome"] == "failed"
    teardown_failed = teardown is not None and teardown["outcome"] == "failed"
    call_failed = call is not None and call["outcome"] == "failed"
    if setup_failed or teardown_failed:
        if setup_failed:
            return "error", setup["detail"], xfail, xpass
        return "error", teardown["detail"], xfail, xpass
    if call_failed:
        return "failed", call["detail"], xfail, xpass
    skipped = next((phase for phase in phases if phase["outcome"] == "skipped"), None)
    if skipped is not None:
        return "skipped", skipped["detail"], xfail, xpass
    return "passed", call["detail"] if call is not None else "", xfail, xpass


def _validate_phases(nodeid: str, phases: Any) -> list[dict[str, Any]]:
    if not isinstance(phases, list) or not phases:
        _fail(f"{nodeid}: phases must be a non-empty list")
    allowed_when = {"setup", "call", "teardown"}
    allowed_outcome = {"passed", "failed", "skipped"}
    seen: set[str] = set()
    xfail_phases = 0
    normalized: list[dict[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, dict) or set(phase) != {"when", "outcome", "wasxfail", "detail", "signature"}:
            _fail(f"{nodeid}: phase keys are not exact")
        when = phase["when"]
        outcome = phase["outcome"]
        if when not in allowed_when or when in seen:
            _fail(f"{nodeid}: phase names are invalid or duplicated")
        if outcome not in allowed_outcome:
            _fail(f"{nodeid}: phase outcome is invalid")
        if phase["wasxfail"] is not None and not isinstance(phase["wasxfail"], str):
            _fail(f"{nodeid}: phase wasxfail must be null or a string")
        if phase["wasxfail"] is not None:
            xfail_phases += 1
            if outcome == "skipped":
                pass
            elif when == "call" and outcome in {"passed", "failed"}:
                pass
            else:
                _fail(f"{nodeid}: wasxfail is illegal for this phase outcome")
        detail = _normalize_detail(phase["detail"])
        candidate = {
            "when": when,
            "outcome": outcome,
            "wasxfail": phase["wasxfail"],
            "detail": detail,
        }
        if phase["signature"] != _phase_signature(candidate):
            _fail(f"{nodeid}: phase signature mismatch")
        candidate["signature"] = phase["signature"]
        normalized.append(candidate)
        seen.add(when)
    if "setup" not in seen:
        _fail(f"{nodeid}: setup phase is required")
    if "teardown" not in seen:
        _fail(f"{nodeid}: teardown phase is required")
    if xfail_phases > 1:
        _fail(f"{nodeid}: more than one phase carries wasxfail")
    if normalized[0]["when"] != "setup":
        _fail(f"{nodeid}: setup must precede other phases")
    if "call" in seen and [p["when"] for p in normalized].index("call") < [p["when"] for p in normalized].index("setup"):
        _fail(f"{nodeid}: call precedes setup")
    if "teardown" in seen and "call" in seen and [p["when"] for p in normalized].index("teardown") < [p["when"] for p in normalized].index("call"):
        _fail(f"{nodeid}: teardown precedes call")
    by_when = {phase["when"]: phase for phase in normalized}
    setup_outcome = by_when["setup"]["outcome"]
    if setup_outcome == "passed" and "call" not in seen:
        _fail(f"{nodeid}: a passing setup must have a call phase")
    if setup_outcome != "passed" and "call" in seen:
        _fail(f"{nodeid}: call phase is illegal after a non-passed setup")
    return normalized


def _recompute_record(record: dict[str, Any]) -> dict[str, Any]:
    nodeid = record["nodeid"]
    phases = _validate_phases(nodeid, record["phases"])
    status, detail, xfail, xpass = _phase_result(phases)
    if record["outcome"] != status:
        _fail(f"{nodeid}: record outcome does not match phase precedence")
    if record["detail"] != _normalize_detail(detail):
        _fail(f"{nodeid}: record detail does not match call/setup/teardown detail")
    if record["xfail"] != xfail or record["xpass"] != xpass:
        _fail(f"{nodeid}: xfail/xpass flags do not match phases")
    normalized = {
        "nodeid": nodeid,
        "outcome": status,
        "detail": _normalize_detail(record["detail"]),
        "xfail": xfail,
        "xpass": xpass,
        "phases": phases,
    }
    if record["signature"] != _record_signature(normalized):
        _fail(f"{nodeid}: record signature mismatch")
    normalized["signature"] = record["signature"]
    return normalized


def project_pytest_9_1_1_junit(records: tuple[dict[str, Any], ...]) -> dict[str, int]:
    """Project pytest 9.1.1 report phases to its JUnit testcase counters.

    Pytest's JUnit plugin counts a call-pass followed by a teardown failure as
    one testcase, while a call failure followed by teardown failure exposes two
    failure/error elements.  The explicit correction below preserves that
    observed 9.1.1 projection without trusting host XML node identity.
    """
    passed = failures = errors = skipped = double_fail_correction = 0
    for record in records:
        reporter_seen = False
        open_call_fail = False
        phases = record.get("phases")
        if not isinstance(phases, list):
            _fail("JUnit projection requires complete record phase lists")
        for phase in phases:
            when = phase["when"]
            outcome = phase["outcome"]
            wasxfail = phase.get("wasxfail")
            if when == "call":
                if outcome == "passed":
                    passed += 1
                elif outcome == "failed":
                    if wasxfail is not None:
                        skipped += 1
                    else:
                        failures += 1
                    open_call_fail = True
                elif outcome == "skipped":
                    skipped += 1
                reporter_seen = True
            elif outcome == "failed":
                errors += 1
                if when == "teardown" and not open_call_fail and reporter_seen:
                    double_fail_correction += 1
                reporter_seen = True
            elif outcome == "skipped":
                skipped += 1
                reporter_seen = True
    tests = passed + failures + errors + skipped - double_fail_correction
    return {
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "double_fail_correction": double_fail_correction,
        "tests": tests,
    }


def validate_collection(payload: bytes) -> dict[str, Any]:
    """Validate the exact collection and candidate-tool identity schema."""
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        _fail(f"collection is not valid UTF-8 JSON: {exc}")
    required = {"schema", "complete", "count", "nodeids", "nodeids_sha256", "effective_config", "candidate_tools"}
    if not isinstance(document, dict) or set(document) != required:
        _fail("collection keys are not exact")
    if document["schema"] != 1 or document["complete"] is not True:
        _fail("collection is incomplete or uses an unsupported schema")
    nodeids = document["nodeids"]
    if not isinstance(nodeids, list) or any(not isinstance(nodeid, str) or not nodeid for nodeid in nodeids):
        _fail("collection nodeids are invalid")
    if nodeids != sorted(nodeids) or len(nodeids) != len(set(nodeids)):
        _fail("collection nodeids are not sorted and unique")
    if not isinstance(document["count"], int) or isinstance(document["count"], bool) or document["count"] != len(nodeids) or not nodeids:
        _fail("collection count is invalid")
    expected_hash = _nodeids_hash(tuple(nodeids))
    if document["nodeids_sha256"] != expected_hash:
        _fail("collection node ID hash mismatch")
    if document["effective_config"] != {
        "SCAI_ENABLE_SNAPSHOT_V2": {
            "value": "true",
            "source": "root conftest.py setdefault",
            "production_default": "disabled",
        }
    }:
        _fail("collection effective configuration is not exact")
    candidate = document["candidate_tools"]
    required_tools = {"pytest", "pytest-asyncio", "pytest-benchmark", "plugins"}
    if not isinstance(candidate, dict) or set(candidate) != required_tools:
        _fail("collection candidate-tool keys are not exact")
    expected_versions = {"pytest": "9.1.1", "pytest-asyncio": "1.3.0", "pytest-benchmark": "5.2.3"}
    for name, expected_version in expected_versions.items():
        tool = candidate[name]
        if not isinstance(tool, dict) or set(tool) != {"version", "origin"}:
            _fail(f"collection candidate-tool schema is invalid: {name}")
        if tool["version"] != expected_version or not isinstance(tool["origin"], str) or not tool["origin"].startswith("/opt/scai-test/venv/"):
            _fail(f"collection candidate-tool identity mismatch: {name}")
    plugins = candidate["plugins"]
    if plugins != {"asyncio": True, "benchmark": True}:
        _fail("collection plugin identity is not exact")
    return document


def validate_outcomes(payload: bytes, collection_nodeids: tuple[str, ...]) -> dict[str, Any]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        _fail(f"outcomes are not valid UTF-8 JSON: {exc}")
    if not isinstance(document, dict) or set(document) != {
        "schema", "complete", "count", "records_sha256", "warnings_sha256", "records", "warnings"
    }:
        _fail("outcomes top-level keys are not exact")
    if document["schema"] != 1 or document["complete"] is not True:
        _fail("outcomes are incomplete or use an unsupported schema")
    expected_nodes = _validate_collection_nodeids(tuple(collection_nodeids))
    records = document["records"]
    warnings = document["warnings"]
    if not isinstance(records, list) or not isinstance(warnings, list):
        _fail("outcome records and warnings must be lists")
    if isinstance(document["count"], bool) or not isinstance(document["count"], int):
        _fail("outcome count must be an integer")
    if any(not isinstance(record, dict) for record in records):
        _fail("outcome records must be objects")
    if document["count"] != len(records) or tuple(record.get("nodeid") for record in records) != expected_nodes:
        _fail("outcome records are not an exact collection bijection")
    if not isinstance(document["records_sha256"], str) or not HEX64.fullmatch(document["records_sha256"]):
        _fail("records hash is invalid")
    if not isinstance(document["warnings_sha256"], str) or not HEX64.fullmatch(document["warnings_sha256"]):
        _fail("warnings hash is invalid")
    normalized_records: list[dict[str, Any]] = []
    for raw in records:
        if not isinstance(raw, dict) or set(raw) != {
            "nodeid", "outcome", "detail", "signature", "xfail", "xpass", "phases"
        }:
            _fail("outcome record keys are not exact")
        nodeid = raw["nodeid"]
        if not isinstance(nodeid, str) or not nodeid:
            _fail("outcome record nodeid is invalid")
        if raw["outcome"] not in {"passed", "failed", "error", "skipped"}:
            _fail(f"{nodeid}: outcome status is invalid")
        if not isinstance(raw["signature"], str) or not HEX64.fullmatch(raw["signature"]):
            _fail(f"{nodeid}: record signature is invalid")
        if not isinstance(raw["xfail"], bool) or not isinstance(raw["xpass"], bool):
            _fail(f"{nodeid}: xfail/xpass must be booleans")
        record = {
            "nodeid": nodeid,
            "outcome": raw["outcome"],
            "detail": _normalize_detail(raw["detail"]),
            "signature": raw["signature"],
            "xfail": raw["xfail"],
            "xpass": raw["xpass"],
            "phases": raw["phases"],
        }
        normalized_records.append(_recompute_record(record))
    if _hash_json(normalized_records) != document["records_sha256"]:
        _fail("records hash does not match canonical records")
    normalized_warnings: list[dict[str, Any]] = []
    for warning in warnings:
        if not isinstance(warning, dict) or set(warning) != {"nodeid", "when", "category", "message", "signature"}:
            _fail("warning keys are not exact")
        if warning["nodeid"] is not None and not isinstance(warning["nodeid"], str):
            _fail("warning nodeid must be null or a string")
        if not all(isinstance(warning[key], str) for key in ("when", "category", "message", "signature")):
            _fail("warning fields must be strings")
        normalized = {
            "nodeid": warning["nodeid"],
            "when": warning["when"],
            "category": warning["category"],
            "message": _normalize_detail(warning["message"]),
        }
        if warning["signature"] != _hash_json(normalized):
            _fail("warning signature mismatch")
        normalized["signature"] = warning["signature"]
        normalized_warnings.append(normalized)
    warning_order = lambda value: (value["nodeid"] or "", value["when"], value["category"], value["message"], value["signature"])
    if normalized_warnings != sorted(normalized_warnings, key=warning_order):
        _fail("warnings are not sorted canonically")
    if _hash_json(normalized_warnings) != document["warnings_sha256"]:
        _fail("warnings hash does not match canonical warnings")
    status_counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    failed_nodes: list[dict[str, Any]] = []
    skipped_nodes: list[dict[str, Any]] = []
    for record in normalized_records:
        status_counts[record["outcome"]] += 1
        if record["outcome"] in {"failed", "error"}:
            failed_nodes.append({
                "nodeid": record["nodeid"],
                "outcome": record["outcome"],
                "signature": record["signature"],
                "detail": record["detail"],
            })
        if record["outcome"] == "skipped":
            skipped_nodes.append({
                "nodeid": record["nodeid"],
                "signature": record["signature"],
                "reason": record["detail"],
            })
    counts = {
        "collected": len(normalized_records),
        "passed": status_counts["passed"],
        "failed": status_counts["failed"],
        "errors": status_counts["error"],
        "skipped": status_counts["skipped"],
        "xfail": sum(1 for record in normalized_records if record["xfail"]),
        "xpass": sum(1 for record in normalized_records if record["xpass"]),
        "warnings": len(normalized_warnings),
    }
    projection = project_pytest_9_1_1_junit(tuple(normalized_records))
    return {
        "schema": 1,
        "collection_sha256": _nodeids_hash(expected_nodes),
        "records_sha256": document["records_sha256"],
        "warnings_sha256": document["warnings_sha256"],
        "records": normalized_records,
        "warnings": normalized_warnings,
        "counts": counts,
        "junit_projection": projection,
        "failed_nodes": sorted(failed_nodes, key=lambda value: value["nodeid"]),
        "skipped_nodes": sorted(skipped_nodes, key=lambda value: value["nodeid"]),
    }


def parse_junit_bytes(payload: bytes) -> dict[str, int]:
    try:
        root = ET.fromstring(payload)
    except (UnicodeDecodeError, ET.ParseError) as exc:
        _fail(f"cannot parse JUnit XML: {exc}")
    suite_nodes = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suite_nodes:
        _fail("JUnit XML has no testsuite")
    result = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suite_nodes:
        counters: dict[str, int] = {}
        for key in result:
            value = suite.attrib.get(key)
            if value is None:
                if key == "tests":
                    _fail("JUnit tests attribute is missing")
                continue
            try:
                parsed = int(value)
            except ValueError:
                _fail(f"JUnit {key} attribute is not an integer")
            if parsed < 0:
                _fail(f"JUnit {key} attribute is negative")
            counters[key] = parsed
        for key in result:
            result[key] += counters.get(key, 0)
    if result["tests"] <= 0:
        _fail("JUnit XML reports zero tests")
    return result


def parse_junit(path: Path) -> dict[str, int]:
    """Compatibility wrapper that performs one no-follow direct-child read."""
    artifacts = read_direct_children_nofollow(path.parent, (path.name,))
    return parse_junit_bytes(artifacts[path.name].payload)


def reject_kernel_gated_live_skips(profile: str, outcome_details: dict[str, Any]) -> None:
    """Reject only the exact kernel environment-gated live-test skip reason."""
    if profile != "kernel":
        return
    gated_reason = "Set SCAI_RUN_KERNEL_TESTS=1"
    for entry in outcome_details.get("skipped_nodes", []):
        reason = entry.get("reason") if isinstance(entry, dict) else None
        if isinstance(reason, str) and gated_reason in _normalize_detail(reason):
            _fail(f"kernel profile contains an environment-gated live skip: {entry.get('nodeid')}")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    return True


def publish_evidence(results_root: Path, run_id: str, artifacts: dict[str, ArtifactBytes]) -> Path:
    """Exclusively publish a validated byte map without deleting destinations."""
    results_root.mkdir(parents=True, exist_ok=True)
    if results_root.is_symlink() or not results_root.is_dir():
        _fail("test-results root is not an exact directory")
    lock_path = results_root / ".publication.lock"
    lock_fd = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    incoming = results_root / f".incoming-{run_id}"
    final = results_root / run_id
    try:
        os.fchmod(lock_fd, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if _lexists(incoming) or _lexists(final):
            _fail("evidence publication destination already exists")
        os.mkdir(incoming, 0o700)
        incoming_fd = os.open(incoming, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            for name in sorted(artifacts):
                if not name or "/" in name or name in {".", ".."}:
                    _fail(f"invalid evidence artifact name: {name!r}")
                item = artifacts[name]
                if item.name != name or item.size != len(item.payload) or item.sha256 != _sha256_bytes(item.payload):
                    _fail(f"evidence artifact bytes are not self-consistent: {name}")
                fd = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=incoming_fd,
                )
                try:
                    offset = 0
                    while offset < len(item.payload):
                        offset += os.write(fd, item.payload[offset:])
                    os.fsync(fd)
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != item.size:
                        _fail(f"published artifact became unsafe: {name}")
                finally:
                    os.close(fd)
            os.fsync(incoming_fd)
        finally:
            os.close(incoming_fd)
        _fsync_directory(results_root)
        if _lexists(final):
            _fail("evidence final destination appeared during publication")
        os.rename(incoming, final)
        _fsync_directory(results_root)
        return final
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _cli() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name, kind in (("worktree", "worktree"), ("index", "index")):
        command = sub.add_parser(name)
        command.add_argument("--base", required=True)
        command.add_argument("--output", type=Path, required=True)
    commit_parser = sub.add_parser("commit")
    commit_parser.add_argument("--base", required=True)
    commit_parser.add_argument("--commit", required=True)
    commit_parser.add_argument("--output", type=Path, required=True)
    summary_parser = sub.add_parser("compare-summary")
    summary_parser.add_argument("--tested", type=Path, required=True)
    summary_parser.add_argument("--post", type=Path, required=True)
    summary_parser.add_argument("--summary-path", type=Path, required=True)
    index_parser = sub.add_parser("compare-index")
    index_parser.add_argument("--post", type=Path, required=True)
    index_parser.add_argument("--index", type=Path, required=True)
    verify_parser = sub.add_parser("verify-commit")
    verify_parser.add_argument("--base", required=True)
    verify_parser.add_argument("--index", type=Path, required=True)
    verify_parser.add_argument("--subject", required=True)
    outcomes_parser = sub.add_parser("validate-outcomes")
    outcomes_parser.add_argument("--collection", type=Path, required=True)
    outcomes_parser.add_argument("--outcomes", type=Path, required=True)
    outcomes_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        repo = Path.cwd()
        if args.command in {"worktree", "index", "commit"}:
            if args.command == "worktree":
                result = build_worktree_manifest(repo, args.base)
            elif args.command == "index":
                result = build_index_manifest(repo, args.base)
            else:
                result = build_commit_manifest(repo, args.base, args.commit)
            _write_json(args.output, result)
        elif args.command == "compare-summary":
            compare_post_summary(args.tested, args.post, args.summary_path)
        elif args.command == "compare-index":
            compare_index(args.post, args.index)
        elif args.command == "verify-commit":
            print(verify_final_commit(repo, args.base, args.index, args.subject))
        elif args.command == "validate-outcomes":
            collection = json.loads(args.collection.read_text(encoding="utf-8"))
            nodeids = tuple(collection["nodeids"])
            _write_json(args.output, validate_outcomes(args.outcomes.read_bytes(), nodeids))
    except ContractError as exc:
        print(f"contract error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
