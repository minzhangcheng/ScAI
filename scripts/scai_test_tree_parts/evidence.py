"""Collection, outcome, JUnit, and evidence publication proofs."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence

from .discovery import ArtifactBytes, HEX64, _fail, _sha256_bytes, read_direct_children_nofollow

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
    effective = document["effective_config"]
    # The build has one configuration, so the recorded set is empty and must be
    # exactly empty: a key appearing here means a switch was reintroduced
    # without extending this contract to say what it measured.
    if effective != {}:
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
