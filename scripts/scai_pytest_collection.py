"""Pytest-side collection and phase evidence plugin for the ScAI runner."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

import pytest

from scripts.scai_test_tree import (
    ContractError,
    _hash_json,
    _normalize_detail,
    _phase_result,
    _phase_signature,
    _record_signature,
    _validate_phases,
)


SCHEMA = 1
EXPECTED_TOOLS = {
    "pytest": ("pytest", "9.1.1"),
    "pytest-asyncio": ("pytest_asyncio", "1.3.0"),
    "pytest-benchmark": ("pytest_benchmark", "5.2.3"),
}
_ACTIVE_EVIDENCE: "_Evidence | None" = None
_PENDING_WARNINGS: list[tuple[Any, str | None, str | None]] = []


def _atomic_write(path_text: str, payload: dict[str, Any]) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _make_readable(path_text: str) -> None:
    path = Path(path_text)
    if not path.exists():
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"unsafe pytest artifact: {path}")
        os.fchmod(fd, 0o644)
    finally:
        os.close(fd)


def _origin(module_name: str) -> str:
    module = importlib.import_module(module_name)
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str):
        raise RuntimeError(f"{module_name} has no module origin")
    path = os.path.realpath(origin)
    if not (path == "/opt/scai-test/venv" or path.startswith("/opt/scai-test/venv/")):
        raise RuntimeError(f"{module_name} was imported outside the immutable test venv: {path}")
    return path


def _candidate_tools() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, (module_name, expected_version) in EXPECTED_TOOLS.items():
        version = importlib.metadata.version(name)
        origin = _origin(module_name)
        if version != expected_version:
            raise RuntimeError(f"candidate tool identity mismatch: {name} {version} != {expected_version}")
        result[name] = {"version": version, "origin": origin}
    return result


def _collection_payload(session: pytest.Session) -> dict[str, Any]:
    nodeids = sorted(item.nodeid for item in session.items)
    candidate_tools = _candidate_tools()
    plugin_names = {name.lower() for name, _plugin in session.config.pluginmanager.list_name_plugin()}
    plugin_repr = " ".join(repr(plugin).lower() for plugin in session.config.pluginmanager.get_plugins())
    asyncio_loaded = "asyncio" in plugin_names or "pytest_asyncio" in plugin_names or "asyncio" in plugin_repr
    benchmark_loaded = "benchmark" in plugin_names or "pytest_benchmark" in plugin_names or "benchmark" in plugin_repr
    if not asyncio_loaded or not benchmark_loaded:
        raise RuntimeError("required asyncio and benchmark plugins are not loaded")
    effective_config = {
        "SCAI_ENABLE_SNAPSHOT_V2": {
            "value": os.environ.get("SCAI_ENABLE_SNAPSHOT_V2"),
            "source": "root conftest.py setdefault",
            "production_default": "disabled",
        }
    }
    if effective_config["SCAI_ENABLE_SNAPSHOT_V2"]["value"] != "true":
        raise RuntimeError("SCAI_ENABLE_SNAPSHOT_V2 is not enabled by the test conftest")
    return {
        "schema": SCHEMA,
        "complete": session.testsfailed == 0,
        "count": len(nodeids),
        "nodeids": nodeids,
        "nodeids_sha256": hashlib.sha256("".join(nodeid + "\n" for nodeid in nodeids).encode()).hexdigest(),
        "effective_config": effective_config,
        "candidate_tools": {**candidate_tools, "plugins": {"asyncio": asyncio_loaded, "benchmark": benchmark_loaded}},
    }


class _Evidence:
    def __init__(self, session: pytest.Session) -> None:
        self.session = session
        self.phases: dict[str, list[dict[str, Any]]] = {}
        self.warnings: list[dict[str, Any]] = []
        self._incomplete: dict[str, Any] | None = None

    def add_report(self, report: pytest.TestReport) -> None:
        detail = ""
        if report.outcome == "failed":
            try:
                detail = report.longreprtext
            except Exception:
                detail = str(getattr(report, "longrepr", ""))
        detail = _normalize_detail(detail)
        phase = {
            "when": report.when,
            "outcome": report.outcome,
            "wasxfail": getattr(report, "wasxfail", None),
            "detail": detail,
        }
        phase["signature"] = _phase_signature(phase)
        self.phases.setdefault(report.nodeid, []).append(phase)

    def add_warning(self, warning_message: Any, nodeid: str | None, when: str | None) -> None:
        category = getattr(warning_message, "category", Warning)
        message = _normalize_detail(str(getattr(warning_message, "message", warning_message)))
        value = {
            "nodeid": nodeid if nodeid else None,
            "when": when or "",
            "category": getattr(category, "__name__", str(category)),
            "message": message,
        }
        value["signature"] = _hash_json({
            "nodeid": value["nodeid"],
            "when": value["when"],
            "category": value["category"],
            "message": value["message"],
        })
        self.warnings.append(value)

    def _records(self, *, strict: bool) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for nodeid in sorted(self.phases):
            phases = self.phases[nodeid]
            normalized_phases = _validate_phases(nodeid, phases)
            outcome, detail, xfail, xpass = _phase_result(normalized_phases)
            record = {
                "nodeid": nodeid,
                "outcome": outcome,
                "detail": _normalize_detail(detail),
                "xfail": xfail,
                "xpass": xpass,
                "phases": normalized_phases,
            }
            record["signature"] = _record_signature(record)
            records.append(record)
        return records

    def _payload(self, records: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
        warnings = sorted(self.warnings, key=lambda warning: (
            warning["nodeid"] or "", warning["when"], warning["category"], warning["message"], warning["signature"]
        ))
        return {
            "schema": SCHEMA,
            "complete": complete,
            "count": len(records),
            "records_sha256": _hash_json(records),
            "warnings_sha256": _hash_json(warnings),
            "records": records,
            "warnings": warnings,
        }

    def _safe_incomplete_payload(self, message: str) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        for nodeid in sorted(self.phases):
            phases = self.phases[nodeid]
            try:
                normalized_phases = _validate_phases(nodeid, phases)
                outcome, detail, xfail, xpass = _phase_result(normalized_phases)
            except ContractError:
                normalized_phases = phases
                outcome, detail, xfail, xpass = "error", _normalize_detail(message), False, False
            record = {
                "nodeid": nodeid,
                "outcome": outcome,
                "detail": _normalize_detail(detail),
                "xfail": xfail,
                "xpass": xpass,
                "phases": normalized_phases,
            }
            record["signature"] = _record_signature(record)
            records.append(record)
        return self._payload(records, complete=False)

    def payload(self) -> dict[str, Any]:
        try:
            records = self._records(strict=True)
        except ContractError as exc:
            self._incomplete = self._safe_incomplete_payload(str(exc))
            raise pytest.UsageError(f"outcome evidence incomplete: {exc}") from exc
        return self._payload(records, complete=True)


def _write_outcomes(target: str, evidence: _Evidence) -> None:
    try:
        payload = evidence.payload()
    except pytest.UsageError:
        if evidence._incomplete is not None:
            _atomic_write(target, evidence._incomplete)
        raise
    _atomic_write(target, payload)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("scai-contract")
    group.addoption("--scai-collection-json", action="store", default=None)
    group.addoption("--scai-outcomes-json", action="store", default=None)


def pytest_collection_finish(session: pytest.Session) -> None:
    target = session.config.getoption("--scai-collection-json")
    if target:
        _atomic_write(target, _collection_payload(session))


def pytest_sessionstart(session: pytest.Session) -> None:
    global _ACTIVE_EVIDENCE
    evidence = _Evidence(session)
    for warning_message, nodeid, when in _PENDING_WARNINGS:
        evidence.add_warning(warning_message, nodeid, when)
    _PENDING_WARNINGS.clear()
    session._scai_evidence = evidence
    _ACTIVE_EVIDENCE = evidence


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _ACTIVE_EVIDENCE is not None:
        _ACTIVE_EVIDENCE.add_report(report)


def pytest_warning_recorded(
    warning_message: Any,
    when: str,
    nodeid: str,
    location: Any,
) -> None:
    if _ACTIVE_EVIDENCE is not None:
        _ACTIVE_EVIDENCE.add_warning(warning_message, nodeid, when)
    else:
        _PENDING_WARNINGS.append((warning_message, nodeid, when))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: pytest.ExitCode) -> None:
    target = session.config.getoption("--scai-outcomes-json")
    if target:
        _write_outcomes(target, session._scai_evidence)
    junit_target = session.config.getoption("junitxml", default=None)
    if junit_target:
        _make_readable(junit_target)


def pytest_unconfigure(config: pytest.Config) -> None:
    global _ACTIVE_EVIDENCE
    _ACTIVE_EVIDENCE = None
    _PENDING_WARNINGS.clear()
