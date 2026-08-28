"""Vitest result parsing for frontend test evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import fail, require_regular, sha256_bytes

VITEST_FIELDS = (
    "numTotalTestSuites", "numPassedTestSuites", "numFailedTestSuites", "numPendingTestSuites",
    "numTotalTests", "numPassedTests", "numFailedTests", "numPendingTests", "numTodoTests",
)

def parse_vitest(path: Path, exit_code: int) -> dict[str, Any]:
    require_regular(path, "Vitest result")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail("Vitest result", f"machine result is malformed: {exc}")
    if not isinstance(value, dict):
        fail("Vitest result", "machine result is not an object")
    for key in VITEST_FIELDS:
        if key not in value or isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0:
            fail("Vitest result", f"missing or invalid count: {key}")
    if not isinstance(value.get("success"), bool) or not isinstance(value.get("testResults"), list):
        fail("Vitest result", "success or testResults is malformed")
    raw_suite_counts = {
        "numTotalTestSuites": value["numTotalTestSuites"],
        "numPassedTestSuites": value["numPassedTestSuites"],
        "numFailedTestSuites": value["numFailedTestSuites"],
        "numPendingTestSuites": value["numPendingTestSuites"],
    }
    if raw_suite_counts["numTotalTestSuites"] != (
        raw_suite_counts["numPassedTestSuites"]
        + raw_suite_counts["numFailedTestSuites"]
        + raw_suite_counts["numPendingTestSuites"]
    ):
        fail("Vitest result", "top-level suite count arithmetic drift")
    suites: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    file_counts = {"passed": 0, "failed": 0, "pending": 0}
    assertion_counts = {"passed": 0, "failed": 0, "pending": 0, "todo": 0}
    diagnostics: list[dict[str, str]] = []
    for raw_suite in value["testResults"]:
        if not isinstance(raw_suite, dict) or not isinstance(raw_suite.get("name"), str) or not raw_suite["name"] or raw_suite["name"] in seen_names:
            fail("Vitest result", "suite names must be nonempty and unique")
        name = raw_suite["name"]
        seen_names.add(name)
        status = raw_suite.get("status")
        assertions = raw_suite.get("assertionResults")
        message = "" if raw_suite.get("message") is None else " ".join(str(raw_suite.get("message")).split())
        if status not in {"passed", "failed", "pending"} or not isinstance(assertions, list):
            fail("Vitest result", f"invalid suite: {name}")
        if status in {"passed", "pending"} and not assertions:
            fail("Vitest result", f"empty assertion list for {status} suite: {name}")
        if status == "failed" and not assertions:
            if not message:
                fail("Vitest result", f"failed suite has no diagnostic: {name}")
            diagnostics.append({"suite": name, "message": message, "signature": sha256_bytes(message.encode()), "kind": "suite-level import/setup/transform application failure"})
        file_counts[status] += 1
        detail_assertions: list[dict[str, Any]] = []
        for assertion in assertions:
            if not isinstance(assertion, dict) or assertion.get("status") not in assertion_counts:
                fail("Vitest result", f"invalid assertion in suite: {name}")
            assertion_counts[assertion["status"]] += 1
            detail_assertions.append({"ancestorTitles": assertion.get("ancestorTitles", []), "fullName": assertion.get("fullName", ""), "status": assertion["status"], "title": assertion.get("title", "")})
        suites.append({"name": name, "status": status, "message": message, "assertions": detail_assertions})
    test_counts = {
        "numTotalTests": sum(assertion_counts.values()),
        "numPassedTests": assertion_counts["passed"],
        "numFailedTests": assertion_counts["failed"],
        "numPendingTests": assertion_counts["pending"],
        "numTodoTests": assertion_counts["todo"],
    }
    for key, expected in test_counts.items():
        if value[key] != expected:
            fail("Vitest result", f"top-level count drift: {key}")
    top_failed = raw_suite_counts["numFailedTestSuites"] > 0
    file_failed = file_counts["failed"] > 0
    if top_failed != file_failed:
        fail("Vitest result", "failed file records do not agree with top-level suite failures")
    expected_success = not top_failed and assertion_counts["failed"] == 0 and not file_failed
    if value["success"] != expected_success or value["success"] != (exit_code == 0):
        fail("Vitest result", "success does not agree with details/process exit")
    actual = {**raw_suite_counts, **test_counts}
    return {"schema": 1, "counts": actual, "success": value["success"], "exit_code": exit_code, "suites": suites, "suite_diagnostics": diagnostics}
