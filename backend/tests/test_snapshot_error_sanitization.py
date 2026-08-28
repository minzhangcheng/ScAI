import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.models import schemas
from backend.models.schemas import AnalysisState
from backend.services.snapshot_v2.errors import (
    SnapshotCaptureError,
    snapshot_error_detail,
)
from backend.services.snapshot_v2.failure_gate import enter_snapshot_failure_gate
from backend.services.snapshot_v2.models import SnapshotFailureRecord
from backend.services.snapshot_v2.service_parts.capture_steps import CaptureStepsMixin
from backend.services.snapshot_v2.runtime import RuntimeCapture
from backend.services.snapshot_v2.runtime_parts.proof import _validate_h5ad_file


SECRET = "synthetic-snapshot-credential-value"
PRIVATE_PATH = "/home/synthetic-user/private-project/frame.py"


def _serialized(value) -> str:
    return json.dumps(value, default=str)


def _assert_redacted(value) -> None:
    exposed = _serialized(value)
    assert SECRET not in exposed
    assert "synthetic-user" not in exposed


def test_snapshot_failure_sanitizer_facade_and_state_roundtrip():
    failure = {"code": "x", "details": {"path": Path("relative/a")}}
    payload = {"snapshot_failure": failure}

    assert schemas._sanitize_snapshot_failure(payload) == {
        "code": "x",
        "details": {"path": "relative/a"},
    }
    state = AnalysisState(name="s", user_id="u", snapshot_failure=failure)
    restored = AnalysisState.model_validate_json(state.model_dump_json())
    assert restored.model_dump(mode="json") == state.model_dump(mode="json")


def test_snapshot_error_detail_recursively_sanitizes_every_string():
    error = SnapshotCaptureError(
        "SYNTHETIC",
        f"api_key='{SECRET}' at {PRIVATE_PATH}",
        details={
            "error": f"Bearer {SECRET}",
            "nested": [{"message": f"sk-{SECRET}", "path": PRIVATE_PATH}],
        },
    )

    detail = snapshot_error_detail(error)

    _assert_redacted(detail)
    assert detail["code"] == "SYNTHETIC"


def test_failure_record_and_gate_sanitize_before_persistence():
    record = SnapshotFailureRecord(
        snapshot_id="snapshot",
        session_id="session",
        failed_at="2026-08-02T00:00:00Z",
        code="SYNTHETIC",
        message=f"api_key='{SECRET}' at {PRIVATE_PATH}",
        details={"nested": {"error": f"Bearer {SECRET}"}},
    )
    _assert_redacted(record.model_dump(mode="json"))

    state = AnalysisState(id="session", user_id="user")
    enter_snapshot_failure_gate(
        state,
        {
            "code": "SYNTHETIC",
            "message": f"api_key='{SECRET}' at {PRIVATE_PATH}",
            "details": {"nested": f"Bearer {SECRET}"},
        },
    )
    _assert_redacted(state.snapshot_failure)


def test_secret_key_semantics_cover_model_dump_snapshot_bytes_and_api_detail():
    record = SnapshotFailureRecord(
        snapshot_id="snapshot",
        session_id="session",
        failed_at="2026-08-02T00:00:00Z",
        code="SYNTHETIC",
        message="capture failed",
        details={"password": SECRET},
    )
    # Nested Pydantic containers remain mutable after validation. The serializer
    # is a second boundary and must not trust that earlier validation still
    # describes the current value.
    record.details["googleApiKey"] = SECRET
    _assert_redacted(record.model_dump(mode="json"))

    state = AnalysisState(
        id="session",
        user_id="user",
        snapshot_failure={
            "code": "SYNTHETIC",
            "message": "capture failed",
            "details": {"access_token": SECRET},
        },
    )
    _assert_redacted(state.snapshot_failure)
    state.snapshot_failure["details"]["client-secret"] = SECRET
    _assert_redacted(state.model_dump(mode="json"))
    persisted = CaptureStepsMixin._serialize_analysis_state(state)
    assert SECRET.encode() not in persisted

    error = SnapshotCaptureError(
        "SYNTHETIC",
        "capture failed",
        details={"google_api_key": SECRET, "refreshToken": SECRET},
    )
    _assert_redacted(snapshot_error_detail(error))


class _IntrospectionSandbox:
    def __init__(self, output: str):
        self.output = output

    async def execute(self, _code, **_kwargs):
        return SimpleNamespace(success=True, output=self.output, error=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        f"api_key='{SECRET}' at {PRIVATE_PATH}",
        (
            "__RUNTIME_GLOBALS_BEGIN__\n"
            f'{{"api_key": "{SECRET}", "path": "{PRIVATE_PATH}" BROKEN\n'
            "__RUNTIME_GLOBALS_END__"
        ),
    ],
)
async def test_introspection_parse_failures_do_not_leak_raw_output(output):
    capture = RuntimeCapture(MagicMock(), MagicMock())

    with pytest.raises(SnapshotCaptureError) as exc_info:
        await capture._introspect_globals(_IntrospectionSandbox(output))

    _assert_redacted(snapshot_error_detail(exc_info.value))


def test_h5ad_validation_failure_sanitizes_exception_and_details(
    tmp_path: Path, monkeypatch,
):
    ad = pytest.importorskip("anndata")
    path = tmp_path / "candidate.h5ad"
    path.write_bytes(b"not-empty")

    def fail(_path, **_kwargs):
        raise RuntimeError(f"api_key='{SECRET}' at {PRIVATE_PATH}")

    monkeypatch.setattr(ad, "read_h5ad", fail)
    with pytest.raises(SnapshotCaptureError) as exc_info:
        _validate_h5ad_file(path)

    _assert_redacted(snapshot_error_detail(exc_info.value))
