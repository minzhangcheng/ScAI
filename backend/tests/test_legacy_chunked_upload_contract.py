"""Public contract tests for the legacy web chunked-upload initializer."""

import pytest
from pydantic import ValidationError

from backend.routes.upload import InitChunkedUploadRequest


def _request_payload() -> dict:
    return {
        "filename": "dataset.h5ad",
        "total_size": 1024,
        "total_chunks": 1,
        "file_type": "snrna_seq",
    }


def test_complete_file_digest_is_explicitly_required_and_documented():
    schema = InitChunkedUploadRequest.model_json_schema()

    assert "expected_sha256" in schema["required"]
    description = schema["properties"]["expected_sha256"]["description"]
    assert "Required SHA-256" in description
    assert "predating ScAI 0.38.11" in description

    with pytest.raises(ValidationError):
        InitChunkedUploadRequest.model_validate(_request_payload())


def test_complete_file_digest_remains_part_of_the_request_model():
    payload = _request_payload()
    payload["expected_sha256"] = "a" * 64

    request = InitChunkedUploadRequest.model_validate(payload)

    assert request.expected_sha256 == "a" * 64
