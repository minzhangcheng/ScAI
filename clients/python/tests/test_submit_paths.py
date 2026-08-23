"""P1 client submit contracts for local paths and placeholder identity."""
import json
from pathlib import Path

import pytest

from scai_client import Client
from scai_client.errors import ScaiError


def _config(*paths: str) -> dict:
    return {
        "schema_version": "1.0",
        "analysis": {
            "name": "P1 client",
            "initial_instruction": "test",
            "round_wait_seconds": 0,
        },
        "api_config": {
            "default_llm": {"llm_base_url": "", "api_key": ""},
            "literature": {"mode": "disabled"},
        },
        "data": {
            "files": [
                {
                    "local_path": path,
                    "section": "sequencing",
                    "data_type": "h5ad",
                }
                for path in paths
            ]
        },
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(config))
    return path


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload
        self.text = "ok"

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def test_submit_rejects_all_missing_paths_before_create(tmp_path, monkeypatch):
    (tmp_path / "present.h5ad").write_bytes(b"present")
    config_path = _write_config(
        tmp_path,
        _config("present.h5ad", "missing-a.h5ad", "missing-b.h5ad"),
    )

    with Client(base_url="http://localhost:8000", api_key="scai_test") as client:
        monkeypatch.setattr(
            client._http,
            "post",
            lambda *_args, **_kwargs: pytest.fail("create must not run for missing files"),
        )
        with pytest.raises(FileNotFoundError) as caught:
            client.submit(config_path)

    message = str(caught.value)
    assert "missing-a.h5ad" in message
    assert "missing-b.h5ad" in message
    assert "2 path(s)" in message


def test_submit_resolves_paths_and_binds_each_placeholder_id(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.h5ad").write_bytes(b"a")
    (data_dir / "b.h5ad").write_bytes(b"b")
    config_path = _write_config(
        tmp_path,
        _config("data/../data/a.h5ad", "data/b.h5ad"),
    )

    uploaded: list[tuple[Path, str]] = []
    created = {
        "job_id": "job-p1",
        "files": [
            {
                "id": "pf-a",
                "filename": "a.h5ad",
                "section": "sequencing",
                "group_id": None,
                "sample_number": None,
            },
            {
                "id": "pf-b",
                "filename": "b.h5ad",
                "section": "sequencing",
                "group_id": None,
                "sample_number": None,
            },
        ],
    }
    with Client(base_url="http://localhost:8000", api_key="scai_test") as client:
        monkeypatch.setattr(client._http, "post", lambda *_args, **_kwargs: _Response(created))
        monkeypatch.setattr(
            "scai_client.client.upload_file",
            lambda _http, _job_id, path, *, file_id=None: uploaded.append((path, file_id)),
        )

        job = client.submit(config_path)

    assert job.job_id == "job-p1"
    assert uploaded == [
        (data_dir / "a.h5ad", "pf-a"),
        (data_dir / "b.h5ad", "pf-b"),
    ]


def test_submit_rejects_an_incomplete_placeholder_manifest(tmp_path, monkeypatch):
    (tmp_path / "a.h5ad").write_bytes(b"a")
    (tmp_path / "b.h5ad").write_bytes(b"b")
    config_path = _write_config(tmp_path, _config("a.h5ad", "b.h5ad"))
    uploaded = []

    manifests = [
        # Incomplete entry: every required identity field must be present.
        [{"id": "pf-a", "filename": "a.h5ad", "section": "sequencing", "group_id": None}],
        # Placeholder IDs are the upload binding key and must be unique.
        [
            {
                "id": "pf-a",
                "filename": "a.h5ad",
                "section": "sequencing",
                "group_id": None,
                "sample_number": None,
            },
            {
                "id": "pf-a",
                "filename": "b.h5ad",
                "section": "sequencing",
                "group_id": None,
                "sample_number": None,
            },
        ],
        # The server manifest must preserve the config's filename and section.
        [
            {
                "id": "pf-a",
                "filename": "other.h5ad",
                "section": "sequencing",
                "group_id": None,
                "sample_number": None,
            },
            {
                "id": "pf-b",
                "filename": "b.h5ad",
                "section": "sequencing",
                "group_id": None,
                "sample_number": None,
            },
        ],
        [
            {
                "id": "pf-a",
                "filename": "a.h5ad",
                "section": "wrong-section",
                "group_id": None,
                "sample_number": None,
            },
            {
                "id": "pf-b",
                "filename": "b.h5ad",
                "section": "sequencing",
                "group_id": None,
                "sample_number": None,
            },
        ],
        [
            {
                "id": "pf-a",
                "filename": "a.h5ad",
                "section": "sequencing",
                "group_id": "unexpected-group",
                "sample_number": None,
            },
            {
                "id": "pf-b",
                "filename": "b.h5ad",
                "section": "sequencing",
                "group_id": None,
                "sample_number": None,
            },
        ],
        [
            {
                "id": "pf-a",
                "filename": "a.h5ad",
                "section": "sequencing",
                "group_id": None,
                "sample_number": 1,
            },
            {
                "id": "pf-b",
                "filename": "b.h5ad",
                "section": "sequencing",
                "group_id": None,
                "sample_number": None,
            },
        ],
    ]

    with Client(base_url="http://localhost:8000", api_key="scai_test") as client:
        monkeypatch.setattr(
            client._http,
            "post",
            lambda *_args, **_kwargs: _Response(
                {"job_id": "job-p1", "files": manifests[0]}
            ),
        )
        monkeypatch.setattr(
            "scai_client.client.upload_file",
            lambda *_args, **_kwargs: uploaded.append(True),
        )

        for manifest in manifests:
            monkeypatch.setattr(
                client._http,
                "post",
                lambda *_args, manifest=manifest, **_kwargs: _Response(
                    {"job_id": "job-p1", "files": manifest}
                ),
            )
            with pytest.raises(ScaiError):
                client.submit(config_path)

    assert uploaded == []
