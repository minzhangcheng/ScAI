"""Focused P1 contracts for placeholder identity and upload destinations."""
import asyncio
import hashlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from backend.models.schemas import AnalysisState, DataFile, TaskStatus, TreatmentGroup
from backend.state import sessions


SID = "p1-upload-test"
CELLRANGER = "filtered_feature_bc_matrix.h5"


def _state(*, session_id=SID, same_group=False):
    group_a = "gcon"
    group_b = group_a if same_group else "gdko"
    return AnalysisState(
        id=session_id,
        user_id="u1",
        name="P1 upload",
        treatment_groups=[
            TreatmentGroup(id="gcon", name="Control", sample_count=1, order=0),
            TreatmentGroup(id="gdko", name="Treatment", sample_count=1, order=1),
        ],
        data_files=[
            DataFile(
                id="pf-con",
                filename=CELLRANGER,
                file_path="",
                section="sequencing",
                data_type="h5ad",
                group_id=group_a,
            ),
            DataFile(
                id="pf-dko",
                filename=CELLRANGER,
                file_path="",
                section="sequencing",
                data_type="h5ad",
                group_id=group_b,
            ),
        ],
    )


def _request(body: dict):
    async def read_json():
        return body

    return SimpleNamespace(json=read_json, state=SimpleNamespace(user_id="u1"))


@contextmanager
def _patches(app_db, temp_dir: Path, upload_dir: Path):
    session_root = upload_dir.parent / "session-store"
    metadata = lambda session_id: session_root / session_id / "session.json"
    with (
        patch("backend.services.upload.get_db", return_value=app_db),
        patch("backend.services.upload.get_session_uploads", return_value=upload_dir),
        patch("backend.services.upload.UPLOAD_TEMP_DIR", temp_dir),
        patch("backend.services.session_store.get_session_metadata_path", side_effect=metadata),
        patch("backend.services.session_store.resolve_session_metadata_path", side_effect=metadata),
        patch("backend.services.session_lifecycle.SESSION_DIR", session_root),
    ):
        yield session_root


def _body(*, file_id="", filename=CELLRANGER, payload=b"bytes"):
    return {
        "file_id": file_id,
        "filename": filename,
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
        "total_size": len(payload),
        "total_chunks": 1,
        "chunk_size": 1024,
    }


async def _stage(app_db, temp_dir: Path, session_id: str, placeholder_id: str, payload: bytes):
    upload_id = await app_db.create_upload_session(
        session_id=session_id,
        filename=CELLRANGER,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        total_size=len(payload),
        total_chunks=1,
        chunk_size=1024,
        expires_at_iso="2099-01-01T00:00:00",
        placeholder_file_id=placeholder_id,
    )
    (temp_dir / f"{upload_id}.chunk.0").write_bytes(payload)
    await app_db.mark_chunk_received(upload_id, 0)
    return upload_id


@pytest.mark.asyncio
async def test_create_returns_stable_placeholder_manifest_in_config_order(monkeypatch):
    from backend.routes import v1

    state = _state()
    monkeypatch.setattr(v1, "_create_pending_session", AsyncMock(return_value=state))
    request = SimpleNamespace(state=SimpleNamespace(user_id="u1"))

    created = await v1.create_analysis(request)
    from_config = await v1.create_from_config(request)

    expected = [
        {
            "id": "pf-con",
            "filename": CELLRANGER,
            "section": "sequencing",
            "group_id": "gcon",
            "sample_number": None,
        },
        {
            "id": "pf-dko",
            "filename": CELLRANGER,
            "section": "sequencing",
            "group_id": "gdko",
            "sample_number": None,
        },
    ]
    assert created == {"job_id": SID, "status": "pending", "files": expected}
    assert from_config == {"id": SID, "status": "pending", "files": expected}


@pytest.mark.asyncio
async def test_init_requires_exact_file_id_and_rejects_invalid_bindings(tmp_path, app_db):
    from backend.services.upload import init_upload

    upload_dir = tmp_path / "uploads"
    state = _state()
    sessions[SID] = state
    try:
        with _patches(app_db, tmp_path, upload_dir):
            valid = await init_upload(SID, _request(_body(file_id="pf-dko")))
            row = await app_db.get_upload_session(valid["upload_id"])
            assert row["placeholder_file_id"] == "pf-dko"

            cases = [
                ({}, "file_id"),
                ({"file_id": "pf-nope"}, "pf-nope"),
                ({"file_id": "pf-dko"}, "already bound"),
                ({"file_id": "pf-con", "filename": "other.h5ad"}, "other.h5ad"),
            ]
            state.data_files[1].file_path = str(upload_dir / "already.h5")
            for overrides, expected in cases:
                body = _body(**overrides)
                with pytest.raises(HTTPException) as caught:
                    await init_upload(SID, _request(body))
                assert caught.value.status_code in {400, 409}
                assert expected.lower() in str(caught.value.detail).lower()
    finally:
        sessions.pop(SID, None)


@pytest.mark.asyncio
async def test_same_basename_out_of_order_concurrent_completion_preserves_identity_on_disk(
    tmp_path, app_db
):
    from backend.services import session_store
    from backend.services.upload import complete_upload, init_upload

    upload_dir = tmp_path / "uploads"
    state = _state()
    sessions[SID] = state
    payloads = {"pf-con": b"control counts", "pf-dko": b"treated counts"}
    try:
        with _patches(app_db, tmp_path, upload_dir):
            # Initialize the second group first; binding must follow the ID.
            upload_ids = {}
            for placeholder_id in ("pf-dko", "pf-con"):
                body = _body(file_id=placeholder_id, payload=payloads[placeholder_id])
                result = await init_upload(SID, _request(body))
                upload_ids[placeholder_id] = result["upload_id"]
                row = await app_db.get_upload_session(result["upload_id"])
                payload = payloads[placeholder_id]
                (tmp_path / f"{result['upload_id']}.chunk.0").write_bytes(payload)
                await app_db.mark_chunk_received(result["upload_id"], 0)

            results = await asyncio.gather(
                complete_upload(SID, upload_ids["pf-dko"]),
                complete_upload(SID, upload_ids["pf-con"]),
            )
            rows = [await app_db.get_upload_session(upload_ids[item]) for item in ("pf-con", "pf-dko")]
            reloaded = session_store.load_session_from_disk(SID)
    finally:
        sessions.pop(SID, None)

    by_id = {row["placeholder_file_id"]: row for row in rows}
    for placeholder_id, payload in payloads.items():
        target = next(df for df in state.data_files if df.id == placeholder_id)
        path = Path(target.file_path)
        assert path.exists()
        assert path.read_bytes() == payload
        assert path.parts[-4:] == ("sequencing", target.group_id, "_unsorted", CELLRANGER)
        assert path != Path(next(df.file_path for df in state.data_files if df.id != placeholder_id))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == by_id[placeholder_id]["expected_sha256"]
        assert by_id[placeholder_id]["completed"] == 1
    assert {Path(result["path"]) for result in results} == {
        Path(df.file_path) for df in state.data_files
    }
    assert reloaded is not None
    assert {
        data_file.id: data_file.file_path
        for data_file in reloaded.data_files
    } == {
        data_file.id: data_file.file_path
        for data_file in state.data_files
    }


@pytest.mark.asyncio
async def test_duplicate_destination_is_rejected_before_and_under_lifecycle_lock(
    tmp_path, app_db
):
    from backend.services.upload import complete_upload, init_upload

    upload_dir = tmp_path / "uploads"
    state = _state(same_group=True)
    sessions[SID] = state
    try:
        with _patches(app_db, tmp_path, upload_dir):
            first = await init_upload(SID, _request(_body(file_id="pf-con", payload=b"first")))
            first_id = first["upload_id"]
            with pytest.raises(HTTPException) as concurrent_init:
                await init_upload(SID, _request(_body(file_id="pf-con", payload=b"first-again")))
            assert concurrent_init.value.status_code == 409
            (tmp_path / f"{first_id}.chunk.0").write_bytes(b"first")
            await app_db.mark_chunk_received(first_id, 0)
            await complete_upload(SID, first_id)

            with pytest.raises(HTTPException) as before_transfer:
                await init_upload(SID, _request(_body(file_id="pf-dko", payload=b"second")))
            assert before_transfer.value.status_code == 409

            second_id = await _stage(app_db, tmp_path, SID, "pf-dko", b"second")
            with pytest.raises(HTTPException) as under_lock:
                await complete_upload(SID, second_id)
            assert under_lock.value.status_code == 409
            assert Path(state.data_files[0].file_path).read_bytes() == b"first"
            assert not state.data_files[1].file_path
    finally:
        sessions.pop(SID, None)


@pytest.mark.asyncio
async def test_start_distinguishes_unbound_from_missing_bound_path(tmp_path, monkeypatch):
    from backend.routes import v1

    state = _state()
    state.status = TaskStatus.PENDING
    state.data_files[0].filename = "never-uploaded.h5ad"
    state.data_files[1].filename = "deleted-after-upload.h5ad"
    state.data_files[1].file_path = str(tmp_path / "deleted-after-upload.h5ad")
    monkeypatch.setattr(v1, "require_session_ownership", AsyncMock(return_value=state))
    monkeypatch.setattr(v1, "resolve_config_for_user", AsyncMock(return_value=object()))
    started = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "backend.services.workflow_runner.start_workflow_if_needed",
        started,
    )
    request = SimpleNamespace(state=SimpleNamespace(user_id="u1"))

    with pytest.raises(HTTPException) as caught:
        await v1.start_analysis(SID, request)

    assert caught.value.status_code == 400
    detail = str(caught.value.detail)
    assert "never uploaded" in detail
    assert "no longer exists" in detail
    assert "never-uploaded.h5ad" in detail
    assert "deleted-after-upload.h5ad" in detail
    started.assert_not_awaited()


@pytest.mark.asyncio
async def test_interrupted_completion_adopts_only_exact_path_size_and_checksum(
    tmp_path, app_db, monkeypatch
):
    from backend.services import session_store, upload as upload_service
    from backend.services.upload import complete_upload

    exact_sid = "p1-interrupted-exact"
    exact_state = _state(session_id=exact_sid)
    exact_uploads = tmp_path / "exact-uploads"
    sessions[exact_sid] = exact_state
    try:
        payload = b"already delivered"
        upload_id = await _stage(app_db, tmp_path, exact_sid, "pf-con", payload)
        exact_path = exact_uploads / "sequencing/gcon/_unsorted" / CELLRANGER
        exact_path.parent.mkdir(parents=True)
        exact_path.write_bytes(payload)
        await app_db.complete_upload_session(upload_id)
        with _patches(app_db, tmp_path, exact_uploads):
            real_save = upload_service._async_save_session_under_lifecycle
            save_calls = 0

            async def fail_once(session_id, state, force=False):
                nonlocal save_calls
                save_calls += 1
                if save_calls == 1:
                    raise OSError("injected persistence failure")
                return await real_save(session_id, state, force=force)

            monkeypatch.setattr(
                upload_service,
                "_async_save_session_under_lifecycle",
                fail_once,
            )
            with pytest.raises(OSError, match="persistence"):
                await complete_upload(exact_sid, upload_id)
            assert exact_state.data_files[0].file_path == ""
            result = await complete_upload(exact_sid, upload_id)
            reloaded = session_store.load_session_from_disk(exact_sid)
        assert result["path"] == str(exact_path.resolve())
        assert exact_state.data_files[0].file_path == str(exact_path.resolve())
        assert reloaded is not None
        assert reloaded.data_files[0].file_path == str(exact_path.resolve())
        assert save_calls == 2
        assert exact_path.stat().st_size == len(payload)
        assert hashlib.sha256(exact_path.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()
    finally:
        sessions.pop(exact_sid, None)

    wrong_sid = "p1-interrupted-wrong"
    wrong_state = _state(session_id=wrong_sid)
    wrong_uploads = tmp_path / "wrong-uploads"
    sessions[wrong_sid] = wrong_state
    try:
        payload = b"expected bytes"
        upload_id = await _stage(app_db, tmp_path, wrong_sid, "pf-con", payload)
        wrong_path = wrong_uploads / "sequencing/gcon/_unsorted" / CELLRANGER
        wrong_path.parent.mkdir(parents=True)
        wrong_path.write_bytes(b"wrong")
        await app_db.complete_upload_session(upload_id)
        with _patches(app_db, tmp_path, wrong_uploads):
            with pytest.raises(HTTPException) as caught:
                await complete_upload(wrong_sid, upload_id)
        assert caught.value.status_code == 409
        assert wrong_state.data_files[0].file_path == ""
        assert wrong_path.read_bytes() == b"wrong"
    finally:
        sessions.pop(wrong_sid, None)
