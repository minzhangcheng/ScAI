"""Transactional contracts for chat attachments and delegation publication."""

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from backend.models.schemas import AnalysisState, WebSocketMessage
from backend.routes.chat_parts import delegation as chat_delegation
from backend.routes.chat_parts import endpoints as chat_endpoints
from backend.services.session_lifecycle import SessionGenerationError
from backend.state import chat_task_lock, chat_tasks


class ChunkUpload:
    def __init__(
        self,
        filename: str,
        chunks: list[bytes],
        *,
        declared_size: int | None = None,
    ):
        self.filename = filename
        self.size = declared_size
        self._chunks = list(chunks)

    async def read(self, _size=-1):
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def _configure_upload(monkeypatch, tmp_path, state):
    uploads = tmp_path / "uploads"

    def assert_writable(session_id, current):
        assert session_id == state.id
        assert current is state
        assert chat_endpoints.get_lifecycle_lock(session_id).locked()

    monkeypatch.setattr(
        chat_endpoints,
        "assert_generation_writable",
        assert_writable,
    )
    monkeypatch.setattr(
        chat_endpoints,
        "get_session_uploads",
        lambda _session_id: uploads,
    )
    return uploads


async def _commit(state, files):
    return await chat_endpoints._commit_chat_uploads(
        state.id,
        state,
        files,
        per_file_agents=["pathology"] * len(files),
        per_file_groups=[],
        per_file_samples=[],
        per_file_types=[],
        message="review these attachments",
    )


@pytest.mark.asyncio
async def test_chat_upload_uses_exclusive_publication_and_commits_state(
    monkeypatch, tmp_path,
):
    state = AnalysisState(id="chat-upload-success", user_id="owner")
    uploads = _configure_upload(monkeypatch, tmp_path, state)
    chat_dir = uploads / "chat"
    chat_dir.mkdir(parents=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (chat_dir / "sample.png").symlink_to(outside)

    saved = []

    async def save(session_id, current, *, force=False):
        saved.append((session_id, current, force))

    monkeypatch.setattr(
        chat_endpoints,
        "_async_save_session_under_lifecycle",
        save,
    )
    payload = b"not-a-real-image-but-bounded"

    files, encoded, previews = await _commit(
        state,
        [ChunkUpload("sample.png", [payload], declared_size=len(payload))],
    )

    final = chat_dir / "sample_1.png"
    assert final.read_bytes() == payload
    assert outside.read_bytes() == b"outside"
    assert (chat_dir / "sample.png").is_symlink()
    assert files[0]["path"] == str(final)
    assert base64.b64decode(encoded[0]) == payload
    assert previews == []
    assert len(state.data_files) == 1
    assert state.data_files[0].file_path == str(final)
    assert state.chat_messages[-1]["files"] == files
    assert saved == [(state.id, state, True)]
    assert list(uploads.rglob("*.uploading")) == []


@pytest.mark.asyncio
async def test_chat_upload_request_limit_rolls_back_prior_files_and_state(
    monkeypatch, tmp_path,
):
    state = AnalysisState(id="chat-upload-total-limit", user_id="owner")
    uploads = _configure_upload(monkeypatch, tmp_path, state)
    monkeypatch.setattr(chat_endpoints, "_CHAT_UPLOAD_MAX_FILE_BYTES", 10)
    monkeypatch.setattr(chat_endpoints, "_CHAT_UPLOAD_MAX_REQUEST_BYTES", 12)
    save = AsyncMock()
    monkeypatch.setattr(
        chat_endpoints,
        "_async_save_session_under_lifecycle",
        save,
    )

    with pytest.raises(HTTPException) as raised:
        await _commit(
            state,
            [
                ChunkUpload("first.bin", [b"a" * 8]),
                ChunkUpload("second.bin", [b"b" * 8]),
            ],
        )

    assert raised.value.status_code == 413
    assert not (uploads / "chat" / "first.bin").exists()
    assert not (uploads / "chat" / "second.bin").exists()
    assert list(uploads.rglob("*.uploading")) == []
    assert state.data_files == []
    assert state.chat_messages == []
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_upload_save_failure_rolls_back_files_and_memory(
    monkeypatch, tmp_path,
):
    state = AnalysisState(id="chat-upload-save-failure", user_id="owner")
    uploads = _configure_upload(monkeypatch, tmp_path, state)

    async def fail_save(*_args, **_kwargs):
        raise OSError("disk failed")

    monkeypatch.setattr(
        chat_endpoints,
        "_async_save_session_under_lifecycle",
        fail_save,
    )

    with pytest.raises(OSError, match="disk failed"):
        await _commit(state, [ChunkUpload("notes.txt", [b"payload"])])

    assert not (uploads / "chat" / "notes.txt").exists()
    assert list(uploads.rglob("*.uploading")) == []
    assert state.data_files == []
    assert state.chat_messages == []


@pytest.mark.asyncio
async def test_chat_upload_publication_race_preserves_competing_symlink(
    monkeypatch, tmp_path,
):
    state = AnalysisState(id="chat-upload-publish-race", user_id="owner")
    uploads = _configure_upload(monkeypatch, tmp_path, state)
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    publish = chat_endpoints._publish_upload_staging

    async def race_publish(staging_path, final_path):
        final_path.symlink_to(outside)
        await publish(staging_path, final_path)

    monkeypatch.setattr(
        chat_endpoints,
        "_publish_upload_staging",
        race_publish,
    )
    save = AsyncMock()
    monkeypatch.setattr(
        chat_endpoints,
        "_async_save_session_under_lifecycle",
        save,
    )

    with pytest.raises(HTTPException) as raised:
        await _commit(state, [ChunkUpload("notes.txt", [b"replacement"])])

    final = uploads / "chat" / "notes.txt"
    assert raised.value.status_code == 409
    assert final.is_symlink()
    assert outside.read_bytes() == b"outside"
    assert list(uploads.rglob("*.uploading")) == []
    assert state.data_files == []
    assert state.chat_messages == []
    save.assert_not_awaited()


@pytest.mark.asyncio
async def test_chat_upload_stale_generation_cannot_create_upload_tree(
    monkeypatch, tmp_path,
):
    state = AnalysisState(id="chat-upload-stale", user_id="owner")
    uploads = tmp_path / "uploads"
    monkeypatch.setattr(
        chat_endpoints,
        "get_session_uploads",
        lambda _session_id: uploads,
    )

    def reject(*_args, **_kwargs):
        raise SessionGenerationError("stale generation")

    monkeypatch.setattr(chat_endpoints, "assert_generation_writable", reject)

    with pytest.raises(SessionGenerationError, match="stale generation"):
        await _commit(state, [ChunkUpload("notes.txt", [b"payload"])])

    assert not uploads.exists()
    assert state.data_files == []
    assert state.chat_messages == []


def test_chat_image_encoding_reads_bounded_chunks(monkeypatch, tmp_path):
    payload = b"z" * (600 * 1024 + 1)
    image = tmp_path / "large.png"
    image.write_bytes(payload)
    observed_sizes = []
    original_encode = base64.b64encode

    def recording_encode(chunk):
        observed_sizes.append(len(chunk))
        return original_encode(chunk)

    monkeypatch.setattr(chat_endpoints.base64, "b64encode", recording_encode)

    encoded = chat_endpoints._encode_chat_image(image)

    assert base64.b64decode(encoded) == payload
    assert len(observed_sizes) > 1
    assert max(observed_sizes) <= 192 * 1024 + 2


def _configure_delegation(monkeypatch, state):
    def assert_writable(session_id, current):
        assert session_id == state.id
        assert current is state
        assert chat_endpoints.get_lifecycle_lock(session_id).locked()

    monkeypatch.setattr(
        chat_endpoints,
        "assert_generation_writable",
        assert_writable,
    )


async def _publish(state):
    return await chat_endpoints._publish_chat_delegation(
        state.id,
        state,
        {"plan": "run checks", "delegates": []},
        "user content",
        [],
        [],
    )


@pytest.mark.asyncio
async def test_chat_delegation_concurrent_publication_has_one_owner(
    monkeypatch,
):
    state = AnalysisState(id="chat-delegation-owner", user_id="owner")
    _configure_delegation(monkeypatch, state)
    save_started = asyncio.Event()
    release_save = asyncio.Event()
    runner_started = asyncio.Event()
    release_runner = asyncio.Event()
    runner_calls = 0

    async def save(*_args, **_kwargs):
        save_started.set()
        await release_save.wait()

    async def run(*_args, **_kwargs):
        nonlocal runner_calls
        runner_calls += 1
        runner_started.set()
        await release_runner.wait()

    monkeypatch.setattr(
        chat_endpoints,
        "_async_save_session_under_lifecycle",
        save,
    )
    monkeypatch.setattr(chat_endpoints, "_run_chat_delegation", run)

    first = asyncio.create_task(_publish(state))
    await asyncio.wait_for(save_started.wait(), timeout=1)
    # The publisher deliberately owns chat_task_lock until its durable state
    # commit finishes, so observe the event-loop-local registry directly here.
    published_owner = chat_tasks.get(state.id)
    assert published_owner is not None
    assert runner_calls == 0

    second = asyncio.create_task(_publish(state))
    await asyncio.sleep(0)
    release_save.set()
    owner = await asyncio.wait_for(first, timeout=1)
    with pytest.raises(HTTPException) as raised:
        await asyncio.wait_for(second, timeout=1)
    assert raised.value.status_code == 409
    assert owner is published_owner

    await asyncio.wait_for(runner_started.wait(), timeout=1)
    assert runner_calls == 1
    release_runner.set()
    await asyncio.wait_for(owner, timeout=1)
    async with chat_task_lock:
        assert state.id not in chat_tasks


@pytest.mark.asyncio
async def test_chat_delegation_save_failure_retires_owner_without_running(
    monkeypatch,
):
    state = AnalysisState(id="chat-delegation-save-failure", user_id="owner")
    _configure_delegation(monkeypatch, state)
    ran = False

    async def fail_save(*_args, **_kwargs):
        raise OSError("disk failed")

    async def run(*_args, **_kwargs):
        nonlocal ran
        ran = True

    monkeypatch.setattr(
        chat_endpoints,
        "_async_save_session_under_lifecycle",
        fail_save,
    )
    monkeypatch.setattr(chat_endpoints, "_run_chat_delegation", run)

    with pytest.raises(OSError, match="disk failed"):
        await _publish(state)

    assert ran is False
    assert not any(
        message.get("delegation_active") for message in state.chat_messages
    )
    async with chat_task_lock:
        assert state.id not in chat_tasks


@pytest.mark.asyncio
async def test_completed_old_delegation_cannot_retire_replacement_owner(
    monkeypatch,
):
    state = AnalysisState(id="chat-delegation-exact-owner", user_id="owner")
    _configure_delegation(monkeypatch, state)
    runner_started = asyncio.Event()
    release_runner = asyncio.Event()
    release_replacement = asyncio.Event()

    async def save(*_args, **_kwargs):
        return None

    async def run(*_args, **_kwargs):
        runner_started.set()
        await release_runner.wait()

    monkeypatch.setattr(
        chat_endpoints,
        "_async_save_session_under_lifecycle",
        save,
    )
    monkeypatch.setattr(chat_endpoints, "_run_chat_delegation", run)

    old_owner = await _publish(state)
    await asyncio.wait_for(runner_started.wait(), timeout=1)
    replacement = asyncio.create_task(release_replacement.wait())
    async with chat_task_lock:
        chat_tasks[state.id] = replacement

    release_runner.set()
    await asyncio.wait_for(old_owner, timeout=1)
    async with chat_task_lock:
        assert chat_tasks.get(state.id) is replacement
        chat_tasks.pop(state.id, None)
    replacement.cancel()
    await asyncio.gather(replacement, return_exceptions=True)


@pytest.mark.asyncio
async def test_chat_delegation_stale_generation_publishes_nothing(monkeypatch):
    state = AnalysisState(id="chat-delegation-stale", user_id="owner")

    def reject(*_args, **_kwargs):
        raise SessionGenerationError("stale generation")

    monkeypatch.setattr(chat_endpoints, "assert_generation_writable", reject)

    with pytest.raises(SessionGenerationError, match="stale generation"):
        await _publish(state)

    assert state.chat_messages == []
    async with chat_task_lock:
        assert state.id not in chat_tasks


@pytest.mark.asyncio
async def test_chat_delegation_callback_persists_durable_capped_history(
    monkeypatch,
):
    from backend.constants import STRIPPED_FIGURE_PLACEHOLDER
    from backend.utils import agent_llm, llm_client

    state = AnalysisState(id="chat-delegation-durable", user_id="owner")
    state.realtime_messages = [
        WebSocketMessage(type="status", content=f"old-{index}")
        for index in range(1000)
    ]
    image = "iVBORw0KGgo" + ("A" * 2048)
    emitted = WebSocketMessage(
        type="graphic_abstract",
        content={"image_base64": image, "raw": b"raw-binary"},
    )

    class _Client:
        async def close(self):
            return None

    class _Manager:
        def __init__(self, **_kwargs):
            self.openrouter = _Client()

        async def close(self):
            return None

    async def resolve(_user_id):
        return SimpleNamespace(
            api=SimpleNamespace(api_key="", llm_base_url="", proxy_url=None)
        )

    async def execute(
        _session_id,
        _state,
        _agent_name,
        _instruction,
        _agent_client,
        message_callback,
        _saved_files,
    ):
        await message_callback(emitted)
        raise asyncio.CancelledError

    async def cancel_after_durable_append(_session_id, live):
        assert len(state.realtime_messages) == 1000
        durable = state.realtime_messages[-1]
        assert durable is not emitted
        assert durable.content["image_base64"] == STRIPPED_FIGURE_PLACEHOLDER
        assert durable.content["raw"] == STRIPPED_FIGURE_PLACEHOLDER
        assert live["content"]["image_base64"] == image
        raise asyncio.CancelledError

    monkeypatch.setattr(llm_client, "LLMManager", _Manager)
    monkeypatch.setattr(agent_llm, "get_agent_llm_client", lambda *_args: _Client())
    monkeypatch.setattr(chat_delegation, "resolve_config_for_user", resolve)
    monkeypatch.setattr(chat_delegation, "_execute_chat_agent", execute)
    monkeypatch.setattr(chat_delegation, "_send_chat_ws", AsyncMock())
    monkeypatch.setattr(
        chat_delegation,
        "send_analysis_json",
        cancel_after_durable_append,
    )
    monkeypatch.setattr(
        chat_delegation,
        "_persist_chat_terminal",
        AsyncMock(return_value=True),
    )

    with pytest.raises(asyncio.CancelledError):
        await chat_delegation._run_chat_delegation(
            state.id,
            state,
            {
                "plan": "domain-neutral check",
                "delegates": [
                    {"agent": "literature", "instruction": "inspect evidence"}
                ],
            },
            "inspect evidence",
            [],
            [],
        )

    assert len(state.realtime_messages) == 1000
    durable = state.realtime_messages[-1]
    assert durable is not emitted
    assert durable.content["image_base64"] == STRIPPED_FIGURE_PLACEHOLDER
    assert durable.content["raw"] == STRIPPED_FIGURE_PLACEHOLDER
    assert emitted.content["image_base64"] == image
