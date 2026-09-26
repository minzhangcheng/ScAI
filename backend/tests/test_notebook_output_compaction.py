"""F4 notebook-media and standalone-report regression tests.

These tests deliberately exercise the retained ``NotebookState`` and report
entry points.  They do not import the F4 helper module: on the unmodified
rebaseline the nodes must collect and fail on the missing behavior contract.
"""

from __future__ import annotations

import base64
import copy
import hashlib
from pathlib import Path

import nbformat
import pytest

from backend.models.schemas import AnalysisState
from backend.services.report.generator import DetailedReportGenerator
from backend.utils.sandbox_parts.shared import NotebookState


_EXTERNAL_IMAGE_MIME = "application/vnd.scai.external-image+json"
_TEXT_LIMIT = 256 * 1024
_TRACEBACK_LIMIT = 200


def _notebook_state(tmp_path: Path, session_id: str = "notebook-compaction"):
    session_root = tmp_path / session_id
    states = session_root / "states"
    workspace = session_root / "workspace"
    states.mkdir(parents=True)
    workspace.mkdir(parents=True)
    return NotebookState(session_id, str(states)), workspace


def _reference(raw: bytes, media_type: str = "image/png") -> dict[str, object]:
    digest = hashlib.sha256(raw).hexdigest()
    suffix = ".jpg" if media_type == "image/jpeg" else ".png"
    return {
        "path": f"{digest}{suffix}",
        "sha256": digest,
        "size": len(raw),
        "media_type": media_type,
    }


def _write_external_notebook(
    states: Path,
    session_id: str,
    reference: dict[str, object] | list[dict[str, object]],
    *,
    placeholder: str = "source placeholder",
    output_types: tuple[str, ...] = ("display_data",),
    references: list[dict[str, object]] | None = None,
) -> Path:
    notebook = nbformat.v4.new_notebook()
    selected_references = references or [reference] * len(output_types)
    outputs = []
    for output_type, output_reference in zip(output_types, selected_references):
        kwargs = {"execution_count": 1} if output_type == "execute_result" else {}
        outputs.append(
            nbformat.v4.new_output(
                output_type=output_type,
                data={
                    _EXTERNAL_IMAGE_MIME: output_reference,
                    "text/plain": placeholder,
                },
                **kwargs,
            )
        )
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            source="print('a sufficiently long analysis cell')",
            outputs=outputs,
        )
    )
    path = states / f"{session_id}.ipynb"
    nbformat.write(notebook, path)
    return path


def _state_for_report(session_id: str) -> AnalysisState:
    return AnalysisState(id=session_id, user_id="report-owner", name="F4 report")


async def _render_detailed_report(
    state: AnalysisState,
    states: Path,
    workspace: Path,
    output_dir: Path,
    monkeypatch,
) -> tuple[dict, Path]:
    import backend.config as config_module
    import backend.services.report.generator as generator_module

    monkeypatch.setattr(
        generator_module,
        "get_session_states",
        lambda _session_id: states,
    )
    monkeypatch.setattr(
        config_module,
        "get_session_states",
        lambda _session_id: states,
    )
    generator = DetailedReportGenerator(
        None,
        "",
        "",
        str(output_dir),
        workspace_dir=str(workspace),
        session_id=state.id,
    )
    report = await generator._generate_report_staged(
        state,
        resume_from_checkpoint=False,
        force_fallback=True,
    )
    return report, output_dir / f"{state.id}.ipynb"


def test_large_images_are_chunk_decoded_content_addressed_and_deduplicated(
    tmp_path, monkeypatch,
):
    from backend.utils import notebook_media

    state, workspace = _notebook_state(tmp_path)
    png_raw = b"png-bytes" * 140000
    jpeg_raw = b"jpeg-bytes" * 120000
    outputs = [
        {
            "type": "display_data",
            "data": {
                "image/png": base64.b64encode(png_raw).decode("ascii"),
                "image/jpeg": base64.b64encode(jpeg_raw).decode("ascii"),
            },
        }
    ]
    original = copy.deepcopy(outputs)
    decode_sizes: list[int] = []
    original_decode = notebook_media.base64.b64decode

    def recording_decode(value, *args, **kwargs):
        decode_sizes.append(len(value))
        return original_decode(value, *args, **kwargs)

    monkeypatch.setattr(notebook_media.base64, "b64decode", recording_decode)

    state.add_cell("plot_one()", outputs, 1)
    state.add_cell("plot_again()", outputs, 2)
    state.save_notebook()

    media = workspace / ".notebook_media"
    png_media = list(media.glob("*.png"))
    jpeg_media = list(media.glob("*.jpg"))
    assert len(png_media) == 1
    assert len(jpeg_media) == 1
    assert png_media[0].read_bytes() == png_raw
    assert jpeg_media[0].read_bytes() == jpeg_raw
    assert outputs == original
    assert state.cells[0]["outputs"] == state.cells[1]["outputs"]
    references = state.cells[0]["outputs"][0]["data"][_EXTERNAL_IMAGE_MIME]
    assert isinstance(references, list)
    assert {item["media_type"] for item in references} == {
        "image/png",
        "image/jpeg",
    }
    assert decode_sizes
    assert max(decode_sizes) <= 1024 * 1024 + 3


@pytest.mark.parametrize(
    "invalid_case",
    ("escape", "absolute", "nested", "basename", "size", "digest", "media", "missing"),
)
@pytest.mark.asyncio
async def test_invalid_or_missing_sidecar_stays_visible_without_external_read(
    tmp_path, monkeypatch, invalid_case
):
    session_id = f"missing-notebook-{invalid_case}"
    states = tmp_path / session_id / "states"
    workspace = tmp_path / session_id / "workspace"
    media = workspace / ".notebook_media"
    media.mkdir(parents=True)
    states.mkdir(parents=True)
    expected = b"expected bytes"
    reference = _reference(expected)
    outside = tmp_path / f"outside-{invalid_case}.png"
    if invalid_case == "escape":
        outside = workspace / "outside.png"
        reference["path"] = "../outside.png"
    elif invalid_case == "absolute":
        reference["path"] = str(outside.resolve())
    elif invalid_case == "nested":
        reference["path"] = f"nested/{reference['path']}"
        outside = media / str(reference["path"])
    elif invalid_case == "basename":
        reference["path"] = "wrong-basename.png"
        outside = media / str(reference["path"])
    elif invalid_case == "size":
        outside = media / str(reference["path"])
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(expected)
        reference["size"] = len(expected) + 1
    elif invalid_case == "digest":
        reference["sha256"] = "0" * 64
        reference["path"] = f"{reference['sha256']}.png"
        outside = media / str(reference["path"])
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(expected)
    elif invalid_case == "media":
        reference["media_type"] = "application/pdf"
        reference["path"] = f"{reference['sha256']}.pdf"
        outside = media / str(reference["path"])
    elif invalid_case != "missing":
        raise AssertionError(f"unknown invalid reference case: {invalid_case}")
    if invalid_case not in {"size", "digest", "missing"}:
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(b"OUTSIDE-SENTINEL")
    _write_external_notebook(states, session_id, reference, placeholder="original")

    _report, output = await _render_detailed_report(
        _state_for_report(session_id),
        states,
        workspace,
        tmp_path / f"report-{invalid_case}",
        monkeypatch,
    )
    exported = nbformat.read(output, as_version=4)
    data = exported.cells[0].outputs[0].data
    assert _EXTERNAL_IMAGE_MIME in data
    assert "omitted" in data["text/plain"].lower() or "unavailable" in data["text/plain"].lower()
    assert b"OUTSIDE-SENTINEL" not in output.read_bytes()


@pytest.mark.asyncio
async def test_sidecar_root_file_symlink_and_directory_exchange_fail_closed(
    tmp_path, monkeypatch
):
    raw = b"safe-media" * 100
    cases = ("root", "file", "exchange")
    for case in cases:
        session_id = f"symlink-{case}"
        root = tmp_path / session_id
        states = root / "states"
        workspace = root / "workspace"
        states.mkdir(parents=True)
        media = workspace / ".notebook_media"
        media.mkdir(parents=True)
        reference = _reference(raw)
        outside = tmp_path / f"outside-{case}.png"
        outside.write_bytes(b"OUTSIDE-SENTINEL")
        if case == "root":
            media.rmdir()
            media.symlink_to(outside.parent, target_is_directory=True)
        elif case == "file":
            (media / reference["path"]).symlink_to(outside)
        else:
            (media / reference["path"]).write_bytes(raw)
            from backend.utils import notebook_media

            original_regular_stat = notebook_media._regular_stat
            exchanged = False

            def _exchange_after_pin(media_fd, name):
                nonlocal exchanged
                if not exchanged:
                    old_media = workspace / ".notebook_media.old"
                    replacement = workspace / ".notebook_media.replacement"
                    replacement.mkdir()
                    (replacement / name).write_bytes(b"OUTSIDE-SENTINEL")
                    media.rename(old_media)
                    replacement.rename(media)
                    exchanged = True
                return original_regular_stat(media_fd, name)

            monkeypatch.setattr(
                notebook_media, "_regular_stat", _exchange_after_pin
            )
        _write_external_notebook(states, session_id, reference, placeholder="original")

        _report, output = await _render_detailed_report(
            _state_for_report(session_id),
            states,
            workspace,
            tmp_path / f"report-{case}",
            monkeypatch,
        )
        if case == "exchange":
            assert exchanged is True
        exported = nbformat.read(output, as_version=4)
        data = exported.cells[0].outputs[0].data
        if "image/png" in data:
            assert data["image/png"] == base64.b64encode(raw).decode("ascii")
        else:
            assert _EXTERNAL_IMAGE_MIME in data
            assert "original" not in data.get("text/plain", "")
        assert b"OUTSIDE-SENTINEL" not in output.read_bytes()
