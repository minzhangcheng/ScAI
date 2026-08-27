"""F4 notebook-media and standalone-report regression tests.

These tests deliberately exercise the retained ``NotebookState`` and report
entry points.  They do not import the F4 helper module: on the unmodified
rebaseline the nodes must collect and fail on the missing behavior contract.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
from pathlib import Path

import nbformat
import pytest

from backend.agents.supervisor.report_html import ReportHtmlMixin
from backend.config import get_session_states
from backend.models.schemas import AnalysisState
from backend.services.report.analysis_log import AnalysisLogMixin
from backend.services.report.generator import DetailedReportGenerator
from backend.utils.sandbox_parts.shared import NotebookState
from backend.utils.sandbox_parts import notebook_state as notebook_state_owner


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


def test_notebook_state_identity_and_serialized_bytes_are_stable(tmp_path):
    state, _workspace = _notebook_state(tmp_path, "rc17-contract")
    assert NotebookState is notebook_state_owner.NotebookState
    assert NotebookState.__module__ == "backend.utils.sandbox_parts.shared"
    assert (
        notebook_state_owner.CodeExecutionResult.__module__
        == "backend.utils.sandbox_parts.shared"
    )

    state.created_at = "2026-08-27T00:00:00"
    state.updated_at = "2026-08-27T00:00:01"
    state.execution_count = 1
    state.cells = [{
        "code": "print('stable')",
        "outputs": [{"type": "stream", "name": "stdout", "text": "stable\\n"}],
        "execution_count": 1,
        "executed_at": "2026-08-27T00:00:01",
    }]

    state.save_notebook()
    state.save_metadata()

    assert hashlib.sha256(state.notebook_path.read_bytes()).hexdigest() == (
        "d59f8142e9762fee2ba5584dfdd126ed6bbb57e159d192da77b3c92fb394f255"
    )
    assert hashlib.sha256(state.state_path.read_bytes()).hexdigest() == (
        "5487f841d71f5f9af1feaecd7a62043a70e1152826f24a2513ae57f832249e1e"
    )


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


def _render_supervisor_report(
    report: dict,
    state: AnalysisState,
    workspace: Path,
    output_dir: Path,
    monkeypatch,
) -> Path:
    import backend.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_session_states",
        lambda _session_id: workspace.parent / "states",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ReportHtmlMixin()._generate_html_report(
        report,
        report.get("sections", []),
        report.get("abstract", ""),
        report.get("conclusions", ""),
        report.get("figures", []),
        {},
        {},
        "en",
        output_dir,
        workspace,
    )
    return output_dir / f"{state.id}.ipynb"


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


def test_text_html_and_traceback_limits_are_independent_and_copy_only(tmp_path):
    state, _workspace = _notebook_state(tmp_path)
    text = "HEAD\n" + ("x" * (_TEXT_LIMIT + 1000)) + "\nTAIL"
    html = "<p>" + ("H" * (_TEXT_LIMIT + 1000)) + "</p>"
    traceback = [f"line-{i}" for i in range(500)]
    outputs = [
        {"type": "stream", "text": text},
        {"type": "execute_result", "data": {"text/plain": text}},
        {"type": "display_data", "data": {"text/html": html}},
        {
            "type": "error",
            "ename": "RuntimeError",
            "evalue": text,
            "traceback": traceback,
        },
    ]
    original = copy.deepcopy(outputs)

    state.add_cell("bound_outputs()", outputs, 1)
    compacted = state.cells[0]["outputs"]

    assert outputs == original
    assert len(compacted[0]["text"]) <= _TEXT_LIMIT
    assert len(compacted[1]["data"]["text/plain"]) <= _TEXT_LIMIT
    assert "HEAD" in compacted[0]["text"] and "TAIL" in compacted[0]["text"]
    assert "text/html" not in compacted[2]["data"]
    assert "omitted" in compacted[2]["data"]["text/plain"].lower()
    assert len(compacted[3]["traceback"]) == _TRACEBACK_LIMIT + 1
    assert compacted[3]["traceback"][0] == "line-0"
    assert compacted[3]["traceback"][-1] == "line-499"

    state.save_notebook()
    saved_once = state.notebook_path.read_bytes()
    state.save_notebook()
    assert state.notebook_path.read_bytes() == saved_once


@pytest.mark.asyncio
async def test_standalone_export_rehydrates_only_verified_sidecars(tmp_path, monkeypatch):
    session_id = "verified-notebook"
    states = tmp_path / session_id / "states"
    workspace = tmp_path / session_id / "workspace"
    media = workspace / ".notebook_media"
    states.mkdir(parents=True)
    media.mkdir(parents=True)
    png_raw = b"verified-png" * 500
    jpeg_raw = b"verified-jpeg" * 500
    png_reference = _reference(png_raw)
    jpeg_reference = _reference(jpeg_raw, "image/jpeg")
    (media / png_reference["path"]).write_bytes(png_raw)
    (media / jpeg_reference["path"]).write_bytes(jpeg_raw)
    _write_external_notebook(
        states,
        session_id,
        [png_reference, jpeg_reference],
    )

    state = _state_for_report(session_id)
    detailed_dir = tmp_path / "detailed-report"
    report, detailed_notebook = await _render_detailed_report(
        state, states, workspace, detailed_dir, monkeypatch
    )
    detailed = nbformat.read(detailed_notebook, as_version=4)
    detailed_data = detailed.cells[0].outputs[0].data
    assert detailed_data["image/png"] == base64.b64encode(png_raw).decode("ascii")
    assert detailed_data["image/jpeg"] == base64.b64encode(jpeg_raw).decode("ascii")
    assert _EXTERNAL_IMAGE_MIME not in detailed_data

    supervisor_input = dict(report)
    supervisor_input["_analysis_state"] = state
    supervisor_notebook = _render_supervisor_report(
        supervisor_input,
        state,
        workspace,
        tmp_path / "supervisor-report",
        monkeypatch,
    )
    supervisor = nbformat.read(supervisor_notebook, as_version=4)
    supervisor_data = supervisor.cells[0].outputs[0].data
    assert supervisor_data["image/png"] == base64.b64encode(png_raw).decode("ascii")
    assert supervisor_data["image/jpeg"] == base64.b64encode(jpeg_raw).decode("ascii")
    assert _EXTERNAL_IMAGE_MIME not in supervisor_data

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


@pytest.mark.parametrize("output_type", ("display_data", "execute_result"))
@pytest.mark.asyncio
async def test_both_report_paths_use_verified_standalone_notebooks(
    tmp_path, monkeypatch, output_type
):
    session_id = "report-paths"
    states = tmp_path / session_id / "states"
    workspace = tmp_path / session_id / "workspace"
    media = workspace / ".notebook_media"
    states.mkdir(parents=True)
    media.mkdir(parents=True)
    raw = b"report-path-media" * 100
    reference = _reference(raw, "image/jpeg")
    (media / reference["path"]).write_bytes(raw)
    _write_external_notebook(
        states,
        session_id,
        reference,
        output_types=(output_type,),
    )
    state = _state_for_report(session_id)

    detailed_report, detailed_path = await _render_detailed_report(
        state, states, workspace, tmp_path / "detailed", monkeypatch
    )
    assert "_analysis_state" not in detailed_report
    supervisor_input = dict(detailed_report)
    supervisor_input["_analysis_state"] = state
    supervisor_path = _render_supervisor_report(
        supervisor_input, state, workspace, tmp_path / "supervisor", monkeypatch
    )
    for path in (detailed_path, supervisor_path):
        notebook = nbformat.read(path, as_version=4)
        data = notebook.cells[0].outputs[0].data
        assert data["image/jpeg"] == base64.b64encode(raw).decode("ascii")
        assert _EXTERNAL_IMAGE_MIME not in data

    import backend.config as config_module
    monkeypatch.setattr(config_module, "get_session_states", lambda _sid: states)
    mixin = AnalysisLogMixin()
    mixin.cluster_annotations = {}
    cells = mixin._read_notebook(state)
    assert cells and cells[0]["has_inline_figure"] is True

    invalid_reference = dict(reference)
    invalid_reference["sha256"] = "0" * 64
    _write_external_notebook(
        states,
        session_id,
        invalid_reference,
        output_types=(output_type,),
    )
    invalid_cells = mixin._read_notebook(state)
    assert invalid_cells and invalid_cells[0]["has_inline_figure"] is False
