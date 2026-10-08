"""Reviewed Chat attachments retain record-bound previews and archival PDFs."""

import base64
import hashlib
from pathlib import Path

import pytest
from PIL import Image

from backend.models.schemas import AnalysisState, FigureReviewStatus, PublicationFigure
from backend.routes.chat_parts import core, figures
from backend.routes.chat_parts.attachments import resolve_figure_references
from backend.services.report import artifacts
from backend.services.report.artifacts import verified_artifact_bytes
from backend.services.session_store import _durable_state_dump
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes
from backend.tests.test_chat_message_identity import _run_simple_chat


@pytest.fixture
def reviewed_figure(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    (workspace / "figures").mkdir(parents=True)
    image_path = workspace / "figures" / "panel.jpg"
    pdf_path = workspace / "figures" / "panel.pdf"
    image_path.write_bytes(valid_image_bytes("JPEG", min_bytes=2200))
    pdf_path.write_bytes(valid_pdf_bytes())
    preview = valid_image_bytes("PNG")
    figure = PublicationFigure(
        name="panel", description="Reviewed panel", figure_type="other",
        png_path=str(image_path), pdf_path=str(pdf_path),
        png_base64=base64.b64encode(preview).decode(),
        render_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        preview_sha256=hashlib.sha256(preview).hexdigest(),
        pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        review_status=FigureReviewStatus.APPROVED, purpose="supplementary",
        producer_run_kind="chat", producer_run_scope_id="chat-current",
        logical_figure_id="panel", is_published_head=True,
    )
    state = AnalysisState(id="chat-attachments", user_id="owner", publication_figures=[figure])
    monkeypatch.setattr(figures, "get_session_workspace", lambda _: workspace)
    monkeypatch.setattr(core, "get_session_workspace", lambda _: workspace)
    monkeypatch.setattr(core, "resolve_published_report_dir", lambda *_: workspace / "reports")
    return state, figure, preview


@pytest.mark.parametrize("preview_format", ["JPEG", "PNG"])
def test_oversized_render_uses_same_record_preview_and_original_pdf(reviewed_figure, preview_format):
    state, figure, _ = reviewed_figure
    preview = valid_image_bytes(preview_format)
    figure.png_base64 = base64.b64encode(preview).decode()
    figure.preview_sha256 = hashlib.sha256(preview).hexdigest()
    image_path = Path(figure.png_path)
    Image.new("RGB", (3917, 4367), (12, 34, 56)).save(image_path, "JPEG")
    original = image_path.read_bytes()
    figure.render_sha256 = hashlib.sha256(original).hexdigest()
    assert verified_artifact_bytes(image_path, figure, kind="image") is None
    for changes in (
        {"id": "previous", "is_published_head": False},
        {"id": "pending", "logical_figure_id": "pending", "review_status": FigureReviewStatus.PENDING},
        {"id": "other-run", "logical_figure_id": "other", "producer_run_scope_id": "chat-other"},
    ):
        state.publication_figures.append(figure.model_copy(update=changes))
    figure.supersedes_id = "previous"

    attached = figures._collect_session_figures(
        state.id, state=state, run_scope_ids=["chat-current"],
    )

    assert len(attached) == 1
    entry = attached[0]
    assert entry["artifact_id"] == figure.id
    assert entry["name"] == figure.name
    assert entry["pdfPath"] == figure.pdf_path
    assert Path(entry["path"]) != image_path
    assert Path(entry["path"]).read_bytes() == preview
    assert verified_artifact_bytes(entry["path"], figure, kind="image") == preview
    assert image_path.read_bytes() == original
    # Durable state strips inline media; the existing preview must suffice.
    restored = AnalysisState.model_validate(_durable_state_dump(state))
    assert restored.publication_figures[0].png_base64 != figure.png_base64
    assert figures._collect_session_figures(
        restored.id, state=restored, run_scope_ids=["chat-current"],
    ) == attached
    # Subsequent Chat retains the logical name, identity and PDF, with no
    # duplicate preview under its physical filename.
    restored.publication_figures = [restored.publication_figures[0]]
    _, inventory = core._build_chat_context(restored)
    assert inventory == attached


@pytest.mark.parametrize("invalid_preview", ["hash_mismatch", "pixel_limit", "missing"])
def test_invalid_preview_cannot_replace_unavailable_render(
    reviewed_figure, monkeypatch, invalid_preview,
):
    state, figure, _ = reviewed_figure
    Path(figure.png_path).write_bytes(b"corrupt render")
    if invalid_preview == "hash_mismatch":
        figure.png_base64 = base64.b64encode(valid_image_bytes(color=(200, 10, 10))).decode()
    elif invalid_preview == "pixel_limit":
        monkeypatch.setattr(artifacts, "MAX_VERIFIED_IMAGE_PIXELS", 255)
    else:
        figure.png_base64 = None

    attached = figures._collect_session_figures(
        state.id, state=state, run_scope_ids=["chat-current"],
    )
    assert len(attached) == 1
    assert attached[0]["pdfPath"] == figure.pdf_path
    assert "path" not in attached[0]
    # An invalid PDF is never attached merely because image delivery failed.
    Path(figure.pdf_path).write_bytes(b"corrupt PDF")
    assert figures._collect_session_figures(
        state.id, state=state, run_scope_ids=["chat-current"],
    ) == []


def test_changed_pdf_is_not_attached_with_valid_image(reviewed_figure):
    state, figure, _ = reviewed_figure
    Path(figure.pdf_path).write_bytes(valid_pdf_bytes(color=(200, 10, 10)))

    attached = figures._collect_session_figures(
        state.id, state=state, run_scope_ids=["chat-current"],
    )

    assert len(attached) == 1
    assert attached[0]["path"] == figure.png_path
    assert "pdfPath" not in attached[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("availability", ["preview", "pdf_only", "image_only"])
@pytest.mark.parametrize("reference", ["panel", "panel.png"])
async def test_followup_reply_preserves_verified_attachments(reviewed_figure, monkeypatch, availability, reference):
    state, figure, _ = reviewed_figure
    if availability == "image_only":
        Path(figure.pdf_path).write_bytes(b"unverified sidecar")
    else:
        Path(figure.png_path).write_bytes(b"unavailable render")
        if availability == "pdf_only":
            figure.png_base64 = None
        else:
            preview = valid_image_bytes("JPEG")
            figure.png_base64 = base64.b64encode(preview).decode()
            figure.preview_sha256 = hashlib.sha256(preview).hexdigest()
    _, inventory = core._build_chat_context(state)
    assert len(inventory) == 1
    if availability == "preview":
        sidecar = Path(inventory[0]["path"]).with_suffix(".pdf")
        if str(sidecar) != figure.pdf_path:
            sidecar.write_bytes(b"unverified sidecar")
    result = await _run_simple_chat(
        monkeypatch, state, figure_inventory=inventory,
        client_content=f"Here is the figure. [[FIGURE:{reference}]] [[FIGURE:panel]]",
    )
    assert result["figures"] == inventory
    assert state.chat_messages[-1]["figures"] == inventory
    assert "were not attached" not in result["content"]
    if availability == "pdf_only":
        assert "path" not in result["figures"][0]
    if availability == "image_only":
        assert "pdfPath" not in result["figures"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("reference, names, expected", [
    ("panel.png", ["panel", "panel.png"], "artifact-1"),
    ("panel.png", ["panel", "panel"], None),
    ("panel", ["panel", "panel"], None),
    ("panel.csv", ["panel"], None),
])
async def test_reference_collision_never_selects_an_arbitrary_artifact(
    monkeypatch, reference, names, expected,
):
    state = AnalysisState(id="chat-figure-collision", user_id="owner")
    inventory = [{"name": name, "artifact_id": f"artifact-{i}", "figure_role": "publication"}
                 for i, name in enumerate(names)]
    result = await _run_simple_chat(
        monkeypatch, state, figure_inventory=inventory,
        client_content=f"[[FIGURE:{reference}]]",
    )
    assert [f["artifact_id"] for f in result.get("figures", [])] == ([expected] if expected else [])
    assert ("were not attached" in result["content"]) is (expected is None)


def test_duplicate_receipts_and_aliases_preserve_one_canonical_attachment():
    figure = {"name": "panel.v2", "artifact_id": "reviewed-id", "pdfPath": "/verified.pdf"}
    selected, unresolved = resolve_figure_references(
        ["panel.v2.png", "panel.v2.pdf", "panel.v2"], [figure, dict(figure)],
    )
    assert selected == [figure] and unresolved == []
    assert selected[0] is not figure
