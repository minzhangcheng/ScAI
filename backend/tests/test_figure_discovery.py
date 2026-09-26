"""Tests for the Results brief-gallery figure discovery helper."""
import base64
import hashlib
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from backend.services.report.figure_discovery import (
    ensure_legacy_artifact_inventory,
    prepare_discovered_figures,
    resolve_state_figure_candidates,
)
from backend.models.schemas import AnalysisState
from backend.services.report.artifacts import (
    ARTIFACT_REGISTRY_VERSION,
)
from backend.tests.figure_image_helpers import valid_pdf_bytes


def _image_bytes(noise: float = 100) -> bytes:
    buffer = io.BytesIO()
    Image.effect_noise((256, 256), noise).convert("RGB").save(buffer, format="PNG")
    payload = buffer.getvalue()
    assert len(payload) > 2000
    return payload


def _render(path: Path, payload: bytes | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload or _image_bytes())
    return path


def _publication_figure(name, *, status="approved", figure_type="barplot",
                        description="authoritative description", **overrides):
    png_path = overrides.pop("png_path", f"/workspace/figures/{name}.png")
    render_sha256 = overrides.pop("render_sha256", None)
    if render_sha256 is None and Path(png_path).is_file():
        render_sha256 = hashlib.sha256(Path(png_path).read_bytes()).hexdigest()
    return SimpleNamespace(
        name=name,
        png_path=png_path,
        png_base64=overrides.pop("png_base64", None),
        render_sha256=render_sha256,
        preview_sha256=overrides.pop("preview_sha256", None),
        pdf_sha256=overrides.pop("pdf_sha256", "2" * 64),
        csv_path=None,
        figure_type=figure_type,
        description=description,
        interpretation="",
        diagnostic=False,
        round_number=1,
        review_status=status,
        is_current=True,
        invalidated_at=None,
        layout_violations=[],
        **overrides,
    )


def test_same_stem_overwrite_must_match_registered_render_hash(tmp_path):
    original = _image_bytes()
    original_hash = hashlib.sha256(original).hexdigest()
    path = _render(tmp_path / "bound.png", original)
    registered = _publication_figure(
        "bound",
        png_path=str(path),
        png_base64=base64.b64encode(original).decode(),
        render_sha256=original_hash,
        preview_sha256=original_hash,
    )
    # A later generic save uses the reviewed stem for different bytes.
    replacement = _image_bytes(noise=20)
    assert hashlib.sha256(replacement).hexdigest() != original_hash
    path.write_bytes(replacement)
    raw = [{
        "name": "bound", "type": "other", "path": str(path),
        "pdf_path": None, "mtime": 2.0, "source": "figures",
        "rel_path": "figures/bound.png", "diagnostic": False,
    }]

    assert prepare_discovered_figures(raw, [registered]) == []


def test_strict_registry_excludes_unregistered_images_without_mtime_guessing(tmp_path):
    path = _render(tmp_path / "strict.png")
    os.utime(path, (1.0, 1.0))
    raw = [{
        "name": "strict", "type": "other", "path": str(path),
        "pdf_path": None, "mtime": 1.0, "source": "figures",
        "rel_path": "figures/strict.png", "diagnostic": False,
    }]

    assert prepare_discovered_figures(
        raw,
        [],
        registry_version=ARTIFACT_REGISTRY_VERSION,
    ) == []


@pytest.mark.parametrize("pdf_existed_at_freeze", [False, True])
def test_frozen_legacy_image_cannot_adopt_an_added_or_replaced_pdf(
    tmp_path, pdf_existed_at_freeze,
):
    workspace = tmp_path / "workspace"
    image = _render(workspace / "figures" / "legacy.png")
    pdf = image.with_suffix(".pdf")
    if pdf_existed_at_freeze:
        pdf.write_bytes(valid_pdf_bytes(color=(10, 20, 30)))
    state = AnalysisState(id="legacy-session", user_id="u")
    ensure_legacy_artifact_inventory(state, workspace)

    pdf.write_bytes(valid_pdf_bytes(color=(200, 30, 20)))
    candidates = resolve_state_figure_candidates(
        state, workspace, report_only=False
    )

    assert len(candidates) == 1
    assert candidates[0].get("pdf_path") in (None, "")
    assert candidates[0].get("pdf_sha256") is None
