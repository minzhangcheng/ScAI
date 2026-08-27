"""Failure-injection contracts for publication pair + record transactions."""
from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.models.schemas import AnalysisState, FigureReviewStatus, PublicationFigure
from backend.services.report import figure_transactions as tx
from backend.services.report.artifacts import (
    eligible_artifacts,
    image_file_sha256,
    pdf_file_sha256,
    preview_render_sha256,
)
from backend.tests.figure_image_helpers import valid_image_bytes, valid_pdf_bytes


def _write_pair(jpg: Path, pdf: Path, *, color) -> tuple[bytes, bytes]:
    jpg.parent.mkdir(parents=True, exist_ok=True)
    jpg_bytes = valid_image_bytes("JPEG", color=color, min_bytes=4000)
    pdf_bytes = valid_pdf_bytes(color=color)
    jpg.write_bytes(jpg_bytes)
    pdf.write_bytes(pdf_bytes)
    return jpg_bytes, pdf_bytes


def _record(
    *,
    record_id: str,
    jpg: Path,
    pdf: Path,
    producer_status: str,
    head: bool,
    revision_of_id: str | None = None,
    name: str = "result",
) -> PublicationFigure:
    preview = base64.b64encode(jpg.read_bytes()).decode()
    return PublicationFigure(
        id=record_id,
        name=name,
        description=record_id,
        figure_type="heatmap",
        png_path=str(jpg),
        pdf_path=str(pdf),
        png_base64=preview,
        render_sha256=image_file_sha256(jpg),
        preview_sha256=preview_render_sha256(preview),
        pdf_sha256=pdf_file_sha256(pdf),
        review_status=FigureReviewStatus.APPROVED,
        logical_figure_id=f"logical-{name}",
        render_attempt_id=record_id,
        producer_status=producer_status,
        revision_of_id=revision_of_id,
        is_published_head=head,
    )


def _setup_attempt(
    tmp_path: Path,
    attempt_id: str = "attempt-new",
    figure_name: str = "result",
):
    figures = tmp_path / "figures"
    final_jpg = figures / f"{figure_name}.jpg"
    final_pdf = figures / f"{figure_name}.pdf"
    old_jpg, old_pdf = _write_pair(final_jpg, final_pdf, color=(80, 10, 10))
    old = _record(
        record_id=f"old-head-{figure_name}",
        jpg=final_jpg,
        pdf=final_pdf,
        producer_status="published",
        head=True,
        name=figure_name,
    )
    attempt_dir = figures / ".attempts" / attempt_id
    staged_jpg = attempt_dir / f"{figure_name}.jpg"
    staged_pdf = attempt_dir / f"{figure_name}.pdf"
    new_jpg, new_pdf = _write_pair(staged_jpg, staged_pdf, color=(10, 80, 10))
    new = _record(
        record_id=attempt_id,
        jpg=staged_jpg,
        pdf=staged_pdf,
        producer_status="pending",
        head=False,
        revision_of_id=old.id,
        name=figure_name,
    )
    state = AnalysisState(
        id="session-tx",
        user_id="user",
        publication_figures=[old, new],
    )
    journal = tx.register_pending_figure_attempt(
        attempt_dir=attempt_dir,
        staged_jpg=staged_jpg,
        staged_pdf=staged_pdf,
        final_jpg=final_jpg,
        final_pdf=final_pdf,
        published_record=new,
        session_id=state.id,
    )
    return {
        "figures": figures,
        "final_jpg": final_jpg,
        "final_pdf": final_pdf,
        "staged_jpg": staged_jpg,
        "staged_pdf": staged_pdf,
        "attempt_dir": attempt_dir,
        "old_jpg": old_jpg,
        "old_pdf": old_pdf,
        "new_jpg": new_jpg,
        "new_pdf": new_pdf,
        "old": old,
        "new": new,
        "state": state,
        "journal": journal,
    }


def _begin(case, *, archived=None):
    return tx.begin_figure_publish_transaction(
        attempt_dir=case["attempt_dir"],
        staged_jpg=case["staged_jpg"],
        staged_pdf=case["staged_pdf"],
        final_jpg=case["final_jpg"],
        final_pdf=case["final_pdf"],
        published_record=case["new"],
        archived_record=case["old"] if archived is None else archived,
    )


def _refreeze(case, transaction=None):
    return tx.refreeze_pending_figure_attempt(
        transaction or case["journal"],
        attempt_dir=case["attempt_dir"],
        staged_jpg=case["staged_jpg"],
        staged_pdf=case["staged_pdf"],
        final_jpg=case["final_jpg"],
        final_pdf=case["final_pdf"],
        published_record=case["new"],
    )


def _apply_record(case, transaction, *, archived=None):
    old = case["old"] if archived is None else archived
    new = case["new"]
    tx.apply_publish_record_paths(transaction, new, old)
    new.producer_status = "published"
    new.render_sha256 = image_file_sha256(new.png_path)
    new.pdf_sha256 = pdf_file_sha256(new.pdf_path)
    new.supersedes_id = old.id
    new.is_published_head = True
    old.is_published_head = False
    tx.mark_publish_record_applied(
        transaction, published_record=new, archived_record=old
    )


def _assert_old_restored(case, *, terminal: str):
    assert case["final_jpg"].read_bytes() == case["old_jpg"]
    assert case["final_pdf"].read_bytes() == case["old_pdf"]
    assert case["old"].is_published_head is True
    assert case["new"].producer_status == terminal
    assert case["new"].png_base64 is None
    assert case["new"].png_path == ""
    assert eligible_artifacts(case["state"].publication_figures) == [case["old"]]
    assert not case["attempt_dir"].exists()


__all__ = [name for name in globals() if not name.startswith("__")]
