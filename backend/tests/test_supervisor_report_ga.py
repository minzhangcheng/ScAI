"""图形摘要必须从 analysis_state.graphic_abstract 落盘到报告目录，
并在 report dict 中暴露 graphic_abstract_path。"""
import base64
from pathlib import Path

from backend.agents.supervisor_agent import persist_graphic_abstract


class _GA:
    def __init__(self, b64): self.image_base64 = b64; self.text_diagram = ""


def test_persists_ga_image(tmp_path: Path):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, "PNG")
    ga = _GA(base64.b64encode(buf.getvalue()).decode())
    path = persist_graphic_abstract(ga, tmp_path)
    assert path is not None
    assert Path(path).exists()
    saved = Image.open(path)
    assert saved.format == "JPEG"  # valid JPEG, not a raw byte dump
    assert saved.mode == "RGB"


def test_returns_none_for_text_mode(tmp_path: Path):
    assert persist_graphic_abstract(_GA(""), tmp_path) is None


def test_returns_none_when_absent(tmp_path: Path):
    assert persist_graphic_abstract(None, tmp_path) is None


def test_persists_raises_on_unopenable_image(tmp_path: Path):
    # The shared PIL saver raises on non-image bytes. The report call-site
    # (generate_detailed_report) wraps persist_graphic_abstract in try/except
    # so a malformed graphic abstract degrades gracefully (ga_path=None) instead
    # of crashing report generation. This test pins the raise contract that the
    # call-site guard relies on (see M1 hardening).
    ga = _GA(base64.b64encode(b"\xff\xd8\xff\xe0fakejpeg").decode())
    raised = False
    try:
        persist_graphic_abstract(ga, tmp_path)
    except Exception:
        raised = True
    assert raised, (
        "persist_graphic_abstract must raise on unopenable image bytes — "
        "the report call-site guard relies on this to degrade gracefully"
    )
