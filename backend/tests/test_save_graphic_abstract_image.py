"""save_graphic_abstract_image must flatten RGBA/palette images to RGB before
writing JPEG (raw byte-dump produced invalid .jpg on the live supervisor path)."""
import base64
import io

from PIL import Image

from backend.utils.image_utils import save_graphic_abstract_image


def _rgba_png_b64() -> str:
    img = Image.new("RGBA", (8, 8), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_none_for_empty_input(tmp_path):
    assert save_graphic_abstract_image("", tmp_path) is None


def test_rgba_is_flattened_and_saved_as_jpeg(tmp_path):
    out = save_graphic_abstract_image(_rgba_png_b64(), tmp_path)
    assert out is not None
    saved = Image.open(out)
    assert saved.format == "JPEG"
    assert saved.mode == "RGB"  # alpha flattened, not a raw byte dump
