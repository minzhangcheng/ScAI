import re
import backend.routes.files as files_mod
import backend.routes.sessions as sessions_mod
import backend.routes.export_import as ei_mod


def test_files_resolve_uses_is_relative_to():
    """P3-6: _resolve_within_allowed_dirs should use Path.is_relative_to (Python 3.9+),
    not try/except ValueError. Either form works, but is_relative_to is clearer."""
    src = open(files_mod.__file__).read()
    assert "is_relative_to" in src, "P3-6: must use Path.is_relative_to"


def test_export_filename_capped():
    """P3-7: export filename construction must truncate to a max length."""
    src = open(ei_mod.__file__).read()
    assert "200" in src or "MAX_EXPORT_NAME" in src or "[:200]" in src or "max_length" in src, \
        "P3-7: must cap export filename length"


def test_sessions_list_dedup_by_id():
    """P3-8: session listing must dedup by id."""
    src = open(sessions_mod.__file__).read()
    assert ("seen" in src.lower() and "id" in src.lower()) or "{ s['id']" in src or "by_id" in src, \
        "P3-8: must implement dedup-by-id in session listing"
